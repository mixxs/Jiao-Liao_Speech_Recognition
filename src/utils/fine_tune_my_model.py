import json
import logging
import os
import sys
import warnings

import torch
import torchaudio
import transformers
from transformers import HfArgumentParser, TrainingArguments, Wav2Vec2Config, set_seed
from transformers.trainer_utils import get_last_checkpoint, is_main_process

from src.models.change_wav2vec import add_wf_adapter
from src.models.ctc_aid.wav2vec2CTC_ecapatdnnAID import Wav2Vec2_CTC_ECAPA_AID
from src.models.model import init_model
from src.utils.data.KeSpeech.dataset import DataCollatorCTCWithPadding
from src.utils.data.KeSpeech.dataset import make_hugging_face_datasets as make_kespeech_datasets
from src.utils.data.MySpeech.dataset import make_hugging_face_datasets as make_myspeech_datasets
from src.utils.data.process import make_processor
from src.utils.fileIO.model import save_model
from src.utils.forTrain.Trainer import CTCTrainer
from src.utils.forTrain.arguments import ModelArguments, DataTrainingArguments, AdditionalTrainingArguments
from src.utils.result.best import get_best, save_if_best
from src.utils.result.metric import compute_cer, compute_wer
from src.utils.result.metric_io import save_metric

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")


def run_train(
        myspeech_dataset_json_path,
        kespeech_dataset_json_path,
        processor,
        myspeech_dataset_dir: str,
        kespeech_dataset_dir: str,
        pretrained_model_dir: str,
        pretrained_model_weight_path: str,
        ecapa_model_path: str,
        model_config_path,
        adapter_embed_dim: int,
        language: str,
        fine_tune_stage: int,
        dialect_feature: torch.Tensor = None,

):
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, AdditionalTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, additional_training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, additional_training_args = parser.parse_args_into_dataclasses()

    if language.lower() == "myspeech":
        output_dir = "../../weights/MySpeechASR/fine_tune_mymodel"
    else:
        output_dir = "../../weights/KeSpeechASR/fine_tune_mymodel"

    if fine_tune_stage == 1:
        output_dir = os.path.join(output_dir, "fine_tune_stage_one")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
    else:
        last_stage_output_dir = os.path.join(output_dir, "fine_tune_stage_one")
        output_dir = os.path.join(output_dir, "fine_tune_stage_two")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        if not os.path.exists(os.path.join(output_dir, "model.pt")):
            import shutil
            shutil.copy(os.path.join(last_stage_output_dir, "model.pt"), os.path.join(output_dir, "model.pt"))

    training_args = TrainingArguments(
        output_dir=output_dir,
        do_train=True, do_eval=True, overwrite_output_dir=True, num_train_epochs=10,
        per_device_train_batch_size=3, per_device_eval_batch_size=1, learning_rate=1e-4, warmup_ratio=0.1,
        save_steps=1000, eval_steps=2000, evaluation_strategy="steps", save_total_limit=1,
        label_names=["labels", "lengths", "input_values", "attention_mask"], dataloader_num_workers=12, seed=42,
        logging_steps=1000, dataloader_drop_last=True, save_safetensors=False)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)
    # Set seed before initializing model.

    set_seed(training_args.seed)
    if language.lower() == "myspeech":
        datasets = make_myspeech_datasets(
            data_json=myspeech_dataset_json_path,
            subsets=["train", "test"],
            dataset_dir=myspeech_dataset_dir,
            num_proc=data_args.preprocessing_num_workers,
            proc=processor)
    else:
        datasets = make_kespeech_datasets(
            data_json=kespeech_dataset_json_path,
            languages=[language, ],
            subsets=["train", "test"],
            dataset_dir=kespeech_dataset_dir,
            num_proc=data_args.preprocessing_num_workers,
            processor=processor,
            merge=True
        )
    train_dataset = datasets["train"]
    test_dataset = datasets["test"]
    del datasets
    # Filtering dataset:

    train_dataset_size = train_dataset.num_rows
    test_dataset_size = test_dataset.num_rows

    if data_args.max_train_samples is not None:
        if train_dataset_size > data_args.max_train_samples:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
    if data_args.max_val_samples is not None:
        if test_dataset_size > data_args.max_val_samples:
            test_dataset = test_dataset.select(range(data_args.max_val_samples))

    train_dataset_final_size = train_dataset.num_rows
    test_dataset_final_size = test_dataset.num_rows

    logger.info(
        f"After filtering {train_dataset_final_size} of {train_dataset_size} samples will be used to train the model")
    logger.info(
        f"After filtering {test_dataset_final_size} of {test_dataset_size} samples will be used to eval the model")

    if additional_training_args.remove_samples_with_oov_from_training:
        vocab = set(processor.tokenizer.encoder.keys())
        train_dataset_size = train_dataset_final_size
        train_dataset = train_dataset.filter(
            lambda example: vocab.issuperset(example["text"].replace(" ", "")),
            num_proc=data_args.preprocessing_num_workers
        )
        train_dataset_final_size = len(train_dataset)
        print(
            f"OOV found in {train_dataset_size - train_dataset_final_size} samples, and they were removed from training set")
        print(f"The final training set size is {train_dataset_final_size}")

    # save the feature_extractor and the tokenizer
    processor.save_pretrained(training_args.output_dir)

    with open(model_config_path, "r", encoding="utf-8") as f:
        model_config = json.load(f)
    model_config = Wav2Vec2Config.from_dict(model_config)
    if dialect_feature is not None:
        dialect_feature = dialect_feature.cuda()
        print(dialect_feature.shape)
        print("加载了语言特征")
    model: Wav2Vec2_CTC_ECAPA_AID = init_model(
        model_name="wav2vec2_ctc_ecapatdnn_aid",
        processor=processor,
        model_args=model_args,
        encoder_dir_path=pretrained_model_dir,
        model_config=model_config,
        aid_model_dir_path=ecapa_model_path,
        dialect_feature=dialect_feature)
    # 添加adapter

    model = add_wf_adapter(model, model_config, adapter_embed_dim=adapter_embed_dim,
                           language_num=1, layer_num=3, freeze=False)
    if os.path.exists(pretrained_model_weight_path):
        model.load_state_dict(torch.load(pretrained_model_weight_path))
        print(f"从{pretrained_model_weight_path}中加载了预先训练的模型的参数")
    del model.aid
    # 从检查点加载模型
    pt_model_path = os.path.join(training_args.output_dir, "model.pt")
    if os.path.exists(pt_model_path):
        model.load_state_dict(torch.load(pt_model_path))
        print(f"\n\nload model state_dict from {pt_model_path}\n\n")
    else:
        print(f"\n\ninitialize model\n\n")

    if fine_tune_stage == 1:
        for param in model.parameters():
            param.requires_grad = True
        model.freeze_share_weights()
    else:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.att_adapter.parameters():
            param.requires_grad = True
        for param in model.wav2vec2.adapter.parameters():
            param.requires_grad = True
        for param in model.lm_head.parameters():
            param.requires_grad = True

    # 设置language_id
    language_id = 0
    print(f"\n\nlanguage_id:{language_id}")
    model.set_language_id(language_id)
    model.cuda()

    for name, param in model.named_parameters():
        print("需要梯度下降？  ", param.requires_grad, "   ", name, "   ", param.device)
    # Data collator

    data_collator = DataCollatorCTCWithPadding(
        processor=processor,
        padding=True,
        apply_gaussian_noise_with_p=additional_training_args.apply_gaussian_noise_with_p,
        apply_gain_with_p=additional_training_args.apply_gain_with_p,
        apply_pitch_shift_with_p=additional_training_args.apply_pitch_shift_with_p,
        apply_time_stretch_with_p=additional_training_args.apply_time_stretch_with_p,
        sample_rate=16_000,
    )

    def compute_metrics(pred):
        pred_ids = pred.predictions
        asr_label = pred.label_ids
        print(asr_label.shape)  # batch_size, len
        print(pred_ids.shape)  # batch_size * len

        del pred

        asr_label[asr_label == -100] = processor.tokenizer.pad_token_id  # 处理填充
        print(processor.tokenizer.word_delimiter_token_id)
        pred_str = processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(asr_label, group_tokens=False)
        print("\nfirst label: ", label_str[0])
        print("\nfirst pred: ", pred_str[0])
        cer = compute_cer(label_str, pred_str)
        wer = compute_wer(label_str, pred_str)
        save_if_best(model, cer, wer, training_args.output_dir)
        print("\ncer", cer)
        print("\nwer", wer)
        torch.cuda.empty_cache()

        save_metric("cer", float(cer), save_dir=training_args.output_dir)
        save_metric("wer", float(wer), save_dir=training_args.output_dir)
        return {"cer": cer, "wer": wer}

    # Initialize our Trainer
    trainer = CTCTrainer(
        model_output_dir=training_args.output_dir,
        length_field_name="length",
        upload_model_to_wandb_each_step=additional_training_args.upload_model_to_wandb_each_step,
        lr_warmup_ratio=additional_training_args.lr_warmup_ratio,
        lr_constant_ratio=additional_training_args.lr_constant_ratio,
        sampling_rate=16_000,
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=processor.feature_extractor,
    )
    # exit(0)
    # Training
    if training_args.do_train:
        checkpoint = os.path.join(training_args.output_dir, "checkpoint-last")
        if not os.path.exists(checkpoint):
            checkpoint = None
        if checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        save_model(model, os.path.join(training_args.output_dir, "model.pt"), save_state_dict=True)
        metrics = train_result.metrics
        print(f"\n\n{metrics}")

        metrics["train_samples"] = train_dataset_final_size

        trainer.log_metrics(split=f" train", metrics=metrics)
        trainer.save_metrics(split=f" train", metrics=metrics)
        trainer.save_state()

    # Evaluation
    metrics = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        metrics["eval_samples"] = test_dataset_final_size
        trainer.log_metrics(split=" eval", metrics=metrics)
        trainer.save_metrics(split=" eval", metrics=metrics)
    best_cer, best_wer = get_best(training_args.output_dir)
    print(f"cer of best model : {best_cer}")
    print(f"wer of best model : {best_wer}")


if __name__ == "__main__":
    torchaudio.set_audio_backend("soundfile")
    MySpeech_dataset_json_path = "../../dataset_metadata/MySpeech_data_metadata/MySpeechDataset.json"
    KeSpeech_dataset_json_path = "../../dataset_metadata/KeSpeech_data_metadata/KeSpeechDataset_shrinked.json"
    Processor = make_processor(True, pretrainModel_path="../../weights/MySpeechASR")
    KeSpeech_dataset_dir = "/input0/"
    MySpeech_dataset_dir = "/input1/"
    # dataset_dir = r"F:\语音数据集\MySpeech"

    pretrainModelDir = "../../weights/wav2vec2_zhCN"
    pretrainModelWeightPath = "../../models/my_Model/my_model.pt"
    ecapaModelPath = "../../weights/SpeechBrain/model"
    modelConfigPath = "../../weights/wav2vec2_zhCN/config.json"

    dialectFeature = torch.tensor(torch.load("../../models/my_Model/mean_dialect_feature.pt"))

    run_train(
        myspeech_dataset_json_path=MySpeech_dataset_json_path,
        kespeech_dataset_json_path=KeSpeech_dataset_json_path,
        processor=Processor,
        myspeech_dataset_dir=MySpeech_dataset_dir,
        kespeech_dataset_dir=KeSpeech_dataset_dir,
        pretrained_model_dir=pretrainModelDir,
        pretrained_model_weight_path=pretrainModelWeightPath,
        ecapa_model_path=ecapaModelPath,
        model_config_path=modelConfigPath,
        adapter_embed_dim=512,
        language="MySpeech",
        fine_tune_stage=1,
        dialect_feature=dialectFeature)
