"""
获得my_Model/my_model_pro.pt
"""

import json
import logging
import os
import sys

import torch
import transformers
import wandb
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Wav2Vec2Config,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process, set_seed

from src.models.change_wav2vec import add_wf_adapter
from src.models.ctc_aid.wav2vec2CTC_ecapatdnnAID import Wav2Vec2_CTC_ECAPA_AID
from src.models.model import init_model
from src.utils.data.KeSpeech.KeSpeech_metadata import get_language_id, DIALECT_LS
from src.utils.data.KeSpeech.dataset import DataCollatorCTCWithPadding
from src.utils.data.KeSpeech.dataset import make_hugging_face_datasets as make_kespeech_dataset
from src.utils.data.MySpeech.dataset import make_hugging_face_datasets as make_myspeech_dataset
from src.utils.data.process import make_processor
from src.utils.fileIO.model import save_model
from src.utils.forTrain.Trainer import CTCTrainer
from src.utils.forTrain.arguments import DataTrainingArguments, ModelArguments, AdditionalTrainingArguments
from src.utils.result.metric import compute_cer, compute_wer
from src.utils.result.metric_io import save_metric

logger = logging.getLogger(__name__)


def run_train(
        kespeech_dataset_json_path,
        myspeech_dataset_json_path,
        processor,
        all_languages: list,
        kespeech_dataset_dir: str,
        myspeech_dataset_dir: str,
        pretrainedModelDirPath: str,
        ecapaModelDirPath: str,
        adapter_dim,
        modelConfigPath,
        model_before_adapt_weight_path: str = None,
        language_now: str = None,
):
    """
    :datasetMetaDirPath:数据集的源文件

    """

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, AdditionalTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, additional_training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, additional_training_args = parser.parse_args_into_dataclasses()

    training_args = TrainingArguments(
        output_dir="../../../weights/KeSpeechASR/KeSpeech_train_mymodel",
        do_train=True, do_eval=True, overwrite_output_dir=True, num_train_epochs=5,
        per_device_train_batch_size=3, per_device_eval_batch_size=1, evaluation_strategy="steps"
        , learning_rate=1e-4, warmup_steps=10000, warmup_ratio=0.1, eval_steps=2000, save_steps=1000,
        save_total_limit=1, label_names=["labels", "lengths", "input_values"],
        logging_steps=1000, dataloader_drop_last=True)
    os.makedirs(training_args.output_dir, exist_ok=True)
    os.makedirs("../wandb_cache", exist_ok=True)

    wandb.init(dir="../wandb_cache")

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

    set_seed(training_args.seed)
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    if language_now is None:
        language_now = all_languages
    else:
        language_now = [language_now]
    kespeech_datasets = make_kespeech_dataset(
        data_json=kespeech_dataset_json_path,
        languages=language_now,
        subsets=["train", "test"],
        dataset_dir=kespeech_dataset_dir,
        num_proc=data_args.preprocessing_num_workers, processor=processor,
        merge=True)
    myspeech_datasets = make_myspeech_dataset(
        data_json=myspeech_dataset_json_path,
        subsets=["train", "test"],
        dataset_dir=myspeech_dataset_dir,
        num_proc=data_args.preprocessing_num_workers,
        proc=processor)

    kespeech_train_dataset = kespeech_datasets["train"]
    kespeech_test_dataset = kespeech_datasets["test"]

    myspeech_train_dataset = myspeech_datasets["train"]
    myspeech_test_dataset = myspeech_datasets["test"]

    del kespeech_datasets, myspeech_datasets
    from datasets import concatenate_datasets

    train_dataset = concatenate_datasets([kespeech_train_dataset, myspeech_train_dataset])
    test_dataset = concatenate_datasets([kespeech_test_dataset, myspeech_test_dataset])
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

    processor.save_pretrained(training_args.output_dir)

    def compute_metrics(pred):
        pred_ids = pred.predictions
        asr_label = pred.label_ids
        print(asr_label.shape)  # batch_size, len
        print(pred_ids.shape)  # batch_size * len

        del pred

        asr_label[asr_label == -100] = processor.tokenizer.pad_token_id  # 处理填充
        print(processor.tokenizer.word_delimiter_token_id)
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(asr_label, group_tokens=False)
        print("\nlabel: ", label_str[0])
        print("\npred: ", pred_str[0])

        print("长度和类型：")
        print(len(label_str), type(label_str))
        print(len(pred_str), type(pred_str))
        cer = compute_cer(label_str, pred_str)
        wer = compute_wer(label_str, pred_str)
        print("\ncer", cer)
        print("\nwer", wer)

        # upload_model_to_wandb(training_args.output_dir, name=f"{wandb.run.name}_eval",
        #                       metadata={"cer": float(cer), "wer": wer})
        torch.cuda.empty_cache()
        save_metric("cer", float(cer), save_dir=training_args.output_dir)
        save_metric("wer", float(wer), save_dir=training_args.output_dir)
        return {"cer": cer, "wer": wer}

    with open(modelConfigPath, "r", encoding="utf-8") as f:
        model_config = json.load(f)
    model_config = Wav2Vec2Config.from_dict(model_config)
    checkpoint_model_path = os.path.join(training_args.output_dir, "model.pt")

    model: Wav2Vec2_CTC_ECAPA_AID = init_model(
        model_name="wav2vec2_ctc_ecapatdnn_aid",
        processor=processor,
        model_args=model_args,
        encoder_dir_path=pretrainedModelDirPath,
        model_config=model_config,
        aid_model_dir_path=ecapaModelDirPath)

    if os.path.exists(model_before_adapt_weight_path):
        model.load_state_dict(torch.load(model_before_adapt_weight_path))
        print(f"从{model_before_adapt_weight_path}中加载了添加adapter前的模型参数")

    # 指定可训练的参数
    for param in model.parameters():
        param.requires_grad = True
    for param in model.aid.parameters():
        param.requires_grad = False
    model.freeze_feature_extractor()

    # 添加adapter
    if len(language_now) > 1:
        adapter_num = 1
    else:
        adapter_num = len(all_languages)
    model = add_wf_adapter(
        myWav2vec=model,
        config=model_config,
        adapter_embed_dim=adapter_dim, language_num=adapter_num,
        layer_num=3,
        freeze=False)

    if os.path.exists(checkpoint_model_path):
        model.load_state_dict(torch.load(checkpoint_model_path))
        print(f"\n\nload model state_dict from {checkpoint_model_path}\n\n")
    else:
        print("\n\ninitialize model\n\n")

    # 设置language_id
    print(f"\n\n\n {len(language_now) > 1}\n\n\n")
    print(language_now)
    if len(language_now) > 1:
        language_id = 0
        print(language_id)
    else:
        language_id = get_language_id(language_now[0], all_languages)
    print(f"\n\nlanguage_id:{language_id}")
    model.set_language_id(language_id)
    model.cuda()
    for name, param in model.named_parameters():
        print("需要梯度下降？  ", param.requires_grad, "   ", name)

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


if __name__ == "__main__":
    kespeech_dataset_json_path = "../../dataset_metadata/KeSpeech_data_metadata/KeSpeechDataset_shrinked.json"
    myspeech_dataset_json_path = "../../dataset_metadata/MySpeech_data_metadata/MySpeechDataset.json"
    processor = make_processor(True, "../../../weights/KeSpeechASR")

    kespeech_dataset_dir = "/input0"
    myspeech_dataset_dir = "/input1"

    pretrain_model_dir = "../../weights/wav2vec2_zhCN"  # 用于生成wav2vec2
    ecapa_model_dir = "../../weights/SpeechBrain/model"
    pretrain_model_weight_path = ""
    model_config_path = "../../weights/wav2vec2_zhCN/config.json"
    languages = DIALECT_LS

    run_train(
        kespeech_dataset_json_path=kespeech_dataset_json_path,
        myspeech_dataset_json_path=myspeech_dataset_json_path,
        processor=processor,
        all_languages=languages,
        kespeech_dataset_dir=kespeech_dataset_dir,
        myspeech_dataset_dir=myspeech_dataset_dir,
        pretrainedModelDirPath=pretrain_model_dir,
        ecapaModelDirPath=ecapa_model_dir,
        modelConfigPath=model_config_path,
        model_before_adapt_weight_path="../../models/ctc_aid/wav2vec_ctc_ecapa_aid_pro.pt",
        language_now=None,
        adapter_dim=512)

    """
    训练模型，以适应adapter和融合模块的加入
    共三个adapter块
    """
