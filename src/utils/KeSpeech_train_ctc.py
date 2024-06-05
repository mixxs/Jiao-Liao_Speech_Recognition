"""
获取
wav2vec/wav2vec2_ASRTrained_model.pt
将这个模型的lm head和wav2vec分开得到下面两个文件
wav2vec/wav2vec2_ASRTrained_LMHead.pt
wav2vec/wav2vec2_ASRTrained_Wav2Vec2Model.pt
"""
import json
import logging
import os
import sys

import torch
import transformers
from transformers import HfArgumentParser, TrainingArguments, Wav2Vec2Config, set_seed
from transformers.trainer_utils import get_last_checkpoint, is_main_process

from src.models.model import init_model
from src.utils.data.KeSpeech.dataset import DataCollatorCTCWithPadding, make_hugging_face_datasets
from src.utils.data.process import make_processor
from src.utils.fileIO.model import save_model
from src.utils.forTrain.Trainer import CTCTrainer
from src.utils.forTrain.arguments import ModelArguments, DataTrainingArguments, AdditionalTrainingArguments
from src.utils.result.metric import compute_cer, compute_wer
from src.utils.result.metric_io import save_metric

logger = logging.getLogger(__name__)


def run_train(
        DatasetJsonPath,
        Processor,
        languages: list,
        DatasetDir: str,
        pretrained_model_path: str,
        ModelConfigPath, ):
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, AdditionalTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, additional_training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, additional_training_args = parser.parse_args_into_dataclasses()
    data_args.max_val_samples = 500
    training_args = TrainingArguments(
        output_dir="../../weights/KeSpeechASR/CTC",
        do_train=True, do_eval=True, overwrite_output_dir=True, num_train_epochs=5,
        per_device_train_batch_size=6, per_device_eval_batch_size=1, learning_rate=2e-4, warmup_ratio=0.1,
        save_steps=1000, eval_steps=5000, evaluation_strategy="steps", save_total_limit=1,
        label_names=["labels"], dataloader_num_workers=12, seed=42,
        logging_steps=1000, dataloader_drop_last=True)
    os.makedirs(training_args.output_dir, exist_ok=True)
    os.makedirs("./wandb_cache", exist_ok=True)
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

    datasets = make_hugging_face_datasets(
        data_json=DatasetJsonPath,
        languages=languages,
        subsets=["train", "test"],
        dataset_dir=DatasetDir,
        num_proc=data_args.preprocessing_num_workers,
        processor=Processor, merge=True)
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
        vocab = set(Processor.tokenizer.encoder.keys())
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
    Processor.save_pretrained(training_args.output_dir)

    def compute_metrics(pred):
        pred_ids = pred.predictions
        asr_label = pred.label_ids
        print(asr_label.shape)  # batch_size, len
        print(pred_ids.shape)  # batch_size * len

        del pred

        asr_label[asr_label == -100] = Processor.tokenizer.pad_token_id  # 处理填充
        print(Processor.tokenizer.word_delimiter_token_id)
        pred_str = Processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = Processor.batch_decode(asr_label, group_tokens=False)
        print("\nlabel: ", label_str[0])
        print("\npred: ", pred_str[0])

        print("长度和类型：")
        print(len(label_str), type(label_str))
        print(len(pred_str), type(pred_str))
        cer = compute_cer(label_str, pred_str)
        wer = compute_wer(label_str, pred_str)
        print("\ncer", cer)
        print("\nwer", wer)

        torch.cuda.empty_cache()
        save_metric("cer", float(cer), save_dir=training_args.output_dir)
        save_metric("wer", float(wer), save_dir=training_args.output_dir)
        return {"cer": cer, "wer": wer}

    with open(ModelConfigPath, "r", encoding="utf-8") as f:
        model_config = json.load(f)
    model_config = Wav2Vec2Config.from_dict(model_config)
    model = init_model(
        model_name="ctc",
        processor=Processor,
        model_args=model_args,
        encoder_dir_path=pretrained_model_path,
        model_config=model_config,
    )
    pt_model_path = os.path.join(training_args.output_dir, "model.pt")
    if os.path.exists(pt_model_path):
        model.load_state_dict(torch.load(pt_model_path))
        print(f"\n\nload model state_dict from {pt_model_path}\n\n")
    else:
        print(f"\n\ninitialize model\n\n")

    if model_args.freeze_feature_extractor:
        print("\n\n冻结特征提取器\n\n")
        model.freeze_feature_extractor()

    save_model(model, os.path.join(training_args.output_dir, "model.pt"), save_state_dict=True)
    model.load_state_dict(torch.load(os.path.join(training_args.output_dir, "model.pt")))
    # Data collator
    data_collator = DataCollatorCTCWithPadding(
        processor=Processor,
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
        tokenizer=Processor.feature_extractor,
    )

    # Training
    if training_args.do_train:
        checkpoint = os.path.join(training_args.output_dir, "checkpoint-last")
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
    from src.utils.data.KeSpeech.KeSpeech_metadata import DIALECT_LS

    dataset_json_path = "../../dataset_metadata/KeSpeech_data_metadata/KeSpeechDataset.json"
    processor = make_processor(True, pretrainModel_path="../../weights/KeSpeechASR")
    dataset_dir = "/input0"
    pretrain_model_path = "../../weights/wav2vec2_zhCN"
    model_config_path = "../../weights/wav2vec2_zhCN/config.json"
    run_train(
        DatasetJsonPath=dataset_json_path,
        Processor=processor,
        languages=DIALECT_LS,
        DatasetDir=dataset_dir,
        pretrained_model_path=pretrain_model_path,
        ModelConfigPath=model_config_path,
    )

"""
lr_warmup_ratio=0.2, lr_constant_ratio=0.2
超参数设置：先是训练了5个epoch，然后又将epoch设置为8,训练了3个epoch
"""
