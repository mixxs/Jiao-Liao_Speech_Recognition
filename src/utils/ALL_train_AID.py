"""
获取ecapa/siamese_pro.pt和ecapa/ecapa_trained_model_pro.pt
"""
import logging
import os
import sys

import torch
import transformers
import wandb
from datasets import concatenate_datasets
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Wav2Vec2Processor
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process, set_seed

from src.models.aid.SiameseEcapaTDNN import Siamese_ECAPA_TDNN
from src.utils.data.KeSpeech.dataset import DataCollatorTripletWithPadding

from src.utils.data.KeSpeech.make_Triplet_dataset import getTripletDataset
from src.utils.data.process import make_processor
from src.utils.fileIO.model import save_model
from src.utils.forTrain.Trainer import upload_model_to_wandb, ECAPATDNNTripletTrainer
from src.utils.forTrain.arguments import DataTrainingArguments, ModelArguments, AdditionalTrainingArguments
from src.utils.result.metric import compute_acc
from src.utils.result.metric_io import save_metric

logger = logging.getLogger(__name__)


def run_train(
        KeSpeechDatasetMetaDirPath,
        MySpeechDatasetMetaDirPath,
        processor: Wav2Vec2Processor,
        KeSpeechDatasetBaseDir: str,
        MySpeechDatasetBaseDir,
        pretrained_model_path: str):
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, AdditionalTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, additional_training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, additional_training_args = parser.parse_args_into_dataclasses()

    training_args = TrainingArguments(
        output_dir="../../weights/ecapa_tdnn_tripletAID",
        do_train=True, do_eval=True, overwrite_output_dir=True, num_train_epochs=5,
        per_device_train_batch_size=8, per_device_eval_batch_size=2, evaluation_strategy="steps", learning_rate=2e-4,
        warmup_steps=10000,
        warmup_ratio=0.1, save_steps=1000,
        eval_steps=5000, save_total_limit=1, label_names=["labelAP", "labelN", "input_values", "lengths"],
        logging_steps=100, dataloader_drop_last=True)
    os.makedirs(training_args.output_dir, exist_ok=True)
    os.makedirs("./wandb_cache", exist_ok=True)

    wandb.init(dir="./wandb_cache")

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

    kespeech_train_dataset = getTripletDataset(os.path.join(KeSpeechDatasetMetaDirPath, "TripletDataset_train.json"),
                                               KeSpeechDatasetBaseDir, 8)
    kespeech_test_dataset = getTripletDataset(os.path.join(KeSpeechDatasetMetaDirPath, "TripletDataset_test.json"),
                                              KeSpeechDatasetBaseDir, 8)
    myspeech_train_dataset = getTripletDataset(os.path.join(MySpeechDatasetMetaDirPath, "TripletDataset_train.json"),
                                               KeSpeechDatasetBaseDir, 8, MySpeechDatasetBaseDir)
    myspeech_test_dataset = getTripletDataset(os.path.join(MySpeechDatasetMetaDirPath, "TripletDataset_test.json"),
                                              KeSpeechDatasetBaseDir, 8, MySpeechDatasetBaseDir)
    train_dataset = concatenate_datasets([kespeech_train_dataset, myspeech_train_dataset])
    test_dataset = concatenate_datasets([kespeech_test_dataset, myspeech_test_dataset])

    train_dataset_original_size = train_dataset.num_rows
    test_dataset_original_size = test_dataset.num_rows

    if data_args.max_train_samples is not None:
        if train_dataset_original_size > data_args.max_train_samples:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if data_args.max_val_samples is not None:
        if test_dataset_original_size > data_args.max_val_samples:
            test_dataset = test_dataset.select(range(data_args.max_val_samples))

    train_dataset_final_size = train_dataset.num_rows
    eval_dataset_final_size = test_dataset.num_rows
    logger.info(
        f"After filtering {train_dataset_final_size} of {train_dataset_original_size} samples will be used to train the model")
    logger.info(
        f"After filtering {eval_dataset_final_size} of {test_dataset_original_size} samples will be used to eval the model")
    set_seed(training_args.seed)
    if additional_training_args.remove_samples_with_oov_from_training:
        vocab = set(processor.tokenizer.encoder.keys())
        train_dataset_size = train_dataset_final_size
        train_dataset = train_dataset.filter(
            lambda example: vocab.issuperset(example["text"].replace(" ", "")),
            num_proc=data_args.preprocessing_num_workers
        )
        train_dataset_final_size = len(train_dataset)
        print(
            f"OOV found in {train_dataset_size - len(train_dataset)} samples, and they were removed from training set")
        print(f"The final training set size is {train_dataset_final_size}")

    # save the feature_extractor and the tokenizer
    processor.save_pretrained(training_args.output_dir)

    def compute_metrics(pred):
        aid_logits = pred.predictions
        aid_label = pred.label_ids
        del pred
        aid_pred_A = aid_logits["logitsA"]
        aid_pred_P = aid_logits["logitsP"]
        aid_pred_N = aid_logits["logitsN"]
        aid_label_A = aid_label["labelA"]
        aid_label_P = aid_label["labelP"]
        aid_label_N = aid_label["labelN"]
        del aid_logits, aid_label
        """
        计算AID指标
        """
        acc_A = compute_acc(aid_pred_A, aid_label_A)
        acc_P = compute_acc(aid_pred_P, aid_label_P)
        acc_N = compute_acc(aid_pred_N, aid_label_N)
        acc = float((acc_A + acc_P + acc_N) / 3)
        save_metric("acc", float(acc), save_dir=training_args.output_dir)
        torch.cuda.empty_cache()
        return {"acc": acc}

    model = Siamese_ECAPA_TDNN(encoder_dir_path=pretrained_model_path)
    pt_model_path = os.path.join(training_args.output_dir, "model.pt")
    if os.path.exists(pt_model_path):
        print(f"文件{pt_model_path}存在，将加载该文件")
        model_dict = torch.load(pt_model_path)
        model.load_state_dict(model_dict)

    model.freeze_feature_extractor()
    # Data collator
    data_collator = DataCollatorTripletWithPadding(
        processor=processor,
        padding=True,
        apply_gaussian_noise_with_p=additional_training_args.apply_gaussian_noise_with_p,
        apply_gain_with_p=additional_training_args.apply_gain_with_p,
        apply_pitch_shift_with_p=additional_training_args.apply_pitch_shift_with_p,
        apply_time_stretch_with_p=additional_training_args.apply_time_stretch_with_p,
        sample_rate=16_000,
    )

    # Initialize our Trainer

    trainer = ECAPATDNNTripletTrainer(
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
        if os.path.exists(os.path.join(training_args.output_dir, "checkpoint-last")):
            checkpoint = os.path.join(training_args.output_dir, "checkpoint-last")
            logger.info(
                f"Checkpoint detected, resuming training at {checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
        else:
            checkpoint = None
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()
    save_model(model, os.path.join(training_args.output_dir, "model.pt"), save_state_dict=True)
    metrics = train_result.metrics
    metrics["train_samples"] = train_dataset_final_size

    trainer.log_metrics(split=f"Triplet AID train", metrics=metrics)
    trainer.save_metrics(split=f"Triplet AID train", metrics=metrics)
    trainer.save_state()
    result = trainer.evaluate()
    print(f"\n{result}")
    torch.cuda.synchronize()  # 等待GPU操作完成
    torch.cuda.empty_cache()

    # Evaluation
    metrics = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        max_val_samples = test_dataset_original_size
        metrics["eval_samples"] = min(max_val_samples, len(test_dataset))
        trainer.log_metrics(split=f"Triplet AID eval", metrics=metrics)
        trainer.save_metrics(split=f"triplet AID eval", metrics=metrics)

    # save model files
    if additional_training_args.upload_final_model_to_wandb:
        upload_model_to_wandb(training_args.output_dir, name=f"{wandb.run.name}_final", metadata=metrics)


if __name__ == "__main__":
    kespeech_dataset_meta_dir_path = "../../dataset_metadata/KeSpeech_data_metadata"  # 用于找到dataset的json文件
    myspeech_dataset_meta_dir_path = "../../dataset_metadata/MySpeech_data_metadata"

    asrProcessor = make_processor(True, "../../weights/KeSpeechASR")
    kespeech_dataset_dir = "/input0/"
    myspeech_dataset_dir = "/input1/"
    pretrain_model_path = "../../weights/SpeechBrain/model"  # 用于生成ecapa-tdnn

    run_train(
        KeSpeechDatasetMetaDirPath=kespeech_dataset_meta_dir_path,
        MySpeechDatasetMetaDirPath=myspeech_dataset_meta_dir_path,
        processor=asrProcessor,
        KeSpeechDatasetBaseDir=kespeech_dataset_dir,
        MySpeechDatasetBaseDir=myspeech_dataset_dir,
        pretrained_model_path=pretrain_model_path,
    )
