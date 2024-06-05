from dataclasses import dataclass, field
from typing import List, Optional


def list_field(default=None, metadata=None):
	return field(default_factory=lambda: default, metadata=metadata)


@dataclass
class AdditionalTrainingArguments:
	"""
	Additional training arguments
	"""
	lr_warmup_ratio: Optional[float] = field(
		default=0.1,
		metadata={"help": "Percentage of steps for LR warmup phase"},
	)
	lr_constant_ratio: Optional[float] = field(
		default=0.4,
		metadata={"help": "Percentage of steps for LR constant phase (after warmup)"},
	)
	upload_final_model_to_wandb: Optional[bool] = field(
		default=None,
		metadata={"help": "Upload the final trained model to the WandB artifacts repository"},
	)
	upload_model_to_wandb_each_step: Optional[int] = field(
		default=None,
		metadata={"help": "Frequency (in steps) to upload the trained model to the WandB artifacts repository"},
	)
	apply_gaussian_noise_with_p: Optional[float] = field(
		default=0.5,
		metadata={"help": "Probability to apply Gaussian Noise in the original samples"},
	)
	apply_gain_with_p: Optional[float] = field(
		default=0.5,
		metadata={"help": "Probability to apply Gain in the original samples"},
	)
	apply_pitch_shift_with_p: Optional[float] = field(
		default=0.5,
		metadata={"help": "Probability to apply Pitch Shift in the original samples"},
	)
	apply_time_stretch_with_p: Optional[float] = field(
		default=0.5,
		metadata={"help": "Probability to apply Time Stretch in the original samples"},
	)
	min_char_occurrence_ratio: Optional[float] = field(
		default=None,
		metadata={"help": "Minimum ratio of character occurrences to be considered for the vocabulary builder"},
	)
	max_dataset_size_vocab_builder: Optional[int] = field(
		default=10000,
		metadata={"help": "Maximum size of the dataset to be considered for vocabulary builder"},
	)
	remove_samples_with_oov_from_training: Optional[bool] = field(
		default=False,
		metadata={"help": "Whether to remove samples from training when there are OOV characters on them"},
	)


@dataclass
class ModelArguments:
	"""
	Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
	"""
	cache_dir: Optional[str] = field(
		default=None,
		metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
	)
	freeze_feature_extractor: Optional[bool] = field(
		default=True, metadata={"help": "Whether to freeze the feature extractor layers of the model."}
	)
	attention_dropout: Optional[float] = field(
		default=0.1, metadata={"help": "The dropout ratio for the attention probabilities."}
	)
	activation_dropout: Optional[float] = field(
		default=0.1, metadata={"help": "The dropout ratio for activations inside the fully connected layer."}
	)
	hidden_dropout: Optional[float] = field(
		default=0.1,
		metadata={
			"help": "The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler."
		},
	)
	feat_proj_dropout: Optional[float] = field(
		default=0.0,
		metadata={"help": "The dropout probabilitiy for all 1D convolutional layers in feature extractor."},
	)
	mask_time_prob: Optional[float] = field(
		default=0.05,
		metadata={
			"help": "Propability of each feature vector along the time axis to be chosen as the start of the vector"
			        "span to be masked. Approximately ``mask_time_prob * sequence_length // mask_time_length`` feature"
			        "vectors will be masked along the time axis. This is only relevant if ``apply_spec_augment is True``."
		},
	)
	
	layerdrop: Optional[float] = field(default=0.1, metadata={"help": "The LayerDrop probability."})


@dataclass
class DataTrainingArguments:
	"""
	Arguments pertaining to what KeSpeech we are going to input our model for training and eval.

	Using `HfArgumentParser` we can turn this class
	into argparse arguments to be able to specify them on
	the command line.
	"""
	
	dataset_config_name: Optional[str] = field(
		default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
	)
	overwrite_cache: bool = field(
		default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
	)
	preprocessing_num_workers: Optional[int] = field(
		default=28,
		metadata={"help": "The number of processes to use for the preprocessing."},
	)
	max_train_samples: Optional[int] = field(
		default=None,
		metadata={
			"help": "For debugging purposes or quicker training, truncate the number of training examples of one dialect to this "
			        "value if set."
		},
	)
	max_val_samples: Optional[int] = field(
		default=200,
		metadata={
			"help": "For debugging purposes or quicker training, truncate the number of validation examples of one dialect to this "
			        "value if set."
		},
	)
	val_ratio: Optional[float] = field(
		default=0.2,
		metadata={
			"help": "Percentage of dataset samples to be used for evaluation, default is 20%"
		},
	)
	chars_to_ignore: List[str] = list_field(
		default=[",", "?", "¿", ".", "!", "¡", ";", "；", ":", '""', "%", '"', "�", "ʿ", "·", "჻", "~", "՞",
		         "؟", "،", "।", "॥", "«", "»", "„", "“", "”", "「", "」", "‘", "’", "《", "》", "(", ")", "[", "]",
		         "{", "}", "=", "`", "_", "+", "<", ">", "…", "–", "°", "´", "ʾ", "‹", "›", "©", "®", "—", "→", "。",
		         "、", "﹂", "﹁", "‧", "～", "﹏", "，", "｛", "｝", "（", "）", "［", "］", "【", "】", "‥", "〽",
		         "『", "』", "〝", "〟", "⟨", "⟩", "〜", "：", "！", "？", "♪", "؛", "/", "\\", "º", "−", "^", "ʻ", "ˆ"],
		metadata={"help": "A list of characters to remove from the transcripts."},
	)
	min_duration: Optional[float] = field(
		default=0.0,
		metadata={
			"help": "The minimum duration (in seconds) that a sample needs to have to be considered for training"
		},
	)
	max_duration: Optional[float] = field(
		default=float("inf"),
		metadata={
			"help": "The maximum duration (in seconds) that a sample needs to have to be considered for training"
		},
	)
