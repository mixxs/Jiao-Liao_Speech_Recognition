import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel, Wav2Vec2Config

from src.modules.decoder.simpleClassifierForCTC import WFforCTCdecoder, MultiforCTCdecoder
from src.utils.fileIO.model import save_model


class Wav2Vec2_CTC(Wav2Vec2PreTrainedModel):
	def __init__(self, model_args, encoder_dir_path, processor, model_config: Wav2Vec2Config):
		super().__init__(model_config)
		self.wav2vec2 = Wav2Vec2Model.from_pretrained(
			encoder_dir_path,
			ignore_mismatched_sizes=True,
			cache_dir=model_args.cache_dir,
			activation_dropout=model_args.activation_dropout,
			attention_dropout=model_args.attention_dropout,
			hidden_dropout=model_args.hidden_dropout,
			feat_proj_dropout=model_args.feat_proj_dropout,
			mask_time_prob=model_args.mask_time_prob,
			layerdrop=model_args.layerdrop,
			ctc_loss_reduction="mean",
			pad_token_id=processor.tokenizer.pad_token_id,
			ctc_zero_infinity=True
		)
		self.pad_token_id = processor.tokenizer.pad_token_id
		self.dropout = nn.Dropout(model_config.final_dropout)
		self.lm_head = nn.Sequential(
			nn.Linear(model_config.hidden_size, len(processor.tokenizer)),
			nn.LayerNorm(len(processor.tokenizer)),
		)
		
		self.language_id = 0
		
		self._keys_to_ignore_on_save = self.wav2vec2._keys_to_ignore_on_save
	
	def freeze_share_weights(self):
		if hasattr(self.wav2vec2.adapter, "freeze_share_weights"):
			self.wav2vec2.adapter.freeze_share_weights()
		for layer in self.wav2vec2.encoder.layers:
			if hasattr(layer, "freeze_share_weights"):
				layer.freeze_share_weights()
	
	def freeze_feature_extractor(self):
		"""
		Calling this function will disable the gradient computation for the feature extractor so that its parameter
		will not be updated during training.
		"""
		self.wav2vec2.freeze_feature_extractor()
	
	def set_language_id(self, language_id: int):
		self.language_id = language_id
		print(f"Wav2vec_CTC language_id changed to {self.language_id}")
		for layer in self.wav2vec2.encoder.layers:
			if hasattr(layer, "set_language_id"):
				layer.set_language_id(language_id)
		if hasattr(self.wav2vec2.adapter, "set_language_id"):
			self.wav2vec2.adapter.set_language_id(language_id)
	
	def forward(self, input_values, attention_mask=None, output_attentions=None, output_hidden_states=None,
	            labels=None, language_id=None):
		if attention_mask is None:
			import warnings
			warnings.warn(
				"pad_mask is None, please make sure you didn't use padding.\n If you padded the sequence, you should use pad_mask"
			)
		outputs = self.wav2vec2(
			input_values,
			attention_mask=attention_mask,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
		)
		hidden_states = outputs[0]
		hidden_states = self.dropout(hidden_states)
		if language_id is not None:
			self.set_language_id(language_id)
		if isinstance(self.lm_head, WFforCTCdecoder) or isinstance(self.lm_head, MultiforCTCdecoder):
			logits = self.lm_head(hidden_states, self.language_id)
		else:
			logits = self.lm_head(hidden_states)
		asr_loss = None
		if labels is not None:
			attention_mask = (
				attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
			)
			input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1))
			
			# assuming that padded tokens are filled with -100
			# when not being attended to
			labels_mask = labels >= 0
			target_lengths = labels_mask.sum(-1)
			flattened_targets = labels.masked_select(labels_mask)
			
			log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
			
			with torch.backends.cudnn.flags(enabled=False):
				asr_loss = F.ctc_loss(
					log_probs,
					flattened_targets,
					input_lengths,
					target_lengths,
					blank=self.pad_token_id,
					reduction="mean",
					zero_infinity=True,
				)
		loss = {"asr": asr_loss, "total": asr_loss}
		pred = {"asr": torch.argmax(logits, dim=-1)}
		return {"loss": loss, "pred": pred}
	
	def transfer(self):
		return {"Wav2vec2Model": self.wav2vec2, "linear": self.lm_head}
	
	def save(self, path, save_state_dict: bool = True):
		save_model(self, path, save_state_dict)
	
	def save_feature_encoder(self, path, save_state_dict: bool = True):
		save_model(self.wav2vec2, path, save_state_dict)
	
	def save_lm_head(self, path, save_state_dict: bool = True):
		save_model(self.lm_head, path, save_state_dict)
