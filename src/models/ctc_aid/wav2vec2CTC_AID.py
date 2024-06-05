import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel, Wav2Vec2Config

from src.models.aid.wav2vecAid import Wav2Vec2_AID
from src.modules.WFAdapter.EncoderAdapterBlock import EncoderWFAdapterLayer, EncoderStableLayerNormWFAdapterLayer
from src.modules.decoder.simpleClassifierForCTC import WFforCTCdecoder, MultiforCTCdecoder
from src.modules.AttAdapter.AttentionAdapter import AttAdapterBlock
from src.utils.fileIO.model import save_model


class Wav2Vec2_CTC_AID(Wav2Vec2PreTrainedModel):
	def __init__(self, model_args, encoder_dir_path, processor, config: Wav2Vec2Config, language_num):
		super().__init__(config)
		
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
		self.dropout = nn.Dropout(config.final_dropout)
		self.lm_head = nn.Sequential(
		    nn.Linear(config.hidden_size, len(processor.tokenizer)),
		    nn.LayerNorm(len(processor.tokenizer)),)
		self.aid = Wav2Vec2_AID(
			model_args=model_args,
			encoder_dir_path=encoder_dir_path,
			config=config,
			language_num=language_num
		)
		self.language_id = 0
		self.fusion_block = AttAdapterBlock(config.hidden_size, config.hidden_size, config.hidden_size)
	
	def set_language_id(self, language_id: int):
		self.language_id = language_id
		print(f"Wav2vec_CTC language_id changed to {self.language_id}")
		for layer in self.wav2vec2.encoder.layers:
			if hasattr(layer, "set_language_id"):
				layer.set_language_id(language_id)
		if hasattr(self.wav2vec2.adapter, "set_language_id"):
			self.wav2vec2.adapter.set_language_id(language_id)
	
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
		# change_wav2vec2时会将wav2vec2所有参数都冻结，这里先冻结CNN部分
		self.wav2vec2.wav2vec._freeze_parameters()
		for param in self.aid.parameters():
			param.requires_grad = False
	
	def forward(self, input_values, attention_mask=None, output_attentions=None, output_hidden_states=None, labels=None,
	            class_labels=None, languageId=None, ):
		if attention_mask is None:
			import warnings
			warnings.warn(
				"pad_mask is None, please make sure you didn't use padding.\n If you padded the sequence, you should use pad_mask"
			)
		class_feature = self.aid.get_feature(input_values)
		class_pred = self.aid.get_prediction(class_feature)
		language_id = 0
		if languageId is not None:
			language_id = languageId
			self.set_language_id(language_id)
		elif class_labels is not None:
			language_id = class_labels[0]
			self.set_language_id(language_id)
		else:
			language_class = torch.argmax(class_pred, dim=-1)
			language_id = language_class if len(language_class.shape) == 0 else language_class[0]
			self.set_language_id(language_id)
		
		outputs = self.wav2vec2(
			input_values,
			attention_mask=attention_mask,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
		)
		
		hidden_states = outputs.last_hidden_state
		hidden_states = self.dropout(hidden_states)
		fused_feature = self.fusion_block.forward(hidden_states, class_feature, language_id=language_id)
		
		logits = self.lm_head(fused_feature, self.language_id)
		asr_loss = None
		if labels is not None:
			# retrieve loss input_lengths from attention_mask
			attention_mask = (
				attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
			)
			input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1))
			
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
		class_loss = None
		if class_labels is not None:
			CEloss = nn.CrossEntropyLoss()
			class_loss = CEloss(class_pred, class_labels)
		
		total_loss = 0
		if asr_loss is None and class_loss is None:
			total_loss = None
		elif asr_loss is not None:
			total_loss = asr_loss
		elif class_loss is not None:
			total_loss = class_loss
		else:
			total_loss = asr_loss + class_loss
		loss = {"asr": asr_loss, "aid": class_loss, "total": total_loss}
		pred = {"asr": torch.argmax(logits, dim=-1), "aid": torch.argmax(class_pred, dim=-1)}
		return {"loss": loss, "pred": pred}
	
	def save(self, path, save_state_dict: bool = True):
		save_model(self, path, save_state_dict)
	
	def save_asr_feature_extractor(self, path, save_state_dict: bool = True):
		save_model(self.wav2vec2, path, save_state_dict)
	
	def save_aid_feature_extrator(self, path, save_state_dict: bool = True):
		save_model(self.aid, path, save_state_dict)
