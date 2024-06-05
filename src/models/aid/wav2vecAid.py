import torch
import torch.nn as nn
from transformers import Wav2Vec2PreTrainedModel, Wav2Vec2Model, Wav2Vec2Config

from src.utils.fileIO.model import save_model


class Wav2Vec2_AID(Wav2Vec2PreTrainedModel):
    def __init__(self, model_args, encoder_dir_path, config: Wav2Vec2Config, language_num):
        super().__init__(config)

        self.wav2vec: Wav2Vec2Model = Wav2Vec2Model.from_pretrained(
            encoder_dir_path,
            ignore_mismatched_sizes=True,
            cache_dir=model_args.cache_dir,
        )  # [bs,seq,hidden(normed)],[]
        self.norm = nn.LayerNorm(config.hidden_size)
        self.act = nn.ELU()
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.ELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 64),
            nn.ELU(),
            nn.LayerNorm(64),
            nn.Linear(64, language_num),
            nn.LayerNorm(language_num),
        )
        self.ceLoss = nn.CrossEntropyLoss()

    def freeze_feature_extractor(self):
        """
		Calling this function will disable the gradient computation for the feature extractor so that its parameter
		will not be updated during training.
		"""
        self.wav2vec.freeze_feature_encoder()

    def get_feature(self, input_values, pad_mask=None):
        if pad_mask is None:
            import warnings
            warnings.warn(
                "pad_mask is None, please make sure you didn't use padding.\n If you padded the sequence, you should use pad_mask"
            )
        out = self.wav2vec.forward(input_values=input_values, attention_mask=pad_mask)
        hidden_states = out.last_hidden_state  # bs,seq,hidden
        feature = torch.mean(hidden_states, dim=1)  # bs,hidden
        feature = self.act(self.norm(feature))
        return feature

    def get_prediction(self, feature):
        return self.classifier(feature)

    def forward(self, input_values, pad_mask=None, class_labels=None):
        feature = self.get_feature(input_values, pad_mask)
        class_pred = self.get_prediction(feature)

        class_loss = None
        if class_labels is not None:
            class_loss = self.ceLoss(class_pred, class_labels)

        loss = {"aid": class_loss, "total": class_loss}
        pred = {"aid": torch.argmax(class_pred, dim=-1)}
        feature = {"dialect_feature": feature}
        return {"loss": loss, "pred": pred, "feature": feature}

    def save(self, path, save_state_dict: bool = True):
        save_model(self, path, save_state_dict)
