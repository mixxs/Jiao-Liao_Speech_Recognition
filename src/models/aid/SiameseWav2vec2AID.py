from typing import Dict

import torch
import torch.nn as nn

from src.models.aid.wav2vecAid import Wav2Vec2_AID
from src.utils.fileIO.model import save_model


class Siamese_Wav2Vec2_AID(nn.Module):
    def __init__(self, model_args, encoder_dir_path, config, language_num, *args, **kwargs):
        super(Siamese_Wav2Vec2_AID, self).__init__(*args, **kwargs)
        self.feature_extractor = Wav2Vec2_AID(model_args, encoder_dir_path, config, language_num)
        self.triplet_loss = nn.TripletMarginLoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, wavP, wavA, wavN, pad_maskP=None, pad_maskA=None, pad_maskN=None, labelP=None, labelN=None,
                alpha=1.0, beta=0.0):
        outP: Dict[str, Dict] = self.feature_extractor.forward(wavP, pad_maskP, labelP)
        outA: Dict[str, Dict] = self.feature_extractor.forward(wavA, pad_maskA, labelP)
        outN: Dict[str, Dict] = self.feature_extractor.forward(wavN, pad_maskN, labelN)
        featureP = outP["feature"]["dialect_feature"]
        featureA = outA["feature"]["dialect_feature"]
        featureN = outN["feature"]["dialect_feature"]
        tripletLoss = self.triplet_loss.forward(featureA, featureP, featureN)

        predP = outP["pred"]["aid"]
        predA = outA["pred"]["aid"]
        predN = outN["pred"]["aid"]

        if (labelP is not None and labelN is None) or (labelP is None and labelN is not None):
            raise ValueError("labelP and labelN must both be None or not None")
        elif labelP is not None:
            aidLossP = outP["loss"]["aid"]
            aidLossA = outA["loss"]["aid"]
            aidLossN = outN["loss"]["aid"]
            aidLoss = (aidLossP + aidLossA + aidLossN) / 3
            loss = {"aid": aidLoss, "triplet": tripletLoss, "total": beta * tripletLoss + alpha * aidLoss}
        else:
            loss = {"aid": None, "triplet": tripletLoss, "total": tripletLoss}

        pred = {"predP": torch.argmax(predP, dim=-1), "predA": torch.argmax(predA, dim=-1),
                "predN": torch.argmax(predN, dim=-1)}
        feature = {"featureP": featureP, "featureA": featureA, "featureN": featureN}
        return {"loss": loss, "pred": pred, "feature": feature}

    def save(self, path: str, save_state_dict: bool = True) -> None:
        save_model(self, path, save_state_dict)

    def save_feature_extractor(self, path: str, save_state_dict: bool = True):
        self.feature_extractor.save(path, save_state_dict)

    def freeze_feature_extractor(self):
        self.feature_extractor.freeze_feature_extractor()
