import torch.nn as nn
from speechbrain.pretrained import EncoderClassifier

from src.utils.fileIO.model import save_model


class Siamese_ECAPA_TDNN(nn.Module):
    def __init__(self, encoder_dir_path, *args, **kwargs):
        super(Siamese_ECAPA_TDNN, self).__init__(*args, **kwargs)
        self.feature_extractor = EncoderClassifier.from_hparams(source=encoder_dir_path, savedir=encoder_dir_path,
                                                                run_opts={"device": "cuda"})
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
        self.triplet_loss = nn.TripletMarginLoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, wavP, wavA, wavN, lens_P=None, lens_A=None, lens_N=None, labelP=None, labelN=None,
                alpha=1.0, beta=1.0):
        featureP = self.feature_extractor.encode_batch(wavP, wav_lens=lens_P)
        featureA = self.feature_extractor.encode_batch(wavA, wav_lens=lens_A)
        featureN = self.feature_extractor.encode_batch(wavN, wav_lens=lens_N)
        tripletLoss = self.triplet_loss.forward(featureA.squeeze(1), featureP.squeeze(1), featureN.squeeze(1))

        predP = self.feature_extractor.mods.classifier(featureP).squeeze(1)
        predA = self.feature_extractor.mods.classifier(featureA).squeeze(1)
        predN = self.feature_extractor.mods.classifier(featureN).squeeze(1)

        if (labelP is not None and labelN is None) or (labelP is None and labelN is not None):
            raise ValueError("labelP and labelN must both be None or not None")
        elif labelP is not None:
            aidLossP = self.ce_loss(predP, labelP)
            aidLossA = self.ce_loss(predA, labelP)
            aidLossN = self.ce_loss(predN, labelN)
            aidLoss = (aidLossP + aidLossA + aidLossN) / 3
            loss = {"aid": aidLoss, "triplet": tripletLoss, "total": beta * tripletLoss + alpha * aidLoss}
        else:
            loss = {"aid": None, "triplet": tripletLoss, "total": tripletLoss}

        pred = {"predP": predP, "predA": predA, "predN": predN}
        feature = {"featureP": featureP, "featureA": featureA, "featureN": featureN}
        return {"loss": loss, "pred": pred, "feature": feature}

    def save(self, path: str, save_state_dict: bool = True) -> None:
        save_model(self, path, save_state_dict)

    def save_feature_extractor(self, path: str, save_state_dict: bool = True):
        save_model(self.feature_extractor, path, save_state_dict)

    def freeze_feature_extractor(self):
        pass

