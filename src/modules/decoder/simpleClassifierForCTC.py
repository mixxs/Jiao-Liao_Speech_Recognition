import torch.nn as nn

from src.modules.WFAdapter.LinearAdapt import WFLinear


class WFforCTCdecoder(nn.Module):
    def __init__(self, in_features, out_features, language_num, *args, **kwargs):
        super(WFforCTCdecoder, self).__init__(*args, **kwargs)
        self.fc1 = WFLinear(in_features, out_features, language_num)
        self.ln1 = nn.LayerNorm(out_features)

    def freeze_share_weights(self):
        self.fc1.freeze_share_weights()

    def forward(self, x, languageID):
        out = self.ln1(self.fc1(x, languageID))
        return out


class forCTCdecoder(nn.Module):
    def __init__(self, in_features, out_features, *args, **kwargs):
        super(forCTCdecoder, self).__init__(*args, **kwargs)
        self.fc1 = nn.Linear(in_features, out_features)
        self.ln1 = nn.LayerNorm(out_features)

    def forward(self, x):
        out = self.ln1(self.fc1(x))
        return out


class MultiforCTCdecoder(nn.Module):
    def __init__(self, in_features, out_features, language_num, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.CTC_list = []
        for language in range(language_num):
            self.CTC_list.append(forCTCdecoder(in_features, out_features))

    def forward(self, x, languageID):
        return self.CTC_list[languageID](x)
