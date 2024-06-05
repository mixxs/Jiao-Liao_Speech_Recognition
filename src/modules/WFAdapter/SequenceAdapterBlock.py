from torch import nn
from transformers import Wav2Vec2Config

from src.modules.WFAdapter.SequenceAdapterLayer import WFLinearAdapter, LinearAdapter


class LinearAdapterBlock(nn.Module):
    def __init__(self, config: Wav2Vec2Config, adapter_embed_dim, language_num, layer_num, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        # self.adapter = WFLinearAdapter(in_dim=config.hidden_size, embed_dim=adapter_embed_dim,
        #                                out_dim=config.hidden_size, language_num=language_num)
        self.in_dim = config.hidden_size
        self.hidden_dim = adapter_embed_dim
        self.language_id = None
        self.layer_num = layer_num
        self.language_num = language_num
        self.layers = nn.ModuleList()
        for i in range(self.layer_num):
            self.layers.append(
                LinearAdapter(in_dim=self.in_dim, embed_dim=self.hidden_dim, language_num=self.language_num)
            )

    def set_language_id(self, language_id):
        self.language_id = int(language_id)

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer.forward(hidden_states, language_id=self.language_id)
        return hidden_states


class WFAdapterBlock(nn.Module):
    def __init__(self, config: Wav2Vec2Config, adapter_embed_dim, language_num, layer_num, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.in_dim = config.hidden_size
        self.hidden_dim = adapter_embed_dim
        self.language_id = None
        self.layer_num = layer_num
        self.language_num = language_num
        self.layers = nn.ModuleList()
        for i in range(self.layer_num):
            self.layers.append(
                WFLinearAdapter(in_dim=self.in_dim, embed_dim=self.hidden_dim, language_num=self.language_num)
            )

    def freeze_share_weights(self):
        for layer in self.layers:
            layer.freeze_share_weights()

    def set_language_id(self, language_id):
        self.language_id = int(language_id)

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer.forward(hidden_states, language_id=self.language_id)
        return hidden_states
