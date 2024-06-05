from torch import nn
from transformers import Wav2Vec2Config
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2EncoderStableLayerNorm, Wav2Vec2EncoderLayer

from src.modules.WFAdapter.SequenceAdapterLayer import WFLinearAdapter, LinearAdapter


class EncoderWFAdapterLayer(nn.Module):
    def __init__(self, config: Wav2Vec2Config, adapter_embed_dim, language_num, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = Wav2Vec2EncoderLayer(config=config)
        self.adapter = WFLinearAdapter(in_dim=config.hidden_size, embed_dim=adapter_embed_dim,
                                       language_num=language_num)
        self.language_id = None

    def set_language_id(self, language_id):
        self.language_id = int(language_id)

    def freeze_share_weights(self):
        self.adapter.freeze_share_weights()

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        outputs = self.encoder(hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)
        # 这个Wav2VecModel对象的encoderlayer模块（也就是这里的encoder）的输出是一个元组子类的对象，
        # 这个元组的第一个元素是hidden_states也就是我们需要的
        # 如果output_attentions指定为True，那么这个元组就会有第二个元素：attention得分
        # 接下来我们将outputs中的hidden_states作为adapter的输入，得到一个tensor类型的adapt_output
        adapt_output = self.adapter.forward(outputs[0], language_id=self.language_id)
        new_outputs = (adapt_output,)  # 我们最终的返回值形式和encoder保持一致
        if len(outputs) > 1:
            new_outputs += new_outputs[1:]
        return new_outputs


class EncoderStableLayerNormWFAdapterLayer(nn.Module):
    def __init__(self, config: Wav2Vec2Config, adapter_embed_dim, language_num, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = Wav2Vec2EncoderStableLayerNorm(config=config)
        self.adapter = WFLinearAdapter(in_dim=config.hidden_size, embed_dim=adapter_embed_dim,
                                       language_num=language_num)
        self.language_id = None

    def set_language_id(self, language_id):
        self.language_id = int(language_id)

    def freeze_share_weights(self):
        self.adapter.freeze_share_weights()

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        outputs = self.encoder(hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)
        # 这个Wav2VecModel对象的encoderlayer模块（也就是这里的encoder）的输出是一个元组子类的对象，
        # 这个元组的第一个元素是hidden_states也就是我们需要的
        # 如果output_attentions指定为True，那么这个元组就会有第二个元素：attention得分
        # 接下来我们将outputs中的hidden_states作为adapter的输入，得到一个tensor类型的adapt_output
        adapt_output = self.adapter.forward(outputs[0], language_id=self.language_id)
        new_outputs = (adapt_output,)
        if len(outputs) > 1:
            new_outputs += new_outputs[1:]
        return new_outputs


class EncoderAdapterLayer(nn.Module):
    def __init__(self, config: Wav2Vec2Config, adapter_embed_dim, language_num, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = Wav2Vec2EncoderLayer(config=config)
        self.adapter = LinearAdapter(in_dim=config.hidden_size, embed_dim=adapter_embed_dim, language_num=language_num)
        self.language_id = None

    def set_language_id(self, language_id):
        self.language_id = int(language_id)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        outputs = self.encoder(hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)
        # 这个Wav2VecModel对象的encoderlayer模块（也就是这里的encoder）的输出是一个元组子类的对象，
        # 这个元组的第一个元素是hidden_states也就是我们需要的
        # 如果output_attentions指定为True，那么这个元组就会有第二个元素：attention得分
        # 接下来我们将outputs中的hidden_states作为adapter的输入，得到一个tensor类型的adapt_output
        adapt_output = self.adapter.forward(outputs[0], language_id=self.language_id)
        new_outputs = (adapt_output,)
        if len(outputs) > 1:
            new_outputs += new_outputs[1:]
        return new_outputs


class EncoderStableLayerNormAdapterLayer(nn.Module):
    def __init__(self, config: Wav2Vec2Config, adapter_embed_dim, language_num, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = Wav2Vec2EncoderStableLayerNorm(config=config)
        self.adapter = WFLinearAdapter(in_dim=config.hidden_size, embed_dim=adapter_embed_dim,
                                       language_num=language_num)
        self.language_id = None

    def set_language_id(self, language_id):
        self.language_id = int(language_id)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        outputs = self.encoder(hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)
        # 这个Wav2VecModel对象的encoderlayer模块（也就是这里的encoder）的输出是一个元组子类的对象，
        # 这个元组的第一个元素是hidden_states也就是我们需要的
        # 如果output_attentions指定为True，那么这个元组就会有第二个元素：attention得分
        # 接下来我们将outputs中的hidden_states作为adapter的输入，得到一个tensor类型的adapt_output
        adapt_output = self.adapter.forward(outputs[0], language_id=self.language_id)
        new_outputs = (adapt_output,)
        if len(outputs) > 1:
            new_outputs += new_outputs[1:]
        return new_outputs
