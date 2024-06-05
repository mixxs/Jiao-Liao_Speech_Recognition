import copy

import torch.nn as nn
import torch.nn.functional as F

from src.modules.WFAdapter.AttentionWF import WFMultiHeadAttention
from src.modules.WFAdapter.LinearAdapt import WFLinear
from src.modules.embeders.BasicEmbeder import BasicEmbedding
from src.modules.encoder.positionalEncoder import PositionalEncoder


# TODO: 测试transformer decoder


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        """
        维度变化：embed_dim -> d_ff -> embed_dim
        :param d_model: attention的向量维度
        :param d_ff: 隐层层向量维度
        :param dropout: dropout
        """
        super().__init__()

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class WFFeedForward(nn.Module):

    def __init__(self, d_model, d_ff=2048, language_num=1, dropout=0.1):
        super().__init__()

        # We set d_ff as a default to 2048
        self.linear_1 = WFLinear(d_model, d_ff, language_num)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = WFLinear(d_ff, d_model, language_num)

    def freeze_share_weights(self):
        self.linear_1.freeze_share_weights()
        self.linear_2.freeze_share_weights()

    def forward(self, x, languageID):
        x = self.dropout(F.relu(self.linear_1(x, languageID)))
        x = self.linear_2(x, languageID)
        return x


class WFTransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, language_num, dim_feedforward=2048, dropout=0.1):
        """

        :param d_model: attention的向量维度
        :param nhead: 注意力头数
        :param language_num:语言数
        :param dim_feedforward: 全连接层中间输出向量维度
        :param dropout: dropout率
        """
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = WFMultiHeadAttention(d_model, nhead, language_num, dropout=dropout)
        self.attn_2 = WFMultiHeadAttention(d_model, nhead, language_num, dropout=dropout)
        self.ff = WFFeedForward(d_model, dim_feedforward, language_num=language_num, dropout=dropout)

    def freeze_share_weights(self):
        self.attn_1.freeze_share_weights()
        self.attn_2.freeze_share_weights()
        self.ff.freeze_share_weights()

    def forward(self, x, e_outputs, src_mask, trg_mask, languageID: int):
        """

        :param x: 输入之一，output embedding
        :param e_outputs: encoder的输出
        :param src_mask: padding mask,padding的部分是没有意义的
        :param trg_mask: 对标签进行mask
        :param languageID: 语言类别
        :return:
        """
        x2 = self.norm_1(x)
        print("output embedding shape: ", x2.shape)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask, languageID))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, src_mask, languageID))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2, languageID))
        return x


class WFTransformerDecoder(nn.Module):
    """
    N层decoderLayer
    """

    def __init__(self, language_num, vocab_size, d_model, N, nhead, dim_feedforward=2048, dropout=0.1, max_seq_len=80):
        """

        :param language_num: 语言数量
        :param vocab_size: 字典大小
        :param d_model: 多头注意力的输入输出大小
        :param N: 多少层
        :param nhead: 注意力的头数
        :param dim_feedforward: 全连接的中间层输出向量大小
        :param dropout: dropout
        :param max_seq_len: 最大序列长度
        """
        super(WFTransformerDecoder, self).__init__()
        self.N = N
        self.embed = BasicEmbedding(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, max_seq_len)
        self.layers = nn.ModuleList([WFTransformerDecoderLayer(d_model, nhead, language_num, dim_feedforward, dropout=dropout) for _ in range(N)])
        self.norm = nn.LayerNorm(d_model)

    def freeze_share_weights(self):
        for layer in self.layers:
            layer.freeze_share_weights()

    def forward(self, x, e_outputs, src_mask, trg_mask, languageID: int):
        """

        :param x: 目标(预训练)/output(推理)的输入，要经过embedding和positional encode
        :param e_outputs: encoder的输出
        :param src_mask: padding mask,padding的部分是没有意义的
        :param trg_mask: 对标签进行mask
        :param languageID: 语言类别
        :return:
        """
        x = self.embed(x)
        x = self.pe(x)
        for i in range(self.N):
            print("输入向量形状", x.shape)
            x = self.layers[i](x, e_outputs, src_mask, trg_mask, languageID)
        return self.norm(x)
