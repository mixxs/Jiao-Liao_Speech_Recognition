import math

from torch import nn
import torch.nn.functional as F
from torch import Tensor
from src.modules.WFAdapter.LinearAdapt import WFLinear


def AttentionWF(Q: Tensor, K: Tensor, V: Tensor, d_k, mask: Tensor = None, dropout=None):
    """
    基于权重分解的attention，参考论文：《attention is all you need》,《Efficient Weight
    factorization for Multilingual Speech Recognition》和
    《exploration of language specific self-attention parameters for multilingual end to end speech recognition》

    shape:
        Q,K,V (batch_size,sequence_len,embed_dim) embed_dim:attention输入和输出的维度（这两个维度要相同，都是d_model）
        mask (batch_size,sequence_len)
    :param Q: q@Wq的结果
    :param K: k@Wk的结果
    :param V: v@(Wv_share*Wv_spefic)的结果
    :param d_k: dk
    :param mask: pad mask
    :param dropout: dropout
    :return:
    """
    alpha = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)  # (batch,nhead,sequence_len,sequence_len)

    if mask is not None: # 考虑batch size
        mask = mask.unsqueeze(1)
        mask = mask.unsqueeze(1)
        # print("alpha shape:", alpha.shape, " mask shape", mask.shape)
        alpha = alpha.masked_fill(mask == 0, -1e9)
    alpha = F.softmax(alpha, dim=-1)
    if dropout is not None:
        alpha = dropout(alpha)
    return alpha @ V


class WFMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, language_num, dropout=0.0, *args, **kwargs):
        """
        :param embed_dim: 模型输入输出的向量维度
        :param num_heads: 头的数量
        :param language_num: 语言的数量
        :param dropout: dropout
        :param args: args
        :param kwargs: kwargs
        """
        super(WFMultiHeadAttention, self).__init__(*args, **kwargs)
        self.embed_dim = embed_dim
        self.nhead = num_heads
        self.head_dim = embed_dim // num_heads  # 单个头的输入输出维度
        self.scaling = self.head_dim ** -0.5
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )

        self.linear_q = nn.Linear(embed_dim, embed_dim)
        self.linear_k = nn.Linear(embed_dim, embed_dim)
        self.linear_v = WFLinear(embed_dim, embed_dim, language_num)
        self.linear_o = WFLinear(embed_dim, embed_dim, language_num)

        self.dropout = nn.Dropout(dropout)

    def freeze_share_weights(self):
        self.linear_v.freeze_share_weights()
        self.linear_o.freeze_share_weights()

    def forward(self, q, k, v, languageID: int, mask=None, ):
        """
        :param q: q，用于计算Q=q@Wq
        :param k: k，用于计算Q=k@Wk
        :param v: v，用于计算Q=v@Wv
        :param mask: mask，
        :param languageID: 语言编号（指定语言）
        :return:
        """

        bs = q.size(0)

        # perform linear operation and split into N heads
        K = self.linear_k(k).view(bs, -1, self.nhead, self.head_dim)
        Q = self.linear_q(q).view(bs, -1, self.nhead, self.head_dim)* self.scaling  # (batch_size,seq_len,nhead,d_k)
        V = self.linear_v(v, languageID).view(bs, -1, self.nhead, self.head_dim)

        # transpose to get dimensions bs * nhead * sl * d_k
        K = K.transpose(1, 2)
        Q = Q.transpose(1, 2)
        V = V.transpose(1, 2)

        # calculate attention using function we will define next
        scores = AttentionWF(Q, K, V, self.head_dim, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.embed_dim)  # bs,sl,embed_dim
        output = self.linear_o(concat, languageID)  # bs,sl,embed_dim
        return output
