import torch.nn as nn

from src.modules.WFAdapter.AttentionWF import WFMultiHeadAttention
from src.modules.WFAdapter.LinearAdapt import WFLinear


class WFLinearAdapter(nn.Module):
    """
    全连接Adapter
    """

    def __init__(self, in_dim, embed_dim, language_num, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linears_in = WFLinear(in_features=in_dim, out_features=embed_dim, language_num=language_num)
        self.linears_out = WFLinear(in_features=embed_dim, out_features=in_dim, language_num=language_num)
        self.norm_inside = nn.LayerNorm(embed_dim)
        self.norm_outside = nn.LayerNorm(in_dim)
        self.act = nn.ELU()

    def freeze_share_weights(self):
        self.linears_in.freeze_share_weights()
        self.linears_out.freeze_share_weights()

    def forward(self, x, language_id):
        out = self.norm_inside(self.act(self.linears_in.forward(x, language_id)))

        out = self.norm_outside(self.act(x + self.linears_out.forward(out, language_id)))
        return out


class WFMHAdapter(nn.Module):
    """
    Multi head attention Adapter
    """

    def __init__(self, in_dim, embed_dim, language_num, head_num, dropout=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear_in = nn.Linear(in_features=in_dim, out_features=embed_dim)
        self.WF_MHAttention = WFMultiHeadAttention(embed_dim=embed_dim, num_heads=head_num, language_num=language_num,
                                                   dropout=dropout)
        self.linear_out = nn.Linear(in_features=embed_dim, out_features=in_dim)
        self.norm = nn.LayerNorm(in_dim)
        self.norm_inside = nn.LayerNorm(embed_dim)
        self.act = nn.ELU()

    def freeze_share_weights(self):
        self.WF_MHAttention.freeze_share_weights()

    def forward(self, x, language_id):
        out = self.act(self.norm(self.linear_in(x)))
        out = self.act(self.norm_inside(self.WF_MHAttention.forward(q=out, k=out, v=out, languageID=language_id)))
        out = self.act(self.norm(x + self.linear_out(out)))
        return out


class LinearAdapter(nn.Module):
    """
    全连接Adapter
    """

    def __init__(self, in_dim, embed_dim, language_num, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linears_in = nn.ModuleList()
        self.linears_out = nn.ModuleList()
        for lan in range(language_num):
            self.linears_in.append(nn.Linear(in_features=in_dim, out_features=embed_dim))
        for lan in range(language_num):
            self.linears_out.append(nn.Linear(in_features=embed_dim, out_features=in_dim))
        self.norm_inside = nn.LayerNorm(embed_dim)
        self.norm_outside = nn.LayerNorm(in_dim)
        self.act = nn.ELU()

    def forward(self, x, language_id):
        out = self.norm_inside(self.act(self.linears_in[language_id].forward(x)))
        out = self.norm_outside(self.act(x + self.linears_out[language_id].forward(out)))
        return out
