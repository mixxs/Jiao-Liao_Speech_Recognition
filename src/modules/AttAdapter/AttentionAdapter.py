import torch.nn as nn


class AttAdapterBlock(nn.Module):

    def __init__(self, voice_feature_dim, dialect_feature_dim, num_head=16, num_language=9, dropout=0.0,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear_in = nn.Linear(in_features=dialect_feature_dim, out_features=voice_feature_dim)

        self.cross1 = nn.MultiheadAttention(embed_dim=voice_feature_dim, num_heads=num_head, dropout=dropout,
                                            batch_first=True)
        self.norm_inside1 = nn.LayerNorm(voice_feature_dim)
        self.norm_inside2 = nn.LayerNorm(voice_feature_dim)
        self.linear_out = nn.Linear(in_features=voice_feature_dim, out_features=voice_feature_dim)
        self.norm_out = nn.LayerNorm(voice_feature_dim)
        self.act = nn.ELU()

    def forward(self, voice_feature, dialect_feature, attention_mask):
        """
        :param voice_feature: 语音序列数特征,形状(batch_size, seq_len, embed_dim)
        :param dialect_feature: 方言特征(batch_size, embed_dim)
        :param attention_mask: 注意力掩码(batch_size, seq_len)
        """
        batch_size, seq_len, _ = voice_feature.shape
        if len(dialect_feature.shape) == 2:
            dialect_feature = dialect_feature.unsqueeze(1)
        if len(dialect_feature.shape) == 1:
            dialect_feature = dialect_feature.unsqueeze(0)
            dialect_feature = dialect_feature.unsqueeze(0)
        d_bs, d_sl, _ = dialect_feature.shape
        if d_bs == 1:
            dialect_feature = dialect_feature.expand(batch_size, -1, -1)
        if d_sl == 1:
            dialect_feature = dialect_feature.expand(-1, seq_len, -1)

        dialect_feature = self.norm_inside1(self.linear_in(dialect_feature))
        out = self.act(self.norm_inside2(voice_feature + self.cross1.forward(query=voice_feature, key=dialect_feature,
                                                                             value=dialect_feature,
                                                                             key_padding_mask=~(attention_mask.bool()))[
            0]))  # pytorch的MHattention会默认返回两个tensor,然后取反是因为True对应的KEY会被忽略
        del dialect_feature
        out = self.act(self.norm_out(voice_feature + self.linear_out(out)))  # 将输入直接横跨整个模块做残差连接
        return out
