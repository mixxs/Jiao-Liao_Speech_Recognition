import math

import torch
from torch import nn


class PositionalEncoder(nn.Module):
	
	def __init__(self, d_model, max_seq_len=80):
		super().__init__()
		self.d_model = d_model
		
		# 根据pos和i创建一个常量pe矩阵
		pe = torch.zeros(max_seq_len, d_model)
		for pos in range(max_seq_len):
			for i in range(0, d_model, 2):
				pe[pos, i] = \
					math.sin(pos / (10000 ** ((2 * i) / d_model)))
				pe[pos, i + 1] = \
					math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
		
		pe = pe.unsqueeze(0)
		self.register_buffer('pe', pe)
	
	def forward(self, x):
		# 让 embeddings vector 相对大一些
		x = x * math.sqrt(self.d_model)
		# 增加位置常量到 embedding 中
		seq_len = x.size(1)
		x = x + torch.tensor(self.pe[:, :seq_len], requires_grad=False)
		return x