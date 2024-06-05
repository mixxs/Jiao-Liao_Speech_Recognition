from torch import nn


class BasicEmbedding(nn.Module):
	"""
	用于output embedding,详见《attention is all you need》
	"""
	def __init__(self, vocab_size, d_model):
		"""
		初始化
		:param vocab_size: 字典大小
		:param d_model: 隐层向量大小，也是attention的向量大小
		"""
		super().__init__()
		self.d_model = d_model
		self.embed = nn.Embedding(vocab_size, d_model)
	
	def forward(self, x):
		return self.embed(x)
