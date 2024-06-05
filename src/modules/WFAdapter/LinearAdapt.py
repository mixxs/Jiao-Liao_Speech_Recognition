import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from src.modules.WFAdapter.WeightFactorization import BasicWF


# TODO:修改初始化的方式

class WFLinear(nn.Module):
    """
    Linear layer based on weights factorization,
    refer to the pytorch documentation and paper 《Efficient Weight factorization for Multilingual Speech Recognition》
    """

    def __init__(self, in_features: int, out_features: int, language_num: int):
        # 初始化参数
        super(WFLinear, self).__init__()
        self.shareWeight = nn.Parameter(torch.randn(out_features, in_features))  # refer to that paper
        self.shareBias = nn.Parameter(torch.randn(out_features))
        self.rm = nn.Parameter(F.softmax(torch.randn(language_num, in_features, 1), dim=1))
        self.sm = nn.Parameter(F.softmax(torch.randn(language_num, out_features, 1), dim=1))
        self.ra = nn.Parameter(torch.randn(language_num, in_features, 1))
        self.sa = nn.Parameter(torch.randn(language_num, out_features, 1))
        # 设置初始值
        init.kaiming_uniform_(self.shareWeight, a=math.sqrt(5))
        init.kaiming_uniform_(self.ra, a=math.sqrt(5))
        init.kaiming_uniform_(self.sa, a=math.sqrt(5))
        init.constant_(self.shareBias, 0.01)

    def freeze_share_weights(self):
        self.shareWeight.requires_grad = False
        self.shareBias.requires_grad = False

    def forward(self, x, languageID: int):
        return BasicWF(x, shareWeight=self.shareWeight, rm=self.rm[languageID], sm=self.sm[languageID],
                       ra=self.ra[languageID], sa=self.sa[languageID]) + self.shareBias  # Y=x@W^T+B,W=(Ws*Wml+Wbl)

# if __name__ == "__main__":
# 	linear = WFLinear(10, 5, language_num=10)
# 	linear.cuda()
# 	x = torch.randn(100, 10).cuda()
# 	lossFunction = nn.CrossEntropyLoss()
# 	label = torch.arange(0, 100) % 5
# 	opti = torch.optim.Adam(linear.parameters())
# 	label = label.cuda()
# 	for i in range(0, 100):
# 		opti.zero_grad()
# 		y = linear(x, 3)
# 		loss = lossFunction(y, label)
# 		print(loss)
# 		loss.backward()
# 		opti.step()
