from torch import Tensor


def BasicWF(x: Tensor, shareWeight: Tensor, rm: Tensor, sm: Tensor, ra: Tensor, sa: Tensor):
	"""
	Weight factorization, a weights-adaptive method,
	refers to the paper：《Efficient Weight factorization for Multilingual Speech Recognition》
	
	(Ws * (rm @ sm^T))^T @ X  +  (ra @ sa^T)^T @ X, * is elem wise product, @ is the inner product Refer to
	formula 18 in the paper But I converted it to Torch format,where Ws : (out_features,in_features),
	x : (batch,in_features),r_ : (in_features,1),s_ : (out_features,1)
		100,10,500
	"""
	# print("x: ", x.shape)
	# print("share_weight: ", shareWeight.shape, " sm: ", sm.shape, " rm ", rm.shape)
	# print("weight1: ", (shareWeight * (sm @ rm.transpose(-1, -2))).transpose(-1, -2).shape)
	# print("weight2: ", (sa @ ra.transpose(-1, -2)).transpose(-1, -2).shape)
	# print("final: ", ((x @ (shareWeight * (sm @ rm.transpose(-1, -2))).transpose(-1, -2)) + (
	# 			x @ (sa @ ra.transpose(-1, -2)).transpose(-1, -2))).shape)
	
	return (x @ (shareWeight * (sm @ rm.transpose(-1, -2))).transpose(-1, -2)) + (
			x @ (sa @ ra.transpose(-1, -2)).transpose(-1, -2))  # (batch,out_features)
