import torch
from torch import nn


def save_model(model: torch.nn.Module, path: str, save_state_dict: bool = True):
    if save_state_dict:
        # 检查模型是否使用了 nn.DataParallel 或 nn.parallel.DistributedDataParallel
        if isinstance(model, nn.DataParallel) or isinstance(model, nn.parallel.DistributedDataParallel):
            torch.save(model.module.state_dict(), path)
        else:
            torch.save(model.state_dict(), path)
    else:
        torch.save(model, path)


