import os

import torch

from src.utils.fileIO.json import write_json, read_json


def save_metric(name: str, value, save_dir: str = "./result", save_name: str = None):
    """

	:param name:指标名称
	:param value:指标值
	:param save_dir:保存指标的路径
	:param save_name:保存指标的文件名
	"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if save_name is None:
        save_name = name + ".json"
    save_path = os.path.join(save_dir, save_name)
    datas = {"name": name}

    if not os.path.exists(save_path):
        datas["data"] = [{"index": 0, "value": value}, ]
    else:
        try:
            datas = read_json(json_path=save_path)
            index = max([data["index"] for data in datas["data"]]) + 1
            datas["data"].append({"index": index, "value": value})
        except Exception:
            datas["data"] = [{"index": 0, "value": value}, ]
        if datas["name"] != name:
            raise Exception(f"名称{datas['name']}和{name} 不匹配")
    write_json(datas, save_path)


def del_metric(name: str, metric_dir: str = "./result", file_name: str = None):
    if file_name is None:
        file_name = name + ".json"
    metric_path = os.path.join(metric_dir, file_name)
    os.remove(metric_path)


def load_metric(metric_name: str, metric_dir: str = "./result", file_name: str = None, return_list: bool = True):
    if file_name is None:
        file_name = metric_name + ".json"
    metric_path = os.path.join(metric_dir, file_name)
    metric_dict = read_json(metric_path)
    if not return_list:
        return metric_dict

    if metric_dict["name"] != metric_name:
        raise Exception(f"名称{metric_dict['name']}和{metric_name} 不匹配")
    metric_dict_list = sorted(metric_dict["data"], key=lambda da: da["index"])
    metric_list = [data["value"] for data in metric_dict_list]
    return metric_list


def show_metric(name: str, metric_dir: str = "./result", file_name: str = None):
    if file_name is None:
        file_name = name + ".json"
    metric_path = os.path.join(metric_dir, file_name)
    datas = read_json(metric_path)

    if datas["name"] != name:
        raise Exception(f"名称{datas['name']}和{name} 不匹配")
    show_name = name
    data_list = sorted(datas["data"], key=lambda da: da["index"])
    show_datas = [data["value"] for data in data_list]
    _show_metric(show_name, show_datas)


def _show_metric(show_name: str, show_datas: list):
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(r'./logs')
    show_datas = torch.tensor(show_datas)
    show_datas_dict = {i: show_datas[i] for i in range(len(show_datas))}
    for step, item in show_datas_dict.items():
        writer.add_scalars(f'{show_name}', {f'{show_name}': item}, global_step=step)

# if __name__=="__main__":
# 	show_metric("test")
# 	del_metric("test")
