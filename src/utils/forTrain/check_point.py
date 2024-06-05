import os
import re

from src.utils.fileIO.json import read_json, write_json


def get_huggingface_checkpoint(output_dir: str, last_checkpoint: str = None, change_checkpoint=True):
    if last_checkpoint is not None:
        if change_checkpoint:
            _change_checkpoint(last_checkpoint)
        return last_checkpoint
    checkpoint = None
    max_num = 0
    for name in os.listdir(output_dir):
        path = os.path.join(output_dir, name)
        if "checkpoint" in name and os.path.isdir(path) and len(os.listdir(path)) != 0:
            num = int(re.findall("\d+\.?\d*", name)[0])
            checkpoint = path if num > max_num else checkpoint
            max_num = num if num > max_num else checkpoint
    if change_checkpoint and checkpoint is not None:
        print(f"检测到checkpoint,位于{checkpoint}")
        _change_checkpoint(checkpoint)
    return checkpoint


def _change_checkpoint(checkpoint):
    path = None
    for name in os.listdir(checkpoint):
        if name == "trainer_state.json":
            path = os.path.join(checkpoint, name)
    if path is None:
        return
    trainer_state = read_json(path)
    trainer_state["global_step"] = trainer_state["global_step"] + 10
    print("\n trainer_state changed!new global step = old global step + 10")
    write_json(trainer_state, path)


def load_my_checkpoint(path):
    return read_json(path)


def make_my_checkpoint(save_path, data_dict):
    write_json(data_dict, save_path)
