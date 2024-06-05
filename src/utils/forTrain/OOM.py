import os.path
import shutil

import torch.cuda


def afterOOM(output_dir_path, error):
    torch.cuda.empty_cache()
    # 规定check_point的名称
    num_list = []
    for name in os.listdir(output_dir_path):
        path = os.path.join(output_dir_path, name)
        if os.path.isdir(path) and "checkpoint-" in name:
            num_str = name.replace("checkpoint-", "")
            if num_str == "last":
                shutil.rmtree(path)  # 删除上一次遗留的checkpoint-last
            else:
                num_list.append(int(num_str))

    max_num = max(num_list)
    last_name = "checkpoint-" + str(max_num)
    old_path = os.path.join(output_dir_path, last_name)
    new_path = os.path.join(output_dir_path, "checkpoint-last")
    os.rename(old_path, new_path)
    raise error
