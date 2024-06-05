import os
import re
from typing import Tuple, List, Dict

from tqdm import tqdm

from src.utils.data.get_token_dict import load_dic, text2index
from src.utils.fileIO.path import get_path
from src.utils.fileIO.txt import read_txt, write_txt


def get_metadata_from_one_dir(info_dir: str, subset="train") -> Tuple[List[Dict[str, str]], str]:
    wav2text_path = os.path.join(info_dir, "wav2text.txt")
    lines = read_txt(wav2text_path)
    metadata_list = []
    for line in lines:
        line = line.strip()
        split_index = re.search("\s", line).span()[0]
        wave_path = line[:split_index:].replace("\\", "/")
        text = line[split_index:]
        text = re.sub("\s", "", text)
        metadata_list.append({"wave_path": wave_path, "text": text})
    return metadata_list, subset


def save_metadata_txt(metadata_dict_list: list, save_dir_path: str = "./", subset: str = ""):
    """
    将元数据写入txt文件
    :param metadata_dict_list: 元数据的字典列表，通过get_metadata_from_dirs生成
    :param save_dir_path:选择目录
    :param subset: train,val,test等
    :return:
    """
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)
    file_path = get_path(save_dir_path, subset, ".txt")

    with open(file_path, "a", encoding="utf-8") as file:
        for dic in tqdm(metadata_dict_list, desc="write KeSpeech to file", ncols=80, unit="KeSpeech"):
            file.write(f"{dic['wave_path']} {dic['text']}\n")


def get_metadata_from_metadatatxt(txt_file_path: str) -> List[Dict[str, str]]:
    lines = read_txt(txt_file_path)
    metadata_dic_list = []
    for line in lines:
        wave_path, text = line.split()
        metadata_dic_list.append({"wave_path": wave_path, "text": text})
    return metadata_dic_list


def split_byKey(metadata_dict_list: List[Dict[str, str]], dict_path: str, save_dir: str):
    text_list = []
    text_index_list = []
    wave_path_list = []
    char_dict = load_dic(dict_path)
    for metadata_dict in tqdm(metadata_dict_list, desc="split metadata", ncols=80):
        text = re.sub("\s", "", metadata_dict["text"])
        wave_path = re.sub("\s", "", metadata_dict["wave_path"])
        text_list.append(text)
        # 数字转文本
        text_index_list.append(str(text2index(char_dict, text)))
        wave_path_list.append(wave_path)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    text_path = get_path(save_dir, "text", ".txt")
    text_index_path = get_path(save_dir, "text_index", ".txt")
    wave_path = get_path(save_dir, "wave_path", ".txt")

    write_txt(text_list, text_path)
    write_txt(text_index_list, text_index_path)
    write_txt(wave_path_list, wave_path)


# if __name__ == "__main__":
# 	dir_list = [
# 		"/media/mixxis/T7/root/语音/MySpeech/Tasks/ASR/test",
# 		"/media/mixxis/T7/root/语音/MySpeech/Tasks/ASR/train",
# 	]
# 	subset_list = ["test", "train"]
# 	train_metadata,test_metadata = get_metadata_from_dirs(dir_list, subset_list, get_metadata_from_one_dir)
# 	print(
# 		f"train_metadata的数量：{len(train_metadata)}\nval_metadata的数量：{len(val_metadata)}\ntest_metadata的数量：{len(test_metadata)}"
# 	)
# 	print("======================================================================================================")
#
# 	save_metadata_txt(train_metadata, "../../../../MySpeech_data_metadata/dialect_metadata", "train")
# 	save_metadata_txt(test_metadata, "../../../../MySpeech_data_metadata/dialect_metadata", "test")

# if __name__ == "__main__":
#     train_metadata = get_metadata_from_metadatatxt(
#         "../../../../dataset_metadata/MySpeech_data_metadata/dialect_metadata/train.txt")
#     test_metadata = get_metadata_from_metadatatxt(
#         "../../../../dataset_metadata/MySpeech_data_metadata/dialect_metadata/test.txt")
#
#     split_byKey(train_metadata, "../../../../dataset_metadata/MySpeech_data_metadata/vocab.txt",
#                 "../../../../dataset_metadata/MySpeech_data_metadata/dialect_metadata/train")
#     split_byKey(test_metadata, "../../../../dataset_metadata/MySpeech_data_metadata/vocab.txt",
#                 "../../../../dataset_metadata/MySpeech_data_metadata/dialect_metadata/test")

if __name__ == "__main__":
    train_metadata = get_metadata_from_metadatatxt(
        "../../../../dataset_metadata/MySpeech_data_metadata/dialect_metadata/train.txt")
    test_metadata = get_metadata_from_metadatatxt(
        "../../../../dataset_metadata/MySpeech_data_metadata/dialect_metadata/test.txt")

    split_byKey(train_metadata, "../../../../dataset_metadata/MySpeech_data_metadata/vocab.txt",
                "../../../../dataset_metadata/MySpeech_data_metadata/dialect_metadata/train")
    split_byKey(test_metadata, "../../../../dataset_metadata/MySpeech_data_metadata/vocab.txt",
                "../../../../dataset_metadata/MySpeech_data_metadata/dialect_metadata/test")
