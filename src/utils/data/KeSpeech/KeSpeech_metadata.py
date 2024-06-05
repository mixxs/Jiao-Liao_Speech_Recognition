import os
import re
from typing import List, Dict, Tuple

from tqdm import tqdm

from src.utils.data.get_token_dict import text2index, load_dic
from src.utils.fileIO.path import get_path
from src.utils.fileIO.txt import write_txt, read_txt

DIALECT_LS = ['Northeastern', 'Mandarin', 'Ji-Lu', 'Zhongyuan', 'Jiao-Liao', 'Jiang-Huai', 'Beijing', 'Southwestern',
              'Lan-Yin']


def get_language_id(language: str, language_list=None) -> int:
    if language_list is None:
        language_list = DIALECT_LS
    return language_list.index(language)


def get_metadata_from_one_dir(info_dir: str, subset="train") -> Tuple[List[Dict[str, str]], str]:
    """
    get info: utt,spk,text,dialect,wave_path
    :param info_dir:
    :param subset:
    :return:List[Dict]
    """
    metadata_dic_list = []
    utt2spk = read_txt(os.path.join(info_dir, "utt2spk"))
    utt2text = read_txt(os.path.join(info_dir, "text"))
    utt2subdialect = read_txt(os.path.join(info_dir, "utt2subdialect"))
    wav = read_txt(os.path.join(info_dir, "wav.scp"))

    for uttAndspk in tqdm(utt2spk, desc=f'Extract from {info_dir}', unit='utt'):
        utt, spk = uttAndspk.split(" ")
        utt = re.sub("\s", "", utt)
        spk = re.sub("\s", "", spk)
        text = ""
        dialect = ""
        wave_path = ""
        for uttAndtext in utt2text:
            utt_temp, text_temp = uttAndtext.split(" ")
            utt_temp = re.sub("\s", "", utt_temp)
            text_temp = re.sub("\s", "", text_temp)
            # 根据utt进行匹配，如果找到，则保存到dict,删除这一项，然后退出循环
            if utt_temp == utt:
                text = text_temp
                utt2text.remove(uttAndtext)
                break
            # 否则直接结束程序
            if uttAndtext == utt2text[-1]:
                print(f"{utt} not found in utt2text")
                exit(1)
        for uttAnddialect in utt2subdialect:
            utt_temp, dialect_temp = uttAnddialect.split(" ")
            utt_temp = re.sub("\s", "", utt_temp)
            dialect_temp = re.sub("\s", "", dialect_temp)
            if utt_temp == utt:
                dialect = dialect_temp
                utt2subdialect.remove(uttAnddialect)
                break
            if uttAnddialect == utt2subdialect[-1]:
                print(f"{utt} not found in utt2subdialect")
                exit(1)
        for uttAndwav in wav:
            utt_temp, wav_temp = uttAndwav.split(" ")
            utt_temp = re.sub("\s", "", utt_temp)
            wav_temp = re.sub("\s", "", wav_temp)
            if utt_temp == utt:
                wave_path = wav_temp
                wav.remove(uttAndwav)
                break
            if uttAndwav == wav[-1]:
                print(f"{utt} not found in wav")
                exit(1)
        metadata_dic_list.append({"utt": utt, "spk": spk, "text": text, "dialect": dialect, "wave_path": wave_path})
    print(f"\n{info_dir}：{len(utt2spk), len(metadata_dic_list)}\n")
    return metadata_dic_list, subset


def get_metadata_from_dirs(dir_list: list, subset_list: list, get_from_one_dir_func) -> Tuple[List, List, List]:
    """

    :param dir_list: info_dirs
    :return: [dict,dict,dict……]
    """
    train_meta = []
    val_meta = []
    test_meta = []
    for (dir, subset) in zip(dir_list, subset_list):
        metadata, subset = get_from_one_dir_func(dir, subset)
        if subset == "train":
            train_meta += metadata
        elif subset == "val":
            val_meta += metadata
        else:
            test_meta += metadata
    return train_meta, val_meta, test_meta


# (metadata_dict_list,subset) = get_metadata_from_one_dir(first_dir)
# for dir in tqdm(dir_list, desc="merge KeSpeech", ncols=80, unit="dir"):
# 	dict_list = get_metadata_from_one_dir(dir)
# 	metadata_dict_list += dict_list
# return metadata_dict_list

def save_metadata_txt(metadata_dict_list: list, file_name: str, save_dir_path: str = "./", subset: str = ""):
    """
    将元数据写入txt文件
    :param file_name:
    :param metadata_dict_list: 元数据的字典列表，通过get_metadata_from_dirs生成
    :param save_dir_path:选择目录
    :param subset: train,val,test等
    :return:
    """
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)
    file_path = get_path(save_dir_path, subset + file_name, ".txt")

    with open(file_path, "a", encoding="utf-8") as file:
        for dic in tqdm(metadata_dict_list, desc="write KeSpeech to file", ncols=80, unit="KeSpeech"):
            file.write(f"{dic['utt']} {dic['spk']} {dic['text']} {dic['dialect']} {dic['wave_path']}\n")


def get_metadata_from_metadatatxt(txt_file_path: str) -> List[Dict[str, str]]:
    lines = read_txt(txt_file_path)

    while lines[0].split(" ")[0] == "subDataset_class:" or lines[0].split(" ")[0] == "dataset_base_dir:":
        if lines[0].split(" ")[0] == "subDataset_class:":
            subDatasetclass = re.sub("\s", "", lines[0].split(" ")[1])
            lines.pop(0)
        if lines[0].split(" ")[0] == "dataset_base_dir:":
            dataset_base_dir = re.sub("\s", "", lines[0].split(" ")[1])
            lines.pop(0)

    metadata_dic_list = []
    for line in lines:
        utt, spk, text, dialect, wave_path = line.split(" ")
        metadata_dic_list.append(
            {"utt": re.sub("\s", "", utt), "spk": re.sub("\s", "", spk), "text": re.sub("\s", "", text),
             "dialect": re.sub("\s", "", dialect), "wave_path": re.sub("\s", "", wave_path)})
    return metadata_dic_list


def split_byLanguage(metadata_dic_list: list):
    """

    :param metadata_dic_list: 通过get_metadata_from_dirs等函数获取的元数据字典列表
    :return: [ [dict,dict……], [dict,dict……] …… ]
    """
    dialect_list = DIALECT_LS
    metadata_list = [[] for _ in DIALECT_LS]
    for dic in tqdm(metadata_dic_list, desc=f"iterate over metadata", ncols=80, unit="unit"):
        metadata_list[dialect_list.index(dic["dialect"])].append(dic)
    return metadata_list


def split_byKey(metadata_dict_list: List[Dict[str, str]], dict_path: str, save_dir: str):
    utt_list = []
    spk_list = []
    text_list = []
    text_index_list = []
    dialect_index_list = []
    wave_path_list = []
    char_dict = load_dic(dict_path)
    for metadata_dict in tqdm(metadata_dict_list, desc="split metadata", ncols=80):
        utt = re.sub("\s", "", metadata_dict["utt"])
        spk = re.sub("\s", "", metadata_dict["spk"])
        text = re.sub("\s", "", metadata_dict["text"])
        wave_path = re.sub("\s", "", metadata_dict["wave_path"])
        dialect = re.sub("\s", "", metadata_dict["dialect"])
        utt_list.append(utt)
        spk_list.append(spk)
        text_list.append(text)
        # 数字转文本
        text_index_list.append(str(text2index(char_dict, text)))
        dialect_index_list.append(str(DIALECT_LS.index(dialect)))
        wave_path_list.append(wave_path)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    utt_path = get_path(save_dir, "utt", ".txt")
    spk_path = get_path(save_dir, "spk", ".txt")
    text_path = get_path(save_dir, "text", ".txt")
    text_index_path = get_path(save_dir, "text_index", ".txt")
    dialect_index_path = get_path(save_dir, "dialect_index", ".txt")
    wave_path = get_path(save_dir, "wave_path", ".txt")

    write_txt(utt_list, utt_path)
    write_txt(spk_list, spk_path)
    write_txt(text_list, text_path)
    write_txt(text_index_list, text_index_path)
    write_txt(dialect_index_list, dialect_index_path)
    write_txt(wave_path_list, wave_path)


# if __name__ == "__main__":
# 	dir_list = [
# 		"/media/mixxis/T7/语音数据集/KeSpeech/Tasks/ASR/test",
# 		"/media/mixxis/T7/语音数据集/KeSpeech/Tasks/ASR/train_phase1",
# 		"/media/mixxis/T7/语音数据集/KeSpeech/Tasks/ASR/train_phase2",
# 		"/media/mixxis/T7/语音数据集/KeSpeech/Tasks/ASR/dev_phase1",
# 		"/media/mixxis/T7/语音数据集/KeSpeech/Tasks/ASR/dev_phase2"
# 	]
# 	subset_list = ["test", "train", "train", "val", "val"]
# 	train_metadata, val_metadata, test_metadata = get_metadata_from_dirs(dir_list, subset_list, get_metadata_from_one_dir)
# 	print(
# 		f"train_metadata的数量：{len(train_metadata)}\nval_metadata的数量：{len(val_metadata)}\ntest_metadata的数量：{len(test_metadata)}"
# 	)
# 	print("======================================================================================================")
# 	train_splited = split_byLanguage(train_metadata)
# 	val_splited = split_byLanguage(val_metadata)
# 	test_splited = split_byLanguage(test_metadata)
# 	print(f"train_splited各部分大小：{[len(train_splited_i) for train_splited_i in train_splited]}")
# 	print(f"val_splited各部分大小：{[len(val_splited_i) for val_splited_i in val_splited]}")
# 	print(f"test_splited各部分大小：{[len(test_splited_i) for test_splited_i in test_splited]}")
# 	print("======================================================================================================")
# 	for meta in tqdm(train_splited, desc="save KeSpeech", ncols=80):
# 		if len(meta) == 0:
# 			break
# 		dialect_name = meta[0]["dialect"]
# 		save_metadata_txt(meta, dialect_name,
# 		                  os.path.join("../../../../KeSpeech_data_metadata/dialect_metadata", dialect_name), "train")
# 	for meta in tqdm(val_splited, desc="save KeSpeech", ncols=80):
# 		if len(meta) == 0:
# 			break
# 		dialect_name = meta[0]["dialect"]
# 		save_metadata_txt(meta, dialect_name,
# 		                  os.path.join("../../../../KeSpeech_data_metadata/dialect_metadata", dialect_name), "val")
# 	for meta in tqdm(test_splited, desc="save KeSpeech", ncols=80):
# 		if len(meta) == 0:
# 			break
# 		dialect_name = meta[0]["dialect"]
# 		save_metadata_txt(meta, dialect_name,
# 		                  os.path.join("../../../../KeSpeech_data_metadata/dialect_metadata", dialect_name), "test")


if __name__ == "__main__":
    txt_file_paths = [
        "../../../../dataset_metadata/KeSpeech_data_metadata/dialect_metadata/Beijing/trainBeijing.txt",
        "../../../../dataset_metadata/KeSpeech_data_metadata/dialect_metadata/Beijing/valBeijing.txt",
        "../../../../dataset_metadata/KeSpeech_data_metadata/dialect_metadata/Beijing/testBeijing.txt",

        "../../../../dataset_metadata/KeSpeech_data_metadata/dialect_metadata/Ji-Lu/trainJi-Lu.txt",
        "../../../../dataset_metadata/KeSpeech_data_metadata/dialect_metadata/Ji-Lu/valJi-Lu.txt",
        "../../../../dataset_metadata/KeSpeech_data_metadata/dialect_metadata/Ji-Lu/testJi-Lu.txt",

        "../../../../dataset_metadata/KeSpeech_data_metadata/dialect_metadata/Jiang-Huai/trainJiang-Huai.txt",
        "../../../../dataset_metadata/KeSpeech_data_metadata/dialect_metadata/Jiang-Huai/valJiang-Huai.txt",
        "../../../../dataset_metadata/KeSpeech_data_metadata/dialect_metadata/Jiang-Huai/testJiang-Huai.txt",

        "../../../../dataset_metadata/KeSpeech_data_metadata/dialect_metadata/Jiao-Liao/trainJiao-Liao.txt",
        "../../../../dataset_metadata/KeSpeech_data_metadata/dialect_metadata/Jiao-Liao/valJiao-Liao.txt",
        "../../../../dataset_metadata/KeSpeech_data_metadata/dialect_metadata/Jiao-Liao/testJiao-Liao.txt",

        "../../../../dataset_metadata/KeSpeech_data_metadata/dialect_metadata/Lan-Yin/trainLan-Yin.txt",
        "../../../../dataset_metadata/KeSpeech_data_metadata/dialect_metadata/Lan-Yin/valLan-Yin.txt",
        "../../../../dataset_metadata/KeSpeech_data_metadata/dialect_metadata/Lan-Yin/testLan-Yin.txt",

        "../../../../dataset_metadata/KeSpeech_data_metadata/dialect_metadata/Mandarin/trainMandarin.txt",
        "../../../../dataset_metadata/KeSpeech_data_metadata/dialect_metadata/Mandarin/valMandarin.txt",
        "../../../../dataset_metadata/KeSpeech_data_metadata/dialect_metadata/Mandarin/testMandarin.txt",

        "../../../../dataset_metadata/KeSpeech_data_metadata/dialect_metadata/Northeastern/trainNortheastern.txt",
        "../../../../dataset_metadata/KeSpeech_data_metadata/dialect_metadata/Northeastern/valNortheastern.txt",
        "../../../../dataset_metadata/KeSpeech_data_metadata/dialect_metadata/Northeastern/testNortheastern.txt",

        "../../../../dataset_metadata/KeSpeech_data_metadata/dialect_metadata/Southwestern/trainSouthwestern.txt",
        "../../../../dataset_metadata/KeSpeech_data_metadata/dialect_metadata/Southwestern/valSouthwestern.txt",
        "../../../../dataset_metadata/KeSpeech_data_metadata/dialect_metadata/Southwestern/testSouthwestern.txt",

        "../../../../dataset_metadata/KeSpeech_data_metadata/dialect_metadata/Zhongyuan/trainZhongyuan.txt",
        "../../../../dataset_metadata/KeSpeech_data_metadata/dialect_metadata/Zhongyuan/valZhongyuan.txt",
        "../../../../dataset_metadata/KeSpeech_data_metadata/dialect_metadata/Zhongyuan/testZhongyuan.txt",
    ]
    subset_list = ["train", "val", "test"] * 9
    train_splited = []
    val_splited = []
    test_splited = []
    for (txt_file_path, subset) in tqdm(zip(txt_file_paths, subset_list), desc="read from files", ncols=100,
                                        unit="file"):
        meta_dic = get_metadata_from_metadatatxt(txt_file_path)
        if subset == "train":
            train_splited.append(meta_dic)
        elif subset == "val":
            val_splited.append(meta_dic)
        else:
            test_splited.append(meta_dic)

    for meta in tqdm(train_splited, desc="save KeSpeech", ncols=80):
        dialect_name = meta[0]["dialect"]
        dialect_dir = os.path.join("../../../../dataset_metadata/KeSpeech_data_metadata/dialect_metadata", dialect_name)
        dialect_subset_dir = os.path.join(dialect_dir, "train")
        if not os.path.exists(dialect_subset_dir):
            os.mkdir(dialect_subset_dir)
        split_byKey(meta, "../../../../dataset_metadata/KeSpeech_data_metadata/vocab.txt", dialect_subset_dir)

    for meta in tqdm(val_splited, desc="save KeSpeech", ncols=80):
        dialect_name = meta[0]["dialect"]
        dialect_dir = os.path.join("../../../../dataset_metadata/KeSpeech_data_metadata/dialect_metadata", dialect_name)
        dialect_subset_dir = os.path.join(dialect_dir, "val")
        if not os.path.exists(dialect_subset_dir):
            os.mkdir(dialect_subset_dir)
        split_byKey(meta, "../../../../dataset_metadata/KeSpeech_data_metadata/vocab.txt", dialect_subset_dir)

    for meta in tqdm(test_splited, desc="save KeSpeech", ncols=80):
        dialect_name = meta[0]["dialect"]
        dialect_dir = os.path.join("../../../../dataset_metadata/KeSpeech_data_metadata/dialect_metadata",
                                   dialect_name, )
        if not os.path.exists(dialect_dir):
            os.mkdir(dialect_dir)
        dialect_subset_dir = os.path.join(dialect_dir, "test")
        if not os.path.exists(dialect_subset_dir):
            os.mkdir(dialect_subset_dir)
        split_byKey(meta, "../../../../dataset_metadata/KeSpeech_data_metadata/vocab.txt", dialect_subset_dir)
