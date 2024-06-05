import json
import os
import random
from typing import List, Dict

from datasets import Dataset
from tqdm import tqdm

from src.utils.data.KeSpeech.KeSpeech_metadata import get_language_id
from src.utils.fileIO.json import read_json, write_json


def readKeSpeechData(KeSpeechJsonPath: str, dialect_list: List[str]) -> List[dict]:
    """
    从KeSpeech中读取数据，并按照方言进行分类读取
    :param KeSpeechJsonPath: KeSpeech的json元数据文件的路径
    :dialect_list:方言列表
    """
    KeSpeechData = read_json(KeSpeechJsonPath)
    dialect_data_list = []
    for dialect in dialect_list:
        dialect_dict = KeSpeechData[dialect]
        dialect_data_list.append(dialect_dict)

    return dialect_data_list


def readMySpeechData(MySpeechJsonPath: str):
    MySpeechData = read_json(MySpeechJsonPath)
    return [MySpeechData]


def getKeSpeechSubsetData(dialect_data_list: List[dict], subset: str) -> List[dict]:
    """
    从KeSpeech中读取训练集、验证集或者测试集的数据
    :param dialect_data_list:经过readKeSpeechData函数处理后得到的结果，是方言数据字典的列表
    :param subset:要读取的数据集划分
    """
    subset_data_list = []
    if subset not in ["train", "val", "test"]:
        raise Exception("subset must in ['train','val','test']")
    for dialect_data in dialect_data_list:
        subset_data_list.append(dialect_data[subset])
    return subset_data_list


def _makeTripletDataset(temp_dataset_dict: Dict[str, List[dict]], a_dialect=None) -> Dict[str, List[dict]]:
    """
    生成三元组数据集
    :param temp_dataset_dict:{"dialect":data_list}格式的字典

    """
    tripletDataset = {"data": []}
    dialect_list = temp_dataset_dict.keys()
    if a_dialect is not None:
        dialect = a_dialect
        dialect_data_list = temp_dataset_dict[dialect]
        for data in tqdm(dialect_data_list, "正在生成三元组数据"):
            a = data
            p = random.choice(dialect_data_list)
            dialect_list_temp = [dia for dia in dialect_list if dia != dialect]
            n_dialect = random.choice(dialect_list_temp)
            n = random.choice(temp_dataset_dict[n_dialect])
            tripletDataset["data"].append(
                {"pathP": p["path"], "pathA": a["path"], "pathN": n["path"], "labelAP": get_language_id(dialect),
                 "labelN": get_language_id(n_dialect)})
        return tripletDataset
    for dialect in tqdm(dialect_list, desc="正在遍历每一种方言"):

        dialect_data_list = temp_dataset_dict[dialect]
        for data in tqdm(dialect_data_list, "正在生成三元组数据"):
            a = data
            p = random.choice(dialect_data_list)
            dialect_list_temp = [dia for dia in dialect_list if dia != dialect]
            n_dialect = random.choice(dialect_list_temp)
            n = random.choice(temp_dataset_dict[n_dialect])
            tripletDataset["data"].append(
                {"pathP": p["path"], "pathA": a["path"], "pathN": n["path"], "labelAP": get_language_id(dialect),
                 "labelN": get_language_id(n_dialect)})
    return tripletDataset


def makeTripletDataset(subset_data_list: List[dict], savePath: str, a_dialect=None):
    """
    保存生成的三元组数据集
    :param subset_data_list:经过getKeSpeechSubsetData函数处理得到的某一个划分的数据列表
    :param savePath:保存的路径
    """
    temp_dataset_dict = {}
    for subsetData in tqdm(subset_data_list, desc="正在处理每一个方言", unit="dialect"):
        dialect = subsetData["dialect"]
        temp_dataset_dict[dialect] = subsetData["data"]
        print(len(temp_dataset_dict[dialect]))
    tri_dataset = _makeTripletDataset(temp_dataset_dict, a_dialect)
    write_json(tri_dataset, savePath)


def process_triplet_dataset(datasetObj: Dataset, dataset_base_dir: str, num_proc: int,
                            a_dataset_base_dir=None) -> Dataset:
    """
    对数据集进行初步的处理
    :param datasetObj: 数据集
    :param dataset_base_dir: 数据集根目录
    :param num_proc: 使用多少线程
    :return:
    """

    def speech_path_fn(batch):
        batch["input_values"] = {}

        if a_dataset_base_dir is None:
            batch["input_values"]["A"] = os.path.join(dataset_base_dir, batch["pathA"])
            batch["input_values"]["P"] = os.path.join(dataset_base_dir, batch["pathP"])
        else:
            batch["input_values"]["A"] = os.path.join(a_dataset_base_dir, batch["pathA"])
            batch["input_values"]["P"] = os.path.join(a_dataset_base_dir, batch["pathP"])
        batch["input_values"]["N"] = os.path.join(dataset_base_dir, batch["pathN"])

        return batch

    datasetObj = datasetObj.map(
        speech_path_fn,
        remove_columns=["pathP", "pathA", "pathN"],
        num_proc=num_proc
    )
    datasetObj = datasetObj.shuffle()
    return datasetObj


def getTripletDataset(file_path: str, dataset_base_dir: str, num_proc: int, a_dataset_base_dir=None) -> Dataset:
    """
    如果已经保存了三元组数据，就可以通过这个函数来获取经过初步处理的三元组数据集
    :param file_path:保存好的三元组json数据文件
    :param dataset_base_dir:数据集根目录
    :param num_proc:处理数据集的进程数
    :param a_dataset_base_dir: 如果非空，则将a的根目录设为该值(考虑数据不在一个根目录)
    """
    with open(file_path, "r", encoding="utf-8") as f:
        dataset = Dataset.from_list(json.load(f)["data"])
    dataset = process_triplet_dataset(dataset, dataset_base_dir, num_proc, a_dataset_base_dir)
    return dataset


# if __name__ == "__main__":
# 	random.seed(42)
# 	path = "../../../../dataset_metadata/KeSpeech_data_metadata/KeSpeechDataset_shrinked.json"
# 	train_save_path = "../../../../dataset_metadata/KeSpeech_data_metadata/TripletDataset_train.json"
# 	val_save_path = "../../../../dataset_metadata/KeSpeech_data_metadata/TripletDataset_val.json"
# 	test_save_path = "../../../../dataset_metadata/KeSpeech_data_metadata/TripletDataset_test.json"
# 	list = DIALECT_LS
# 	makeTripletDataset(getKeSpeechSubsetData(readKeSpeechData(path, list), "train"), train_save_path)
# 	makeTripletDataset(getKeSpeechSubsetData(readKeSpeechData(path, list), "val"), val_save_path)
# 	makeTripletDataset(getKeSpeechSubsetData(readKeSpeechData(path, list), "test"), test_save_path)

if __name__ == "__main__":
    random.seed(42)
    mySpeechPath = "../../../../dataset_metadata/MySpeech_data_metadata/MySpeechDataset.json"
    keSpeechPath = "../../../../dataset_metadata/KeSpeech_data_metadata/KeSpeechDataset_shrinked.json"
    train_save_path = "../../../../dataset_metadata/MySpeech_data_metadata/TripletDataset_train.json"
    test_save_path = "../../../../dataset_metadata/MySpeech_data_metadata/TripletDataset_test.json"
    list = ['Northeastern', 'Mandarin', 'Ji-Lu', 'Zhongyuan', 'Jiang-Huai', 'Beijing', 'Southwestern',
            'Lan-Yin']
    mySpeechData = readMySpeechData(mySpeechPath)
    my_train_set_data = getKeSpeechSubsetData(mySpeechData, "train")
    my_test_set_data = getKeSpeechSubsetData(mySpeechData, "test")

    keSpeechData = readKeSpeechData(keSpeechPath, list)
    keSpeech_train_set_data = getKeSpeechSubsetData(keSpeechData, "train")
    keSpeech_test_set_data = getKeSpeechSubsetData(keSpeechData, "test")

    train_set_data = my_train_set_data + keSpeech_train_set_data
    test_set_data = my_test_set_data + keSpeech_test_set_data
    makeTripletDataset(train_set_data, train_save_path, a_dialect="Jiao-Liao")
    makeTripletDataset(test_set_data, test_save_path, a_dialect="Jiao-Liao")

# if __name__ == "__main__":
#     dataset = getTripletDataset(
#         "/media/mixxis/T7/code/python/speech_recognition/dataset_metadata/KeSpeech_data_metadata/TripletDataset_val.json",
#         "/media/mixxis/T7/语音数据集/KeSpeech/", 8)
#     print(dataset.num_rows)
#     print(dataset.num_columns)
#     print(dataset.features)

# if __name__ == "__main__":
# 	data = read_json("../../../../dataset_metadata/KeSpeech_data_metadata/TripletDataset_train.json")["data"]
# 	print(len(data))
