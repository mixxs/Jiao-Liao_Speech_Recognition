import json
import os.path
from typing import Dict, List, Optional, Union

import numpy as np
import soundfile as sf
import torch
from audiomentations import (
    Compose,
    AddGaussianNoise,
    Gain,
    PitchShift,
    TimeStretch,
)
from transformers import Wav2Vec2Processor

from src.utils.data.KeSpeech.load_data import load_one_language_infos
from src.utils.data.process import make_processor
from src.utils.fileIO.json import write_json


def make_huggingface_data(languagesToLoad: str, subset_name, language2dir_dict, save_path, max_example_num: int = None):
    data_list = load_one_language_infos(["wave_path", "dialect_index", "text"],
                                        languagesToLoad, subset_name, language2dir_dict)
    path = data_list[0]
    dialect_index = data_list[1]
    text = data_list[2]
    data_dict_dict = {"dialect": languagesToLoad, "dialect_index": dialect_index[0], "subset": subset_name,
                      "data": []}
    for index in range(len(path)):
        if max_example_num is not None and index >= max_example_num:
            break
        data_dict_dict["data"].append({"path": path[index], "text": text[index], "dialect_index": dialect_index[0]})
    write_json(data_dict_dict, save_path)


def merge_hugging_face_dataset(dialect_metadata_dir_path, save_path):
    dataset_dict = {}
    for dialect_name in os.listdir(dialect_metadata_dir_path):
        dialect_dir = os.path.join(dialect_metadata_dir_path, dialect_name)
        dialect_dict = {}
        if not os.path.isdir(dialect_dir):
            continue
        for subset_name in os.listdir(dialect_dir):
            subset_dir = os.path.join(dialect_dir, subset_name)
            if not os.path.isdir(subset_dir):
                continue
            dataset_json_path = os.path.join(subset_dir, "KeSpeechDataset.json")
            with open(dataset_json_path, "r", encoding="utf-8") as f:
                dataset = json.load(f)
            dialect_dict[subset_name] = dataset
        dataset_dict[dialect_name] = dialect_dict
    write_json(dataset_dict, save_path)


def process_hugging_face_dataset(dataset, datasetbase_dir, num_proc, processor: Wav2Vec2Processor):
    """

    :param dataset: 数据集
    :param datasetbase_dir: 数据集根目录
    :param num_proc: 使用多少线程
    :param processor: Wav2Vec2Processor对象
    :return:
    """

    def speech_path_fn(batch):
        batch["input_values"] = os.path.join(datasetbase_dir, batch["path"])
        return batch

    dataset = dataset.map(
        speech_path_fn,
        remove_columns=["path"],
        num_proc=num_proc
    )

    def prepare_dataset_fn(batch):
        # check that all files have the correct sampling rate
        # Setup the processor for targets

        batch["labels"] = processor(text=batch["text"], ).input_ids
        return batch

    dataset = dataset.map(
        prepare_dataset_fn,
        remove_columns=["text"],
        num_proc=num_proc
    )
    dataset = dataset.shuffle()
    return dataset


def make_hugging_face_datasets(data_json: str, languages: list, subsets: list, dataset_dir, num_proc, processor,
                               merge: bool = False):
    from datasets import Dataset
    datasets_dict = {}
    for key in languages:
        temp = {}
        for subset in subsets:
            with open(data_json, "r", encoding="utf-8") as f:
                temp[subset] = process_hugging_face_dataset(Dataset.from_list(json.load(f)[key][subset]["data"]),
                                                            dataset_dir, num_proc, processor)
        datasets_dict[key] = temp
        del temp
    if merge:
        merged_datasets_dict = {}
        for sub_set in subsets:
            sub_datasets = [datasets_dict[k][sub_set] for k in datasets_dict.keys()]  # train,val,test
            from datasets import concatenate_datasets
            subset_dataset = concatenate_datasets(sub_datasets)
            merged_datasets_dict[sub_set] = subset_dataset
        return merged_datasets_dict  # train,val,test
    return datasets_dict  # Mandarin,Lan-Yin,Jiang-Huai,Jiao-Liao……


class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None  # 默认pad到整个序列的最大长度
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __init__(self, processor, padding=True, apply_gaussian_noise_with_p=0.5, apply_gain_with_p=0.5,
                 apply_pitch_shift_with_p=0.5,
                 apply_time_stretch_with_p=0.5, sample_rate=16_000):
        self.processor = processor
        self.padding = padding
        self.apply_gaussian_noise_with_p = apply_gaussian_noise_with_p
        self.apply_gain_with_p = apply_gain_with_p
        self.apply_pitch_shift_with_p = apply_pitch_shift_with_p
        self.apply_time_stretch_with_p = apply_time_stretch_with_p
        self.sample_rate = sample_rate

        self.augmentator = None
        if self.apply_gaussian_noise_with_p + self.apply_gain_with_p + self.apply_pitch_shift_with_p + self.apply_time_stretch_with_p > 0:
            self.augmentator = Compose([
                TimeStretch(min_rate=0.8, max_rate=1.2, leave_length_unchanged=False, p=self.apply_time_stretch_with_p),
                PitchShift(min_semitones=-1, max_semitones=1, p=self.apply_pitch_shift_with_p),
                Gain(min_gain_in_db=-1, max_gain_in_db=1, p=self.apply_gain_with_p),
                AddGaussianNoise(min_amplitude=0.0001, max_amplitude=0.001, p=self.apply_gaussian_noise_with_p),
            ])

    def _apply_augmentation(self, input_values: List[float]):
        """apply some audio augmentations in the given input_values"""
        if self.augmentator is not None:
            return self.augmentator(samples=np.array(input_values), sample_rate=self.sample_rate).tolist()
        else:
            return input_values

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = []
        lens = []
        for feature in features:
            wav, sr = sf.read(feature["input_values"])
            leng = wav.shape[0]
            lens.append(leng)
            wav_feature = self.processor(audio=wav, sampling_rate=sr).input_values[0]
            input_features.append({"input_values": self._apply_augmentation(wav_feature)})
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # 应该返回一个列表字典
        max_len = min((max(lens), 150000))
        batch = self.processor.pad(
            input_features,
            padding=True,
            max_length=max_len,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():  # 对text进行pad要使用这个
            labels_batch = self.processor.pad(
                label_features,
                padding=True,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        if "dialect_index" in feature.keys():
            dialect_index = [feature["dialect_index"] for feature in features]
            dialect_index = torch.tensor(dialect_index)
            batch["dialect_index"] = dialect_index
        batch["lengths"] = torch.tensor(lens)
        return batch


class DataCollatorTripletWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None  # 默认pad到整个序列的最大长度
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __init__(self, processor: Wav2Vec2Processor, padding=True, apply_gaussian_noise_with_p=0.5,
                 apply_gain_with_p=0.5,
                 apply_pitch_shift_with_p=0.5,
                 apply_time_stretch_with_p=0.5, sample_rate=16_000):
        self.processor = processor
        self.padding = padding
        self.apply_gaussian_noise_with_p = apply_gaussian_noise_with_p
        self.apply_gain_with_p = apply_gain_with_p
        self.apply_pitch_shift_with_p = apply_pitch_shift_with_p
        self.apply_time_stretch_with_p = apply_time_stretch_with_p
        self.sample_rate = sample_rate

        self.augmentator = None
        if self.apply_gaussian_noise_with_p + self.apply_gain_with_p + self.apply_pitch_shift_with_p + self.apply_time_stretch_with_p > 0:
            self.augmentator = Compose([
                TimeStretch(min_rate=0.8, max_rate=1.2, leave_length_unchanged=False, p=self.apply_time_stretch_with_p),
                PitchShift(min_semitones=-1, max_semitones=1, p=self.apply_pitch_shift_with_p),
                Gain(min_gain_in_db=-1, max_gain_in_db=1, p=self.apply_gain_with_p),
                AddGaussianNoise(min_amplitude=0.0001, max_amplitude=0.001, p=self.apply_gaussian_noise_with_p),
            ])

    def _apply_augmentation(self, input_values: List[float]):
        """apply some audio augmentations in the given input_values"""
        if self.augmentator is not None:
            return self.augmentator(samples=np.array(input_values), sample_rate=self.sample_rate).tolist()
        else:
            return input_values

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features_A = []
        input_features_P = []
        input_features_N = []
        lensA = []
        lensP = []
        lensN = []
        batch = {"labelAP": [], "labelN": [], "input_values": {}, "lengths": {}, "attention_mask": {}}  # 返回值,后续还会继续添加键
        for feature in features:
            wavP, srP = sf.read(feature["input_values"]["P"])
            wavA, srA = sf.read(feature["input_values"]["A"])
            wavN, srN = sf.read(feature["input_values"]["N"])
            batch["labelAP"].append(feature["labelAP"])
            batch["labelN"].append(feature["labelN"])

            lensA.append(wavA.shape[0])
            lensP.append(wavP.shape[0])
            lensN.append(wavN.shape[0])

            wav_feature_A = self.processor(audio=wavA, sampling_rate=srA).input_values[0]
            wav_feature_P = self.processor(audio=wavP, sampling_rate=srP).input_values[0]
            wav_feature_N = self.processor(audio=wavN, sampling_rate=srN).input_values[0]
            input_features_A.append({"input_values": self._apply_augmentation(wav_feature_A)})
            input_features_P.append({"input_values": self._apply_augmentation(wav_feature_P)})
            input_features_N.append({"input_values": self._apply_augmentation(wav_feature_N)})
        # label_features = [{"input_ids": feature["labels"]} for feature in features]
        # 应该返回一个列表字典
        max_lenA = min((max(lensA), 150000))
        max_lenP = min((max(lensP), 150000))
        max_lenN = min((max(lensN), 150000))
        batch_A = self.processor.pad(
            input_features_A,
            padding=True,
            max_length=max_lenA,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors="pt",
        )

        batch_P = self.processor.pad(
            input_features_P,
            padding=True,
            max_length=max_lenP,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors="pt",
        )

        batch_N = self.processor.pad(
            input_features_N,
            padding=True,
            max_length=max_lenN,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors="pt",
        )
        batch["input_values"]["A"] = batch_A["input_values"]
        batch["attention_mask"]["A"] = batch_A["attention_mask"]
        lensA = torch.tensor(lensA)
        batch["lengths"]["A"] = lensA / torch.max(lensA)

        batch["input_values"]["P"] = batch_P["input_values"]
        batch["attention_mask"]["P"] = batch_P["attention_mask"]
        lensP = torch.tensor(lensP)
        batch["lengths"]["P"] = lensP / torch.max(lensP)

        batch["input_values"]["N"] = batch_N["input_values"]
        batch["attention_mask"]["N"] = batch_N["attention_mask"]
        lensN = torch.tensor(lensN)
        batch["lengths"]["N"] = lensN / torch.max(lensN)

        batch["labelAP"] = torch.tensor(batch["labelAP"])
        batch["labelN"] = torch.tensor(batch["labelN"])
        return batch


if __name__ == "__main__":
    LANGUAGE2DIR_path = {
        "Beijing": "../../../../dataset_metadata/KeSpeech_data_metadata/dialect_metadata/Beijing",
        "Ji-Lu": "../../../../dataset_metadata/KeSpeech_data_metadata/dialect_metadata/Ji-Lu",
        "Jiang-Huai": "../../../../dataset_metadata/KeSpeech_data_metadata/dialect_metadata/Jiang-Huai",
        "Jiao-Liao": "../../../../dataset_metadata/KeSpeech_data_metadata/dialect_metadata/Jiao-Liao",
        "Lan-Yin": "../../../../dataset_metadata/KeSpeech_data_metadata/dialect_metadata/Lan-Yin",
        "Mandarin": "../../../../dataset_metadata/KeSpeech_data_metadata/dialect_metadata/Mandarin",
        "Northeastern": "../../../../dataset_metadata/KeSpeech_data_metadata/dialect_metadata/Northeastern",
        "Southwestern": "../../../../dataset_metadata/KeSpeech_data_metadata/dialect_metadata/Southwestern",
        "Zhongyuan": "../../../../dataset_metadata/KeSpeech_data_metadata/dialect_metadata/Zhongyuan",
    }
    dialect_metadata = "../../../../dataset_metadata/KeSpeech_data_metadata/dialect_metadata/"
    # for dialect in DIALECT_LS:
    #     for subset in ["train","val","test"]:
    #         make_huggingface_data(dialect, subset, LANGUAGE2DIR_path, os.path.join(dialect_metadata,dialect,subset,"KeSpeechDataset.json"))
    make_huggingface_data("Mandarin", "train", LANGUAGE2DIR_path,
                          "../../../../dataset_metadata/KeSpeech_data_metadata/dialect_metadata/Mandarin/train/KeSpeechDataset.json",
                          max_example_num=30000)
    merge_hugging_face_dataset("../../../../dataset_metadata/KeSpeech_data_metadata/dialect_metadata",
                               "../../../../dataset_metadata/KeSpeech_data_metadata/KeSpeechDataset_shrinked.json")

# if __name__ == "__main__":
#     data_file_json = "../../../KeSpeech_data_metadata/KeSpeechDataset.json"
#     dataset_base_dir_path = "/srv/wangyiqun"
#     processor_path = "../../../weights/myWav2vec2"
#     processor = make_processor(True, processor_path)
#
#     dataset = make_hugging_face_datasets(data_file_json, ["Mandarin", "Ji-Lu"], ["train", "test"],
#                                             dataset_base_dir_path,
#                                             8, processor)
#     combined_dataset = concatenate_datasets([dataset[key]["train"] for key in dataset.keys()])
#
#     print(combined_dataset)
#     print(dataset["Mandarin"]["train"])
#     print(dataset["Ji-Lu"]["train"])


# if __name__ == "__main__":
#     from src.utils.data.process import make_processor
#     from src.utils.data.KeSpeech.KeSpeech_metadata import DIALECT_LS
#
#     json_path = "../../../../weights/KeSpeechASR/KeSpeechDataset.json"
#     dataset_base_path = "F:/语音数据集/KeSpeech/"
#     pretrained_path = "../../../../weights/KeSpeechASR"
#     processor = make_processor(True, pretrainModel_path=pretrained_path)
#     subset = ["train", "val", "test"]
#     dataset = make_hugging_face_datasets(data_json=json_path, languages=DIALECT_LS, subsets=subset,
#                                          dataset_dir=dataset_base_path, num_proc=12, processor=processor, merge=True)
#     train_dataset = dataset["train"]
#     print(train_dataset.data)
#     print(train_dataset.num_rows)
#     print(len(train_dataset))
#     print(train_dataset.info)


if __name__ == "__main__":
    processor_path = "../../../../weights/KeSpeechASR"
    processor = make_processor(True, processor_path)
    data_file_json = "../../../../dataset_metadata/KeSpeech_data_metadata/KeSpeechDataset_shrinked.json"
    dataset = make_hugging_face_datasets(data_file_json, ["Mandarin"], ["test"],
                                         "F:/语音数据集/KeSpeech/", 8, processor)
    dataset = dataset["Mandarin"]["train"]
    print(dataset)
    paths = dataset["input_values"]
    del dataset
