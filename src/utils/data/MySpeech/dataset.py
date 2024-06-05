import json
import os.path

from transformers import Wav2Vec2Processor

from src.utils.data.MySpeech.load_data import load_infos
from src.utils.fileIO.json import write_json


def make_huggingface_data(info_dir, save_path):
    from src.utils.data.KeSpeech.KeSpeech_metadata import DIALECT_LS, get_language_id
    data_list = load_infos(["wave_path", "text"], info_dir)
    path = data_list[0]
    text = data_list[1]
    data_dict_dict = {"dialect": "Jiao-Liao", "dialect_index": 4, "data": []}
    for index in range(len(path)):
        data_dict_dict["data"].append(
            {"path": path[index], "text": text[index], "dialect_index": get_language_id("Jiao-Liao", DIALECT_LS)})
    write_json(data_dict_dict, save_path)


def merge_hugging_face_dataset(metadata_dir_path, save_path):
    dataset_dict = {}
    for subset_name in os.listdir(metadata_dir_path):
        subset_dir = os.path.join(metadata_dir_path, subset_name)
        if not os.path.isdir(subset_dir):
            continue
        dataset_json_path = os.path.join(subset_dir, "MySpeechDataset.json")
        with open(dataset_json_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
            dataset_dict[subset_name] = dataset
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
        with processor.as_target_processor():
            batch["labels"] = processor(batch["text"], ).input_ids
        return batch

    dataset = dataset.map(
        prepare_dataset_fn,
        remove_columns=["text"],
        num_proc=num_proc
    )
    dataset = dataset.shuffle()
    return dataset


def make_hugging_face_datasets(data_json: str, subsets: list, dataset_dir, num_proc, proc):
    from datasets import Dataset
    datasets_dict = {}
    for subset in subsets:
        with open(data_json, "r", encoding="utf-8") as f:
            datasets_dict[subset] = process_hugging_face_dataset(Dataset.from_list(json.load(f)[subset]["data"]),
                                                                 dataset_dir, num_proc, proc)
    return datasets_dict


# class DataCollatorCTCWithPadding:
#     processor: Wav2Vec2Processor
#     padding: Union[bool, str] = True
#     max_length: Optional[int] = None  # 默认pad到整个序列的最大长度
#     max_length_labels: Optional[int] = None
#     pad_to_multiple_of: Optional[int] = None
#     pad_to_multiple_of_labels: Optional[int] = None
#
#     def __init__(self, processor, padding=True, apply_gaussian_noise_with_p=0.5, apply_gain_with_p=0.5,
#                  apply_pitch_shift_with_p=0.5,
#                  apply_time_stretch_with_p=0.5, sample_rate=16_000):
#         self.processor = processor
#         self.padding = padding
#         self.apply_gaussian_noise_with_p = apply_gaussian_noise_with_p
#         self.apply_gain_with_p = apply_gain_with_p
#         self.apply_pitch_shift_with_p = apply_pitch_shift_with_p
#         self.apply_time_stretch_with_p = apply_time_stretch_with_p
#         self.sample_rate = sample_rate
#
#         self.augmentator = None
#         if self.apply_gaussian_noise_with_p + self.apply_gain_with_p + self.apply_pitch_shift_with_p + self.apply_time_stretch_with_p > 0:
#             self.augmentator = Compose([
#                 TimeStretch(min_rate=0.8, max_rate=1.2, leave_length_unchanged=False, p=self.apply_time_stretch_with_p),
#                 PitchShift(min_semitones=-1, max_semitones=1, p=self.apply_pitch_shift_with_p),
#                 Gain(min_gain_in_db=-1, max_gain_in_db=1, p=self.apply_gain_with_p),
#                 AddGaussianNoise(min_amplitude=0.0001, max_amplitude=0.001, p=self.apply_gaussian_noise_with_p),
#             ])
#
#     # TODO: 数据增强，使用augmentator
#     def _apply_augmentation(self, input_values: List[float]):
#         """apply some audio augmentations in the given input_values"""
#         if self.augmentator is not None:
#             return self.augmentator(samples=np.array(input_values), sample_rate=self.sample_rate).tolist()
#         else:
#             return input_values
#
#     # TODO:进行填充
#     def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
#         # split inputs and labels since they have to be of different lenghts and need
#         # different padding methods
#         input_features = []
#         lens = []
#         for feature in features:
#             wav, sr = sf.read(feature["input_values"])
#             leng = wav.shape[0]
#             lens.append(leng)
#             wav_feature = self.processor(wav, sampling_rate=sr).input_values[0]
#             input_features.append({"input_values": self._apply_augmentation(wav_feature)})
#         label_features = [{"input_ids": feature["labels"]} for feature in features]
#         # 应该返回一个列表字典
#         max_len = min((max(lens), 150000))
#         batch = self.processor.pad(
#             input_features,
#             padding=True,
#             max_length=max_len,
#             pad_to_multiple_of=self.pad_to_multiple_of,
#             return_tensors="pt",
#         )
#         with self.processor.as_target_processor():  # 对text进行pad要使用这个
#             labels_batch = self.processor.pad(
#                 label_features,
#                 padding=True,
#                 max_length=self.max_length_labels,
#                 pad_to_multiple_of=self.pad_to_multiple_of_labels,
#                 return_tensors="pt",
#             )
#
#         # replace padding with -100 to ignore loss correctly
#         labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
#         batch["labels"] = labels
#         if "dialect_index" in feature.keys():
#             dialect_index = [feature["dialect_index"] for feature in features]
#             dialect_index = torch.tensor(dialect_index)
#             batch["dialect_index"] = dialect_index
#         return batch


if __name__ == "__main__":
    dialect_metadata_dir_path = "../../../../dataset_metadata/MySpeech_data_metadata/dialect_metadata"
    for subset_name in os.listdir(dialect_metadata_dir_path):
        subset_dir_path = os.path.join(dialect_metadata_dir_path, subset_name)
        if not os.path.isdir(subset_dir_path):
            continue
        make_huggingface_data(subset_dir_path, os.path.join(subset_dir_path, "MySpeechDataset.json"))
    merge_hugging_face_dataset("../../../../dataset_metadata/MySpeech_data_metadata/dialect_metadata",
                               "../../../../dataset_metadata/MySpeech_data_metadata/MySpeechDataset.json")

# if __name__ == "__main__":
# 	data_file_json = "../../../../MySpeech_data_metadata/MySpeechDataset.json"
# 	dataset_base_dir_path = "/media/mixxis/T7/root/语音/MySpeech/"
# 	processor_path = "../../../../weights/MySpeechASR"
# 	processor = make_processor(True, processor_path)
#
# 	dataset = make_hugging_face_datasets(data_file_json, ["train", "test"], dataset_base_dir_path, 8, processor)
#
# 	print(dataset)

# if __name__ == "__main__":
# 	dialect_metadata_dir_path = "../../../../dataset_metadata/MySpeech_data_metadata/dialect_metadata"
# 	for subset_name in os.listdir(dialect_metadata_dir_path):
# 		subset_dir_path = os.path.join(dialect_metadata_dir_path, subset_name)
# 		if not os.path.isdir(subset_dir_path):
# 			continue
# 		make_huggingface_data(subset_dir_path, os.path.join(subset_dir_path, "MySpeechDataset.json"))
# 	merge_hugging_face_dataset("../../../../dataset_metadata/MySpeech_data_metadata/dialect_metadata","../../../../MySpeech_data_metadata/MySpeechDataset_own.json")
