import ast
import os
import os.path
import re
from itertools import repeat, chain
from typing import Union

from src.utils.fileIO.txt import read_txt

LANGUAGE2DIR = {
	"Beijing": "../../../KeSpeech_data_metadata/dialect_metadata/Beijing",
	"Ji-Lu": "../../../KeSpeech_data_metadata/dialect_metadata/Ji-Lu",
	"Jiang-Huai": "../../../KeSpeech_data_metadata/dialect_metadata/Jiang-Huai",
	"Jiao-Liao": "../../../KeSpeech_data_metadata/dialect_metadata/Jiao-Liao",
	"Lan-Yin": "../../../KeSpeech_data_metadata/dialect_metadata/Lan-Yin",
	"Mandarin": "../../../KeSpeech_data_metadata/dialect_metadata/Mandarin",
	"Northeastern": "../../../KeSpeech_data_metadata/dialect_metadata/Northeastern",
	"Southwestern": "../../../KeSpeech_data_metadata/dialect_metadata/Southwestern",
	"Zhongyuan": "../../../KeSpeech_data_metadata/dialect_metadata/Zhongyuan",
}


def languageNames2dirs(names: Union[str, list], language2dir_dict=None):
	if language2dir_dict is None:
		language2dir_dict = LANGUAGE2DIR
	if type(names) == str:
		if names not in language2dir_dict:
			print(f"invalid language name{names}")
			exit(1)
		return language2dir_dict[names]
	dir_list = []
	for name in names:
		if name not in language2dir_dict:
			print(f"invalid language name{names}")
			exit(1)
		dir_list.append(language2dir_dict[name])
	return dir_list


def label_list_process(label_index_list: list, BOS: int, EOS: int, GAP: int):
	return [label_process(index_list, BOS, EOS, GAP) for index_list in label_index_list]


def label_process(index_list: list, BOS: int, EOS: int, GAP: int):
	lis = list(chain([BOS], chain(chain.from_iterable(zip(repeat(GAP), index_list))), [EOS]))
	lis.pop(1)  # 这个位置是个多余的GAP
	return lis


def load_one_info(infoToLoad: str, info_dir: str) -> list:
	"""

	:param infoToLoad: 要加载哪一种信息
	:param info_dir: 信息txt所在的目录
	:return:整数或者字符串的列表
	"""
	info_list = []
	file_name = infoToLoad + ".txt"
	file_path = os.path.join(info_dir, file_name)
	# BOS, EOS, GAP = (char_dict.index("<s>"), char_dict.index("</s>"), char_dict.index("<GAP>"))
	lines = read_txt(file_path)
	for line in lines:
		item = re.sub("\s", "", line)
		if infoToLoad == "dialect_index":
			item = int(item)
		if infoToLoad == "text_index":
			item = ast.literal_eval(item)  # ast负责将字符串转为数组
		info_list.append(item)
	return info_list


def load_one_language_infos(infosToLoad: Union[str, list], languageToLoad: str, subset_name: str = "train", language2dir_dict=None) -> list:
	"""

	:param languageToLoad: 要加载的语言
	:param infosToLoad: 要加载哪一种或哪几种信息
	:param subset_name: train,val或者test
	:return: [info1list,info2list,……]，二维的list
	:param language2dir_dict: 语言对应路径的字典
	"""
	info_dir = os.path.join(languageNames2dirs(languageToLoad, language2dir_dict), subset_name)
	if type(infosToLoad) == str:
		return [load_one_info(infosToLoad, info_dir), ]
	elif type(infosToLoad) == list:
		result = []
		for info_class in infosToLoad:
			result.append(load_one_info(info_class, info_dir))
		return result


def load_multi_language_infos(infosToLoad: Union[str, list], languageToLoad: list, subset_name: str = "train", language2dir_dict=None) -> list:
	"""

	:param language2dir_dict: 语言对应路径的字典
	:param vocab: 字典
	:param languageToLoad:要加载的语言（多种语言）
	:param infosToLoad: 要加载哪一种或哪几种信息
	:param subset_name: train,val或者test
	:return: [language1_infos_list,language2_infos_list,……]，三维的list
	"""
	result = []
	for language in languageToLoad:
		result.append(load_one_language_infos(infosToLoad, language, subset_name, language2dir_dict))
	return result

# if __name__ == "__main__":
# 	dict_path = "/home/mixxis/code/speech_recognition/KeSpeech_data_metadata/vocab.txt"
# 	list = load_multi_language_infos(["text_index", "text"], ["Beijing", "Ji-Lu"],  "train")
# 	print(len(list), len(list[0]), len(list[0][0]))
