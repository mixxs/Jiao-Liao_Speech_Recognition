import ast
import os
import os.path
import re
from itertools import repeat, chain
from typing import Union

from src.utils.fileIO.txt import read_txt


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


def load_infos(infosToLoad: Union[str, list], info_dir: str) -> list:
	"""

	:param infosToLoad: 要加载哪一种或哪几种信息
	:param subset_name: train,val或者test
	:param info_dir: 信息所在目录
	:return: [info1list,info2list,……]，二维的list
	"""
	if type(infosToLoad) == str:
		return [load_one_info(infosToLoad, info_dir), ]
	elif type(infosToLoad) == list:
		result = []
		for info_class in infosToLoad:
			result.append(load_one_info(info_class, info_dir))
		return result


if __name__ == "__main__":
	dict_path = "../../../../dataset_metadata/MySpeech_data_metadata_own/vocab.txt"
	lis = load_infos(["text_index", "text"],
					 "../../../../dataset_metadata/MySpeech_data_metadata_own/dialect_metadata/test")
	print(len(lis), len(lis[0]))
