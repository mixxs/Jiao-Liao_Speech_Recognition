import os.path
from typing import List


def write_txt(data_list: list, save_path: str) -> None:
	"""
	将列表写入txt文件
	:param data_list:列表，其中元素可以是int或者str
	:param save_path: 保存路径
	:return:
	"""
	dir = os.path.dirname(save_path)
	if not os.path.exists(dir):
		os.makedirs(dir)
	with open(save_path, "w", encoding="utf-8") as file:
		for data in data_list:
			if type(data) == str:
				file.write(data)
				if not data.endswith("\n"):
					file.write("\n")
			elif type(data) in [int, float]:
				file.write(str(data) + "\n")
			else:
				print(f"invalid type {type(data)}")
				print(data)
				exit(1)


def read_txt(path: str) -> List[str]:
	"""
	从txt文件中读取列表
	:param path:文件路径
	"""
	with open(path, "r", encoding="utf-8") as file:
		lines = file.readlines()
	return lines
