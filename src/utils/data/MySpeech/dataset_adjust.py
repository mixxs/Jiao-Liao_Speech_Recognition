import math
import os
import re
from typing import Tuple, List, Dict
import soundfile as sf
import librosa
import warnings
import shutil

from tqdm import tqdm

from src.utils.fileIO.txt import read_txt, write_txt


def match_phase_and_wav2text(dataset_baseDir_path: str) -> Tuple[List[str], List[str]]:
	"""
	:param dataset_baseDir_path:数据集根目录
	:return: 返回匹配好的phase和 wav路径-标签 映射文件两者组成的元组。相对应的phase和映射文件的索引相同
	"""
	phase_list = []
	wav2text_list = []
	audio_path = os.path.join(dataset_baseDir_path, "audio")
	for personName in tqdm(os.listdir(audio_path), desc="匹配中", unit="dir_path", ncols=80):
		personDir = os.path.join(audio_path, personName)
		phases = os.listdir(personDir)
		phaseDir_list = []
		wav2textDir_list = []
		for phasename in phases:
			path = os.path.join(personDir, phasename)
			# 如果是文件夹，直接添加到phaseDir_list
			if os.path.isdir(path):
				phaseDir_list.append(path)
			# 如果是文件
			elif os.path.isfile(path):
				# 添加到待匹配列表中
				wav2textDir_list.append(path)
		for phase_path in phaseDir_list:
			phaseName = os.path.basename(phase_path)
			find = False
			for wav2text_path in wav2textDir_list:
				wav2text_fileName = os.path.basename(wav2text_path)
				if wav2text_fileName.find(phaseName) != -1:
					find = True
					phase_list.append(phase_path)
					wav2text_list.append(wav2text_path)
					wav2textDir_list.remove(wav2text_path)
					break
			if not find:
				raise Exception(
					f"phase文件夹名称匹配失败，失败的phase文件夹路径：\n {phase_path}\n此时映射文件列表：{wav2textDir_list}")
	print("\nmatch over!")
	print("=========================================================================")
	return phase_list, wav2text_list


def get_appendix_num(dir_path: str) -> Dict[str, int]:
	"""
	获取指定目录中所有后缀名的数目
	:param dir_path:要统计的目录路径
	"""
	appendix_list = []
	appendix_num_dict = {}
	for path_name in os.listdir(dir_path):
		path = os.path.join(dir_path, path_name)
		if os.path.isfile(path):
			appendix = path.split(".")[-1]
			if appendix not in appendix_list:
				appendix_list.append(appendix)
				appendix_num_dict[appendix] = 1
			else:
				appendix_num_dict[appendix] += 1
	return appendix_num_dict


def check_data(dataset_baseDir_path: str):
	"""
	检查数据。该函数会检查映射文件中的路径是否正确，并尝试读取音频文件
	:param dataset_baseDir_path:数据集根目录
	"""
	phase_list, wav2text_list = match_phase_and_wav2text(dataset_baseDir_path)
	for phase, wav2text in tqdm(zip(phase_list, wav2text_list), desc="检查进度", unit="文件夹", ncols=80):
		lines = read_txt(wav2text)
		# 先检查数目是否对应
		wav_num = get_appendix_num(phase)["wav"]
		text_num = len(lines)
		if wav_num != text_num:
			raise Exception(f"{phase}中语音数量和{wav2text}中文本数量不统一")
		# 获取映射文件中每一个语音的路径
		for line in tqdm(lines, desc=f"检查{phase}", unit="文件"):
			line_pure = line.strip()
			split_index = re.search("\s", line_pure).span()[0]
			wav_path_rela = line_pure[:split_index:].replace("\\", "/")
			wav_path = os.path.join(dataset_baseDir_path, wav_path_rela)
			if not os.path.exists(wav_path):
				raise Exception(
					f"{wav2text}第{lines.index(line) + 1}行指示的路径{wav_path}不存在")
			
			wav, sr = sf.read(wav_path)
			if len(wav) == 0:
				raise Exception(
					f"文件{wav_path}读取失败")
			if sr != 16000:
				new_wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
				new_dataset_dir = dataset_baseDir_path + "ori"
				new_path = os.path.join(new_dataset_dir, wav_path_rela)
				new_dir = os.path.dirname(new_path)
				if not os.path.exists(new_dir):
					os.makedirs(new_dir)
				warnings.warn(
					f"{wav_path}文件的采样率是{sr}!=16000，将会被重采样到16000Hz\n，原文件会被移动到{new_path}")
				shutil.move(wav_path, new_path)
				sf.write(wav_path, new_wav, 16000)
				wav, sr = sf.read(wav_path)
				if len(wav) == 0:
					raise Exception(
						f"文件{wav_path}写入后读取失败")
				if sr != 16000:
					raise Exception(
						f"文件{wav_path}采样率{sr}!=16000,写入错误")
			del wav, sr, line_pure, wav_path, wav_path_rela
	print("\ncheck passed!")
	print("=========================================================================")


def merge_wav2text_file(dataset_baseDir_path: str):
	"""
	拼接wav路径-标签 映射文件
	:param dataset_baseDir_path:数据集根目录
	"""
	_, wav2text_list = match_phase_and_wav2text(dataset_baseDir_path)
	final_wav2text_list = []
	for file_path in tqdm(wav2text_list, desc="拼接中", unit="文件", ncols=80):
		lines = read_txt(file_path)
		for line in lines:
			line = line.strip()
			line = line.replace("\\", "/")
			final_wav2text_list.append(line)
	task_path = os.path.join(dataset_baseDir_path, "Tasks")
	asr_path = os.path.join(task_path, "ASR")
	metadata_path = os.path.join(dataset_baseDir_path, "Metadata")
	wav2text_path = os.path.join(metadata_path, "wav2text.txt")
	if not os.path.exists(asr_path):
		os.makedirs(asr_path)
	if not os.path.exists(metadata_path):
		os.makedirs(metadata_path)
	write_txt(final_wav2text_list, wav2text_path)
	shutil.copy(wav2text_path, os.path.join(asr_path, "wav2text.txt"))
	print("\nmerge over!")
	print("=========================================================================")


def split_dataset(dataset_baseDir: str, split_ratio_list: List[float]):
	if sum(split_ratio_list) != 1:
		raise Exception(
			f"{split_ratio_list}切割比例之和{sum(split_ratio_list)}不等于1"
		)
	if len(split_ratio_list) < 1 or len(split_ratio_list) > 3:
		raise Exception(
			f"split_ratio个数{len(split_ratio_list)}错误"
		)
	wav2text_path = os.path.join(dataset_baseDir, "Tasks/ASR/wav2text.txt")
	asr_path = os.path.join(dataset_baseDir, "Tasks/ASR")
	lines = read_txt(wav2text_path)
	line_num = len(lines)
	total_ratio = 0
	split_point_list = []
	split_point_list.sort(reverse=True)  # 排序，从大到小
	last_point = 0
	for split_ratio in split_ratio_list:
		total_ratio += split_ratio
		split_point_index = last_point + math.ceil(line_num * split_ratio)
		last_point = split_point_index
		split_point_list.append(split_point_index)
	last_point = 0
	splited = []
	print(split_point_list)
	for split_point in split_point_list:
		splited.append(lines[last_point:split_point])  # 从last_point到split_point-1
		last_point = split_point

	for name in os.listdir(asr_path):
		if name in ["train", "val", "test"]:
			warnings.warn(
				f"查找到名为{name}的文件夹，即将删除"
			)
			shutil.rmtree(os.path.join(asr_path, name))
	for index in range(len(splited)):
		if index == 0:
			train_dir = os.path.join(asr_path, "train")
			train_wav2text_path = os.path.join(train_dir, "wav2text.txt")
			if not os.path.exists(train_dir):
				os.makedirs(train_dir)
			write_txt(splited[index], train_wav2text_path)
		elif index == 1:
			val_dir = os.path.join(asr_path, "val")
			val_wav2text_path = os.path.join(val_dir, "wav2text.txt")
			if not os.path.exists(val_dir):
				os.makedirs(val_dir)
			write_txt(splited[index], val_wav2text_path)
		else:
			test_dir = os.path.join(asr_path, "test")
			test_wav2text_path = os.path.join(test_dir, "wav2text.txt")
			if not os.path.exists(test_dir):
				os.makedirs(test_dir)
			write_txt(splited[index], test_wav2text_path)
	print("\nsplit over!")
	print("=========================================================================")


if __name__ == "__main__":
	# check_data("/media/mixxis/T7/root/语音/MySpeech")
	# merge_wav2text_file("/media/mixxis/T7/root/语音/MySpeech")
	# split_dataset("/media/mixxis/T7/root/语音/MySpeech", [0.8, 0.1, 0.1])
	pass
