import re

from tqdm import tqdm

from src.utils.fileIO.json import write_json, read_json
from src.utils.fileIO.txt import write_txt, read_txt


def load_dic(dict_path: str):
    """
    加载字典
    :param dict_path: 字典文件（txt）的路径
    :return: 字符列表
    """
    file_format = dict_path.split(".")[-1]
    if file_format == "txt":
        dic_list = read_txt(dict_path)
        for i in range(len(dic_list)):
            dic_list[i] = re.sub("\s", "", dic_list[i])
        return dic_list
    elif file_format == "json":
        return read_json(dict_path)


def list2dict(ls: list) -> dict:
    i = 0
    dic = {}
    for data in ls:
        key = data
        dic[key] = i
        i += 1
    return dic


def dict2list(dic: dict) -> list:
    return list(dic.keys())


def make_dic(txt_path: str) -> list:
    """
    制作字典
    :param txt_path: 文本的路径
    :return: 字符列表
    """
    dic = ['<SPOKEN_NOISE>', '<GAP>', '<PAD>', '<S>', '</S>']  # 未知、分隔、噪声、填充、开始、终止

    lines = read_txt(txt_path)

    for line in tqdm(lines, desc=f'get dict from file', ncols=80, unit='line'):
        # 去除空白、数字、字母、下划线
        line = re.sub(r'[^\u4e00-\u9fff]+', '', line)
        for char in line:
            if char not in dic:
                dic.append(char)
    return dic


def merge_dicts(dict_list: list) -> list:
    """
    拼接字典
    :param dict_list: 要拼接的字典列表，字典是一种字符列表，由make_dic生成
    :return:字符列表
    """
    base_dic = dict_list[0]
    dict_list.pop(0)
    for dic in tqdm(dict_list, desc=f'merge dicts to the first dict ~ing', ncols=80, unit='dict'):
        for char in dic:
            if char not in base_dic:
                base_dic.append(char)
    return base_dic


def text2index(dic: list, text) -> list:
    """
    将字符序列转换为索引序列
    :param dic: 字符列表（也就是我们建立的字典）
    :param text:字符序列
    :return:索引列表
    """
    index_list = []
    string_temp = ""
    begin = False
    for char in text:
        if char != "<" and not begin:  # 如果是普通字符
            if char not in dic:  # 标点符号
                print(f"unknow char{char}, change to SPOKEN_NOISE")
                index_list.append(dic.index("<SPOKEN_NOISE>"))
                continue
            index_list.append(dic.index(char))
        else:  # 如果文本中有特殊字符，例如spoken_noise
            string_temp += char  # 用于保存特殊字符
            if char == "<":
                begin = True
            elif char == ">":
                begin = False
                if string_temp not in dic:
                    index_list.append(dic.index("<SPOKEN_NOISE>"))
                    print(f"unknow char{string_temp}, change to SPOKEN_NOISE")
                    continue
                index_list.append(dic.index(string_temp))

    return index_list


def save_dic(dic: list, save_path: str) -> None:
    file_format = save_path.split(".")[-1]
    if file_format == "txt":
        write_txt(dic, save_path)
    elif file_format == "json":
        dic = list2dict(dic)
        write_json(dic, save_path)
    else:
        print(f"invalid file format : {file_format}")
        exit(1)


# if __name__ == "__main__":
#     txt_list = [
#         "/media/mixxis/T7/root/语音/KeSpeech/Tasks/ASR/test/text",
#         "/media/mixxis/T7/root/语音/KeSpeech/Tasks/ASR/train_phase1/text",
#         "/media/mixxis/T7/root/语音/KeSpeech/Tasks/ASR/train_phase2/text",
#         "/media/mixxis/T7/root/语音/KeSpeech/Tasks/ASR/dev_phase1/text",
#         "/media/mixxis/T7/root/语音/KeSpeech/Tasks/ASR/dev_phase2/text",
#     ]
#
#     dir_path = "/media/mixxis/T7/root/语音/MySpeech/fhw-text/text"
#     for name in os.listdir(dir_path):
#         txt_path = os.path.join(dir_path, name)
#         txt_list.append(txt_path)
#
#     dict_list = []
#     for txt in txt_list:
#         dic = make_dic(txt)
#         dict_list.append(dic)
#     vocab = merge_dicts(dict_list)
#     save_dic(vocab, "../../../KeSpeech_data_metadata/vocab.txt")
#     save_dic(vocab, "../../../KeSpeech_data_metadata/vocab.json")

# if __name__ == "__main__":
# 	text_path = "../../../KeSpeech_data_metadata/vocab.txt"
# 	vocab = load_dic(text_path)
# 	vocab_dict=list2dict(vocab)
# 	json_path = "../../../KeSpeech_data_metadata/vocab.json"
# 	save_dic(vocab, "../../../KeSpeech_data_metadata/vocab.json")
#
#
# 	dialect_Dir_path = "../../../KeSpeech_data_metadata/dialect_metadata"
# 	for dialect_name in tqdm(os.listdir(dialect_Dir_path)):
# 		dialect_path = os.path.join(dialect_Dir_path, dialect_name)
# 		for sub_name in tqdm(os.listdir(dialect_path)):
# 			sub_set_dir = os.path.join(dialect_path, sub_name)
# 			text_path = os.path.join(sub_set_dir, "text.txt")
# 			text_index_path = os.path.join(sub_set_dir, "text_index.txt")
# 			index_list = []
# 			with open(text_path, "r", encoding="utf-8") as f:
# 				lines = f.readlines()
# 				for line in lines:
# 					text = re.sub("\s", "", line)
# 					index = text2index(vocab, text)
# 					index_list.append(str(index))
# 			write_list(index_list, text_index_path)


if __name__ == "__main__":
    txt_path = r"F:\语音数据集\MySpeech\Metadata\text.txt"
    txt_list = [txt_path]

    dict_list = []
    for txt in txt_list:
        dic = make_dic(txt)
        dict_list.append(dic)
    vocab = merge_dicts(dict_list)
    save_dic(vocab, "../../../MySpeech_data_metadata/own_vocab.txt")
    save_dic(vocab, "../../../MySpeech_data_metadata/own_vocab.json")
