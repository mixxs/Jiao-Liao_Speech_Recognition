import os
import os.path


def get_path(dir: str, name: str, append_name):
	"""

	:param dir: 选择的目录
	:param name: 文件的名称（不包含后缀）
	:param append_name: 文件的后缀名，例如".txt",".yaml",".py"等
	:return:
	"""
	
	file_path = os.path.join(dir, name + append_name)
	i = 0
	while os.path.exists(file_path):
		i += 1
		newname = name + str(i)
		file_path = os.path.join(dir, newname + append_name)
	return file_path
