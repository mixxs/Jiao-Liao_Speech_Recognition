import json
import os


def write_json(dic: dict, json_path: str):
	dir = os.path.dirname(json_path)
	if not os.path.exists(dir):
		os.makedirs(dir)
	with open(json_path, "w", encoding="utf-8") as f:
		json.dump(dic, f, indent=4, ensure_ascii=False)


def read_json(json_path: str) -> dict:
	with open(json_path, "r", encoding="utf-8") as f:
		json_dict = json.load(f)
	return json_dict
