import os.path

from src.utils.fileIO.model import save_model
from src.utils.result.metric_io import load_metric


def is_best(cer, wer, output_dir):
    cer_json_path = os.path.join(output_dir, "cer.json")
    if not os.path.exists(cer_json_path):
        # print("not exist")
        return True
    cer_list = load_metric("cer", output_dir)
    best_cer = min(cer_list)
    if cer > best_cer:
        return False
    if cer < best_cer:
        # print("best cer")
        return True
    ind_of_best = cer_list.index(best_cer)
    wer_list = load_metric("wer", output_dir)
    best_wer = wer_list[ind_of_best]
    if wer <= best_wer:
        # print("best wer")
        return True
    return False


def save_if_best(model, cer, wer, output_dir):
    if is_best(cer, wer, output_dir):
        save_model(model, os.path.join(output_dir, "best.pt"))


def get_best(output_dir):
    cer_list = load_metric("cer", output_dir)
    wer_list = load_metric("wer", output_dir)
    best_cer = min(cer_list)
    ind_of_best = cer_list.index(best_cer)
    best_wer = wer_list[ind_of_best]
    return best_cer, best_wer


if __name__ == "__main__":
    dir_path = "../../../weights/KeSpeechASR/fine_tune_adapter_without_wf"
    print(is_best(0.31, 0.1, dir_path))
    print(get_best(dir_path))
