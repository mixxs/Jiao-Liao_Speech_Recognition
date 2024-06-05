import re

import jieba
import jiwer


def keep_chinese(text):
    pattern = re.compile('[\u4e00-\u9fa5]+')  # 匹配所有中文汉字的Unicode范围
    result = pattern.findall(text)
    return ''.join(result)


def myCut(sentence: str):
    jiebaResult = list(jieba.cut(sentence))
    myResult = ""
    for word in jiebaResult:
        myResult += word
        myResult += " "
    myResult = myResult.strip()
    return myResult


def compute_cer(references, predictions, chunk_size=None):
    """"""
    preds = [keep_chinese(seq) for seq in predictions]
    refs = [keep_chinese(seq) for seq in references]
    print(preds)
    print(refs)
    if chunk_size is None:
        return jiwer.cer(refs, preds)
    start = 0
    end = chunk_size
    H, S, D, I = 0, 0, 0, 0
    while start < len(references):
        chunk_metrics = jiwer.compute_measures(refs, preds)
        H = H + chunk_metrics["hits"]
        S = S + chunk_metrics["substitutions"]
        D = D + chunk_metrics["deletions"]
        I = I + chunk_metrics["insertions"]
        start += chunk_size
        end += chunk_size
        del preds
        del refs
        del chunk_metrics
    return float(S + D + I) / float(H + S + D)


def compute_wer(references, predictions, chunk_size=None):
    references = [myCut(keep_chinese(seq)) for seq in references]
    predictions = [myCut(keep_chinese(seq)) for seq in predictions]

    if chunk_size is None:
        return jiwer.wer(references, predictions)
    start = 0
    end = chunk_size
    H, S, D, I = 0, 0, 0, 0
    while start < len(references):
        chunk_metrics = jiwer.compute_measures(references[start:end], predictions[start:end])
        H = H + chunk_metrics["hits"]
        S = S + chunk_metrics["substitutions"]
        D = D + chunk_metrics["deletions"]
        I = I + chunk_metrics["insertions"]
        start += chunk_size
        end += chunk_size
    return float(S + D + I) / float(H + S + D)


def compute_acc(preds, labels):
    if labels.shape != preds.shape:
        raise ValueError(
            f"shape unmatch,{labels.shape}, {preds.shape}")
    total = 0.0
    right = 0.0
    for index in range(len(preds)):
        total += 1.0
        if preds[index] == labels[index]:
            right += 1.0
    return float(right / total)
