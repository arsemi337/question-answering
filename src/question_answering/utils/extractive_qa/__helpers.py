import re
import string
from collections import Counter
from pathlib import Path


def create_dirs_if_not_exists(directory: Path):
    if not directory.is_dir():
        directory.mkdir(parents=True)


def ensure_same_sizes(*args):
    length = len(args[0])
    if not all([len(arg) == length for arg in args]):
        raise Exception("List lengths are not the same!")
    else:
        return length


def precision_score(prediction: str, valid_answer: str, normalize: bool):
    tp, fp, fn = __tp_fp_fn(
        prediction=prediction, valid_answer=valid_answer, normalize=normalize
    )
    if tp == 0:
        return 0
    return (1.0 * tp) / (tp + fp)


def recall_score(prediction: str, valid_answer: str, normalize: bool):
    tp, fp, fn = __tp_fp_fn(
        prediction=prediction, valid_answer=valid_answer, normalize=normalize
    )
    if tp == 0:
        return 0
    return (1.0 * tp) / (tp + fn)


def f1_score(prediction: str, valid_answer: str, normalize: bool):
    precision = precision_score(
        prediction=prediction, valid_answer=valid_answer, normalize=normalize
    )
    recall = recall_score(
        prediction=prediction, valid_answer=valid_answer, normalize=normalize
    )
    if precision == 0 or recall == 0:
        return 0
    return (2 * precision * recall) / (precision + recall)


def exact_match_score(prediction: str, valid_answer: str, normalize: bool):
    if normalize:
        return __normalize_text(prediction) == __normalize_text(valid_answer)
    else:
        return prediction == valid_answer


def __tp_fp_fn(prediction: str, valid_answer: str, normalize: bool):
    if normalize:
        prediction_tokens = __normalize_text(prediction).split()
        ground_truth_tokens = __normalize_text(valid_answer).split()
    else:
        prediction_tokens = prediction.split()
        ground_truth_tokens = valid_answer.split()

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    tp = num_same
    fp = len(prediction_tokens) - num_same
    fn = len(ground_truth_tokens) - num_same
    return tp, fp, fn


def __normalize_text(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punctuation(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punctuation(lower(s))))
