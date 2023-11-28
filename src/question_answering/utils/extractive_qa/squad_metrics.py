import re
import string
from collections import Counter

import evaluate


def calculate_squad_accuracies(
    start_actual: list[list[int]],
    end_actual: list[list[int]],
    start_preds: list[int],
    end_preds: list[int],
):
    length = __ensure_same_sizes(start_actual, end_actual, start_preds, end_preds)

    # Prepare data for comparison
    start_end_pred_pairs = list(zip(start_preds, end_preds))

    start_end_actual_pairs = []
    for i, sample_starts in enumerate(start_actual):
        temp = []
        sample_ends = end_actual[i]

        for j, start in enumerate(sample_starts):
            end = sample_ends[j]
            temp.append((start, end))

        start_end_actual_pairs.append(temp)

    start_good_predictions_count = 0.
    end_good_predictions_count = 0.
    full_good_predictions_count = 0.
    for i in range(length):
        sample_start_pred = start_preds[i]
        sample_end_pred = end_preds[i]
        sample_start_end_actual_pairs = start_end_actual_pairs[i]
        sample_start_end_pred_pair = start_end_pred_pairs[i]

        if sample_start_pred in start_actual[i]:
            start_good_predictions_count += 1

        if sample_end_pred in end_actual[i]:
            end_good_predictions_count += 1

        if sample_start_end_pred_pair in sample_start_end_actual_pairs:
            full_good_predictions_count += 1

    return {
        "start_accuracy": start_good_predictions_count / length,
        "end_accuracy": end_good_predictions_count / length,
        "full_accuracy": full_good_predictions_count / length
    }


def calculate_original_squad_metrics(
    ids: list[str], answers: list[dict], predicted_texts: list[str]
) -> dict:
    __ensure_same_sizes(ids, answers, predicted_texts)

    metric = evaluate.load("squad")

    predictions = []
    references = []
    for i, example_id in enumerate(ids):
        predictions.append({"id": example_id, "prediction_text": predicted_texts[i]})
        references.append({"id": example_id, "answers": answers[i]})

    result = metric.compute(predictions=predictions, references=references)
    result["exact_match"] = result["exact_match"] / 100
    result["f1"] = result["f1"] / 100
    return result


def calculate_squad_qa_metrics(
    answers: list[list[str]], predicted_texts: list[str], normalize: bool
):
    def tp_fp_fn(prediction: str, valid_answer: str):
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

    def precision_score(prediction: str, valid_answer: str):
        tp, fp, fn = tp_fp_fn(valid_answer, prediction)
        if tp == 0:
            return 0
        return (1.0 * tp) / (tp + fp)

    def recall_score(prediction: str, valid_answer: str):
        tp, fp, fn = tp_fp_fn(valid_answer, prediction)
        if tp == 0:
            return 0
        return (1.0 * tp) / (tp + fn)

    def f1_score(prediction: str, valid_answer: str):
        precision = precision_score(prediction, valid_answer)
        recall = recall_score(prediction, valid_answer)
        if precision == 0 or recall == 0:
            return 0
        return (2 * precision * recall) / (precision + recall)

    def exact_match_score(prediction: str, valid_answer: str):
        if normalize:
            return __normalize_text(prediction) == __normalize_text(valid_answer)
        else:
            return prediction == valid_answer

    length = __ensure_same_sizes(answers, predicted_texts)
    precision_metric = 0.
    recall_metric = 0.
    f1_metric = 0.
    exact_match_metric = 0.

    for i in range(length):
        valid_answers = answers[i]
        predicted_text = predicted_texts[i]

        precision_metric += __metric_max_over_ground_truths(
            precision_score, predicted_text, valid_answers
        )
        recall_metric += __metric_max_over_ground_truths(
            recall_score, predicted_text, valid_answers
        )
        f1_metric += __metric_max_over_ground_truths(
            f1_score, predicted_text, valid_answers
        )
        exact_match_metric += __metric_max_over_ground_truths(
            exact_match_score, predicted_text, valid_answers
        )

    return {
        "precision": precision_metric / length,
        "recall": recall_metric / length,
        "f1": f1_metric / length,
        "exact_match": exact_match_metric / length,
    }


def __metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def __normalize_text(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def __ensure_same_sizes(*args):
    length = len(args[0])
    if not all([len(arg) == length for arg in args]):
        raise Exception("List lengths are not the same!")
    else:
        return length
