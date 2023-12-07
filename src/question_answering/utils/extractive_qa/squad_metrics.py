import evaluate

from .__helpers import (
    ensure_same_sizes,
    exact_match_score,
    f1_score,
    precision_score,
    recall_score,
)


def calculate_squad_accuracies(
    start_actual: list[list[int]],
    end_actual: list[list[int]],
    start_preds: list[int],
    end_preds: list[int],
):
    length = ensure_same_sizes(start_actual, end_actual, start_preds, end_preds)

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

    start_good_predictions_count = 0.0
    end_good_predictions_count = 0.0
    full_good_predictions_count = 0.0
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
        "full_accuracy": full_good_predictions_count / length,
    }


def calculate_original_squad_metrics(
    ids: list[str], answers: list[dict], predicted_texts: list[str]
) -> dict:
    ensure_same_sizes(ids, answers, predicted_texts)

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
    length = ensure_same_sizes(answers, predicted_texts)
    precision_metric = 0.0
    recall_metric = 0.0
    f1_metric = 0.0
    exact_match_metric = 0.0

    for i in range(length):
        valid_answers = answers[i]
        predicted_text = predicted_texts[i]

        precision_metric += __metric_max_over_ground_truths(
            metric_fn=precision_score,
            prediction=predicted_text,
            ground_truths=valid_answers,
            normalize=normalize,
        )
        recall_metric += __metric_max_over_ground_truths(
            metric_fn=recall_score,
            prediction=predicted_text,
            ground_truths=valid_answers,
            normalize=normalize,
        )
        f1_metric += __metric_max_over_ground_truths(
            metric_fn=f1_score,
            prediction=predicted_text,
            ground_truths=valid_answers,
            normalize=normalize,
        )
        exact_match_metric += __metric_max_over_ground_truths(
            metric_fn=exact_match_score,
            prediction=predicted_text,
            ground_truths=valid_answers,
            normalize=normalize,
        )

    return {
        "precision": precision_metric / length,
        "recall": recall_metric / length,
        "f1": f1_metric / length,
        "exact_match": exact_match_metric / length,
    }


def get_is_correctly_predicted(
    answers: list[list[str]], predicted_texts: list[str], normalize: bool
):
    length = ensure_same_sizes(answers, predicted_texts)
    is_correctly_predicted = []

    for i in range(length):
        valid_answers = answers[i]
        predicted_text = predicted_texts[i]

        is_correctly_predicted.append(
            __metric_max_over_ground_truths(
                metric_fn=exact_match_score,
                prediction=predicted_text,
                ground_truths=valid_answers,
                normalize=normalize,
            )
        )

    return is_correctly_predicted


def __metric_max_over_ground_truths(
    metric_fn, prediction, ground_truths, normalize: bool
):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(
            prediction=prediction, valid_answer=ground_truth, normalize=normalize
        )
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)
