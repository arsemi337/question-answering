from .__metrics_helpers import (
    ensure_same_sizes,
    f1_score,
    precision_score,
    recall_score,
    exact_match_score,
)


def calculate_pure_accuracies(
    start_actual: list[int],
    end_actual: list[int],
    start_preds: list[int],
    end_preds: list[int],
):
    length = ensure_same_sizes(start_actual, end_actual, start_preds, end_preds)

    # Prepare data for comparison
    start_end_pred_pairs = list(zip(start_preds, end_preds))
    start_end_actual_pairs = list(zip(start_actual, end_actual))

    start_good_predictions_count = 0.0
    end_good_predictions_count = 0.0
    full_good_predictions_count = 0.0
    for i in range(length):
        sample_start_pred = start_preds[i]
        sample_end_pred = end_preds[i]
        sample_start_end_actual_pairs = start_end_actual_pairs[i]
        sample_start_end_pred_pair = start_end_pred_pairs[i]

        if sample_start_pred == start_actual[i]:
            start_good_predictions_count += 1

        if sample_end_pred == end_actual[i]:
            end_good_predictions_count += 1

        if sample_start_end_pred_pair == sample_start_end_actual_pairs:
            full_good_predictions_count += 1

    return {
        "start_accuracy": start_good_predictions_count / length,
        "end_accuracy": end_good_predictions_count / length,
        "full_accuracy": full_good_predictions_count / length,
    }


def calculate_pure_qa_metrics(
    answers: list[str], predicted_texts: list[str], normalize: bool
):
    length = ensure_same_sizes(answers, predicted_texts)
    precision_metric = 0.0
    recall_metric = 0.0
    f1_metric = 0.0
    exact_match_metric = 0.0

    for i in range(length):
        valid_answer = answers[i]
        predicted_text = predicted_texts[i]

        precision_metric += precision_score(
            prediction=predicted_text, valid_answer=valid_answer, normalize=normalize
        )
        recall_metric += recall_score(
            prediction=predicted_text, valid_answer=valid_answer, normalize=normalize
        )
        f1_metric += f1_score(
            prediction=predicted_text, valid_answer=valid_answer, normalize=normalize
        )
        exact_match_metric += exact_match_score(
            prediction=predicted_text, valid_answer=valid_answer, normalize=normalize
        )

    return {
        "precision": precision_metric / length,
        "recall": recall_metric / length,
        "f1": f1_metric / length,
        "exact_match": exact_match_metric / length,
    }
