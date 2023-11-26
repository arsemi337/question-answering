import evaluate


def calculate_squad_exact_match(
    ids: list[str], answers: list[dict], predicted_texts: list[str]
):
    metric = evaluate.load("squad")

    predictions = []
    references = []
    for i, example_id in enumerate(ids):
        predictions.append({"id": example_id, "prediction_text": predicted_texts[i]})
        references.append({"id": example_id, "answers": answers[i]})

    result = metric.compute(predictions=predictions, references=references)
    result["exact_match"] = result["exact_match"] / 100
    result["f1"] = result["f1"] / 100


def calculate_pure_exact_match(
    answers: list[str],
    predicted_texts: list[str],
    variant: str,
):
    allowed_variants = ["classic", "lowercase", "lowercase_no_punctuation"]

    if variant in allowed_variants:
        metric = evaluate.load("exact_match")

        match variant:
            case "classic":
                return metric.compute(predictions=predicted_texts, references=answers)
            case "lowercase":
                return metric.compute(
                    predictions=predicted_texts,
                    references=answers,
                    ignore_case=True,
                )
            case "lowercase_no_punctuation":
                return metric.compute(
                    predictions=predicted_texts,
                    references=answers,
                    ignore_case=True,
                    ignore_punctuation=True,
                )


def calculate_pure_precision():
    print("TBD")


def calculate_pure_recall():
    print("TBD")


def calculate_pure_f1():
    print("TBD")


def calculate_squad_basic_accuracy(
        start_actual: list[list[int]],
        end_actual: list[list[int]],
        start_preds: list[int],
        end_preds: list[int]
):
    length = len(start_actual)
    if len(end_actual) != length and len(start_preds) != length and len(end_preds) != length:
        raise Exception("List lengths are not the same!")

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

    good_predictions_count = 0
    for i in range(length):
        sample_start_end_actual_pairs = start_end_actual_pairs[i]
        sample_start_end_pred_pair = start_end_pred_pairs[i]

        if sample_start_end_pred_pair in sample_start_end_actual_pairs:
            good_predictions_count += 1

    return good_predictions_count / length


def calculate_basic_accuracy(
        start_actual: list[int],
        end_actual: list[int],
        start_preds: list[int],
        end_preds: list[int]
):
    length = len(start_actual)
    if len(end_actual) != length and len(start_preds) != length and len(end_preds) != length:
        raise Exception("List lengths are not the same!")

    # Prepare data for comparison
    start_end_pred_pairs = list(zip(start_preds, end_preds))
    start_end_actual_pairs = list(zip(start_actual, end_actual))

    # Calculate metric
    good_predictions_count = 0
    for i in range(length):
        if start_end_pred_pairs[i] == start_end_actual_pairs[i]:
            good_predictions_count += 1

    return good_predictions_count / length
