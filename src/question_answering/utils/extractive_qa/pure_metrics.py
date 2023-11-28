import evaluate


def calculate_basic_accuracy(
    start_actual: list[int],
    end_actual: list[int],
    start_preds: list[int],
    end_preds: list[int],
):
    length = len(start_actual)
    if (
        len(end_actual) != length
        and len(start_preds) != length
        and len(end_preds) != length
    ):
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


def calculate_pure_f1():
    print("TBD")


def calculate_pure_precision():
    print("TBD")


def calculate_pure_recall():
    print("TBD")
