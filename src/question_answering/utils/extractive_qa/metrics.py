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

    return metric.compute(predictions=predictions, references=references)


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
