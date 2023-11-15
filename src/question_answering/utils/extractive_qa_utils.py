import numpy as np
import sklearn.metrics as skmetrics
import tensorflow as tf
from datasets import Dataset
from evaluate import load
from transformers import TFAutoModelForQuestionAnswering

from question_answering.paths import extractive_qa_paths


def save_model(
    model: tf.keras.Model, model_name: str, weights_name: str = "model_weights"
):
    model.save_weights(extractive_qa_paths.saved_models_dir / model_name / weights_name)


def load_weights_into_model(
    model: tf.keras.Model, model_name: str, weights_name: str = "model_weights"
) -> tf.keras.Model:
    model.load_weights(extractive_qa_paths.saved_models_dir / model_name / weights_name)
    return model


def load_model(
    model_checkpoint: str, model_name: str, weights_name: str = "model_weights"
) -> tf.keras.Model:
    model = TFAutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
    model.load_weights(extractive_qa_paths.saved_models_dir / model_name / weights_name)
    return model


def get_classification_evaluation_metrics(
    class_actual: list, class_preds: list, average: str = "binary"
):
    precision = skmetrics.precision_score(class_actual, class_preds, average=average)
    recall = skmetrics.recall_score(class_actual, class_preds, average=average)
    f1 = skmetrics.f1_score(class_actual, class_preds, average=average)
    return precision, recall, f1


def get_class_preds(predictions, output_key="start_logits", return_classes=True):
    predictions = predictions[output_key]
    probabilities = tf.nn.softmax(predictions)
    if return_classes:
        return np.argmax(probabilities, axis=1)
    else:
        return probabilities.numpy()


def extract_answer_tokens(tokenized_dataset_row: dict):
    start = tokenized_dataset_row["start_positions"]
    end = tokenized_dataset_row["end_positions"]
    if start == 0 and end == 0:
        tokenized_dataset_row["answer_tokens"] = None
    else:
        tokenized_dataset_row["answer_tokens"] = tokenized_dataset_row["input_ids"][
            start : end + 1
        ]
    return tokenized_dataset_row


def decode_answer_tokens(tokenized_dataset_row: dict, tokenizer):
    tokens = tokenized_dataset_row["answer_tokens"]
    if tokens is not None:
        answer = tokenizer.decode(tokens)
    else:
        answer = None
    tokenized_dataset_row["predicted_answer_text"] = answer
    return tokenized_dataset_row


def mark_correct_predictions(tokenized_dataset_row: dict):
    actual_result = tokenized_dataset_row["answer_text"]
    predicted_result = tokenized_dataset_row["predicted_answer_text"]

    if actual_result is not None and predicted_result is not None:
        # Both are not None
        if actual_result == predicted_result:
            tokenized_dataset_row["correctly_predicted"] = 1
        else:
            tokenized_dataset_row["correctly_predicted"] = 0

        if actual_result.lower() == predicted_result.lower():
            tokenized_dataset_row["correctly_predicted_lowercase"] = 1
        else:
            tokenized_dataset_row["correctly_predicted_lowercase"] = 0
    else:
        if actual_result == predicted_result:
            # Both are None
            tokenized_dataset_row["correctly_predicted"] = 1
            tokenized_dataset_row["correctly_predicted_lowercase"] = 1
        else:
            # One is None but not the other
            tokenized_dataset_row["correctly_predicted"] = 0
            tokenized_dataset_row["correctly_predicted_lowercase"] = 0

    return tokenized_dataset_row


def calculate_general_accuracy(actual: list[str | None], predicted: list[str | None]):
    count = len(actual)
    good_preds_count = 0
    good_preds_lowercase_count = 0

    for index, actual_result in enumerate(actual):
        predicted_result = predicted[index]
        if actual_result is not None and predicted_result is not None:
            if actual_result == predicted_result:
                good_preds_count = good_preds_count + 1
            if actual_result.lower() == predicted_result.lower():
                good_preds_lowercase_count = good_preds_lowercase_count + 1
        else:
            if actual_result == predicted[index]:
                good_preds_count = good_preds_count + 1
                good_preds_lowercase_count = good_preds_lowercase_count + 1

    return good_preds_count / count, good_preds_lowercase_count / count


def calculate_pure_exact_match(actual: list[str], predicted: list[str]):
    evaluator = load("exact_match")

    exact_match_original = evaluator.compute(predictions=actual, references=predicted)
    exact_match_lowercase = evaluator.compute(
        predictions=actual,
        references=predicted,
        ignore_case=True,
    )
    exact_match_lowercase_no_punctuation = evaluator.compute(
        predictions=actual,
        references=predicted,
        ignore_case=True,
        ignore_punctuation=True,
    )

    return (
        exact_match_original,
        exact_match_lowercase,
        exact_match_lowercase_no_punctuation,
    )


def calculate_squad_exact_match(model_predictions: Dataset):
    evaluator = load("squad")

    actual_answers = [
        {
            "id": row["id"],
            "answers": {
                "text": [row["answer_text"]],
                "answer_start": [row["answer_start"]],
            },
        }
        for row in model_predictions
    ]
    predicted_answers = [
        {"id": row["id"], "prediction_text": row["predicted_answer_text"]}
        for row in model_predictions
    ]

    results = evaluator.compute(
        predictions=predicted_answers, references=actual_answers
    )
    return results
