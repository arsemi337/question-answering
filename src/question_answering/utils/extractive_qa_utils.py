import tensorflow as tf
from transformers import TFAutoModelForQuestionAnswering
from datasets import Dataset
from evaluate import load
import sklearn.metrics as skmetrics
import numpy as np

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
    tokenized_dataset_row["answer_tokens"] = tokenized_dataset_row["input_ids"][
        start : end + 1
    ]
    return tokenized_dataset_row


def decode_answer_tokens(tokenized_dataset_row: dict, tokenizer):
    tokens = tokenized_dataset_row["answer_tokens"]
    answer = tokenizer.decode(tokens)
    return answer


def calculate_pure_exact_match(test_dataset: Dataset, predicted):
    evaluator = load("exact_match")

    exact_match_original = evaluator.compute(
        predictions=test_dataset["answer_text"], references=predicted["answer_text"]
    )
    exact_match_lowercase = evaluator.compute(
        predictions=test_dataset["answer_text"],
        references=predicted["answer_text"],
        ignore_case=True,
    )
    exact_match_lowercase_no_punctuation = evaluator.compute(
        predictions=test_dataset["answer_text"],
        references=predicted["answer_text"],
        ignore_case=True,
        ignore_punctuation=True,
    )

    return (
        exact_match_original,
        exact_match_lowercase,
        exact_match_lowercase_no_punctuation,
    )


def calculate_squad_exact_match(test_dataset: Dataset, predicted):
    evaluator = load("squad")

    theoretical_answers_squad = [
        {
            "id": row["id"],
            "answers": {
                "text": [row["answer_text"]],
                "answer_start": [row["answer_start"]],
            },
        }
        for row in test_dataset
    ]
    temp_dataset = test_dataset.select_columns(["id", "answer_text"])
    temp_dataset = temp_dataset.add_column("predicted_answers", predicted)
    predicted_answers = [
        {"id": row["id"], "prediction_text": row["predicted_answers"]["answer_text"]}
        for row in temp_dataset
    ]
    results = evaluator.compute(
        predictions=predicted_answers, references=theoretical_answers_squad
    )
    return results
