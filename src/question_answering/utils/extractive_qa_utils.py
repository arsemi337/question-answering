import tensorflow as tf
from transformers import TFAutoModelForQuestionAnswering
import sklearn.metrics as skmetrics
import numpy as np
import evaluate

from question_answering.paths import extractive_qa_paths


def save_model(
    model: tf.keras.Model, model_name: str, weights_name: str = "model_weights"
):
    model.save_weights(extractive_qa_paths.saved_models_dir / model_name / weights_name)


def load_weights_into_model(
    model: tf.keras.Model, model_name: str, weights_name: str = "model_weights"
):
    model.load_weights(extractive_qa_paths.saved_models_dir / model_name / weights_name)
    return model


def load_model(
    model_checkpoint: str, model_name: str, weights_name: str = "model_weights"
):
    model = TFAutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
    model.load_weights(extractive_qa_paths.saved_models_dir / model_name / weights_name)
    return model

def get_classification_evaluation_metrics(class_actual, class_preds, average='binary'):
    precision = skmetrics.precision_score(class_actual, class_preds, average=average)
    recall = skmetrics.recall_score(class_actual, class_preds, average=average)
    f1 = skmetrics.f1_score(class_actual, class_preds, average=average)
    return precision, recall, f1

def get_class_preds(predictions, type='start_logits', return_classes=True):
    predictions = predictions[type]
    probabilities = tf.nn.softmax(predictions)
    if return_classes:
        return np.argmax(probabilities, axis=1)
    else:
        return probabilities.numpy()
    
def extract_answer_tokens(tokenized_dataset_row):
    start = tokenized_dataset_row['start_positions']
    end = tokenized_dataset_row['end_positions']
    tokenized_dataset_row['answer_tokens'] = tokenized_dataset_row['input_ids'][start : end + 1]
    return tokenized_dataset_row

def decode_answer_tokens(tokenized_dataset_row, tokenizer):
    tokens = tokenized_dataset_row['answer_tokens']
    answer = tokenizer.decode(tokens)
    return answer

def calculatePureExactMatch(evaluator, test_dataset, predicted):
    exact_match_original = evaluator.compute(predictions=test_dataset['answer_text'], references=predicted['answer_text'])
    exact_match_lowercase = evaluator.compute(predictions=test_dataset['answer_text'], references=predicted['answer_text'], ignore_case=True)
    exact_match_lowercase_no_punctuation = evaluator.compute(predictions=test_dataset['answer_text'], references=predicted['answer_text'], ignore_case=True, ignore_punctuation=True)
    return exact_match_original, exact_match_lowercase, exact_match_lowercase_no_punctuation

def calculateSquadExactMatch(evaluator, test_dataset, predicted):
    theoretical_answers_squad = [{'id': row['id'], 'answers': {'text': [row['answer_text']], 'answer_start': [row['answer_start']]}} for row in test_dataset]
    temporaryDataset = test_dataset.select_columns(['id', 'answer_text'])
    temporaryDataset = temporaryDataset.add_column("predicted_answers", predicted)
    predicted_answers = [{'id': row['id'], 'prediction_text': row['predicted_answers']['answer_text']} for row in temporaryDataset]
    results = evaluator.compute(predictions=predicted_answers, references=theoretical_answers_squad)
    return results