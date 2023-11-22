import tensorflow as tf
from transformers import TFAutoModelForSeq2SeqLM, BartTokenizerFast
from datasets import Dataset
from evaluate import load
import sklearn.metrics as skmetrics
import numpy as np
from tqdm import tqdm
import pandas as pd
import evaluate

from question_answering.paths import generative_qa_paths


def save_model(
        model: tf.keras.Model, model_name: str, weights_name: str = "model_weights"
):
    model.save_weights(generative_qa_paths.saved_models_dir / model_name / weights_name)


def load_weights_into_model(
        model: tf.keras.Model, model_name: str, weights_name: str = "model_weights"
) -> tf.keras.Model:
    model.load_weights(generative_qa_paths.saved_models_dir / model_name / weights_name)
    return model


def load_model(
        model_checkpoint: str, model_name: str, weights_name: str = "model_weights"
) -> tf.keras.Model:
    model = TFAutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    model.load_weights(generative_qa_paths.saved_models_dir / model_name / weights_name)
    return model


def generate_predictions(
        model: tf.keras.Model, batch, max_length: int
):
    return model.generate(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        max_new_tokens=max_length,
    )


def get_dataset_dataframe_with_predictions(
        model: tf.keras.Model, tokenizer: BartTokenizerFast, tf_dataset, max_length: int,
):
    predictions_list = []
    labels_list = []
    question_contexts_list = []

    for batch, labels in tqdm(tf_dataset):
        predictions = generate_predictions(model, batch, max_length)
        decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = labels.numpy()
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_predictions = [prediction.strip() for prediction in decoded_predictions]
        decoded_labels = [label.strip() for label in decoded_labels]

        predictions_list.extend(decoded_predictions)
        labels_list.extend(decoded_labels)
        question_contexts_list.extend(tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True))

    return pd.DataFrame(list(zip(question_contexts_list, predictions_list, labels_list)),
                        columns=['question_contexts', 'predictions', 'labels'])


def split_questions_and_contexts_into_two_columns(
        dataframe
):
    questions = []
    contexts = []

    if 'question_contexts' in dataframe:
        for index, row in dataframe.iterrows():
            questions.append(row['question_contexts'].split('?')[0] + '?')
            contexts.append(row['question_contexts'].split('?')[1])

        data = {
            'questions': questions,
            'contexts': contexts,
            'labels': dataframe['labels'],
            'predictions': dataframe['predictions']
        }
        return pd.DataFrame(data)
    else:
        return pd.DataFrame().empty

def get_metrics(
        dataset_dataframe_with_predictions
):
    predictions_list = dataset_dataframe_with_predictions['predictions']
    labels_list = dataset_dataframe_with_predictions['labels']

    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    meteor_metric = evaluate.load("meteor")
    bertscore_metric = evaluate.load("bertscore")
    sacrebleu_metric = evaluate.load("sacrebleu")

    bleu_result = bleu_metric.compute(predictions=predictions_list, references=labels_list)
    rogue_result = rouge_metric.compute(predictions=predictions_list, references=labels_list)
    meteor_result = meteor_metric.compute(predictions=predictions_list, references=labels_list)
    bertscore_result = bertscore_metric.compute(predictions=predictions_list, references=labels_list, lang='en')
    sacrebleu_result = sacrebleu_metric.compute(predictions=predictions_list, references=labels_list)

    return bleu_result, rogue_result, meteor_result, bertscore_result, sacrebleu_result
