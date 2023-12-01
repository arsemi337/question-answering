import tensorflow as tf
from transformers import TFAutoModelForSeq2SeqLM, BartTokenizerFast, DataCollatorForSeq2Seq
from evaluate import load
import sklearn.metrics as skmetrics
import numpy as np
from tqdm import tqdm
import pandas as pd
from pandas import DataFrame
import evaluate
from tqdm import tqdm

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
        model: tf.keras.Model, batch, max_length: int,
):
    return model.generate(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        max_new_tokens=max_length,
    )


def get_dataset_dataframe_with_predictions(
        model: tf.keras.Model,
        tokenizer: BartTokenizerFast,
        tf_dataset_list,
        dataframe: DataFrame,
        max_length: int,
        index_to_start_from: int = 0
):
    predictions_list = []
    labels_list = []
    question_contexts_list = []

    i = 0
    for dataset in tqdm(tf_dataset_list):
        if i < index_to_start_from:
            i = i + 1
            continue
        for batch, labels in tqdm(dataset):
            predictions = generate_predictions(model, batch, max_length)
            decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            labels = labels
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            decoded_predictions = [prediction.strip() for prediction in decoded_predictions]
            decoded_labels = [label.strip() for label in decoded_labels]
            predictions_list.extend(decoded_predictions)
            labels_list.extend(decoded_labels)
            question_contexts_list.extend(tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True))

            data = {
                'question_contexts': tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True),
                'labels': decoded_labels,
                'predictions': decoded_predictions
            }
            dataframe = pd.concat([dataframe, pd.DataFrame(data)], ignore_index=True)
        i = i + 1

    return dataframe


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

    bleu_result = bleu_metric.compute(predictions=predictions_list, references=labels_list)
    rogue_result = rouge_metric.compute(predictions=predictions_list, references=labels_list)
    meteor_result = meteor_metric.compute(predictions=predictions_list, references=labels_list)

    return bleu_result, rogue_result, meteor_result


def calculate_rouges_for_each_sample(dataframe):
    rouge_metric = evaluate.load("rouge")
    rogue1_list = []
    rogue2_list = []
    rogueL_list = []

    for _, row in tqdm(dataframe.iterrows()):
        rogue1_list.append(rouge_metric.compute(
            predictions=[row['predictions']], references=[row['labels']])['rouge1']
                           )
        rogue2_list.append(rouge_metric.compute(
            predictions=[row['predictions']], references=[row['labels']])['rouge2']
                           )
        rogueL_list.append(rouge_metric.compute(
            predictions=[row['predictions']], references=[row['labels']])['rougeL']
                           )

    dataframe['rouge1'] = rogue1_list
    dataframe['rouge2'] = rogue2_list
    dataframe['rougeL'] = rogueL_list

    return dataframe


def calculate_bleus_for_each_sample(dataframe):
    bleu_metric = evaluate.load("bleu")
    bleu_list = []
    bleu1_list = []
    bleu2_list = []

    for _, row in tqdm(dataframe.iterrows()):
        bleu_list.append(bleu_metric.compute(
            predictions=[row['predictions']], references=[row['labels']])['bleu']
                         )
        bleu1_list.append(bleu_metric.compute(
            predictions=[row['predictions']], references=[row['labels']])['precisions'][0]
                          )
        bleu2_list.append(bleu_metric.compute(
            predictions=[row['predictions']], references=[row['labels']])['precisions'][1]
                          )

    dataframe['bleu'] = bleu_list
    dataframe['bleu1'] = bleu1_list
    dataframe['bleu2'] = bleu2_list

    return dataframe


def calculate_meteor_for_each_sample(dataframe):
    meteor_metric = evaluate.load("meteor")
    meteor_list = []

    for _, row in tqdm(dataframe.iterrows()):
        meteor_list.append(meteor_metric.compute(
            predictions=[row['predictions']], references=[row['labels']])['meteor']
                           )

    dataframe['meteor'] = meteor_list

    return dataframe


def calculate_prediction_numbers_per_metric_range(
        df, metric_name, thresholds
):
    prediction_numbers = []

    for index, _ in enumerate(thresholds):
        if index == (len(thresholds) - 1):
            break

        if index == 0:
            prediction_numbers.append(
                len(df[(df[metric_name] >= thresholds[index]) &
                       (df[metric_name] <= thresholds[index + 1])])
            )
        else:
            prediction_numbers.append(
                len(df[(df[metric_name] > thresholds[index]) &
                       (df[metric_name] <= thresholds[index + 1])])
            )

    return prediction_numbers
