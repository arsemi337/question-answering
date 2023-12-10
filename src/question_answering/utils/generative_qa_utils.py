import tensorflow as tf
from transformers import TFAutoModelForSeq2SeqLM, BartTokenizerFast
import numpy as np
import pandas as pd
from pandas import DataFrame
import evaluate
from tqdm import tqdm
from pathlib import Path
from datasets import Dataset
import matplotlib.pyplot as plt

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


def calculate_prediction_counts_per_metric_range(
        df, thresholds
):
    lists_dictionary = {}
    metric_list = ['bleu', 'bleu1', 'bleu2', 'rouge1', 'rouge2', 'rougeL', 'meteor']

    for metric in metric_list:
        prediction_numbers = []
        for index, _ in enumerate(thresholds):
            if index == (len(thresholds) - 1):
                break

            if index == 0:
                prediction_numbers.append(
                    len(df[(df[metric] >= thresholds[index]) &
                           (df[metric] <= thresholds[index + 1])])
                )
            else:
                prediction_numbers.append(
                    len(df[(df[metric] > thresholds[index]) &
                           (df[metric] <= thresholds[index + 1])])
                )

        lists_dictionary.update(
            {metric: prediction_numbers
             }
        )

    return lists_dictionary


def count_prediction_numbers_per_metric_range_for_specific_question_type(
        dataframe_predictions_and_question_types: pd.DataFrame
):
    dataframes_dictionary = {}
    question_type_list = ['whats', 'wheres', 'hows', 'for_whats', 'whens', 'closed', 'others']
    ranges_column = ['(0.0, 0.2)', '(0.2, 0.4)', '(0.4, 0.6)', '(0.6, 0.8)', '(0.8, 1.0)']
    threshold_list = [-0.1, 0.2, 0.4, 0.6, 0.8, 1.1]
    metric_list = ['bleu', 'bleu1', 'bleu2', 'rouge1', 'rouge2', 'rougeL', 'meteor']

    for question_type in question_type_list:
        predictions_for_question_type = dataframe_predictions_and_question_types.groupby(['question_type']).get_group(
            question_type).drop('question_type', axis=1).reset_index(drop=True)
        temp_dataframe = pd.DataFrame()
        temp_dataframe['ranges'] = ranges_column
        for metric in metric_list:
            temp_dataframe[metric] = pd.Series(predictions_for_question_type.groupby(
                pd.cut(predictions_for_question_type[metric], threshold_list), observed=False).count()[metric].values)
            dataframes_dictionary.update(
                {question_type: temp_dataframe}
            )

    return dataframes_dictionary


def get_closed_questions_split_according_to_answer_correctness(
        dataframe: pd.DataFrame
):
    closed_questions_dataframes_dictionary = {}

    rows_with_closed_questions = dataframe[
        (dataframe['labels'] == 'Yes') |
        (dataframe['labels'] == 'No')].reset_index(drop=True)
    closed_questions_dataframes_dictionary.update(
        {"closed_all": rows_with_closed_questions}
    )

    rows_with_closed_questions_correct_answers = rows_with_closed_questions[
        (rows_with_closed_questions['labels'] == rows_with_closed_questions['predictions'])].reset_index(drop=True)
    closed_questions_dataframes_dictionary.update(
        {"closed_correct_answer": rows_with_closed_questions_correct_answers}
    )

    rows_with_closed_questions_wrong_answers = rows_with_closed_questions[
        (rows_with_closed_questions['labels'] != rows_with_closed_questions['predictions'])].reset_index(drop=True)
    closed_questions_dataframes_dictionary.update(
        {"closed_wrong_answer": rows_with_closed_questions_wrong_answers}
    )

    rows_with_long_answers_for_closed_questions = rows_with_closed_questions[
        (rows_with_closed_questions['labels'] != 'Yes') &
        (rows_with_closed_questions['labels'] != 'No')].reset_index(drop=True)
    closed_questions_dataframes_dictionary.update(
        {"closed_long_answer": rows_with_long_answers_for_closed_questions}
    )

    return closed_questions_dataframes_dictionary


def return_question_type(question: str, answer: str):
    if question.startswith("What"):
        return "whats"
    elif question.startswith("Where"):
        return "wheres"
    elif question.startswith("How"):
        return "hows"
    elif question.startswith("For what"):
        return "for_whats"
    elif question.startswith("When"):
        return "whens"
    elif answer == "Yes" or answer == "No":
        return "closed"
    else:
        return "others"


def add_question_types_to_dataset_dataframe(
        dataset_predictions_dataframe: pd.DataFrame
):
    default_type_values = ['others'] * len(dataset_predictions_dataframe)
    dataset_predictions_dataframe['question_type'] = default_type_values

    for i in dataset_predictions_dataframe.index:
        dataset_predictions_dataframe.at[i, 'question_type'] = return_question_type(
            question=dataset_predictions_dataframe['questions'][i],
            answer=dataset_predictions_dataframe['labels'][i]
        )

    return dataset_predictions_dataframe


def save_question_type_metrics_dictionary_to_csv(
        model_evaluation_dir: Path, question_type_metrics_dictionary: pd.DataFrame
):
    question_type_list = ['whats', 'wheres', 'hows', 'for_whats', 'whens', 'closed', 'others']
    save_path = model_evaluation_dir / "question_type_numbers"

    for question_type in question_type_list:
        if not save_path.exists():
            save_path.mkdir(parents=True)
        question_type_metrics_dictionary[question_type].to_csv(
            save_path / f"question_type_numbers_{question_type}.csv",
            index=True,
            index_label="index",
            escapechar="\\",
        )


def read_question_type_metrics_dictionary_from_csv(
        model_evaluation_dir: Path
):
    dataframes_dictionary = {}
    question_type_list = ['whats', 'wheres', 'hows', 'for_whats', 'whens', 'closed', 'others']
    save_path = model_evaluation_dir / "question_type_numbers"

    for question_type in question_type_list:
        dataframes_dictionary.update(
            {question_type: pd.read_csv(save_path / f"question_type_numbers_{question_type}.csv")}
        )

    return dataframes_dictionary


def count_questions_per_question_type(
        dataset: Dataset
):
    question_numbers_per_type = {}
    question_type_list = ['what', 'where', 'how', 'for what', 'when']
    sum = 0
    for question_type in question_type_list:
        questions = dataset['questions']
        questions = [question.lower() for question in questions]

        questions_number = len(list(filter(lambda x: x.startswith(question_type), questions)))
        sum = sum + questions_number
        question_numbers_per_type.update({
            question_type: questions_number
        })

    rows_with_closed_questions = dataset.filter(
        lambda row: row['answers'] == 'Yes' or row['answers'] == 'No'
    )
    closed_questions_number = len(rows_with_closed_questions)
    sum = sum + closed_questions_number
    question_numbers_per_type.update({
        'closed': closed_questions_number
    })

    others_number = len(dataset) - sum
    question_numbers_per_type.update({
        'others': others_number
    })

    return question_numbers_per_type


def plot_question_type_diagram(
        question_numbers_per_type: dict,
        figure_path: Path,
        x_label: str = "Questions count per type",
        y_label: str = "Questions types",
):
    names = list(question_numbers_per_type.keys())
    values = list(question_numbers_per_type.values())

    diagram = plt.bar(range(len(question_numbers_per_type)), values, tick_label=names, color="dimgray", zorder=3)
    plt.title("Questions count per questions type")
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    for rect1 in diagram:
        value = round(rect1.get_height() / sum(values) * 100, 1)
        height = rect1.get_height()
        plt.annotate("{}%".format(value), (rect1.get_x() + rect1.get_width() / 2, height), ha="center",
                     va="bottom", fontsize=10)

    if not figure_path.parent.is_dir():
        figure_path.parent.mkdir(parents=True)

    plt.grid(axis='y')
    plt.savefig(figure_path)
    plt.show()


def plot_prediction_counts_per_metric_range_diagrams(
        prediction_counts_per_metric: dict,
        thresholds: list,
        figure_directory_path: Path,
        x_label: str = "Questions count per range",
        y_label: str = "Threshold ranges",
):
    names = []
    for index, _ in enumerate(thresholds):
        if index == len(thresholds) - 1:
            continue
        names.append(f'({thresholds[index]}, {thresholds[index + 1]})')

    for key, value in prediction_counts_per_metric.items():
        values = value

        diagram = plt.bar(names, values, color="dimgray", zorder=3)
        plt.title(f"{key} - prediction counts per metric range")
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        for rect1 in diagram:
            value = round(rect1.get_height() / sum(values) * 100, 1)
            height = rect1.get_height()
            plt.annotate("{}%".format(value), (rect1.get_x() + rect1.get_width() / 2, height), ha="center", va="bottom",
                         fontsize=10)

        figure_path = figure_directory_path / 'figures' / 'counts-per-metric-range' / f"{key}_prediction_counts_per_metric_range_diagram.png"

        if not figure_path.parent.is_dir():
            figure_path.parent.mkdir(parents=True)

        plt.grid(axis='y')
        plt.savefig(figure_path)
        plt.show()


def plot_prediction_metrics_per_question_type_diagram(
        metric_mean_values_dataframe: pd.DataFrame,
        figure_directory_path: Path,
        x_label: str = "Questions type",
        y_label: str = "Metric value",
):
    question_type_list = []
    metric_list = []
    prediction_metrics_list = []

    for column in metric_mean_values_dataframe:
        column_object = metric_mean_values_dataframe[column]

        if column == 'question_type':
            question_type_list = column_object.values
        else:
            metric_list.append(column)
            prediction_metrics_list.append(column_object.values)

    for index, metric in enumerate(metric_list):
        names = question_type_list
        values = prediction_metrics_list[index]

        diagram = plt.bar(names, values, color="dimgray", zorder=3)
        plt.ylim([0.0, 1.0])
        plt.title(f"{metric} - prediction metrics per question type")
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        for rect1 in diagram:
            value = round(rect1.get_height(), 4)
            height = rect1.get_height()
            plt.annotate("{}%".format(value), (rect1.get_x() + rect1.get_width() / 2, height), ha="center", va="bottom",
                         fontsize=10)

        figure_path = figure_directory_path / f"{metric}_prediction_metrics_per_question_type_diagram.png"

        if not figure_path.parent.is_dir():
            figure_path.parent.mkdir(parents=True)

        plt.grid(axis='y', zorder=0)
        plt.savefig(figure_path)
        plt.show()


def plot_prediction_counts_per_metric_range_per_question_type_diagram(
        question_type_metrics_dictionary: dict,
        figure_directory_path: Path,
        x_label: str = "Questions count per range",
        y_label: str = "Threshold ranges",
):
    threshold_ranges = []
    metric_list = []
    prediction_metrics_list = []

    for question_type, question_type_metrics_dataframe in question_type_metrics_dictionary.items():
        if 'index' in question_type_metrics_dataframe:
            question_type_metrics_dataframe = question_type_metrics_dataframe.drop(['index'], axis=1)

        for column in question_type_metrics_dataframe:
            column_object = question_type_metrics_dataframe[column]

            if column == 'ranges':
                threshold_ranges = column_object.values
            else:
                metric_list.append(column)
                prediction_metrics_list.append(column_object.values)

        for index, metric in enumerate(metric_list):
            names = threshold_ranges
            values = prediction_metrics_list[index]

            diagram = plt.bar(names, values, color="dimgray", zorder=3)
            plt.title(f"{metric} - prediction metrics per question type")
            plt.xlabel(x_label)
            plt.ylabel(y_label)

            for rect1 in diagram:
                value = round(rect1.get_height() / sum(values) * 100, 1)
                height = rect1.get_height()
                plt.annotate("{}%".format(value), (rect1.get_x() + rect1.get_width() / 2, height), ha="center",
                             va="bottom",
                             fontsize=10)

            figure_path = figure_directory_path / question_type / f"{metric}_prediction_counts_per_metric_per_question_type_diagram.png"

            if not figure_path.parent.is_dir():
                figure_path.parent.mkdir(parents=True)

            plt.grid(axis='y', zorder=0)
            plt.savefig(figure_path)
            plt.close()
