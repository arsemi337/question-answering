import json
import shutil
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from datasets import Dataset, concatenate_datasets
from matplotlib.ticker import MaxNLocator, PercentFormatter

from question_answering.constants import constants
from question_answering.paths import extractive_qa_paths, generative_qa_paths


def load_datasets_from_csv(dataset_path: Path, filenames=None):
    if filenames is None:
        filenames = ["train.csv", "val.csv", "test.csv"]

    csvs = [pd.read_csv(dataset_path / filename).dropna() for filename in filenames]
    return csvs


def load_datasets_from_json(dataset_path: Path, filenames: list[str]):
    jsons = [pd.read_json(dataset_path / filename).dropna() for filename in filenames]
    return jsons


def convert_dataframes_to_datasets(dataframes: list[pd.DataFrame]):
    return tuple(
        [
            Dataset.from_pandas(dataframe, preserve_index=False)
            for dataframe in dataframes
        ]
    )


def concatenate_hf_datasets(datasets: list[Dataset]):
    return concatenate_datasets(datasets)


def plot_sentence_lengths_histogram(
    sentences: list[str],
    figure_path: Path,
    figure_title: str,
    divider: int,
    min_threshold: int,
    max_threshold: int,
    reverse_sort: bool = False,
    x_label: str = "Words count per sentence",
    y_label: str = "Number of sentences",
):
    word_count_groups = []
    for sentence in sentences:
        word_count = len(sentence.split())
        num_word_count_group = int(word_count / divider) + 1
        lower_group_boundary = divider * num_word_count_group - divider
        upper_group_boundary = divider * num_word_count_group - 1
        word_count_group = f"{lower_group_boundary}-{upper_group_boundary}"
        word_count_groups.append(word_count_group)

    counter = Counter(word_count_groups)
    filtered_counter = {
        x: count
        for x, count in counter.items()
        if min_threshold <= int(x.split("-")[0])
        and int(x.split("-")[1]) <= max_threshold
    }
    sorted_counter = sorted(
        filtered_counter.items(),
        key=lambda pair: int(pair[0].split("-")[0]),
        reverse=reverse_sort,
    )
    labels, values = zip(*sorted_counter)

    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.bar(labels, values, color="dimgray")
    plt.title(figure_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    _create_dirs_if_not_exists(figure_path.parent)

    plt.savefig(figure_path)
    plt.show()


def plot_correct_predictions_by_sentence_length(
    sentences: list[str],
    correctly_predicted: list[bool],
    figure_path: Path,
    figure_title: str,
    divider: int,
    min_threshold: int,
    max_threshold: int,
    x_label: str = "Words count per sentence",
    y_label: str = "Correct predictions",
):
    # Create word count groups for x labels
    word_count_groups = []
    for sentence in sentences:
        word_count = len(sentence.split())
        num_word_count_group = int(word_count / divider) + 1
        lower_group_boundary = divider * num_word_count_group - divider
        upper_group_boundary = divider * num_word_count_group - 1
        if (
            min_threshold <= lower_group_boundary
            and upper_group_boundary <= max_threshold
        ):
            word_count_group = f"{lower_group_boundary}-{upper_group_boundary}"
            word_count_groups.append(word_count_group)

    word_count_groups = list(dict.fromkeys(word_count_groups))
    word_count_groups = sorted(
        word_count_groups, key=lambda group: int(group.split("-")[0])
    )

    # Create dictionaries to store results
    range_to_valid_predictions_count_dict = {
        word_count_group: 0.0 for word_count_group in word_count_groups
    }
    range_to_element_count_dict = {
        word_count_group: 0.0 for word_count_group in word_count_groups
    }
    range_to_pred_accuracy_dict = {
        word_count_group: 0.0 for word_count_group in word_count_groups
    }

    # Manipulate dictionaries
    for index, sentence in enumerate(sentences):
        word_count = len(sentence.split())
        num_word_count_group = int(word_count / divider) + 1
        lower_group_boundary = divider * num_word_count_group - divider
        upper_group_boundary = divider * num_word_count_group - 1
        word_count_group = f"{lower_group_boundary}-{upper_group_boundary}"
        is_sentence_correctly_predicted = correctly_predicted[index]

        # Take the sentence into account if it is between min and max threshold
        if word_count_group in word_count_groups:
            range_to_element_count_dict[word_count_group] = (
                range_to_element_count_dict[word_count_group] + 1
            )

            if is_sentence_correctly_predicted:
                range_to_valid_predictions_count_dict[word_count_group] = (
                    range_to_valid_predictions_count_dict[word_count_group] + 1
                )

    for key in range_to_pred_accuracy_dict.keys():
        range_to_pred_accuracy_dict[key] = (
            range_to_valid_predictions_count_dict[key]
            / range_to_element_count_dict[key]
        )

    # Plot
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    plt.bar(
        word_count_groups,
        [value for value in range_to_pred_accuracy_dict.values()],
        color="dimgray",
    )
    plt.title(figure_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    _create_dirs_if_not_exists(figure_path.parent)

    plt.savefig(figure_path)
    plt.show()


def convert_to_tf_dataset(
    hf_dataset: Dataset,
    columns: list[str],
    label_cols: list[str],
    collator,
    batch_size: int,
    shuffle: bool = False,
):
    return hf_dataset.to_tf_dataset(
        columns=columns,
        label_cols=label_cols,
        collate_fn=collator,
        batch_size=batch_size,
        shuffle=shuffle,
    )

def prepare_tf_dataset(
    model,
    hf_dataset: Dataset,
    collator,
    batch_size: int,
    shuffle: bool = False,
):
    return model.prepare_tf_dataset(
        dataset=hf_dataset,
        collate_fn=collator,
        shuffle=shuffle,
        batch_size=batch_size,
    )


def get_best_model_from_checkpoints(
    trained_model: tf.keras.Model,
    history: dict,
    model_name: str,
    metric: str = "val_loss",
    remove_checkpoints: bool = True,
    model_type: str = "extractive"
):
    best_epoch = int(np.argmin(history.history[metric]) + 1)
    if model_type == "extractive":
        best_model_checkpoints_path = (
            extractive_qa_paths.training_checkpoints_dir / model_name
        )
    else:
        best_model_checkpoints_path = (
            generative_qa_paths.training_checkpoints_dir / model_name
        )
    best_model_weights_path = (
        best_model_checkpoints_path
        / constants.checkpoint_filename_template.format(epoch=best_epoch)
    )
    best_model = trained_model
    best_model.load_weights(best_model_weights_path)

    if remove_checkpoints:
        shutil.rmtree(best_model_checkpoints_path)

    return best_model, best_epoch


def plot_and_save_fig_from_history(
    history: dict,
    attributes: list[str],
    title: str,
    y_label: str,
    x_label: str,
    legend_descriptors: list[str],
    figure_dir_path: Path,
    figure_filename: str,
):
    for attribute in attributes:
        plt.plot(history[attribute])

    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(legend_descriptors, loc="upper left")

    _create_dirs_if_not_exists(figure_dir_path)

    plt.savefig(figure_dir_path / figure_filename)
    plt.show()


def save_dict_as_json(dictionary: dict, dir_path: Path, filename: str):
    if not dir_path.exists() or not dir_path.is_dir():
        dir_path.mkdir(parents=True)

    with open(dir_path / filename, "w") as fp:
        json.dump(dictionary, fp, sort_keys=True, indent=4)


def read_json_as_dict(path: Path) -> dict:
    with open(path, "r") as fp:
        data = json.load(fp)
        return data


def get_gpu_name():
    gpu_devices = tf.config.list_physical_devices("GPU")
    if gpu_devices:
        details = tf.config.experimental.get_device_details(gpu_devices[0])
        return details.get("device_name", "Unknown GPU")


def _create_dirs_if_not_exists(directory: Path):
    if not directory.is_dir():
        directory.mkdir(parents=True)
