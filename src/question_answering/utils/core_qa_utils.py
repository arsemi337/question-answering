from pathlib import Path

import matplotlib.ticker
from datasets import Dataset
import pandas as pd
import shutil
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import Counter

from question_answering.constants import constants
from question_answering.paths import extractive_qa_paths


def load_train_val_test_datasets(dataset_path: Path):
    train = pd.read_csv(dataset_path / "train.csv").dropna()
    val = pd.read_csv(dataset_path / "val.csv").dropna()
    test = pd.read_csv(dataset_path / "test.csv").dropna()
    return train, val, test


def convert_dataframes_to_datasets(dataframes: list[pd.DataFrame]):
    return tuple(
        [
            Dataset.from_pandas(dataframe, preserve_index=False)
            for dataframe in dataframes
        ]
    )


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
        filtered_counter.items(), key=lambda pair: pair[0], reverse=reverse_sort
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


def get_best_model_from_checkpoints(
    trained_model: tf.keras.Model,
    history: tf.keras.callbacks.History,
    model_name: str,
    metric: str = "val_loss",
    remove_checkpoints: bool = True,
):
    best_model_index = np.argmin(history.history[metric]) + 1
    best_model_checkpoints_path = (
        extractive_qa_paths.training_checkpoints_dir / model_name
    )
    best_model_weights_path = (
        best_model_checkpoints_path
        / constants.checkpoint_filename_template.format(epoch=best_model_index)
    )
    best_model = trained_model
    best_model.load_weights(best_model_weights_path)

    if remove_checkpoints:
        shutil.rmtree(best_model_checkpoints_path)

    return best_model


def _create_dirs_if_not_exists(directory: Path):
    if not directory.is_dir():
        directory.mkdir(parents=True)
