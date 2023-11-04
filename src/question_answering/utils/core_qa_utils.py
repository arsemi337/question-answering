import matplotlib.pyplot as plt
from datasets import Dataset
import pandas as pd
import shutil
import numpy as np
import tensorflow as tf
from pathlib import Path

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


def plot_and_save_fig_from_history(
    history: tf.keras.callbacks.History,
    attributes: list[str],
    title: str,
    y_label: str,
    x_label: str,
    legend_descriptors: list[str],
    figure_dir_path: Path,
    figure_filename: str,
):
    for attribute in attributes:
        plt.plot(history.history[attribute])
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.legend(legend_descriptors, loc="upper left")

    _create_dirs_if_not_exists(figure_dir_path)

    plt.savefig(figure_dir_path / figure_filename)
    plt.show()


def _create_dirs_if_not_exists(directory: Path):
    if not directory.is_dir():
        directory.mkdir()
