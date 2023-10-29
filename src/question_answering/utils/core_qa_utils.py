import os

import datasets
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from question_answering.constants import constants


def load_train_val_test_datasets(dataset_path):
    train = pd.read_csv(dataset_path / "train.csv").dropna()
    val = pd.read_csv(dataset_path / "val.csv").dropna()
    test = pd.read_csv(dataset_path / "test.csv").dropna()
    return train, val, test


def convert_dataframes_to_datasets(dataframes: list):
    return tuple(
        [
            datasets.Dataset.from_pandas(dataframe, preserve_index=False)
            for dataframe in dataframes
        ]
    )


def convert_to_tf_dataset(
    hf_dataset, columns, label_cols, collator, batch_size, shuffle=False
):
    return hf_dataset.to_tf_dataset(
        columns=columns,
        label_cols=label_cols,
        collate_fn=collator,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def plot_and_save_fig_from_history(
    history,
    attributes,
    title,
    y_label,
    x_label,
    legend_descriptors,
    figure_dir,
    figure_filename,
):
    for attribute in attributes:
        plt.plot(history.history[attribute])
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.legend(legend_descriptors, loc="upper left")

    _create_dirs_if_not_exists(figure_dir)

    plt.savefig(figure_dir / figure_filename)
    plt.show()


def save_model(
    model,
    full_model_name,
    models_dir,
    default_model_version=constants.default_model_version,
):
    model.save(models_dir / full_model_name / default_model_version)


def load_model(model_path, model_compile=True):
    return tf.keras.models.load_model(
        model_path / constants.default_model_version, compile=model_compile
    )


def _create_dirs_if_not_exists(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)
