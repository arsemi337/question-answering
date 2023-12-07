from shutil import rmtree

import tensorflow as tf
from transformers import TFAutoModelForQuestionAnswering

from question_answering.constants import constants
from question_answering.paths import extractive_qa_paths


def load_best_model_from_checkpoints(
    model: tf.keras.Model, model_name: str, epoch: int, remove_checkpoints: bool
):
    model_checkpoints_path = extractive_qa_paths.training_checkpoints_dir / model_name
    best_model_weights_path = (
        model_checkpoints_path
        / constants.checkpoint_filename_template.format(epoch=epoch)
    )

    best_model = model
    best_model.load_weights(best_model_weights_path)

    if remove_checkpoints:
        rmtree(model_checkpoints_path)

    return best_model


def save_model(model: tf.keras.Model, model_name: str):
    model.save_weights(
        extractive_qa_paths.saved_models_dir
        / model_name
        / constants.saved_model_weights_name
    )


def load_weights_into_model(model: tf.keras.Model, model_name: str) -> tf.keras.Model:
    model.load_weights(
        extractive_qa_paths.saved_models_dir
        / model_name
        / constants.saved_model_weights_name
    )
    return model


def load_model(model_checkpoint: str, model_name: str) -> tf.keras.Model:
    model = TFAutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
    model.load_weights(
        extractive_qa_paths.saved_models_dir
        / model_name
        / constants.saved_model_weights_name
    )
    return model
