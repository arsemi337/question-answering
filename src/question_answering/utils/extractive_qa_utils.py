import tensorflow as tf
from transformers import TFAutoModelForQuestionAnswering

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
