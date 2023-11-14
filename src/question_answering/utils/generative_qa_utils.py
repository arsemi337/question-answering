import tensorflow as tf
from transformers import TFAutoModelForSeq2SeqLM
from datasets import Dataset
from evaluate import load
import sklearn.metrics as skmetrics
import numpy as np

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