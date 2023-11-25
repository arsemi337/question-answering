from collections import defaultdict

import numpy as np
import tensorflow as tf
from datasets import Dataset


def get_preds(
    outputs,
    output_key: str,
    return_type: str,
):
    allowed_return_types = ["class", "logits", "probability"]

    predictions = outputs[output_key]

    if return_type in allowed_return_types:
        if return_type == "logits":
            return predictions
        else:
            probabilities = tf.nn.softmax(predictions)
            if return_type == "probability":
                return probabilities
            else:
                classes = np.argmax(probabilities, axis=1)
                return classes


def get_predicted_texts(
    start_logits: np.ndarray,
    end_logits: np.ndarray,
    features: Dataset,
    examples: Dataset,
    n_best: int = 20,
    max_answer_length: int = 30,
):
    example_to_features = defaultdict(list)

    for idx, feature in enumerate(features):
        example_id = feature["example_id"]
        example_to_features[example_id].append(idx)

    predicted_answers = []
    for example in examples:
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            best_start_indices = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            best_end_indices = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in best_start_indices:
                for end_index in best_end_indices:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[
                            offsets[start_index][0] : offsets[end_index][1]
                        ],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(best_answer["text"])
        else:
            predicted_answers.append("")

    return predicted_answers
