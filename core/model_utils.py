import matplotlib.pyplot as plt
import os
import shutil
import numpy as np
import tensorflow as tf
import sklearn.metrics as skmetrics


def split_train_val_test(tokenized_dataset, train_size, test_size_of_val_test_batch, shuffle=False, seed=42):
    split_train_val_and_test_dataset = tokenized_dataset.train_test_split(train_size=train_size, shuffle=shuffle,
                                                                          seed=seed)
    split_val_test_dataset = split_train_val_and_test_dataset["test"].train_test_split(
        test_size=test_size_of_val_test_batch, shuffle=shuffle, seed=seed)

    tokenized_train_dataset = split_train_val_and_test_dataset["train"]
    tokenized_val_dataset = split_val_test_dataset["train"]
    tokenized_test_dataset = split_val_test_dataset["test"]

    return tokenized_train_dataset, tokenized_val_dataset, tokenized_test_dataset


def convert_to_tf_dataset(hf_dataset, columns, label_cols, collator, batch_size, shuffle=False):
    return hf_dataset.to_tf_dataset(
        columns=columns,
        label_cols=label_cols,
        collate_fn=collator,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def plot_and_save_fig_from_history(history, attributes, title, y_label, x_label, legend_descriptors, figure_dir_path,
                                   figure_filename):
    for attribute in attributes:
        plt.plot(history.history[attribute])
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.legend(legend_descriptors, loc='upper left')

    figure_dir = os.path.join(os.curdir, figure_dir_path)
    _create_dirs_if_not_exists(figure_dir)

    plt.savefig(os.path.join(figure_dir, figure_filename))
    plt.show()


def get_best_model_from_checkpoints(trained_model, history, checkpoints_dir, checkpoint_filename_template,
                                    metric='val_loss'):
    best_model_index = np.argmin(history.history[metric]) + 1
    best_model_weights_path = os.path.join(checkpoints_dir, checkpoint_filename_template.format(epoch=best_model_index))
    best_model = trained_model
    best_model.load_weights(best_model_weights_path)
    shutil.rmtree(checkpoints_dir)
    return best_model


def save_model(model, model_name, training_number, saved_models_dir, default_model_version):
    saved_model_name = f"{model_name}_{training_number}"
    model.save(os.path.join(saved_models_dir, saved_model_name, default_model_version))


def get_class_preds(model, test_dataset, return_classes=True, model_returns_logits=True):
    if model_returns_logits:
        predictions = model.predict(test_dataset)['logits']
    else:
        predictions = model.predict(test_dataset)
    probabilities = tf.nn.softmax(predictions)
    if return_classes:
        return np.argmax(probabilities, axis=1)
    else:
        return probabilities.numpy()


def get_classification_evaluation_metrics(class_actual, class_preds, average='binary'):
    precision = skmetrics.precision_score(class_actual, class_preds, average=average)
    recall = skmetrics.recall_score(class_actual, class_preds, average=average)
    f1 = skmetrics.f1_score(class_actual, class_preds, average=average)
    return precision, recall, f1


def plot_and_save_conf_matrix(class_actual, class_preds, figure_dir_path, figure_filename):
    confusion_matrix = skmetrics.confusion_matrix(class_actual, class_preds)
    disp = skmetrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    disp.plot()

    figure_dir = os.path.join(os.curdir, figure_dir_path)
    _create_dirs_if_not_exists(figure_dir)

    plt.savefig(os.path.join(figure_dir, figure_filename))
    plt.show()


def print_incorrectly_predicted_texts(texts, class_actual, class_preds, number_of_incorrect_preds=10):
    bad_prediction_count = 0
    predictions_index = 0

    while bad_prediction_count < number_of_incorrect_preds and predictions_index < len(texts):
        predictions_index += 1

        # if bad prediction then show text and increase count
        if class_preds[predictions_index] != class_actual[predictions_index]:
            print(f"""
            BAD PREDICTION:
            - INDEX: {predictions_index}
            - TEXT: {texts[predictions_index]}
            - PREDICTED VALUE: {class_preds[predictions_index]}
            - CORRECT VALUE: {class_actual[predictions_index]}
            """)
            bad_prediction_count += 1


def _create_dirs_if_not_exists(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)
