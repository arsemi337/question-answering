from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from .__helpers import create_dirs_if_not_exists


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
    names = word_count_groups
    values = [value for value in range_to_pred_accuracy_dict.values()]

    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    diagram = plt.bar(
        names,
        values,
        color="dimgray",
        zorder=3
    )
    plt.title(figure_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(axis='y')

    for i, rect in enumerate(diagram):
        height = rect.get_height()
        exact_percent = round(values[i] * 100, 1)
        plt.annotate("{}%".format(exact_percent), (rect.get_x() + rect.get_width() / 2, height), ha="center",
                     va="bottom",
                     fontsize=10)

    create_dirs_if_not_exists(figure_path.parent)

    plt.savefig(figure_path)
    plt.show()
