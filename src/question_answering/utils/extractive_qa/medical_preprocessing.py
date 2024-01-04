from datasets import Dataset


def span_dataset(
    dataset: Dataset,
    window: int,
    batched: bool = True,
    remove_columns: list[str] = None,
):
    def get_context_span(context: str, answer_start: int, answer_text: str):
        start_span = max(0, answer_start - window)
        end_span = min(len(context), answer_start + len(answer_text) + window)

        context_span = context[start_span:end_span]

        return {"context_span": context_span, "answer_start_shift": start_span}

    def span_samples(samples):
        answer_starts = samples["answer_start"]
        answer_texts = samples["answer_text"]
        contexts = samples["context"]

        for i, context in enumerate(contexts):
            context_span_result = get_context_span(
                context=context,
                answer_start=answer_starts[i][0],
                answer_text=answer_texts[i][0],
            )

            contexts[i] = context_span_result["context_span"]
            answer_starts[i] = [
                answer_starts[i][0] - context_span_result["answer_start_shift"]
            ]

        samples["context"] = contexts
        samples["answer_start"] = answer_starts
        return samples

    return dataset.map(span_samples, batched=batched, remove_columns=remove_columns)


def preprocess_medical_training_dataset(
        dataset: Dataset,
        tokenizer,
        max_length: int,
        stride: int,
        batched: bool = True,
        remove_columns: list[str] = None,
):
    def preprocess_samples(samples):
        questions = [q.strip() for q in samples["question"]]
        contexts = [c.strip() for c in samples["context"]]

        inputs = tokenizer(
            questions,
            contexts,
            max_length=max_length,
            padding="max_length",
            truncation="only_second",
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
        )

        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answer_starts = samples["answer_start"]
        answer_texts = samples["answer_text"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer_start = answer_starts[sample_idx][0]
            answer_text = answer_texts[sample_idx][0]
            start_char = answer_start
            end_char = start_char + len(answer_text)
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label is (0, 0)
            if (
                    offset[context_start][0] > start_char
                    or offset[context_end][1] < end_char
            ):
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end TOKEN positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    return dataset.map(
        preprocess_samples, batched=batched, remove_columns=remove_columns
    )


def preprocess_medical_test_dataset(
    dataset: Dataset,
    tokenizer,
    max_length: int,
    stride: int,
    batched: bool = True,
    remove_columns: list[str] = None,
):
    def preprocess_samples(samples):
        questions = [q.strip() for q in samples["question"]]
        contexts = [c.strip() for c in samples["context"]]

        inputs = tokenizer(
            questions,
            contexts,
            max_length=max_length,
            padding="max_length",
            truncation="only_second",
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
        )

        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answer_starts = samples["answer_start"]
        answer_texts = samples["answer_text"]
        start_positions = []
        end_positions = []
        example_ids = []
        new_offset_mapping = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            example_id = samples["id"][sample_idx]
            answer_start = answer_starts[sample_idx][0]
            answer_text = answer_texts[sample_idx][0]
            start_char = answer_start
            end_char = start_char + len(answer_text)
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label is (0, 0)
            if (
                offset[context_start][0] > start_char
                or offset[context_end][1] < end_char
            ):
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end TOKEN positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

            example_ids.append(example_id)
            new_offset_mapping.append(
                [o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)]
            )

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        inputs["example_id"] = example_ids
        inputs["offset_mapping"] = new_offset_mapping
        return inputs

    return dataset.map(
        preprocess_samples, batched=batched, remove_columns=remove_columns
    )


def preprocess_medical_training_dataset_no_stride(
        dataset: Dataset,
        tokenizer,
        max_length: int,
        batched: bool = True,
        remove_columns: list[str] = None,
):
    def preprocess_samples(samples):
        questions = [q.strip() for q in samples["question"]]
        contexts = [c.strip() for c in samples["context"]]

        inputs = tokenizer(
            questions,
            contexts,
            max_length=max_length,
            padding="max_length",
            return_offsets_mapping=True,
        )

        offset_mapping = inputs.pop("offset_mapping")
        answer_starts = samples["answer_start"]
        answer_texts = samples["answer_text"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            answer_start = answer_starts[i][0]
            answer_text = answer_texts[i][0]
            start_char = answer_start
            end_char = start_char + len(answer_text)
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label is (0, 0)
            if (
                    offset[context_start][0] > start_char
                    or offset[context_end][1] < end_char
            ):
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end TOKEN positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    return dataset.map(
        preprocess_samples, batched=batched, remove_columns=remove_columns
    )


def preprocess_medical_test_dataset_no_stride(
        dataset: Dataset,
        tokenizer,
        max_length: int,
        batched: bool = True,
        remove_columns: list[str] = None,
):
    def preprocess_samples(samples):
        questions = [q.strip() for q in samples["question"]]
        contexts = [c.strip() for c in samples["context"]]

        inputs = tokenizer(
            questions,
            contexts,
            max_length=max_length,
            padding="max_length",
            return_offsets_mapping=True,
        )

        offset_mapping = inputs.pop("offset_mapping")
        answer_starts = samples["answer_start"]
        answer_texts = samples["answer_text"]
        start_positions = []
        end_positions = []
        example_ids = []
        new_offset_mapping = []

        for i, offset in enumerate(offset_mapping):
            example_id = samples["id"][i]
            answer_start = answer_starts[i][0]
            answer_text = answer_texts[i][0]
            start_char = answer_start
            end_char = start_char + len(answer_text)
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

            example_ids.append(example_id)
            new_offset_mapping.append(
                [o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)]
            )

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        inputs["example_id"] = example_ids
        inputs["offset_mapping"] = new_offset_mapping
        return inputs

    return dataset.map(
        preprocess_samples, batched=batched, remove_columns=remove_columns
    )
