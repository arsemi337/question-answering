from datasets import Dataset


def filter_samples_below_number_of_tokens(tokenizer, dataset: Dataset, max_tokens: int):
    def tokenize_sample(sample):
        question = sample["question"].strip()
        context = sample["context"].strip()

        return tokenizer(question, context)

    def is_sample_exceeds_max_tokens(sample):
        tokenized_sample = tokenize_sample(sample)
        return len(tokenized_sample["input_ids"]) > max_tokens

    return dataset.filter(lambda sample: not is_sample_exceeds_max_tokens(sample))
