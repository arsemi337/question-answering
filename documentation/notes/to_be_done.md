# Things to be done

## High priority
- Evaluation of stride model on medical dataset (smaller text spans)
- General accuracy and altered test set preprocessing
- How many start_preds before end_preds and how many of these cases can be turned into actual valid predictions using the logits processing

## Medium priority
- Hyperparameter tuning of extractive qa models

## Low priority
- Train squad_bert-uncased_1 again

## Optional
- Training models with varying stride and comparing their performance

## To be remembered
- test_dataset sample with index 4 has "gold" as answer, but it's a part of word golden
- https://huggingface.co/spaces/evaluate-metric/squad (human performance etc.)
- https://kierszbaumsamuel.medium.com/f1-score-in-nlp-span-based-qa-task-5b115a5e7d41 (precision, recall, f1 in case of extractive qa)