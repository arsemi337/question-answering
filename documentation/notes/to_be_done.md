# Things to be done

## Common:
- Hyperparameter tuning

## Extractive qa:

### This semester
- Train BERT on medical data
- Hyperparameter tuning: `{
  "per_gpu_batch_size": [16, 32],
  "learning_rate": [2e-5, 3e-5, 5e-5],
  "num_epochs": [2, 3, 4]
  }`
- 

### Next semester?

##### 1. How many of bad predictions have start pred greater than end pred?
- Retrieve bad preds
- How many have start pred greater than end pred?
- How many of these can be turned into good predictions?

##### 2. Could check if accuracy == exact match, when using same amounts of samples for calculating these metrics

##### 3. Take a deeper look into predictions and ground truth - especially at the normalization and the way it improves the results
- Get predictions where normalized result is good but normal one isn't
- In case of correct normal predictions, are normalized ones always good? - Is it always an improvement?

## Generative qa:
- trash

