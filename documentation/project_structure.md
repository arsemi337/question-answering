# Project structure

The project concerns both generative and extractive QA. The work on these is separated into two directories: [extractive-qa](./../extractive-qa) & [generative-qa](./../generative-qa). 
Both of these directories share similar structure, with each one consisting of:
* ***data*** directory holding data referenced in the notebooks
* ***figures*** directory with figures regarding general data analysis etc.
* ***model-evaluation*** directory with figures and graphs related to models, their training, and evaluation
* ***notebooks*** directory holding various notebooks for training and examining various models
* ***tf-models*** directory with best versions of models saved and trained from specific notebooks
* ***training-checkpoints*** directory with model training checkpoints (typically they are stored there temporarily until the best checkpoint is saved)

Additionally, there is a local package called ***question_answering*** with utility functions, constants and paths used all across the project. 