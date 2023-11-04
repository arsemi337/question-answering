# Project structure

The project concerns both generative and extractive QA. The work on these is separated into two directories: [extractive-qa](./../extractive-qa) & [generative-qa](./../generative-qa). 
Both of these directories share similar structure, with each one consisting of:
* data directory holding data referenced in the notebooks
* notebooks directory holding various notebooks for training and examining various models

Additionally, there is a local package called ***question_answering*** with utility functions, constants and paths used all across the project. 