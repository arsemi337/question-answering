# Generative QA
For generative QA, a CodeQA dataset is used. It contains java and python related code with questions and answers regarding it. 

### Getting the datasets
The CodeQA dataset is available under this [link](https://github.com/jadecxliu/CodeQA). In this repository's README, there should be a link to Google Drive with the dataset.

Next do the following:
1. Extract the dataset to the data directory in [generative-qa directory](./../generative-qa), while keeping its internal directory structure. It should go like this: ***generative-qa/data/codeqa/...***
2. Run all the cells in [this notebook](./../generative-qa/notebooks/extract_data.ipynb). 

# Extractive QA
In case of extractive QA, two datasets are used: 
* a very popular and large SQuAD dataset with questions and answers regarding broad spectrum of topics
* a medical QA dataset with questions and answers regarding COVID-19

### Getting the datasets
In order to generate json files for extractive QA go through:
- [this notebook](./../extractive-qa/notebooks/squad/other/extract_data.ipynb) for SQuAD dataset,
- [this notebook](./../extractive-qa/notebooks/medical/other/extract_data.ipynb) for medical dataset.

Nothing else is required as datasets are read from the [Hugging face datasets](https://huggingface.co/) site.
