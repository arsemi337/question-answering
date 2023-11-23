from pyprojroot import find_root, has_dir

root = find_root(has_dir(".git"))

generative_qa_dir = root / "generative-qa"
training_checkpoints_dir = generative_qa_dir / "training-checkpoints"
saved_models_dir = generative_qa_dir / "tf-models"
hub_models_location = generative_qa_dir / "hub-models"
data_dir = generative_qa_dir / "data"
code_qa_data_dir = data_dir / "codeqa"
datasets_dir = data_dir / "datasets"
code_qa_dataset_dir = datasets_dir / "codeqa"
java_dataset_dir = code_qa_dataset_dir / "java"
python_dataset_dir = code_qa_dataset_dir / "python"
model_evaluation_dir = generative_qa_dir / "model-evaluation"
