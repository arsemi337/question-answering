from pyprojroot import find_root, has_dir

root = find_root(has_dir(".git"))

extractive_qa_dir = root / "extractive-qa"
training_checkpoints_dir = extractive_qa_dir / "training-checkpoints"
saved_models_dir = extractive_qa_dir / "tf-models"
hub_models_location = extractive_qa_dir / "hub-models"
data_dir = extractive_qa_dir / "data"
datasets_dir = data_dir / "datasets"
squad_dataset_dir = datasets_dir / "squad"
medical_dataset_dir = datasets_dir / "medical"
figures_dir = extractive_qa_dir / "figures"
