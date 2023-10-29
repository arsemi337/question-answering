from pyprojroot import find_root, has_dir

root = find_root(has_dir(".git"))

generative_qa_dir = root / "generative-qa"
training_checkpoints_dir = generative_qa_dir / "training-checkpoints"
saved_models_dir = generative_qa_dir / "tf-models"
hub_models_location = generative_qa_dir / "hub-models"
