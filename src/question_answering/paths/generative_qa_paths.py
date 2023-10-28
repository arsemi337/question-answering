from pyprojroot import find_root, has_dir

root = find_root(has_dir(".git"))

generative_qa_dir = root / "generative_qa"
training_checkpoints_dir = generative_qa_dir / "training_checkpoints"
saved_models_dir = generative_qa_dir / "tf_models"
hub_models_location = generative_qa_dir / "hub_models"
