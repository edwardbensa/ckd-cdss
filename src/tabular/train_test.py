"""Modular training and testing script for tabular data."""

# Imports
#from src.tabular.utils.train_optuna.train import train_pipeline
from src.tabular.utils.train_botorch.train import train_pipeline
from src.tabular.utils.test.test import evaluate_model

train_specs = {
    "model_key": "nb",
    "dataset_name": "hiddenckd",
    "train_file": "hiddenckd_train",
    "target": "ckd_status",
    "max_features": 20,
    "n_initial_features": 10,
    "n_trials_per_step": 25,
    "patience": 2,
    "n_trials_final": 75,
}

# Train model
model_dir = train_pipeline(train_specs)

test_specs = {
    "model_dir": model_dir,
    "test_file": train_specs["train_file"][:-5] + "test",
    "target": train_specs["target"],
    "compute_shap": False
}

evaluate_model(test_specs)
