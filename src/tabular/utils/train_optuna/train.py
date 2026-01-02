"""Training utility functions for tabular data models."""

# Imports
from datetime import datetime
import json
import optuna
import joblib
import pandas as pd
from loguru import logger
from src.tabular.utils.train_optuna.feature_selection import boed_feature_selection
from src.tabular.utils.train_optuna.hyperopt import create_objective
from src.tabular.utils.train_optuna.model_registry import MODEL_REGISTRY
from src.config import INTERIM_DATA_DIR, MODELS_DIR


def train_pipeline(specs: dict):
    """Modular model training pipeline."""
    # Extract variables
    model_key = specs["model_key"]
    if model_key not in MODEL_REGISTRY.keys():
        raise ValueError(f"{model_key} not in model registry. Recheck.")
    dataset_name = specs["dataset_name"]

    # Load dataset
    df = pd.read_csv(INTERIM_DATA_DIR / f"tabular_data/{specs["train_file"]}.csv")
    X = df.drop(specs["target"], axis=1)
    y = df[specs["target"]]
    feature_names = X.columns.to_list()
    logger.info(f"Loaded dataset with shape: {df.shape}")

    # Model path
    subfolder_dir = MODELS_DIR / f"{dataset_name}"

    # Perform BOED feature selection
    selected_features, model_dir = boed_feature_selection(
        X,
        y,
        feature_names=feature_names,
        max_features=specs["max_features"],
        model_key=model_key,
        n_initial_features=specs["n_initial_features"],
        n_trials_per_step=specs["n_trials_per_step"],
        patience=specs["patience"],
        subfolder_dir=subfolder_dir
    )

    logger.info(f"Final selected features: {selected_features}")

    # Hyperparameter tuning with Optuna
    logger.info(f"Starting Bayesian optimisation for {model_key} model...")

    X_sel = X[selected_features].values
    study = optuna.create_study(direction="maximize")
    objective = create_objective(X_sel, y, model_key=model_key)
    study.optimize(objective, n_trials=specs["n_trials_final"])
    best_params = study.best_params
    best_score = study.best_value
    logger.info(f"Best Optuna score: {best_score:.5f}")
    logger.info(f"Best parameters: {best_params}")

    # Train final model
    logger.info(f"Training final {model_key} model with best parameters...")
    entry = MODEL_REGISTRY[model_key]
    params = dict(best_params)

    if "requires" in entry and "input_dim" in entry["requires"]:
        params["input_dim"] = X_sel.shape[1]

    model = entry["model_fn"](params)
    model.fit(X_sel, y)

    # Save model and training provenance
    model_path = model_dir / "model.joblib"
    joblib.dump(model, model_path)
    logger.info(f"Saved trained {model_key} model to {model_path}")

    provenance = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "selected_features": selected_features,
        "optuna_best_params": best_params,
        "optuna_best_score": best_score,
        "n_training_samples": len(df),
    }

    prov_path = model_dir / "provenance_training.json"
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=4)
    logger.info(f"Saved training provenance to {prov_path}")
    logger.success("Training complete.")

    return model_dir
