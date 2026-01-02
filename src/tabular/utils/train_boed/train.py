"""Train pipeline script"""

import json
from datetime import datetime
from loguru import logger
import pandas as pd
import joblib
from src.config import INTERIM_DATA_DIR, MODELS_DIR
from .feature_selection import boed_feature_selection
from .hyperopt import botorch_tuning
from .model_registry import MODEL_REGISTRY


def train_pipeline(specs: dict):
    """Modular model training pipeline with BoTorch feature selection + BoTorch hyperopt."""

    # Validate model key
    model_key = specs["model_key"]
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"{model_key} not in model registry. Recheck.")

    dataset_name = specs["dataset_name"]

    # Load dataset
    df = pd.read_csv(INTERIM_DATA_DIR / f"tabular_data/{specs['train_file']}.csv")
    X = df.drop(specs["target"], axis=1)
    y = df[specs["target"]]
    feature_names = X.columns.to_list()

    logger.info(f"Loaded dataset with shape: {df.shape}")

    # Model output directory
    subfolder_dir = MODELS_DIR / f"{dataset_name}"

    # Feature Selection (BoTorch)
    logger.info("Starting BOED feature selection with BoTorch qNEI...")

    selected_features, model_dir = boed_feature_selection(
        X=X,
        y=y,
        feature_names=feature_names,
        model_key=model_key,
        subfolder_dir=subfolder_dir,
        max_features=specs["max_features"],
        n_initial_features=specs["n_initial_features"],
        n_trials_per_step=specs["n_trials_per_step"],
        batch_size=specs.get("batch_size", 4),
        patience=specs["patience"],
        random_state=specs.get("random_state", 42),
    )

    logger.info(f"Final selected features: {selected_features}")

    # Hyperparameter Tuning (BoTorch)
    logger.info(f"Starting BoTorch hyperparameter optimisation for {model_key}...")

    X_sel = X[selected_features].values

    best_params, best_score = botorch_tuning(
        X=X_sel,
        y=y,
        model_key=model_key,
        n_trials=specs["n_trials_final"],
        batch_size=specs.get("batch_size", 4),
        random_state=specs.get("random_state", 42),
        final_tuning=True,
    ) # type: ignore

    logger.info(f"Best BoTorch score: {best_score:.5f}")
    logger.info(f"Best parameters: {best_params}")

    # Train final model
    logger.info(f"Training final {model_key} model with best parameters...")

    entry = MODEL_REGISTRY[model_key]
    params = dict(best_params)

    # Inject input_dim if required
    if "requires" in entry and "input_dim" in entry["requires"]:
        params["input_dim"] = X_sel.shape[1]

    model = entry["model_fn"](params)
    model.fit(X_sel, y)

    # Save model + provenance
    model_path = model_dir / "model.joblib"
    joblib.dump(model, model_path)

    logger.info(f"Saved trained {model_key} model to {model_path}")

    provenance = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "selected_features": selected_features,
        "botorch_best_params": best_params,
        "botorch_best_score": best_score,
        "n_training_samples": len(df),
    }

    prov_path = model_dir / "provenance_training.json"
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=4)

    logger.info(f"Saved training provenance to {prov_path}")
    logger.success("Training complete.")

    return model_dir
