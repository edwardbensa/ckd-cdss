"""Training utility functions for tabular data models."""

# Imports
import json
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
import optuna
import joblib
from src.tabular.utils.model_registry import MODEL_REGISTRY
from src.config import INTERIM_DATA_DIR, MODELS_DIR


def create_objective(X, y, model_key, n_splits=5, random_state=42):
    """
    Returns an Optuna objective function for a model in MODEL_REGISTRY.
    Automatically injects input_dim for neural models.
    """

    entry = MODEL_REGISTRY[model_key]
    param_space = entry["param_space"]

    def objective(trial):
        params = param_space(trial)

        # Inject input_dim if model requires it
        if "input_dim" in entry.get("requires", []):
            params["input_dim"] = X.shape[1]

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scores = []

        for train_idx, valid_idx in cv.split(X, y):
            X_train, X_valid = X[train_idx], X[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]

            # Build model
            if "requires" in entry and "input_dim" in entry["requires"]:
                model = entry["model_fn"](params)
            else:
                model = entry["model_fn"](params)

            # Fit model
            model.fit(X_train, y_train)

            # Handle classifiers vs anomaly detectors vs neural nets
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_valid)[:, 1]
            elif hasattr(model, "decision_function"):
                y_proba = model.decision_function(X_valid)
            else:
                y_proba = model.predict(X_valid)

            score = average_precision_score(y_valid, y_proba)
            scores.append(score)

        return float(np.mean(scores))

    return objective


def boed_feature_selection(
    X,
    y,
    feature_names,
    model_key,
    subfolder_dir,
    max_features=20,
    n_initial_features=5,
    n_trials_per_step=20,
    patience=2,
    random_state=42,
):
    """
    BOED-style sequential feature selection with Optuna optimization.
    Works with any model in MODEL_REGISTRY via model_key.
    """

    rng = np.random.default_rng(random_state)
    n_total_features = X.shape[1]

    if max_features is None:
        max_features = n_total_features

    all_indices = np.arange(n_total_features)

    # Seed with random subset
    initial_indices = rng.choice(all_indices, size=n_initial_features, replace=False)
    selected_indices = list(initial_indices)
    remaining_indices = [i for i in all_indices if i not in selected_indices]

    logger.info(f"Initial feature subset: {[feature_names[i] for i in selected_indices]}")

    # Evaluate initial subset
    X_init = X.iloc[:, selected_indices].values
    study_init = optuna.create_study(direction="maximize")
    objective_init = create_objective(X_init, y, model_key=model_key)
    study_init.optimize(objective_init, n_trials=n_trials_per_step)

    best_score = study_init.best_value
    best_subset = deepcopy(selected_indices)

    history = [{
        "step": 0,
        "selected_features": [feature_names[i] for i in selected_indices],
        "score": best_score,
        "added_feature": None,
    }]

    no_improve_steps = 0
    step = 1

    while remaining_indices and len(selected_indices) < max_features:
        logger.info(f"BOED Step {step}")
        candidate_results = []

        for idx in remaining_indices:
            trial_indices = selected_indices + [idx]
            X_candidate = X.iloc[:, trial_indices].values

            study = optuna.create_study(direction="maximize")
            objective = create_objective(X_candidate, y, model_key=model_key)
            study.optimize(objective, n_trials=n_trials_per_step)

            score = study.best_value
            candidate_results.append((idx, score))

            logger.info(f"Candidate feature '{feature_names[idx]}' → Score: {score:.5f}")

        # Select best candidate
        idx_best, score_best = max(candidate_results, key=lambda t: t[1])
        selected_indices.append(idx_best)
        remaining_indices.remove(idx_best)

        logger.info(f"Selected feature: {feature_names[idx_best]} with score {score_best:.5f}")

        history.append({
            "step": step,
            "selected_features": [feature_names[i] for i in selected_indices],
            "score": score_best,
            "added_feature": feature_names[idx_best],
        })

        if score_best > best_score:
            best_score = score_best
            best_subset = deepcopy(selected_indices)
            no_improve_steps = 0
        else:
            no_improve_steps += 1

        if no_improve_steps >= patience:
            logger.info("Stopping early due to no improvement.")
            break

        step += 1

    selected_features = [feature_names[i] for i in best_subset]

    # Save provenance JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = Path(subfolder_dir / f"{model_key}_{timestamp}")
    model_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = model_dir / "provenance_boed.json"

    provenance = {
        "timestamp": timestamp,
        "selected_features": selected_features,
        "history": history,
        "max_features": max_features,
        "n_initial_features": n_initial_features,
        "n_trials_per_step": n_trials_per_step,
        "patience": patience,
        "model_key": model_key,
        }

    with open(provenance_path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=4)

    logger.info(f"Saved BOED provenance to {provenance_path}")

    return selected_features, model_dir

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
