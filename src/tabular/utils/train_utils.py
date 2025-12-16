"""Training utility functions for tabular data models."""

# Imports
import json
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from loguru import logger
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import optuna


def create_objective(X, y, n_splits=5, random_state=42):
    """
    Returns an Optuna objective function for XGBClassifier
    over a fixed feature set (columns of X).
    """

    def objective(trial):
        # Hyperparameter search space
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "gamma": trial.suggest_float("gamma", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 10.0),
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "random_state": random_state,
        }

        # Stratified CV on the (possibly SMOTEENN‑ed) data
        cv = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )

        ap_scores = []

        for train_idx, valid_idx in cv.split(X, y):
            X_train, X_valid = X[train_idx], X[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]

            model = XGBClassifier(**params)
            model.fit(X_train, y_train)

            y_proba = model.predict_proba(X_valid)[:, 1]
            ap = average_precision_score(y_valid, y_proba)
            ap_scores.append(ap)

        # We maximise mean average precision (AUPRC surrogate)
        return float(np.mean(ap_scores))

    return objective


def boed_feature_selection(
    X,
    y,
    feature_names,
    max_features=20,
    n_initial_features=5,
    n_trials_per_step=20,
    patience=2,
    random_state=42,
    provenance_dir=None,
):
    """
    BOED-style sequential feature selection with detailed logging
    and optional provenance JSON logging.
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
    objective_init = create_objective(X_init, y)
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
        logger.info(f"=== BOED Step {step} ===")
        candidate_results = []

        for idx in remaining_indices:
            trial_indices = selected_indices + [idx]
            X_candidate = X.iloc[:, trial_indices].values

            study = optuna.create_study(direction="maximize")
            objective = create_objective(X_candidate, y)
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
    if provenance_dir:
        provenance_dir = Path(provenance_dir)
        provenance_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        provenance_path = provenance_dir / f"boed_provenance_{timestamp}.json"

        provenance = {
            "timestamp": timestamp,
            "selected_features": selected_features,
            "history": history,
            "max_features": max_features,
            "n_initial_features": n_initial_features,
            "n_trials_per_step": n_trials_per_step,
            "patience": patience,
        }

        with open(provenance_path, "w", encoding="utf-8") as f:
            json.dump(provenance, f, indent=4)

        logger.info(f"Saved BOED provenance to {provenance_path}")

    return selected_features, history
