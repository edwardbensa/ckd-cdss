"""Hyperparameter optimisation utility functions for Optuna."""

# Imports
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from src.tabular.utils.train_optuna.model_registry import MODEL_REGISTRY


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
