"""Hyperparameter tuning utils for BoTorch."""

# Imports
import torch
import numpy as np
from loguru import logger
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from src.tabular.utils.train_boed.model_registry import MODEL_REGISTRY


# Device selection
def select_device():
    """PyTorch GPU/MPS acceleration if available."""
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "cpu"
    else:
        device = "cpu"
    logger.info(f"Device selected: {device}")
    return device


def evaluate_model(X, y, model_key, **params):
    """
    Train + CV evaluate a model with given hyperparameters.
    
    FIXES:
    1. Manual CV loop to ensure proper model reinitialization
    2. Check for training convergence
    3. Add debugging for zero-variance issues
    """
    entry = MODEL_REGISTRY[model_key]

    # Inject input_dim for neural models
    if "requires" in entry and "input_dim" in entry["requires"]:
        params["input_dim"] = X.shape[1]

    # Manual stratified K-fold for better control
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Create fresh model instance for each fold
        model = entry["model_fn"](params)

        try:
            # Fit model
            model.fit(X_train, y_train)

            # Get predictions
            if hasattr(model, 'predict_proba'):
                y_pred = model.predict_proba(X_val)[:, 1]
            else:
                y_pred = model.predict(X_val)

            # Calculate AUC
            if len(np.unique(y_val)) >= 2:
                fold_score = roc_auc_score(y_val, y_pred)
                scores.append(fold_score)

                # Debug: Check if model is learning
                if fold_score < 0.52:  # Nearly random
                    logger.warning(
                        f"Fold {fold_idx}: Low AUC={fold_score:.4f}. "
                        f"Model may not be training properly!"
                    )
            else:
                logger.warning(f"Fold {fold_idx}: Only one class present in validation set")
                scores.append(0.5)

        except (ValueError, RuntimeError, AttributeError) as e:
            logger.error(f"Fold {fold_idx} failed: {e}")
            scores.append(0.5)

    mean_score = np.mean(scores)

    # Debug: Alert if all scores are nearly identical (sign of broken training)
    # Only warn if variance is zero AND mean score is poor (< 0.6)
    if len(scores) > 1:
        score_std = np.std(scores)
        if score_std < 0.001 and mean_score < 0.6:
            logger.warning(
                f"CV scores have zero variance: {scores}. "
                f"Model training may be broken!"
            )
        elif score_std < 0.002 and mean_score > 0.95:
            logger.debug(
                f"CV scores show excellent consistency: "
                f"mean={mean_score:.4f}, std={score_std:.6f}"
            )

    return mean_score


def get_botorch_space(model_key):
    """Retrieve the BoTorch hyperparameter space."""
    entry = MODEL_REGISTRY[model_key]
    if "botorch_space" not in entry:
        raise ValueError(f"Model '{model_key}' has no 'botorch_space' defined in MODEL_REGISTRY.")
    return entry["botorch_space"]


def build_bounds_and_specs(model_key, device):
    """Build BoTorch bounds tensor and param specs list from MODEL_REGISTRY['botorch_space']."""
    space = get_botorch_space(model_key)

    bounds_list = []
    specs = []

    for spec in space:
        spec = dict(spec)
        ptype = spec.get("type", "float")

        if ptype in ("float", "int"):
            low = float(spec["low"])
            high = float(spec["high"])
            bounds_list.append([low, high])
            specs.append(spec)

        elif ptype == "categorical":
            n = len(spec["choices"])
            bounds_list.append([0.0, float(n - 1)])
            specs.append(spec)

        else:
            raise ValueError(f"Unknown parameter type '{ptype}' in botorch_space for model '{model_key}'.")

    bounds = torch.tensor(bounds_list, dtype=torch.double, device=device).T
    return bounds, specs


def sample_random_point(specs, rng):
    """Sample one random point in param space (in continuous index space)."""
    x = []
    for spec in specs:
        ptype = spec.get("type", "float")
        if ptype == "float":
            low, high = float(spec["low"]), float(spec["high"])
            if spec.get("log", False):
                log_low, log_high = np.log10(low), np.log10(high)
                val = 10 ** rng.uniform(log_low, log_high)
            else:
                val = rng.uniform(low, high)
            x.append(val)

        elif ptype == "int":
            low, high = int(spec["low"]), int(spec["high"])
            val = rng.integers(low, high + 1)
            x.append(float(val))

        elif ptype == "categorical":
            n = len(spec["choices"])
            idx = rng.integers(0, n)
            x.append(float(idx))

        else:
            raise ValueError(f"Unknown parameter type '{ptype}' in specs.")

    return x


def tensor_to_params(point, specs):
    """Convert a tensor point in R^d into a hyperparameter dict."""
    point = point.detach().cpu().numpy()
    params = {}
    hidden_dims = []

    for i, spec in enumerate(specs):
        name = spec["name"]
        ptype = spec.get("type", "float")

        if name.startswith("hidden_dim_"):
            low, high = int(spec["low"]), int(spec["high"])
            val = int(round(point[i]))
            val = max(low, min(high, val))
            hidden_dims.append(val)
            continue

        if ptype == "float":
            params[name] = float(point[i])

        elif ptype == "int":
            low, high = int(spec["low"]), int(spec["high"])
            val = int(round(point[i]))
            params[name] = max(low, min(high, val))

        elif ptype == "categorical":
            choices = spec["choices"]
            idx = int(round(point[i]))
            params[name] = choices[max(0, min(len(choices) - 1, idx))]

    if hidden_dims:
        params["hidden_dims"] = hidden_dims

    return params
