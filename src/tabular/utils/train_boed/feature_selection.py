"""BOED feature selection function"""

# Imports
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import numpy as np
from loguru import logger
from .hyperopt import botorch_tuning



def boed_feature_selection(
    X,
    y,
    feature_names,
    model_key,
    subfolder_dir,
    max_features=20,
    n_initial_features=5,
    n_trials_per_step=20,
    batch_size=4,
    patience=2,
    random_state=42,
):
    """
    BOED sequential feature selection using BoTorch with qEI batch Bayesian optimisation.
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
    best_score = botorch_tuning(
        X_init,
        y,
        model_key,
        n_trials=n_trials_per_step,
        batch_size=batch_size,
        random_state=random_state,
    )
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

            score = botorch_tuning(
                X_candidate,
                y,
                model_key,
                n_trials=n_trials_per_step,
                batch_size=batch_size,
                random_state=random_state,
            )

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
    provenance_path = model_dir / "provenance_boed_botorch_qei.json"

    provenance = {
        "timestamp": timestamp,
        "selected_features": selected_features,
        "history": history,
        "max_features": max_features,
        "n_initial_features": n_initial_features,
        "n_trials_per_step": n_trials_per_step,
        "batch_size": batch_size,
        "patience": patience,
        "model_key": model_key,
    }

    with open(provenance_path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=4)

    logger.info(f"Saved BOED provenance to {provenance_path}")

    return selected_features, model_dir
