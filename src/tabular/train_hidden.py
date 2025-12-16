"""Train Hidden CKD model. Select features and tune hyperparameters using BO"""

# Imports
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
from loguru import logger
from xgboost import XGBClassifier
import optuna
import joblib
from src.tabular.utils.train_utils import create_objective, boed_feature_selection
from src.config import MODELS_DIR, INTERIM_DATA_DIR

# Load dataset
df = pd.read_csv(INTERIM_DATA_DIR / "tabular_data/hiddenckd_train.csv")
X = df.drop("ckd_status", axis=1)
y = df["ckd_status"]
feature_names = X.columns.to_list()
logger.info(f"Loaded dataset with shape: {df.shape}")


# Perform BOED feature selection
selected_features, history = boed_feature_selection(
    X,
    y,
    feature_names=feature_names,
    max_features=20,
    n_initial_features=5,
    n_trials_per_step=75,
    patience=2,
    provenance_dir=MODELS_DIR / "hidden_ckd/provenance"
)

logger.info(f"Final selected features: {selected_features}")

# Hyperparameter Tuning with Optuna
logger.info("Starting Bayesian optimisation for XGBoost...")

X_sel = X[selected_features].values
study = optuna.create_study(direction="maximize")
objective = create_objective(X_sel, y)
study.optimize(objective, n_trials=100)
best_params = study.best_params
best_score = study.best_value
logger.info(f"Best Optuna score: {best_score:.5f}")
logger.info(f"Best parameters: {best_params}")

# Train final model
logger.info("Training final model with best parameters...")
model = XGBClassifier(
    **best_params,
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    random_state=42,
)
model.fit(X_sel, y)

# Save model
model_dir = Path(MODELS_DIR / "hidden_ckd")
model_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = model_dir / f"xgb_hidden_ckd_{timestamp}.joblib"
joblib.dump(model, model_path)

logger.info(f"Saved trained model to {model_path}")

# Save training provenance
provenance = {
    "timestamp": timestamp,
    "selected_features": selected_features,
    "optuna_best_params": best_params,
    "optuna_best_score": best_score,
    "n_training_samples": len(df),
}

prov_path = model_dir / f"training_provenance_{timestamp}.json"
with open(prov_path, "w", encoding="utf-8") as f:
    json.dump(provenance, f, indent=4)
logger.info(f"Saved training provenance to {prov_path}")
logger.success("Training complete.")
