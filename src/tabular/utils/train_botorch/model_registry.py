"""Model registry for tabular data models."""

# Imports
from sklearn.svm import OneClassSVM
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from .nn import BayesianNN, DeepBNN, SimpleCNN, Autoencoder, TorchWrapper


MODEL_REGISTRY = {
    # XGBoost
    "xgb": {
        "model_fn": lambda params: __import__("xgboost").XGBClassifier(**params),
        "botorch_space": [
            {"name": "n_estimators",      "type": "int",   "low": 100, "high": 1000},
            {"name": "max_depth",         "type": "int",   "low": 2,   "high": 8},
            {"name": "learning_rate",     "type": "float", "low": 0.01, "high": 0.3, "log": True},
            {"name": "subsample",         "type": "float", "low": 0.5,  "high": 1.0},
            {"name": "colsample_bytree",  "type": "float", "low": 0.5,  "high": 1.0},
            {"name": "min_child_weight",  "type": "float", "low": 1.0,  "high": 10.0},
            {"name": "gamma",             "type": "float", "low": 0.0,  "high": 10.0},
            {"name": "reg_lambda",        "type": "float", "low": 1e-3, "high": 10.0, "log": True},
            {"name": "reg_alpha",         "type": "float", "low": 1e-3, "high": 10.0, "log": True},
            {"name": "scale_pos_weight",  "type": "float", "low": 0.5,  "high": 10.0},
        ],
    },
    # Isolation Forest
    "iforest": {
        "model_fn": lambda params: IsolationForest(**params),
        "botorch_space": [
            {"name": "n_estimators", "type": "int",   "low": 50,  "high": 300},
            {"name": "max_samples",  "type": "float", "low": 0.5, "high": 1.0},
            {"name": "contamination","type": "float", "low": 0.01,"high": 0.05},
        ],
    },
    # Balanced Random Forest
    "brf": {
        "model_fn": lambda params: BalancedRandomForestClassifier(**params),
        "botorch_space": [
            {"name": "n_estimators", "type": "int",   "low": 100, "high": 500},
            {"name": "max_depth",    "type": "int",   "low": 2,   "high": 10},
            {"name": "max_features", "type": "float", "low": 0.5, "high": 1.0},
        ],
    },
    # Easy Ensemble
    "easy": {
        "model_fn": lambda params: EasyEnsembleClassifier(**params),
        "botorch_space": [
            {"name": "n_estimators", "type": "int", "low": 10, "high": 50},
        ],
    },
    # One-Class SVM
    "ocsvm": {
        "model_fn": lambda params: OneClassSVM(**params),
        "botorch_space": [
            {"name": "kernel", "type": "categorical", "choices": ["rbf", "sigmoid"]},
            {"name": "nu",     "type": "float",       "low": 0.01, "high": 0.1},
            {"name": "gamma",  "type": "float",       "low": 1e-4, "high": 1e-1, "log": True},
        ],
    },
    # Logistic Regression
    "logreg": {
        "model_fn": lambda params: LogisticRegression(**params),
        "botorch_space": [
            {"name": "C",       "type": "float", "low": 0.01, "high": 10.0, "log": True},
            {"name": "penalty", "type": "categorical", "choices": ["l2"]},
        ],
    },
    # K-Nearest Neighbors
    "knn": {
        "model_fn": lambda params: KNeighborsClassifier(**params),
        "botorch_space": [
            {"name": "n_neighbors", "type": "int", "low": 1, "high": 50},
            {"name": "weights",     "type": "categorical", "choices": ["uniform", "distance"]},
            {"name": "p",           "type": "int", "low": 1, "high": 2},
        ],
    },
    # LightGBM
    "lgbm": {
        "model_fn": lambda params: __import__("lightgbm").LGBMClassifier(**params),
        "botorch_space": [
            {"name": "n_estimators",      "type": "int",   "low": 200, "high": 2000},
            {"name": "learning_rate",     "type": "float", "low": 1e-3, "high": 0.3, "log": True},
            {"name": "num_leaves",        "type": "int",   "low": 16,   "high": 256},
            {"name": "max_depth",         "type": "int",   "low": -1,   "high": 12},
            {"name": "min_child_samples", "type": "int",   "low": 5,    "high": 50},
            {"name": "subsample",         "type": "float", "low": 0.5,  "high": 1.0},
            {"name": "colsample_bytree",  "type": "float", "low": 0.5,  "high": 1.0},
            {"name": "reg_lambda",        "type": "float", "low": 1e-3, "high": 10.0, "log": True},
            {"name": "reg_alpha",         "type": "float", "low": 1e-3, "high": 10.0, "log": True},
        ],
    },
    # CatBoost
    "catboost": {
        "model_fn": lambda params: __import__("catboost").CatBoostClassifier(**params),
        "botorch_space": [
            {"name": "iterations",         "type": "int",   "low": 200, "high": 2000},
            {"name": "depth",              "type": "int",   "low": 2,    "high": 10},
            {"name": "learning_rate",      "type": "float", "low": 1e-3, "high": 0.3, "log": True},
            {"name": "l2_leaf_reg",        "type": "float", "low": 1e-3, "high": 10.0, "log": True},
            {"name": "random_strength",    "type": "float", "low": 0.1,  "high": 10.0},
            {"name": "bagging_temperature","type": "float", "low": 0.0,  "high": 1.0},
            {"name": "border_count",       "type": "int",   "low": 32,   "high": 255},
        ],
    },
    # Random Forest
    "rf": {
        "model_fn": lambda params: RandomForestClassifier(**params),
        "botorch_space": [
            {"name": "n_estimators",      "type": "int",   "low": 100, "high": 500},
            {"name": "max_depth",         "type": "int",   "low": 2,   "high": 20},
            {"name": "max_features",      "type": "categorical", "choices": ["sqrt", "log2", None]},
            {"name": "min_samples_split", "type": "int",   "low": 2,   "high": 20},
            {"name": "min_samples_leaf",  "type": "int",   "low": 1,   "high": 10},
        ],
    },
    # Gaussian Process Classifier
    "gpc": {
        "model_fn": lambda params: GaussianProcessClassifier(**params),
        "botorch_space": [
            {"name": "max_iter_predict",     "type": "int", "low": 50, "high": 200},
            {"name": "n_restarts_optimizer", "type": "int", "low": 0,  "high": 5},
        ],
    },
    # TabNet
    "tabnet": {
        "model_fn": lambda params: TabNetClassifier(**params),
        "botorch_space": [
            {"name": "n_d",            "type": "int",   "low": 8,   "high": 64},
            {"name": "n_a",            "type": "int",   "low": 8,   "high": 64},
            {"name": "n_steps",        "type": "int",   "low": 3,   "high": 10},
            {"name": "gamma",          "type": "float", "low": 1.0, "high": 2.0},
            {"name": "lambda_sparse",  "type": "float", "low": 1e-5,"high": 1e-2, "log": True},
            {"name": "lr",             "type": "float", "low": 1e-4,"high": 1e-2, "log": True},
        ],
    },
    # Naive Bayes
    "nb": {
        "model_fn": lambda params: GaussianNB(**params),
        "botorch_space": [],
    },
    # Bayesian NN
    "bnn": {
        "model_fn": lambda params: TorchWrapper(BayesianNN, **params),
        "botorch_space": [
            {"name": "hidden_dim_1", "type": "int",   "low": 32,  "high": 256},
            {"name": "hidden_dim_2", "type": "int",   "low": 16,  "high": 128},
            {"name": "dropout",      "type": "float", "low": 0.2, "high": 0.6},
            {"name": "lr",           "type": "float", "low": 1e-3, "high": 1e-2, "log": True},
            {"name": "epochs",       "type": "int",   "low": 50,  "high": 150},
            {"name": "batch_size",   "type": "int",   "low": 32,  "high": 128},
        ],
        "requires": ["input_dim"],
    },
    # Ensemble BNN
    "deep_bnn": {
        "model_fn": lambda params: TorchWrapper(DeepBNN, **params),
        "botorch_space": [
            {"name": "n_models",     "type": "int",   "low": 3,   "high": 10},
            {"name": "hidden_dim_1", "type": "int",   "low": 32,  "high": 256},
            {"name": "hidden_dim_2", "type": "int",   "low": 16,  "high": 128},
            {"name": "dropout",      "type": "float", "low": 0.2, "high": 0.6},
            {"name": "lr",           "type": "float", "low": 1e-4, "high": 1e-2, "log": True},
            {"name": "epochs",       "type": "int",   "low": 50,  "high": 100},
            {"name": "batch_size",   "type": "int",   "low": 32,  "high": 128},
        ],
        "requires": ["input_dim"],
    },
    # CNN (Tabular 1D)
    "cnn": {
        "model_fn": lambda params: TorchWrapper(SimpleCNN, **params),
        "botorch_space": [
            {"name": "num_filters", "type": "int",   "low": 16,  "high": 128},
            {"name": "kernel_size", "type": "int",   "low": 2,   "high": 5},
            {"name": "lr",          "type": "float", "low": 1e-4, "high": 1e-2, "log": True},
            {"name": "epochs",      "type": "int",   "low": 20,  "high": 100},
            {"name": "batch_size",  "type": "int",   "low": 32,  "high": 128},
        ],
    },
    # Autoencoder
    "autoencoder": {
        "model_fn": lambda params: TorchWrapper(Autoencoder, **params),
        "botorch_space": [
            {"name": "hidden_dim_1",   "type": "int",   "low": 32,  "high": 256}, 
            {"name": "bottleneck_dim", "type": "int",   "low": 8,   "high": 64},
            {"name": "lr",             "type": "float", "low": 1e-4, "high": 1e-2, "log": True},
            {"name": "epochs",         "type": "int",   "low": 20,  "high": 100},
            {"name": "batch_size",     "type": "int",   "low": 32,  "high": 128},
        ],
        "requires": ["input_dim"],
    },
}
