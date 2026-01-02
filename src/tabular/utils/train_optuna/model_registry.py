"""Model registry for tabular data models."""

# Imports
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import OneClassSVM
from pytorch_tabnet.tab_model import TabNetClassifier
from src.tabular.utils.train_optuna.nn import (
    BayesianNN, DeepBNN, SimpleCNN, Autoencoder, TorchWrapper
    )


# Registry of models and their parameter spaces
MODEL_REGISTRY = {
    "xgb": {
        "model_fn": lambda params: __import__("xgboost").XGBClassifier(**params),
        "param_space": lambda trial: {
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
            "random_state": 42,
        },
    },
    "iforest": {
        "model_fn": lambda params: IsolationForest(**params),
        "param_space": lambda trial: {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_samples": trial.suggest_float("max_samples", 0.5, 1.0),
            "contamination": trial.suggest_float("contamination", 0.01, 0.05),
            "random_state": 42,
        },
    },
    "brf": {
        "model_fn": lambda params: BalancedRandomForestClassifier(**params),
        "param_space": lambda trial: {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "max_features": trial.suggest_float("max_features", 0.5, 1.0),
            "random_state": 42,
        },
    },
    "easy": {
        "model_fn": lambda params: EasyEnsembleClassifier(**params),
        "param_space": lambda trial: {
            "n_estimators": trial.suggest_int("n_estimators", 10, 50),
            "random_state": 42,
        },
    },
    "ocsvm": {
        "model_fn": lambda params: OneClassSVM(**params),
        "param_space": lambda trial: {
            "kernel": trial.suggest_categorical("kernel", ["rbf", "sigmoid"]),
            "nu": trial.suggest_float("nu", 0.01, 0.1),
            "gamma": trial.suggest_float("gamma", 1e-4, 1e-1, log=True),
        },
    },
    "logreg": {
        "model_fn": lambda params: LogisticRegression(**params),
        "param_space": lambda trial: {
            "C": trial.suggest_float("C", 0.01, 10.0, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l2"]),
            "solver": "lbfgs",
            "class_weight": "balanced",
            "max_iter": 1000,
            "random_state": 42,
        },
    },
    "lgbm": {
        "model_fn": lambda params: __import__("lightgbm").LGBMClassifier(**params),
        "param_space": lambda trial: {
            "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "max_depth": trial.suggest_int("max_depth", -1, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "random_state": 42,
        },
    },
    "catboost": {
        "model_fn": lambda params: __import__("catboost").CatBoostClassifier(**params),
        "param_space": lambda trial: {
            "iterations": trial.suggest_int("iterations", 200, 2000),
            "depth": trial.suggest_int("depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.1, 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "verbose": False,
        },
    },
    "rf": {
        "model_fn": lambda params: RandomForestClassifier(**params),
        "param_space": lambda trial: {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "random_state": 42,
        },
    },
    "gpc": {
        "model_fn": lambda params: GaussianProcessClassifier(**params),
        "param_space": lambda trial: {
            "max_iter_predict": trial.suggest_int("max_iter_predict", 50, 200),
            "n_restarts_optimizer": trial.suggest_int("n_restarts_optimizer", 0, 5),
        },
    },
    "tabnet": {
        "model_fn": lambda params: TabNetClassifier(**params),
        "param_space": lambda trial: {
            "n_d": trial.suggest_int("n_d", 8, 64),
            "n_a": trial.suggest_int("n_a", 8, 64),
            "n_steps": trial.suggest_int("n_steps", 3, 10),
            "gamma": trial.suggest_float("gamma", 1.0, 2.0),
            "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-5, 1e-2, log=True),
            "optimizer_params": {"lr": trial.suggest_float("lr", 0.0001, 0.01, log=True)},
        },
    },
    "nb": {
        "model_fn": lambda params: GaussianNB(**params),
        "param_space": lambda trial: {},
    },
    # Bayesian NN
    "bnn": {
        "model_fn": lambda params: TorchWrapper(BayesianNN, params),
        "param_space": lambda trial: {
                "hidden_dims": [
                    trial.suggest_int("hidden_dim_1", 32, 256),
                    trial.suggest_int("hidden_dim_2", 16, 128)
                ],
                "dropout": trial.suggest_float("dropout", 0.2, 0.6),
                "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                "epochs": trial.suggest_int("epochs", 20, 100),
                "batch_size": trial.suggest_int("batch_size", 32, 128),
                "n_mc_samples": 100,
                },
        "requires": ["input_dim"],
    },
    # Ensemble BNN
    "deep_bnn": {
        "model_fn": lambda params: TorchWrapper(DeepBNN, params),
        "param_space": lambda trial: {
            "n_models": trial.suggest_int("n_models", 3, 10),
            "hidden_dims": [
                trial.suggest_int("hidden_dim_1", 32, 256),
                trial.suggest_int("hidden_dim_2", 16, 128)
            ],
            "dropout": trial.suggest_float("dropout", 0.2, 0.6),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "epochs": trial.suggest_int("epochs", 20, 100),
            "batch_size": trial.suggest_int("batch_size", 32, 128),
            "n_mc_samples": 50,
        },
        "requires": ["input_dim"],
    },
    # CNN (Tabular 1D)
    "cnn": {
        "model_fn": lambda params: TorchWrapper(SimpleCNN, params),
        "param_space": lambda trial: {
            "num_filters": trial.suggest_int("num_filters", 16, 128),
            "kernel_size": trial.suggest_int("kernel_size", 2, 5),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "epochs": trial.suggest_int("epochs", 10, 50),
            "batch_size": trial.suggest_int("batch_size", 32, 128),
        },
    },
    # Autoencoder
    "autoencoder": {
        "model_fn": lambda params: TorchWrapper(Autoencoder, params),
        "param_space": lambda trial: {
            "hidden_dim": trial.suggest_int("hidden_dim", 32, 256),
            "bottleneck_dim": trial.suggest_int("bottleneck_dim", 8, 64),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "epochs": trial.suggest_int("epochs", 10, 50),
            "batch_size": trial.suggest_int("batch_size", 32, 128),
        },
        "requires": ["input_dim"],
    }
}