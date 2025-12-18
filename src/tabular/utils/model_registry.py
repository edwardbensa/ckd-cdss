"""Model registry for tabular data models."""

# Imports
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
from sklearn.linear_model import LogisticRegression
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


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
}


# === Neural network models ===

class BayesianNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return torch.sigmoid(self.fc2(x)).squeeze()


class SimpleCNN(nn.Module):
    # Add input_dim to the signature (even if just to catch the argument)
    def __init__(self, input_channels=1, num_filters=32, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, num_filters, kernel_size)
        # Note: In a real 1D CNN for tabular data, the output width of conv1 
        # depends on the input_dim. 
        self.fc = nn.Linear(num_filters, 1)

    def forward(self, x):
        # If x is (batch, features), CNN expects (batch, channels, features)
        if len(x.shape) == 2:
            x = x.unsqueeze(1) 

        x = F.relu(self.conv1(x))
        x = torch.mean(x, dim=2)  # global average pooling handles variable width
        return torch.sigmoid(self.fc(x)).squeeze()


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, bottleneck_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


# Torch Wrapper to provide sklearn-like interface
class TorchWrapper:
    def __init__(self, net_class, params):
        # Only pass relevant params to the network constructor
        net_params = {k: v for k, v in params.items()
                      if k in ["input_dim", "hidden_dim", "dropout",
                               "num_filters", "kernel_size", "bottleneck_dim"]}
        self.net = net_class(**net_params)
        self.lr = params.get("lr", 1e-3)
        self.epochs = params.get("epochs", 10)
        self.batch_size = params.get("batch_size", 32)

    def fit(self, X, y):
        # Convert pandas DataFrame/Series to numpy arrays if needed
        if hasattr(X, "values"):
            X = X.values
        if hasattr(y, "values"):
            y = y.values

        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32),
                                torch.tensor(y, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

        self.net.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                preds = self.net(xb).view(-1)       # ensure shape (batch_size,)
                loss = criterion(preds, yb.float()) # targets as float
                loss.backward()
                optimizer.step()

    def predict_proba(self, X):
        # Convert pandas DataFrame to numpy if needed
        if hasattr(X, "values"):
            X = X.values

        self.net.eval()
        with torch.no_grad():
            preds = self.net(torch.tensor(X, dtype=torch.float32)).view(-1)
        probs = preds.numpy()
        # Always return shape (n_samples, 2)
        return np.vstack([1 - probs, probs]).T

    def predict(self, X, threshold=0.5):
        """Return binary predictions based on predict_proba."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)

# Registry entries for neural models using TorchWrapper

MODEL_REGISTRY["bnn"] = {
    "model_fn": lambda params: TorchWrapper(BayesianNN, params),
    "param_space": lambda trial: {
        "hidden_dim": trial.suggest_int("hidden_dim", 32, 256),
        "dropout": trial.suggest_float("dropout", 0.2, 0.6),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "epochs": trial.suggest_int("epochs", 10, 50),
        "batch_size": trial.suggest_int("batch_size", 32, 128),
    },
    "requires": ["input_dim"],
}

MODEL_REGISTRY["cnn"] = {
    "model_fn": lambda params: TorchWrapper(SimpleCNN, params),
    "param_space": lambda trial: {
        "num_filters": trial.suggest_int("num_filters", 16, 128),
        "kernel_size": trial.suggest_int("kernel_size", 2, 5),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "epochs": trial.suggest_int("epochs", 10, 50),
        "batch_size": trial.suggest_int("batch_size", 32, 128),
    },
}

MODEL_REGISTRY["autoencoder"] = {
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