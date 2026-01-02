"""Neural net models constructed with PyTorch"""

# Imports
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# Neural network models
class BayesianNN(nn.Module):
    """Bayesian NN using MC Dropout for uncertainty quantification."""

    def __init__(self, input_dim, hidden_dims=None, dropout=0.5):
        super().__init__()

        # Build variable-depth network
        if hidden_dims is None:
            hidden_dims = [64, 32]

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        self.features = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, 1)
        self.dropout = dropout

    def forward(self, x, return_logits=False):
        x = self.features(x)
        logits = self.output(x).squeeze()
        if return_logits:
            return logits
        return torch.sigmoid(logits)

    def enable_dropout(self):
        """Enable dropout for MC sampling at test time."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def predict_with_uncertainty(self, x, n_samples=100):
        """
        Perform MC Dropout to estimate prediction uncertainty.
        
        Returns:
            mean: Mean prediction across samples
            std: Standard deviation (epistemic uncertainty)
            samples: All prediction samples
        """
        self.eval()
        self.enable_dropout()

        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x)
                samples.append(pred.numpy())

        samples = np.array(samples)
        mean = samples.mean(axis=0)
        std = samples.std(axis=0)

        return mean, std, samples


class DeepBNN:
    """
    Deep Ensemble of Bayesian Neural Networks.
    Combines ensemble diversity with MC Dropout uncertainty.
    """

    def __init__(self, input_dim, n_models=7, hidden_dims=[64, 32], dropout=0.5):
        self.n_models = n_models
        self.models = [
            BayesianNN(input_dim, hidden_dims, dropout)
            for _ in range(n_models)
        ]
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

    def fit(self, X, y, lr=1e-3, epochs=50, batch_size=32, verbose=False):
        """Train all models in the ensemble independently."""
        if hasattr(X, "values"):
            X = X.values
        if hasattr(y, "values"):
            y = y.values

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        for i, model in enumerate(self.models):
            if verbose:
                print(f"Training model {i+1}/{self.n_models}...")

            # Use different random seeds for each model
            torch.manual_seed(42 + i)

            # Create data loader with shuffling for diversity
            dataset = TensorDataset(X_tensor, y_tensor)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            model.train()
            for epoch in range(epochs):
                epoch_loss = 0
                for xb, yb in loader:
                    optimizer.zero_grad()
                    preds = model(xb)
                    # Ensure shapes match - handle both batch and single sample cases
                    if preds.dim() == 0:
                        preds = preds.unsqueeze(0)
                    if yb.dim() == 0:
                        yb = yb.unsqueeze(0)
                    loss = criterion(preds, yb)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                if verbose and (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(loader):.4f}")

    def predict_with_uncertainty(self, X, n_mc_samples=50):
        """
        Predict with full uncertainty quantification.
        
        Combines:
        - Epistemic uncertainty from ensemble disagreement
        - Epistemic uncertainty from MC dropout within each model
        - Total predictive uncertainty
        
        Returns:
            predictions: Mean prediction across all models and MC samples
            total_uncertainty: Total predictive uncertainty (std)
            ensemble_uncertainty: Uncertainty from ensemble disagreement
            mc_uncertainty: Average MC dropout uncertainty within models
        """
        if not isinstance(X, torch.Tensor):
            if hasattr(X, "values"):
                X = X.values
            X_tensor = torch.tensor(X, dtype=torch.float32)
        else:
            X_tensor = X

        # Collect predictions from each ensemble member
        ensemble_means = []
        ensemble_stds = []

        for model in self.models:
            mean, std, _ = model.predict_with_uncertainty(X_tensor, n_mc_samples)
            ensemble_means.append(mean)
            ensemble_stds.append(std)

        ensemble_means = np.array(ensemble_means)  # Shape: (n_models, n_samples)
        ensemble_stds = np.array(ensemble_stds)

        # Overall prediction
        predictions = ensemble_means.mean(axis=0)

        # Ensemble uncertainty (disagreement between models)
        ensemble_uncertainty = ensemble_means.std(axis=0)

        # Average MC dropout uncertainty within models
        mc_uncertainty = ensemble_stds.mean(axis=0)

        # Total predictive uncertainty (law of total variance)
        total_uncertainty = np.sqrt(ensemble_uncertainty**2 + mc_uncertainty**2)

        return predictions, total_uncertainty, ensemble_uncertainty, mc_uncertainty

    def predict_proba(self, X):
        """sklearn-compatible predict_proba using mean predictions."""
        predictions, _, _, _ = self.predict_with_uncertainty(X, n_mc_samples=50)
        return np.vstack([1 - predictions, predictions]).T

    def predict(self, X, threshold=0.5):
        """sklearn-compatible predict."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)


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
    """Torch wrapper that supports uncertainty quantification."""

    def __init__(self, net_class, params):
        net_params = {k: v for k, v in params.items()
                      if k in ["input_dim", "hidden_dims", "dropout", 
                               "n_models"]}

        self.is_ensemble = (net_class == DeepBNN)
        self.net = net_class(**net_params)
        self.lr = params.get("lr", 1e-3)
        self.epochs = params.get("epochs", 50)
        self.batch_size = params.get("batch_size", 32)
        self.n_mc_samples = params.get("n_mc_samples", 100)

    def fit(self, X, y):
        """Fit neural network."""
        if hasattr(X, "values"):
            X = X.values
        if hasattr(y, "values"):
            y = y.values

        if self.is_ensemble:
            self.net.fit(X, y, lr=self.lr, epochs=self.epochs, 
                        batch_size=self.batch_size, verbose=False)
        else:
            # Single BNN training
            dataset = TensorDataset(
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32)
            )
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            criterion = nn.BCELoss()
            optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

            self.net.train()
            for _ in range(self.epochs):
                for xb, yb in loader:
                    optimizer.zero_grad()
                    preds = self.net(xb)
                    loss = criterion(preds, yb)
                    loss.backward()
                    optimizer.step()

    def predict_with_uncertainty(self, X):
        """Return predictions with uncertainty estimates."""
        if hasattr(X, "values"):
            X = X.values

        if not isinstance(X, torch.Tensor):
            X_tensor = torch.tensor(X, dtype=torch.float32)
        else:
            X_tensor = X

        if self.is_ensemble:
            return self.net.predict_with_uncertainty(X_tensor, self.n_mc_samples)
        else:
            mean, std, samples = self.net.predict_with_uncertainty(
                X_tensor, n_samples=self.n_mc_samples
            )
            return mean, std, None, std

    def predict_proba(self, X):
        if hasattr(X, "values"):
            X = X.values

        if self.is_ensemble:
            return self.net.predict_proba(X)
        else:
            X_tensor = torch.tensor(X, dtype=torch.float32)
            mean, _, _ = self.net.predict_with_uncertainty(X_tensor, self.n_mc_samples)
            return np.vstack([1 - mean, mean]).T

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)
