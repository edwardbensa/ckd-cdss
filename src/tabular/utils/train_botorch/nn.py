"""Neural net models constructed with PyTorch"""

# Imports
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.base import BaseEstimator, ClassifierMixin
from loguru import logger
import numpy as np

# Neural network models
class BayesianNN(nn.Module):
    """Bayesian NN using MC Dropout for uncertainty quantification."""
    def __init__(self, input_dim, hidden_dims=None, dropout=0.5):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        self.features = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, 1)

    def forward(self, x):
        """Forward pass with shape handling for batch and single samples."""
        # Store original shape info
        was_1d = x.dim() == 1

        # Ensure 2D for BatchNorm
        if was_1d:
            x = x.unsqueeze(0)

        x = self.features(x)
        logits = self.output(x)
        probs = torch.sigmoid(logits)

        # Return consistent shape
        if was_1d:
            return probs.squeeze()
        else:
            return probs.squeeze(-1)

    def enable_dropout(self):
        """Enable dropout for MC sampling at test time."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def predict_with_uncertainty(self, x, n_samples=100):
        """Perform MC Dropout to estimate prediction uncertainty."""
        self.eval()
        self.enable_dropout()
        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x)
                samples.append(pred.cpu().numpy())
        samples = np.array(samples)
        return samples.mean(axis=0), samples.std(axis=0), samples


class DeepBNN:
    """
    Deep Ensemble of Bayesian Neural Networks.
    Combines ensemble diversity with MC Dropout uncertainty.
    """

    def __init__(self, input_dim, n_models=7, hidden_dims=None, dropout=0.5):
        if hidden_dims is None:
            hidden_dims = [64, 32]

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

        X_tensor = torch.tensor(X, dtype=torch.double)
        y_tensor = torch.tensor(y, dtype=torch.double)

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

    def __call__(self, x):
        """Make the ensemble callable to handle forward passes."""
        # Average the outputs of all models in the ensemble
        preds_list = [model(x) for model in self.models]
        return torch.stack(preds_list).mean(dim=0)

    def to(self, *args, **kwargs):
        """Pass .to() calls (device, dtype) to all underlying models."""
        for model in self.models:
            model.to(*args, **kwargs)
        return self

    def parameters(self):
        """Yield parameters from all models in the ensemble for the optimizer."""
        for model in self.models:
            for param in model.parameters():
                yield param

    def train(self):
        """Set all ensemble members to training mode."""
        for model in self.models:
            model.train()

    def eval(self):
        """Set all ensemble members to evaluation mode."""
        for model in self.models:
            model.eval()

    def predict_with_uncertainty(self, X, n_samples=50):
        """Predict with full uncertainty quantification."""
        if not isinstance(X, torch.Tensor):
            if hasattr(X, "values"):
                X = X.values
            X_tensor = torch.tensor(X, dtype=torch.double)
        else:
            X_tensor = X

        # Collect predictions from each ensemble member
        ensemble_means = []
        ensemble_stds = []

        for model in self.models:
            mean, std, _ = model.predict_with_uncertainty(X_tensor, n_samples)
            ensemble_means.append(mean)
            ensemble_stds.append(std)

        ensemble_means = np.array(ensemble_means)
        ensemble_stds = np.array(ensemble_stds)

        # Overall prediction
        predictions = ensemble_means.mean(axis=0)

        # Ensemble uncertainty (disagreement between models)
        ensemble_uncertainty = ensemble_means.std(axis=0)

        # Average MC dropout uncertainty within models
        mc_uncertainty = ensemble_stds.mean(axis=0)

        # Total predictive uncertainty (law of total variance)
        total_uncertainty = np.sqrt(ensemble_uncertainty**2 + mc_uncertainty**2)

        return predictions, total_uncertainty, ensemble_means

    def predict_proba(self, X):
        """sklearn-compatible predict_proba using mean predictions."""
        # Update to use the new argument name
        predictions, _, _ = self.predict_with_uncertainty(X, n_samples=50) 
        return np.vstack([1 - predictions, predictions]).T

    def predict(self, X, threshold=0.5):
        """sklearn-compatible predict."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)


class SimpleCNN(nn.Module):
    def __init__(self, input_channels=1, num_filters=32, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, num_filters, kernel_size)
        self.fc = nn.Linear(num_filters, 1)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1) 

        x = F.relu(self.conv1(x))
        x = torch.mean(x, dim=2)
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


# Torch Wrapper for sklearn-like interface
class TorchWrapper(BaseEstimator, ClassifierMixin):
    """
    Torch wrapper with loss tracking, early stopping on training loss,
    convergence detection
    """
    def __init__(
        self,
        net_class,
        input_dim=None,
        hidden_dims=None,
        dropout=0.5,
        n_models=None,
        lr=1e-3,
        epochs=50,
        batch_size=32,
        n_mc_samples=100,
        **extra_params
    ):
        self.net_class = net_class
        self.input_dim = input_dim
        self.dropout = dropout
        self.n_models = n_models
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_mc_samples = n_mc_samples

        # Handle flattened hidden dims
        if hidden_dims is not None:
            self.hidden_dims = hidden_dims
        else:
            h1 = extra_params.get("hidden_dim_1")
            h2 = extra_params.get("hidden_dim_2")
            self.hidden_dims = [int(h1), int(h2)] if (h1 and h2) else [64, 32]

        self.net = None
        if self.input_dim is not None:
            self._build_net()

    def _build_net(self):
        net_params = {
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
        }
        self.net = self.net_class(**net_params)
        self.net.to(dtype=torch.double)

    def fit(self, X, y):
        """Training with convergence checking."""
        if hasattr(X, "values"):
            X = X.values
        if hasattr(y, "values"):
            y = y.values

        if self.net is None:
            self.input_dim = X.shape[1]
            self._build_net()

        assert self.net is not None
        self.net.to(dtype=torch.double)

        X_tensor = torch.tensor(X, dtype=torch.double)
        y_tensor = torch.tensor(y, dtype=torch.double)

        # Check for class balance
        unique, _ = np.unique(y, return_counts=True)
        if len(unique) < 2:
            logger.error(f"Only one class in training data: {unique}")
            return self

        loader = DataLoader(
            TensorDataset(X_tensor, y_tensor),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False
        )

        optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion = nn.BCELoss()

        self.net.train()
        epoch_losses = []

        # Training loop with convergence tracking
        for epoch in range(self.epochs):
            batch_losses = []
            for xb, yb in loader:
                optimizer.zero_grad()
                preds = self.net(xb)

                # Ensure shape compatibility
                if preds.shape != yb.shape:
                    logger.warning(f"Shape mismatch: preds {preds.shape} vs yb {yb.shape}")
                    if preds.dim() == 0:
                        preds = preds.unsqueeze(0)
                    if yb.dim() == 0:
                        yb = yb.unsqueeze(0)

                loss = criterion(preds, yb)

                # Check for NaN loss
                if torch.isnan(loss):
                    logger.error("NaN loss detected! Training failed.")
                    return self

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                optimizer.step()
                batch_losses.append(loss.item())

            scheduler.step()
            avg_loss = sum(batch_losses) / len(batch_losses)
            epoch_losses.append(avg_loss)

            # Log progress every 100 epochs, and always log the final epoch
            if (epoch + 1) % 100 == 0 or (epoch + 1) == self.epochs:
                logger.debug(f"Epoch {epoch+1}/{self.epochs}: Loss={avg_loss:.4f}")

        # Check if training converged
        final_loss = epoch_losses[-1]
        if final_loss > 0.69:
            logger.warning(
                f"Training may have failed: final loss={final_loss:.4f} "
                f"(random guessing ≈ 0.693)"
            )

        # Check if loss decreased
        if len(epoch_losses) > 10:
            initial_avg = np.mean(epoch_losses[:10])
            final_avg = np.mean(epoch_losses[-10:])
            improvement = initial_avg - final_avg

            if improvement < 0.01:
                logger.warning(
                    f"Loss barely improved: {initial_avg:.4f} -> {final_avg:.4f}. "
                    f"Model may need different hyperparameters."
                )

        return self

    def predict_proba(self, X):
        """Probability predictions."""
        if hasattr(X, "values"):
            X = X.values
        X_tensor = torch.tensor(X, dtype=torch.double)

        assert self.net is not None
        self.net.eval()
        mean, _, _ = self.net.predict_with_uncertainty(X_tensor, n_samples=self.n_mc_samples)

        # Ensure proper shape: (n_samples, 2)
        mean = np.asarray(mean).ravel()
        return np.vstack([1 - mean, mean]).T

    def score(self, X, y):
        """AUC score with error handling."""
        try:
            proba = self.predict_proba(X)[:, 1]
            if len(np.unique(y)) < 2:
                logger.warning("Only one class in validation set")
                return 0.5
            auc = {
                "pr_auc": average_precision_score(y, proba),
                "roc_auc": roc_auc_score(y, proba)
            }
            return auc["roc_auc"]
        except (ValueError, RuntimeError, AttributeError) as e:
            logger.error(f"Scoring failed: {e}")
            return 0.5

    def predict(self, X, threshold=0.5):
        """Predict class labels."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)

    def predict_with_uncertainty(self, X, n_mc_samples=None):
        """
        Expose uncertainty quantification from the inner network
        to allow is_bnn_model() to detect BNN capability.
        """
        if n_mc_samples is None:
            n_mc_samples = self.n_mc_samples

        if hasattr(X, "values"): 
            X = X.values
        X_tensor = torch.tensor(X, dtype=torch.double)

        assert self.net is not None, "Model must be fitted first"

        # Call the inner network's uncertainty method
        return self.net.predict_with_uncertainty(X_tensor, n_samples=n_mc_samples)
