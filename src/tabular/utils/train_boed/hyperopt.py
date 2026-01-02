"""Bayesian optimised hyperparameter tuning with BoTorch (qLogEI)."""

import matplotlib.pyplot as plt
from loguru import logger
from tqdm import tqdm
import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood
from src.tabular.utils.train_boed.utils import (
    select_device,
    evaluate_model,
    build_bounds_and_specs,
    sample_random_point,
    tensor_to_params,
)


def botorch_tuning(
    X,
    y,
    model_key,
    n_trials=20,
    batch_size=4,
    random_state=42,
    final_tuning=False,
    early_stop=True,
    ei_tol=1e-4,
    std_tol=1e-3,
    plot_gp=False,
    verbose=True,
):
    """
    BoTorch qLogNoisyExpectedImprovement hyperparameter optimisation.

    Args:
        early_stop: stop when EI + posterior std collapse
        ei_tol: EI threshold for early stopping
        std_tol: posterior std threshold
        plot_gp: if True, plot GP posterior mean/std each iteration
        verbose: if True, log iteration details
    """

    device = select_device()
    torch.manual_seed(random_state)
    rng = np.random.default_rng(random_state)

    bounds, specs = build_bounds_and_specs(model_key, device=device)

    X_train = []
    y_train = []

    if verbose:
        logger.info(f"[{model_key}] Initialising BoTorch hyperopt with {n_trials} trials...")

    # Initial random points
    n_init = max(3, batch_size)
    for i in range(n_init):
        x0 = sample_random_point(specs, rng)
        x0_tensor = torch.tensor(x0, dtype=torch.double, device=device)

        params0 = tensor_to_params(x0_tensor, specs)
        score0 = evaluate_model(X, y, model_key, **params0)

        if verbose:
            logger.debug(f"[{model_key}] Init {i+1}/{n_init} → score={score0:.4f}, params={params0}")

        X_train.append(x0)
        y_train.append(score0)

    X_train = torch.tensor(X_train, dtype=torch.double, device=device)
    y_train = torch.tensor(y_train, dtype=torch.double, device=device).unsqueeze(-1)

    # Progress bar
    pbar = tqdm(range(n_trials), disable=not verbose)

    for t in pbar:

        # GP with transforms
        gp = SingleTaskGP(
            X_train,
            y_train,
            input_transform=Normalize(d=X_train.shape[-1]),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        # Log GP hyperparameters
        if verbose:
            try:
                ls = gp.covar_module.base_kernel.lengthscale.detach().cpu().numpy() # type: ignore
                os = gp.covar_module.outputscale.item() # type: ignore
                logger.debug(f"[{model_key}] Iter {t}: GP lengthscale={ls}, outputscale={os:.4f}")
            except (AttributeError, RuntimeError):
                logger.debug(f"[{model_key}] Iter {t}: GP hyperparameters unavailable")

        # Posterior uncertainty at last point
        posterior = gp.posterior(X_train)
        std_last = float(posterior.variance.sqrt().detach().cpu().numpy()[-1])
        if verbose:
            logger.debug(f"[{model_key}] Iter {t}: posterior std={std_last:.4f}")

        # Optional GP posterior plot
        if plot_gp:
            plt.figure(figsize=(6, 4))
            plt.plot(y_train.cpu().numpy(), label="Observed y")
            plt.fill_between(
                range(len(y_train)),
                (posterior.mean - posterior.variance.sqrt()).cpu().numpy().flatten(),
                (posterior.mean + posterior.variance.sqrt()).cpu().numpy().flatten(),
                alpha=0.3,
                label="Posterior ± std",
            )
            plt.title(f"GP Posterior at Iter {t}")
            plt.legend()
            plt.show()

        # Acquisition
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        qlognei = qLogNoisyExpectedImprovement(
            model=gp,
            X_baseline=X_train,
            sampler=sampler,
        )

        # EI for early stopping
        with torch.no_grad():
            ei_vals = qlognei(X_train.unsqueeze(1))
            max_ei = float(ei_vals.max().item())

        if verbose:
            logger.debug(f"[{model_key}] Iter {t}: max qLogNEI={max_ei:.6f}")

        # Early stopping
        if early_stop and max_ei < ei_tol and std_last < std_tol:
            if verbose:
                logger.info(
                    f"[{model_key}] Early stopping at iter {t}: "
                    f"EI={max_ei:.2e}, std={std_last:.2e}"
                )
            break

        # Optimise acquisition
        candidates, _ = optimize_acqf(
            acq_function=qlognei,
            bounds=bounds,
            q=batch_size,
            num_restarts=10,
            raw_samples=64,
        )

        new_X = []
        new_y = []

        for i in range(batch_size):
            params_i = tensor_to_params(candidates[i], specs)
            score_i = evaluate_model(X, y, model_key, **params_i)

            if verbose:
                logger.debug(
                    f"[{model_key}] Iter {t}: cand {i+1}/{batch_size} "
                    f"→ score={score_i:.4f}, params={params_i}"
                )

            new_X.append(candidates[i].detach().cpu().numpy())
            new_y.append(score_i)

        new_X = np.stack(new_X, axis=0)
        new_X = torch.tensor(new_X, dtype=torch.double, device=device)
        new_y = torch.tensor(new_y, dtype=torch.double, device=device).unsqueeze(-1)

        X_train = torch.cat([X_train, new_X], dim=0)
        y_train = torch.cat([y_train, new_y], dim=0)

        pbar.set_description(f"Best={float(y_train.max()):.4f}")

    # Extract best result
    best_idx = torch.argmax(y_train).item()
    best_score = float(y_train[best_idx].item()) # type: ignore
    best_point = X_train[best_idx] # type: ignore

    if verbose:
        logger.success(f"[{model_key}] Finished BO. Best score={best_score:.4f}")

    if not final_tuning:
        return best_score

    best_params = tensor_to_params(best_point, specs)

    if verbose:
        logger.success(f"[{model_key}] Best params: {best_params}")

    return best_params, best_score
