from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.gaussian_process.kernels import RBF

# vendored bless (same dir)
from .bless import bless

# ----------------
# Configuration
# ----------------
@dataclass
class Config:
    # distribution parameters
    mu_tr: np.ndarray = field(default_factory=lambda: np.array([0.7, 0.7]))
    var_tr: np.ndarray = field(default_factory=lambda: np.diag(np.array([0.7, 0.7])))
    mu_te: np.ndarray = field(default_factory=lambda: np.array([1.8, 1.8]))
    var_te: np.ndarray = field(default_factory=lambda: np.diag(np.array([0.5, 0.5])))
    noise: float = 0.2

    # train and test sizes
    n_tr: int = 2000
    n_te: int = 200

    # hyperparameter grids
    lambda_grid: Tuple[float, ...] = (1e-5, 1e-4, 1e-3, 1e-2)
    gamma_grid: Tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0)

    def __post_init__(self):
        self.train_dist = multivariate_normal(mean=self.mu_tr, cov=self.var_tr)
        self.test_dist = multivariate_normal(mean=self.mu_te, cov=self.var_te)

    def regression_function(self, X: np.ndarray, k: int = 50) -> np.ndarray:
        # same functional form as notebook
        return 10 * np.exp(-10 * np.linalg.norm(X, axis=1) ** (-2 * k))

# ----------------
# Data utilities
# ----------------
def sample_train(cfg: Config, n_tr: Optional[int] = None, random_state: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    n_tr = cfg.n_tr if n_tr is None else n_tr
    rng = np.random.default_rng(random_state)
    X = rng.multivariate_normal(cfg.mu_tr, cfg.var_tr, n_tr)
    y = cfg.regression_function(X) + rng.normal(0, cfg.noise, n_tr)
    return X, y

def sample_test(cfg: Config, n_te: Optional[int] = None, random_state: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    n_te = cfg.n_te if n_te is None else n_te
    rng = np.random.default_rng(random_state)
    X = rng.multivariate_normal(cfg.mu_te, cfg.var_te, n_te)
    y = cfg.regression_function(X) + rng.normal(0, cfg.noise, n_te)
    return X, y

def weights(cfg: Config, X: np.ndarray) -> np.ndarray:
    # importance weights w = p_te(x)/p_tr(x)
    return cfg.test_dist.pdf(X) / cfg.train_dist.pdf(X)

def get_kernel_matrix(X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
    return rbf_kernel(X, Y, gamma=gamma)

# ----------------
# Models
# ----------------
def KRR(K: np.ndarray, reg: float, y: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    return np.linalg.inv(K + n * reg * np.eye(n)) @ y

def KRR_w(K: np.ndarray, reg: float, w: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    M_w_inv = np.diag(1.0 / w)
    return np.linalg.inv(K + n * reg * M_w_inv) @ y

def KRR_w_nyst(Knm: np.ndarray, Kmm: np.ndarray, reg: float, w: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = Knm.shape[0]
    M_w = np.diag(w)
    # (K_nm^T M_w K_nm + n*reg*K_mm)^+ K_nm^T M_w y
    A = Knm.T @ M_w @ Knm + n * reg * Kmm
    return np.linalg.pinv(A) @ (Knm.T @ M_w @ y)

# ----------------
# CV & Training
# ----------------
def cross_val(cfg: Config, X: np.ndarray, Y: np.ndarray, model: str, n_splits: int = 5) -> Tuple[float, float]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    best_mse = float("inf")
    best_lambda = cfg.lambda_grid[0]
    best_gamma = cfg.gamma_grid[0]

    for lambda_ in cfg.lambda_grid:
        for gamma in cfg.gamma_grid:
            fold_mse: List[float] = []
            for tr_ind, val_ind in kf.split(X):
                X_tr, X_val = X[tr_ind], X[val_ind]
                Y_tr, Y_val = Y[tr_ind], Y[val_ind]

                if model == "KRR":
                    K = get_kernel_matrix(X_tr, X_tr, gamma)
                    c = KRR(K, lambda_, Y_tr)
                    Y_pr = c @ get_kernel_matrix(X_tr, X_val, gamma)

                elif model == "KRR_w":
                    w = weights(cfg, X_tr)
                    K = get_kernel_matrix(X_tr, X_tr, gamma)
                    c = KRR_w(K, lambda_, w, Y_tr)
                    Y_pr = c @ get_kernel_matrix(X_tr, X_val, gamma)

                elif model == "KRR_w_nyst":
                    w = weights(cfg, X_tr)
                    centers = bless(X_tr, RBF(length_scale=1.0), 3, 10, force_cpu=True, verbose=False)
                    X_nyst = centers.X
                    K_nm = get_kernel_matrix(X_tr, X_nyst, gamma)
                    K_mm = get_kernel_matrix(X_nyst, X_nyst, gamma)
                    c = KRR_w_nyst(K_nm, K_mm, lambda_, w, Y_tr)
                    Y_pr = c @ get_kernel_matrix(X_nyst, X_val, gamma)

                else:
                    raise ValueError(f"Unknown model: {model}")

                fold_mse.append(mean_squared_error(Y_val, Y_pr))

            avg_mse = float(np.mean(fold_mse))
            if avg_mse < best_mse:
                best_mse, best_lambda, best_gamma = avg_mse, lambda_, gamma

    return best_lambda, best_gamma

def train(cfg: Config, model: str, num_runs: int = 5, *, random_seed_offset: int = 0):
    mse_list: List[float] = []
    last = None

    for run in range(num_runs):
        X_tr, Y_tr = sample_train(cfg, random_state=run + random_seed_offset)
        X_te, Y_te = sample_test(cfg, random_state=run + random_seed_offset)

        best_lambda, best_gamma = cross_val(cfg, X_tr, Y_tr, model)

        if model == "KRR":
            K = get_kernel_matrix(X_tr, X_tr, best_gamma)
            c = KRR(K, best_lambda, Y_tr)
            Y_pr = c @ get_kernel_matrix(X_tr, X_te, best_gamma)

        elif model == "KRR_w":
            w = weights(cfg, X_tr)
            K = get_kernel_matrix(X_tr, X_tr, best_gamma)
            c = KRR_w(K, best_lambda, w, Y_tr)
            Y_pr = c @ get_kernel_matrix(X_tr, X_te, best_gamma)

        elif model == "KRR_w_nyst":
            w = weights(cfg, X_tr)
            centers = bless(X_tr, RBF(length_scale=1.0), 3, 10, force_cpu=True, verbose=False)
            X_nyst = centers.X
            K_nm = get_kernel_matrix(X_tr, X_nyst, best_gamma)
            K_mm = get_kernel_matrix(X_nyst, X_nyst, best_gamma)
            c = KRR_w_nyst(K_nm, K_mm, best_lambda, w, Y_tr)
            Y_pr = c @ get_kernel_matrix(X_nyst, X_te, best_gamma)
        else:
            raise ValueError(f"Unknown model: {model}")

        mse = float(mean_squared_error(Y_te, Y_pr))
        mse_list.append(mse)
        last = dict(Y_pred=Y_pr, X_tr=X_tr, Y_tr=Y_tr, X_te=X_te, Y_te=Y_te)

    return mse_list, last

# ----------------
# CLI
# ----------------
def cli_compare():
    import argparse, time

    parser = argparse.ArgumentParser(description="Compare KRR variants under covariate shift.")
    parser.add_argument("--runs", type=int, default=5, help="Number of random seeds/runs")
    args = parser.parse_args()

    cfg = Config()
    models = [("KRR", "KRR"), ("IW-KRR", "KRR_w"), ("IW–Nyström KRR", "KRR_w_nyst")]

    records = []
    for display_name, model_key in models:
        t0 = time.perf_counter()
        mse_list, _ = train(cfg, model_key, num_runs=args.runs)
        dt = time.perf_counter() - t0
        mu = float(np.mean(mse_list))
        sd = float(np.std(mse_list, ddof=1)) if len(mse_list) > 1 else 0.0
        records.append(dict(Model=display_name, Mean_MSE=mu, Std_MSE=sd, Runs=args.runs, Time_s=dt))

    df = pd.DataFrame.from_records(records).set_index("Model").sort_values("Mean_MSE")
    pd.options.display.float_format = "{:,.4f}".format
    print("=== Test Performance under Covariate Shift ===")
    print(df.to_string())
    best = df["Mean_MSE"].idxmin()
    print(f"\nBest (by mean MSE): {best}  →  {df.loc[best, 'Mean_MSE']:.4f} ± {df.loc[best, 'Std_MSE']:.4f}")
