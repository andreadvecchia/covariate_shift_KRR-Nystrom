from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
except Exception as e:  # pragma: no cover
    raise ImportError("plotting requires matplotlib; install with `pip install covshift-correct[plot]`") from e

def plot_data_3d(
    cfg,
    n_tr: int = 80,
    n_te: int = 80,
    *,
    k: int = 2,
    grid_limits: Tuple[float, float] = (-5.0, 5.0),
    grid_size: int = 100,
    train_color: str = "red",
    test_color: str = "blue",
    surface_alpha: float = 0.1,
    ax: Optional["Axes3D"] = None,
    random_state: Optional[int] = None,
    title: str = "Training and Test Data (Covariate Shift) with Regression Function",
    show: bool = True,
) -> Tuple[plt.Figure, "Axes3D"]:
    rng = np.random.default_rng(random_state)

    # Sample
    X_tr, _ = cfg.sample_train(cfg, n_tr, rng.integers(0, 10_000))
    X_te, _ = cfg.sample_test(cfg, n_te, rng.integers(0, 10_000))

    # Grid & surface
    grid = np.linspace(*grid_limits, grid_size)
    GX, GY = np.meshgrid(grid, grid, indexing="xy")
    G = np.c_[GX.ravel(), GY.ravel()]
    Z = cfg.regression_function(G, k=k).reshape(GX.shape)

    created_ax = False
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        created_ax = True
    else:
        fig = ax.get_figure()

    ax.scatter(X_tr[:, 0], X_tr[:, 1], cfg.regression_function(X_tr, k=k),
               c=train_color, alpha=0.6, label="Train")
    ax.scatter(X_te[:, 0], X_te[:, 1], cfg.regression_function(X_te, k=k),
               c=test_color, alpha=0.6, label="Test")
    ax.plot_surface(GX, GY, Z, alpha=surface_alpha)

    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("f(x)")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.3)

    if show and created_ax:
        plt.show()

    return fig, ax
