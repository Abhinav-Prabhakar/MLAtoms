"""Template for a new atom implementation.

Contract
--------
- numpy only, plus the standard library. No sklearn, no torch, no pandas.
- Deterministic given a seed.
- Every public function states its shapes in the docstring.
- `python <file>` runs a self-contained demo on synthetic data.
"""

from __future__ import annotations

import numpy as np


class Algorithm:
    """One family, every variant, behind a single parameter.

    Parameters
    ----------
    variant : str
        One of VARIANTS. Selects the specialised algorithm.
    seed : int
        Seed for any stochastic component.

    Notes
    -----
    Fit stores everything needed by predict; no global state.
    """

    VARIANTS: tuple[str, ...] = ()

    def __init__(self, variant: str, *, seed: int = 0) -> None:
        if variant not in self.VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {self.VARIANTS}")
        self.variant = variant
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> Algorithm:
        """X: (n_samples, n_features), y: (n_samples,) or (n_samples, n_targets)."""
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """X: (n_samples, n_features) -> (n_samples, ...)."""
        raise NotImplementedError


def main() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = X @ np.array([1.0, -2.0, 0.5]) + rng.normal(scale=0.1, size=200)
    model = Algorithm("default").fit(X, y)
    print(model.predict(X[:5]))


if __name__ == "__main__":
    main()
