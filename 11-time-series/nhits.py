"""N-HiTS — multi-rate sampling with hierarchical interpolation.

Chapter:   11-time-series
Companion: nhits.md
Tier 3 · 3 variants: nhits, multi_rate, hierarchical_interp
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("nhits", "multi_rate", "hierarchical_interp")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _check_series(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("y must be one-dimensional")
    if y.size < 40:
        raise ValueError("y needs at least forty observations")
    if not np.all(np.isfinite(y)):
        raise ValueError("y must be finite")
    return y


def _windows(values: np.ndarray, seq_len: int, horizon: int
             ) -> tuple[np.ndarray, np.ndarray]:
    count = values.size - seq_len - horizon + 1
    if count < 8:
        raise ValueError("need at least eight training windows")
    starts = np.arange(count)
    left = values[starts[:, None] + np.arange(seq_len)[None, :]]
    right = values[starts[:, None] + seq_len + np.arange(horizon)[None, :]]
    return left, right


def _standardise(left: np.ndarray, right: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = left.mean(axis=1, keepdims=True)
    scale = left.std(axis=1, keepdims=True)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return (left - mean) / scale, (right - mean) / scale, mean, scale


def _interp_matrix(fine: int, coarse: int) -> np.ndarray:
    """Linear interpolation from `coarse` evenly spaced points to `fine`."""
    if coarse == 1:
        return np.ones((fine, 1))
    position = np.linspace(0.0, coarse - 1.0, fine)
    left = np.floor(position).astype(int)
    frac = position - left
    left = np.clip(left, 0, coarse - 1)
    right = np.clip(left + 1, 0, coarse - 1)
    matrix = np.zeros((fine, coarse))
    rows = np.arange(fine)
    matrix[rows, left] += 1.0 - frac
    matrix[rows, right] += frac
    return matrix


def _max_pool(values: np.ndarray, rate: int, pooled: int) -> np.ndarray:
    """Left-pad, reshape and take a maximum per group of `rate` steps."""
    padding = pooled * rate - values.shape[1]
    if padding > 0:
        edge = np.repeat(values[:, :1], padding, axis=1)
        values = np.concatenate([edge, values], axis=1)
    grouped = values.reshape(values.shape[0], pooled, rate)
    return grouped.max(axis=2)


# --------------------------------------------------------------------------- #
# One block
# --------------------------------------------------------------------------- #
class _Block:
    """A one-hidden-layer MLP whose heads emit coefficients on a coarse grid."""

    def __init__(self, seq_len: int, horizon: int, rate: int, coarse: int,
                 width: int, rng: np.random.Generator):
        self.seq_len = seq_len
        self.horizon = horizon
        self.rate = rate
        self.pooled = -(-seq_len // rate)
        self.coarse = coarse
        self.back_matrix = _interp_matrix(seq_len, self.pooled)
        self.fore_matrix = _interp_matrix(horizon, coarse)
        self.w1 = rng.normal(0.0, np.sqrt(2.0 / self.pooled),
                             (width, self.pooled))
        self.b1 = np.zeros(width)
        self.wf = rng.normal(0.0, np.sqrt(2.0 / width), (coarse, width))
        self.cf = np.zeros(coarse)
        self.wb = rng.normal(0.0, np.sqrt(2.0 / width),
                             (self.pooled, width))
        self.cb = np.zeros(self.pooled)

    def _parameters(self) -> list[np.ndarray]:
        return [self.w1, self.b1, self.wf, self.cf, self.wb, self.cb]

    def _hidden(self, values: np.ndarray):
        pooled = _max_pool(values, self.rate, self.pooled)
        pre = pooled @ self.w1.T + self.b1
        return pooled, pre, np.maximum(pre, 0.0)

    def fit(self, inputs: np.ndarray, targets: np.ndarray, *, epochs: int,
            learning_rate: float, backcast_weight: float) -> float:
        parameters = self._parameters()
        moment1 = [np.zeros_like(value) for value in parameters]
        moment2 = [np.zeros_like(value) for value in parameters]
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        batch = inputs.shape[0]
        loss = float("inf")
        for epoch in range(1, epochs + 1):
            pooled, pre, activation = self._hidden(inputs)
            theta_fore = activation @ self.wf.T + self.cf
            theta_back = activation @ self.wb.T + self.cb
            forecast = theta_fore @ self.fore_matrix.T
            backcast = theta_back @ self.back_matrix.T
            loss = float(np.mean((forecast - targets) ** 2)
                         + backcast_weight * np.mean((backcast - inputs) ** 2))
            grad_fore = 2.0 * (forecast - targets) / (batch * self.horizon)
            grad_back = (2.0 * backcast_weight * (backcast - inputs)
                         / (batch * self.seq_len))
            theta_fore_grad = grad_fore @ self.fore_matrix
            theta_back_grad = grad_back @ self.back_matrix
            hidden_grad = (theta_fore_grad @ self.wf
                           + theta_back_grad @ self.wb)
            delta = hidden_grad * (pre > 0.0)
            gradients = [
                delta.T @ pooled,
                delta.sum(axis=0),
                theta_fore_grad.T @ activation,
                theta_fore_grad.sum(axis=0),
                theta_back_grad.T @ activation,
                theta_back_grad.sum(axis=0),
            ]
            for index, (parameter, gradient) in enumerate(
                    zip(parameters, gradients, strict=True)):
                moment1[index] = beta1 * moment1[index] + (1.0 - beta1) * gradient
                moment2[index] = beta2 * moment2[index] + (1.0 - beta2) * gradient ** 2
                corrected1 = moment1[index] / (1.0 - beta1 ** epoch)
                corrected2 = moment2[index] / (1.0 - beta2 ** epoch)
                parameter -= (learning_rate * corrected1
                              / (np.sqrt(corrected2) + epsilon))
        return loss

    def backcast(self, values: np.ndarray) -> np.ndarray:
        _, _, activation = self._hidden(values)
        theta_back = activation @ self.wb.T + self.cb
        return theta_back @ self.back_matrix.T

    def forecast(self, values: np.ndarray) -> np.ndarray:
        _, _, activation = self._hidden(values)
        theta_fore = activation @ self.wf.T + self.cf
        return theta_fore @ self.fore_matrix.T


# --------------------------------------------------------------------------- #
# A stack of blocks
# --------------------------------------------------------------------------- #
class _Stack:
    def __init__(self, rates: list[int], coarse: list[int], seq_len: int,
                 horizon: int, width: int, epochs: int,
                 learning_rate: float, backcast_weight: float, seed: int):
        rng = np.random.default_rng(seed)
        self.blocks = [
            _Block(seq_len, horizon, rate, size, width, rng)
            for rate, size in zip(rates, coarse, strict=True)
        ]
        self.horizon = horizon
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.backcast_weight = backcast_weight
        self.loss_ = float("inf")

    def fit(self, inputs: np.ndarray, targets: np.ndarray) -> _Stack:
        residual = inputs.copy()
        remaining = targets.copy()
        forecast = np.zeros_like(targets)
        for block in self.blocks:
            block.fit(residual, remaining, epochs=self.epochs,
                      learning_rate=self.learning_rate,
                      backcast_weight=self.backcast_weight)
            contribution = block.forecast(residual)
            forecast = forecast + contribution
            remaining = remaining - contribution
            residual = residual - block.backcast(residual)
        self.loss_ = float(np.mean((forecast - targets) ** 2))
        return self

    def predict(self, values: np.ndarray) -> np.ndarray:
        residual = values.copy()
        output = np.zeros((values.shape[0], self.horizon))
        for block in self.blocks:
            output = output + block.forecast(residual)
            residual = residual - block.backcast(residual)
        return output


# --------------------------------------------------------------------------- #
# The public model
# --------------------------------------------------------------------------- #
class NHITS:
    """Multi-rate N-HiTS forecaster with hierarchical interpolation.

    Parameters
    ----------
    variant : str
        One of `VARIANTS`. `nhits` uses both mechanisms, `multi_rate` pools
        the input without coarsening the forecast grid, and
        `hierarchical_interp` keeps rate one and coarsens only the grid.
    seq_len, horizon : int
        Look-back window and forecast horizon.
    blocks : int
        Number of blocks; block 0 is the coarsest.
    width : int
        Hidden width of the one-layer perceptron in each block.
    epochs, learning_rate, backcast_weight : int, float
        Full-batch Adam schedule and the weight on the backcast term.
    seed : int
        Seed for the weight initialisation.

    Notes
    -----
    Block `k` pools its input by `2 ** (blocks - 1 - k)` (`multi_rate` and
    `nhits`) and emits `horizon / rate` forecast coefficients, which a fixed
    linear interpolation matrix expands to the horizon (`nhits` and
    `hierarchical_interp`). `fit` stores `rates_`, `coarse_`, `n_windows_`
    and `train_loss_`; `predict` takes the last `seq_len` observations.
    """

    def __init__(self, variant: str = "nhits", *, seq_len: int = 48,
                 horizon: int = 12, blocks: int = 3, width: int = 32,
                 epochs: int = 120, learning_rate: float = 0.01,
                 backcast_weight: float = 1.0, seed: int = 0):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        for name, value in (("seq_len", seq_len), ("horizon", horizon),
                            ("blocks", blocks), ("width", width),
                            ("epochs", epochs)):
            if not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive and finite")
        if not np.isfinite(backcast_weight) or backcast_weight < 0.0:
            raise ValueError("backcast_weight must be finite and non-negative")
        if not isinstance(seed, (int, np.integer)):
            raise ValueError("seed must be an integer")
        self.variant = variant
        self.seq_len = int(seq_len)
        self.horizon = int(horizon)
        self.blocks = int(blocks)
        self.width = int(width)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.backcast_weight = float(backcast_weight)
        self.seed = int(seed)
        self._fitted = False

    def _schedule(self) -> tuple[list[int], list[int]]:
        rates = []
        coarse = []
        for index in range(self.blocks):
            power = 2 ** (self.blocks - 1 - index)
            rate = 1 if self.variant == "hierarchical_interp" else power
            rates.append(rate)
            if self.variant == "multi_rate":
                size = self.horizon
            elif self.variant == "hierarchical_interp":
                size = max(1, self.horizon // power)
            else:
                size = max(1, int(round(self.horizon / power)))
            coarse.append(min(size, self.horizon))
        return rates, coarse

    # -- fitting ---------------------------------------------------------- #
    def fit(self, y: np.ndarray) -> NHITS:
        """Train the stack on sliding windows of `y`."""
        y = _check_series(y)
        self.rates_, self.coarse_ = self._schedule()
        left, right = _windows(y, self.seq_len, self.horizon)
        inputs, targets, _, _ = _standardise(left, right)
        self.n_windows_ = int(inputs.shape[0])
        stack = _Stack(self.rates_, self.coarse_, self.seq_len, self.horizon,
                       self.width, self.epochs, self.learning_rate,
                       self.backcast_weight, self.seed)
        self.stack_ = stack.fit(inputs, targets)
        self.train_loss_ = self.stack_.loss_
        self._fitted = True
        return self

    # -- prediction ------------------------------------------------------- #
    def predict(self, y: np.ndarray) -> np.ndarray:
        """Forecast `horizon` steps from the last `seq_len` observations."""
        if not self._fitted:
            raise RuntimeError("fit the model before predicting")
        values = _check_series(y)
        if values.size < self.seq_len:
            raise ValueError("history is shorter than the look-back window")
        window = values[-self.seq_len:]
        mean = float(window.mean())
        spread = float(window.std())
        if spread <= 1e-8:
            spread = 1.0
        normalised = (window - mean) / spread
        forecast = self.stack_.predict(normalised[None, :])[0]
        return mean + spread * forecast


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _series(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n, dtype=float)
    return (10.0 + 0.02 * t + 2.0 * np.sin(2.0 * np.pi * t / 12.0)
            + 0.5 * np.cos(4.0 * np.pi * t / 12.0)
            + rng.normal(0.0, 0.25, n))


def _mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - forecast)))


def demo() -> None:
    rng = np.random.default_rng(0)
    print("nhits. 360 points, trend and two harmonics, 12 held out:")
    values = _series(360, rng)
    split = 348
    model = NHITS("nhits", seq_len=48, horizon=12, seed=0).fit(values[:split])
    forecast = model.predict(values[:split])
    naive = np.repeat(values[split - 1], 12)
    seasonal = values[split - 12:split]
    assert model.train_loss_ < 0.15
    assert _mae(values[split:], forecast) < _mae(values[split:], seasonal)
    print(f"  {model.n_windows_} windows, pooling rates {model.rates_}, "
          f"forecast grids {model.coarse_}")
    print(f"  train MSE {model.train_loss_:.4f}, held-out MAE "
          f"{_mae(values[split:], forecast):.4f}")
    print(f"  against {_mae(values[split:], naive):.4f} for naive and "
          f"{_mae(values[split:], seasonal):.4f} for seasonal naive")

    print("multi_rate. the same pooling without a coarse forecast grid:")
    pooled = NHITS("multi_rate", seq_len=48, horizon=12, seed=0)
    pooled.fit(values[:split])
    pooled_error = _mae(values[split:], pooled.predict(values[:split]))
    print(f"  rates {pooled.rates_}, grids {pooled.coarse_}, train MSE "
          f"{pooled.train_loss_:.4f}, held-out MAE {pooled_error:.4f}")

    print("hierarchical_interp. rate one, but the grids coarsen with depth:")
    interp = NHITS("hierarchical_interp", seq_len=48, horizon=12, seed=0)
    interp.fit(values[:split])
    interp_error = _mae(values[split:], interp.predict(values[:split]))
    print(f"  rates {interp.rates_}, grids {interp.coarse_}, train MSE "
          f"{interp.train_loss_:.4f}, held-out MAE {interp_error:.4f}")

    print("  interpolation reproduces a line exactly:")
    matrix = _interp_matrix(12, 4)
    coarse = 2.0 + 3.0 * np.arange(4)
    fitted = matrix @ coarse
    exact = 2.0 + 3.0 * np.linspace(0.0, 3.0, 12)
    assert float(np.max(np.abs(fitted - exact))) < 1e-12
    print(f"  {matrix.shape[0]} points from {matrix.shape[1]} nodes, "
          f"maximum deviation {float(np.max(np.abs(fitted - exact))):.1e}")

    print("  max pooling picks the largest value in each group:")
    block = np.arange(48, dtype=float)[None, :]
    pooled_values = _max_pool(block, 4, 12)
    assert pooled_values.shape == (1, 12)
    assert pooled_values[0, 0] == 3.0
    assert pooled_values[0, -1] == 47.0
    print(f"  first pooled value {pooled_values[0, 0]:.1f}, last "
          f"{pooled_values[0, -1]:.1f}")

    print("  a grid the size of the horizon needs no interpolation:")
    single = NHITS("hierarchical_interp", seq_len=48, horizon=12, blocks=1,
                   seed=0).fit(values[:split])
    identity = single.stack_.blocks[0].fore_matrix
    assert np.array_equal(identity, np.eye(12))
    print(f"  one block, grids {single.coarse_}, forecast matrix is the "
          f"identity")
    plain = _max_pool(values[:split][None, -48:], 1, 48)
    assert np.array_equal(plain[0], values[split - 48:split])
    print("  and a pooling rate of one returns the window unchanged")

    bad = [
        lambda: NHITS("nhits_lite"),
        lambda: NHITS("nhits", seq_len=0),
        lambda: NHITS("nhits", horizon=0),
        lambda: NHITS("nhits", blocks=0),
        lambda: NHITS("nhits", width=0),
        lambda: NHITS("nhits", epochs=0),
        lambda: NHITS("nhits", learning_rate=0.0),
        lambda: NHITS("nhits", learning_rate=np.nan),
        lambda: NHITS("nhits", backcast_weight=-1.0),
        lambda: NHITS("nhits").fit(np.zeros((40, 2))),
        lambda: NHITS("nhits").fit(np.arange(30.0)),
        lambda: NHITS("nhits").fit(np.r_[np.arange(60.0), np.nan]),
        lambda: NHITS("nhits").predict(values[:split]),
        lambda: NHITS("nhits", seq_len=48).fit(values[:split]).predict(
            np.arange(20.0)),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except (ValueError, RuntimeError):
            refused += 1
    print(f"{refused} malformed calls refused out of {len(bad)}: an unknown")
    print("  variant, a non-positive look-back, horizon, block count, width,")
    print("  epoch count or learning rate, a non-finite learning rate or a")
    print("  negative backcast weight, a 2-d or short series, a non-finite")
    print("  series, prediction before fitting, and a history shorter than")
    print("  the look-back")
    print("nhits demo passed")


if __name__ == "__main__":
    demo()
