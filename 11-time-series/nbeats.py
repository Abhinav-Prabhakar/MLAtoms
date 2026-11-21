"""N-BEATS — residual basis expansion with small neural blocks.

Chapter:   11-time-series
Companion: nbeats.md
Tier 3 · 4 variants: generic, interpretable, nbeatsx, ensemble
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("generic", "interpretable", "nbeatsx", "ensemble")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _check_series(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("y must be one-dimensional")
    if y.size < 30:
        raise ValueError("y needs at least thirty observations")
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
    """Centre and scale every window by its own mean and standard deviation."""
    mean = left.mean(axis=1, keepdims=True)
    scale = left.std(axis=1, keepdims=True)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return (left - mean) / scale, (right - mean) / scale, mean, scale


def _basis(kind: str, length: int, horizon: int, period: int, degree: int,
           harmonics: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the backcast and forecast basis matrices for one block kind."""
    if kind == "generic":
        return np.eye(length), np.eye(horizon)
    if kind == "trend":
        scale = float(length)
        back = (np.arange(1, length + 1, dtype=float) - length) / scale
        fore = np.arange(1, horizon + 1, dtype=float) / scale
        powers = range(degree + 1)
        return (np.column_stack([back ** power for power in powers]),
                np.column_stack([fore ** power for power in powers]))
    back = 2.0 * np.pi * np.arange(1, length + 1, dtype=float) / period
    fore = 2.0 * np.pi * np.arange(length + 1, length + horizon + 1,
                                   dtype=float) / period
    back_columns = []
    fore_columns = []
    for harmonic in range(1, harmonics + 1):
        back_columns.append(np.cos(harmonic * back))
        back_columns.append(np.sin(harmonic * back))
        fore_columns.append(np.cos(harmonic * fore))
        fore_columns.append(np.sin(harmonic * fore))
    return np.column_stack(back_columns), np.column_stack(fore_columns)


# --------------------------------------------------------------------------- #
# One block
# --------------------------------------------------------------------------- #
class _Block:
    """A small ReLU MLP with a linear backcast head and a linear forecast head."""

    def __init__(self, kind: str, seq_len: int, horizon: int, width: int,
                 depth: int, period: int, degree: int, harmonics: int,
                 rng: np.random.Generator):
        back_basis, fore_basis = _basis(kind, seq_len, horizon, period, degree,
                                        harmonics)
        self.back_basis = back_basis
        self.fore_basis = fore_basis
        self.seq_len = seq_len
        self.horizon = horizon
        self.depth = depth
        shapes = [seq_len] + [width] * depth
        self.weights = []
        self.biases = []
        for fan_in, fan_out in zip(shapes[:-1], shapes[1:], strict=True):
            self.weights.append(
                rng.normal(0.0, np.sqrt(2.0 / fan_in), (fan_out, fan_in)))
            self.biases.append(np.zeros(fan_out))
        self.head_fore = rng.normal(0.0, np.sqrt(2.0 / width),
                                    (fore_basis.shape[1], width))
        self.head_back = rng.normal(0.0, np.sqrt(2.0 / width),
                                    (back_basis.shape[1], width))
        self.offset_fore = np.zeros(fore_basis.shape[1])
        self.offset_back = np.zeros(back_basis.shape[1])

    def _parameters(self) -> list[np.ndarray]:
        return (self.weights + self.biases
                + [self.head_fore, self.head_back,
                   self.offset_fore, self.offset_back])

    def _forward(self, values: np.ndarray):
        activations = [values]
        pre = []
        current = values
        for weight, bias in zip(self.weights, self.biases, strict=True):
            current = current @ weight.T + bias
            pre.append(current)
            current = np.maximum(current, 0.0)
            activations.append(current)
        theta_fore = current @ self.head_fore.T + self.offset_fore
        theta_back = current @ self.head_back.T + self.offset_back
        return theta_fore, theta_back, activations, pre

    def fit(self, inputs: np.ndarray, targets: np.ndarray, *, epochs: int,
            learning_rate: float, backcast_weight: float) -> float:
        parameters = self._parameters()
        moment1 = [np.zeros_like(value) for value in parameters]
        moment2 = [np.zeros_like(value) for value in parameters]
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        batch = inputs.shape[0]
        loss = float("inf")
        for epoch in range(1, epochs + 1):
            theta_fore, theta_back, activations, pre = self._forward(inputs)
            forecast = theta_fore @ self.fore_basis.T
            backcast = theta_back @ self.back_basis.T
            loss = float(np.mean((forecast - targets) ** 2)
                         + backcast_weight * np.mean((backcast - inputs) ** 2))
            grad_fore = 2.0 * (forecast - targets) / (batch * self.horizon)
            grad_back = (2.0 * backcast_weight * (backcast - inputs)
                         / (batch * self.seq_len))
            theta_fore_grad = grad_fore @ self.fore_basis
            theta_back_grad = grad_back @ self.back_basis
            top = activations[-1]
            weight_grads: list[np.ndarray | None] = [None] * self.depth
            bias_grads: list[np.ndarray | None] = [None] * self.depth
            hidden = (theta_fore_grad @ self.head_fore
                      + theta_back_grad @ self.head_back)
            for layer in range(self.depth - 1, -1, -1):
                delta = hidden * (pre[layer] > 0.0)
                weight_grads[layer] = delta.T @ activations[layer]
                bias_grads[layer] = delta.sum(axis=0)
                if layer > 0:
                    hidden = delta @ self.weights[layer]
            gradients = (
                weight_grads + bias_grads
                + [theta_fore_grad.T @ top, theta_back_grad.T @ top,
                   theta_fore_grad.sum(axis=0), theta_back_grad.sum(axis=0)])
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
        theta_back = self._forward(values)[1]
        return theta_back @ self.back_basis.T

    def forecast(self, values: np.ndarray) -> np.ndarray:
        theta_fore = self._forward(values)[0]
        return theta_fore @ self.fore_basis.T


# --------------------------------------------------------------------------- #
# A stack of blocks
# --------------------------------------------------------------------------- #
class _Stack:
    """Blocks applied to their own residual, with forecasts added together."""

    def __init__(self, kinds: list[str], seq_len: int, horizon: int,
                 width: int, depth: int, period: int, degree: int,
                 harmonics: int, epochs: int, learning_rate: float,
                 backcast_weight: float, seed: int):
        rng = np.random.default_rng(seed)
        self.blocks = [
            _Block(kind, seq_len, horizon, width, depth, period, degree,
                   harmonics, rng)
            for kind in kinds
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
class NBEATS:
    """Residual basis-expansion forecaster with small MLP blocks.

    Parameters
    ----------
    variant : str
        One of `VARIANTS`.
    seq_len, horizon : int
        Look-back window and forecast horizon.
    blocks : int
        Blocks per stack. `interpretable` alternates a polynomial trend basis
        and a Fourier seasonal basis; the other variants use generic bases.
    width, depth : int
        Hidden width and number of hidden ReLU layers per block.
    period : int
        Seasonal period used by the Fourier basis.
    degree, harmonics : int
        Polynomial degree and number of Fourier harmonics.
    epochs : int
        Full-batch Adam epochs per block.
    learning_rate, backcast_weight : float
        Adam step size and the weight on the backcast reconstruction term.
    ensemble_size : int
        Seed count averaged by the `ensemble` variant.
    ridge : float
        Penalty on the exogenous head of `nbeatsx`; the intercept is free.
    seed : int
        Base seed; ensemble members use `seed + 1000 * index`.

    Notes
    -----
    `fit(y, exog=None)` trains each block on the residual of the previous one.
    Every window is standardised by its own mean and standard deviation, so
    `predict` does the same to the last `seq_len` observations. `train_loss_`
    is the mean squared error of the stacked forecast on the standardised
    training windows. `nbeatsx` fits a ridge head on the exogenous rows that
    lie inside the forecast window; `predict` needs those same future rows.
    """

    def __init__(self, variant: str = "generic", *, seq_len: int = 36,
                 horizon: int = 12, blocks: int = 2, width: int = 24,
                 depth: int = 2, period: int = 12, degree: int = 2,
                 harmonics: int = 2, epochs: int = 150,
                 learning_rate: float = 0.02, backcast_weight: float = 1.0,
                 ensemble_size: int = 3, ridge: float = 1e-3, seed: int = 0):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        for name, value in (("seq_len", seq_len), ("horizon", horizon),
                            ("blocks", blocks), ("width", width),
                            ("depth", depth), ("period", period),
                            ("degree", degree), ("harmonics", harmonics),
                            ("epochs", epochs),
                            ("ensemble_size", ensemble_size)):
            if not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive and finite")
        if not np.isfinite(backcast_weight) or backcast_weight < 0.0:
            raise ValueError("backcast_weight must be finite and non-negative")
        if not np.isfinite(ridge) or ridge < 0.0:
            raise ValueError("ridge must be finite and non-negative")
        if not isinstance(seed, (int, np.integer)):
            raise ValueError("seed must be an integer")
        self.variant = variant
        self.seq_len = int(seq_len)
        self.horizon = int(horizon)
        self.blocks = int(blocks)
        self.width = int(width)
        self.depth = int(depth)
        self.period = int(period)
        self.degree = int(degree)
        self.harmonics = int(harmonics)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.backcast_weight = float(backcast_weight)
        self.ensemble_size = int(ensemble_size)
        self.ridge = float(ridge)
        self.seed = int(seed)
        self._fitted = False

    def _kinds(self) -> list[str]:
        if self.variant != "interpretable":
            return ["generic"] * self.blocks
        return ["trend" if index % 2 == 0 else "seasonal"
                for index in range(self.blocks)]

    def _new_stack(self, seed: int) -> _Stack:
        return _Stack(self._kinds(), self.seq_len, self.horizon, self.width,
                      self.depth, self.period, self.degree, self.harmonics,
                      self.epochs, self.learning_rate, self.backcast_weight,
                      seed)

    # -- fitting ---------------------------------------------------------- #
    def fit(self, y: np.ndarray, exog: np.ndarray | None = None) -> NBEATS:
        """Train the stack on sliding windows of `y`."""
        y = _check_series(y)
        if self.variant == "nbeatsx" and exog is None:
            raise ValueError("nbeatsx needs exogenous variables")
        if self.variant != "nbeatsx" and exog is not None:
            raise ValueError("exogenous variables are only used by nbeatsx")
        left, right = _windows(y, self.seq_len, self.horizon)
        inputs, targets, mean, scale = _standardise(left, right)
        self.n_windows_ = int(inputs.shape[0])
        self.scale_ = scale
        if self.variant == "nbeatsx":
            adjusted = self._fit_exog(y, exog, inputs, targets, mean, scale)
            combined = (self._normalised_forecasts(inputs) * scale + mean
                        + adjusted)
            error = (combined - targets * scale - mean) / scale
            self.train_loss_ = float(np.mean(error ** 2))
        else:
            if self.variant == "ensemble":
                self.stacks_ = [self._new_stack(self.seed + 1000 * index).fit(
                    inputs, targets) for index in range(self.ensemble_size)]
                self.stack_ = self.stacks_[0]
            else:
                self.stack_ = self._new_stack(self.seed).fit(inputs, targets)
            self.train_loss_ = float(np.mean(
                (self._normalised_forecasts(inputs) - targets) ** 2))
        self._fitted = True
        return self

    def _normalised_forecasts(self, inputs: np.ndarray) -> np.ndarray:
        if self.variant == "ensemble":
            members = [stack.predict(inputs) for stack in self.stacks_]
            return np.mean(members, axis=0)
        return self.stack_.predict(inputs)

    def _fit_exog(self, y: np.ndarray, exog: np.ndarray, inputs: np.ndarray,
                  targets: np.ndarray, mean: np.ndarray,
                  scale: np.ndarray) -> np.ndarray:
        """Alternate a ridge head on the covariates and the block stack.

        The head is fitted first so that it soaks up the part of the target
        the covariates explain; the blocks are then retrained on what is left.
        Three rounds are enough for the coefficient to settle.
        """
        exog = np.asarray(exog, dtype=float)
        if exog.ndim != 2 or exog.shape[0] != y.size:
            raise ValueError("exog must be two-dimensional with len(y) rows")
        if not np.all(np.isfinite(exog)):
            raise ValueError("exog must be finite")
        rows = self._future_rows(exog, self.n_windows_)
        flat = rows.reshape(-1, rows.shape[-1])
        self.exog_mean_ = flat.mean(axis=0)
        spread = flat.std(axis=0)
        self.exog_scale_ = np.where(spread > 1e-12, spread, 1.0)
        design = self._exog_design(flat)
        penalty = self.ridge * np.eye(design.shape[1])
        penalty[0, 0] = 0.0
        gram = design.T @ design + penalty
        adjusted = targets.copy()
        contribution = np.zeros_like(targets)
        for _ in range(3):
            self.stack_ = self._new_stack(self.seed).fit(inputs, adjusted)
            base = self._normalised_forecasts(inputs) * scale + mean
            residual = targets * scale + mean - base
            coef = np.linalg.solve(gram, design.T @ residual.reshape(-1))
            contribution = (design @ coef).reshape(targets.shape)
            adjusted = targets - contribution / scale
        self.exog_coef_ = coef
        return contribution

    def _future_rows(self, exog: np.ndarray, count: int) -> np.ndarray:
        starts = np.arange(count)
        return exog[starts[:, None] + self.seq_len
                    + np.arange(self.horizon)[None, :]]

    def _exog_design(self, features: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones(features.shape[0]),
                                (features - self.exog_mean_) / self.exog_scale_])

    # -- prediction ------------------------------------------------------- #
    def predict(self, y: np.ndarray,
                exog: np.ndarray | None = None) -> np.ndarray:
        """Forecast `horizon` steps from the last `seq_len` observations."""
        if not self._fitted:
            raise RuntimeError("fit the model before predicting")
        if self.variant == "nbeatsx":
            if exog is None:
                raise ValueError("nbeatsx needs future exogenous variables")
            exog = np.asarray(exog, dtype=float)
            if exog.ndim != 2 or exog.shape[0] != self.horizon:
                raise ValueError("future exog must have horizon rows")
            if not np.all(np.isfinite(exog)):
                raise ValueError("future exog must be finite")
        elif exog is not None:
            raise ValueError("exogenous variables are only used by nbeatsx")
        values = _check_series(y)
        if values.size < self.seq_len:
            raise ValueError("history is shorter than the look-back window")
        window = values[-self.seq_len:]
        mean = float(window.mean())
        spread = float(window.std())
        if spread <= 1e-8:
            spread = 1.0
        normalised = (window - mean) / spread
        forecast = self._normalised_forecasts(normalised[None, :])[0]
        out = mean + spread * forecast
        if self.variant == "nbeatsx":
            design = self._exog_design(exog)
            out = out + design @ self.exog_coef_
        return out


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
    print("generic. 360 points, trend and two harmonics, 12 held out:")
    values = _series(360, rng)
    split = 348
    model = NBEATS("generic", seq_len=36, horizon=12, seed=0)
    model.fit(values[:split])
    forecast = model.predict(values[:split])
    naive = np.repeat(values[split - 1], 12)
    seasonal = values[split - 12:split]
    assert model.train_loss_ < 0.2
    assert _mae(values[split:], forecast) < _mae(values[split:], seasonal)
    print(f"  {model.n_windows_} windows, train MSE {model.train_loss_:.4f}")
    print(f"  held-out MAE {_mae(values[split:], forecast):.4f} against "
          f"{_mae(values[split:], naive):.4f} for naive and "
          f"{_mae(values[split:], seasonal):.4f} for seasonal naive")

    print("interpretable. alternating trend and Fourier blocks:")
    explained = NBEATS("interpretable", seq_len=36, horizon=12, blocks=4,
                       degree=1, harmonics=2, seed=0)
    explained.fit(values[:split])
    explained_forecast = explained.predict(values[:split])
    explained_error = _mae(values[split:], explained_forecast)
    assert explained_error < _mae(values[split:], seasonal)
    print(f"  held-out MAE {explained_error:.4f} against "
          f"{_mae(values[split:], seasonal):.4f} for seasonal naive and "
          f"{_mae(values[split:], forecast):.4f} for generic bases")
    print(f"  train MSE {explained.train_loss_:.4f} against "
          f"{model.train_loss_:.4f} for generic bases; the parametric bases "
          f"cost accuracy")

    rng = np.random.default_rng(1)
    print("nbeatsx. an unpredictable driver that is known in advance:")
    t = np.arange(360, dtype=float)
    driver = rng.normal(0.0, 1.0, t.size)
    driven = (10.0 + 0.02 * t + 2.0 * np.sin(2.0 * np.pi * t / 12.0)
              + 3.0 * driver + rng.normal(0.0, 0.2, t.size))
    exog = driver[:, None]
    plain = NBEATS("generic", seq_len=36, horizon=12, seed=0)
    plain.fit(driven[:split])
    augmented = NBEATS("nbeatsx", seq_len=36, horizon=12, seed=0)
    augmented.fit(driven[:split], exog[:split])
    plain_error = _mae(driven[split:], plain.predict(driven[:split]))
    augmented_error = _mae(driven[split:],
                           augmented.predict(driven[:split], exog[split:]))
    assert augmented_error < plain_error
    print(f"  held-out MAE {augmented_error:.4f} against {plain_error:.4f} "
          f"without the driver")
    print(f"  exogenous head {np.round(augmented.exog_coef_, 4)} on a "
          f"standardised driver; the slope is the effect of one driver "
          f"standard deviation")

    print("ensemble. three seeds averaged on the first series:")
    members = []
    for seed in range(3):
        member = NBEATS("generic", seq_len=36, horizon=12, seed=seed)
        members.append(_mae(values[split:],
                            member.fit(values[:split]).predict(values[:split])))
    ensemble = NBEATS("ensemble", seq_len=36, horizon=12, seed=0)
    ensemble_error = _mae(values[split:],
                          ensemble.fit(values[:split]).predict(values[:split]))
    assert ensemble_error < float(np.max(members))
    assert ensemble_error < members[0]
    print(f"  member MAE {np.round(members, 4)} mean "
          f"{float(np.mean(members)):.4f}, worst {float(np.max(members)):.4f}")
    print(f"  ensemble MAE {ensemble_error:.4f} from "
          f"{len(ensemble.stacks_)} stacks")

    print("  a deeper stack keeps improving the in-sample fit and loses")
    print("  ground out of sample:")
    deep = NBEATS("generic", seq_len=36, horizon=12, blocks=4, width=32,
                  epochs=200, seed=0)
    deep.fit(values[:split])
    deep_error = _mae(values[split:], deep.predict(values[:split]))
    print(f"  four blocks of width 32: train MSE {deep.train_loss_:.4f} "
          f"against {model.train_loss_:.4f}, held-out MAE {deep_error:.4f} "
          f"against {_mae(values[split:], forecast):.4f}")

    bad = [
        lambda: NBEATS("nbeats_lite"),
        lambda: NBEATS("generic", seq_len=0),
        lambda: NBEATS("generic", horizon=0),
        lambda: NBEATS("generic", blocks=0),
        lambda: NBEATS("generic", depth=0),
        lambda: NBEATS("generic", period=0),
        lambda: NBEATS("generic", degree=0),
        lambda: NBEATS("generic", harmonics=0),
        lambda: NBEATS("generic", epochs=0),
        lambda: NBEATS("generic", ensemble_size=0),
        lambda: NBEATS("generic", learning_rate=0.0),
        lambda: NBEATS("generic", backcast_weight=-1.0),
        lambda: NBEATS("generic", ridge=-1.0),
        lambda: NBEATS("generic").fit(np.zeros((30, 2))),
        lambda: NBEATS("generic").fit(np.arange(20.0)),
        lambda: NBEATS("generic").fit(np.r_[np.arange(40.0), np.nan]),
        lambda: NBEATS("nbeatsx").fit(values[:split]),
        lambda: NBEATS("generic").fit(values[:split], np.ones((split, 1))),
        lambda: NBEATS("generic").predict(values[:split]),
        lambda: NBEATS("nbeatsx").fit(values[:split],
                                      np.ones((split, 1))).predict(
            values[:split]),
        lambda: NBEATS("generic", seq_len=36).fit(values[:split]).predict(
            np.arange(10.0)),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except (ValueError, RuntimeError):
            refused += 1
    print(f"{refused} malformed calls refused out of {len(bad)}: an unknown")
    print("  variant, a non-positive look-back, horizon, block count, depth,")
    print("  period, basis size, epoch count or ensemble size, a non-positive")
    print("  learning rate, a negative backcast weight or ridge, a 2-d or")
    print("  short series, a non-finite series, missing exog for nbeatsx, exog")
    print("  passed to another variant, prediction before fitting, nbeatsx")
    print("  prediction without future exog, and a history shorter than the")
    print("  look-back")
    print("nbeats demo passed")


if __name__ == "__main__":
    demo()
