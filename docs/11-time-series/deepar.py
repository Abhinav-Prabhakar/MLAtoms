"""DeepAR — an autoregressive RNN that predicts a distribution, not a number.

Chapter:   11-time-series
Companion: deepar.md
Tier 3 · 5 variants: deepar, gaussian_likelihood, negbin_likelihood, student_t,
quantile_output
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

VARIANTS = ("deepar", "gaussian_likelihood", "negbin_likelihood", "student_t",
            "quantile_output")

TAUS = (0.1, 0.25, 0.5, 0.75, 0.9)

_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)

_LANCZOS = (
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-06,
    1.5056327351493116e-07,
)


# --------------------------------------------------------------------------- #
# Small numerical helpers
# --------------------------------------------------------------------------- #
def _softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _gammaln(x: np.ndarray) -> np.ndarray:
    """Lanczos approximation, vectorised, for positive arguments."""
    values = np.asarray(x, dtype=float)
    if np.any(values <= 0.0):
        raise ValueError("gammaln needs positive arguments")
    result = np.empty_like(values)
    small = values < 0.5
    if small.any():
        part = values[small]
        result[small] = (np.log(np.pi / np.abs(np.sin(np.pi * part)))
                         - _gammaln(1.0 - part))
    large = ~small
    shifted = values[large] - 1.0
    series = _LANCZOS[0] + sum(_LANCZOS[i] / (shifted + i)
                               for i in range(1, 9))
    pivot = shifted + 7.5
    result[large] = (0.5 * math.log(2.0 * math.pi) + (shifted + 0.5)
                     * np.log(pivot) - pivot + np.log(series))
    return result


def _digamma(x: np.ndarray) -> np.ndarray:
    """Recurrence up to six, then the asymptotic series."""
    shifted = np.array(x, dtype=float, copy=True)
    if np.any(shifted <= 0.0):
        raise ValueError("digamma needs positive arguments")
    correction = np.zeros_like(shifted)
    while True:
        small = shifted < 6.0
        if not small.any():
            break
        correction[small] -= 1.0 / shifted[small]
        shifted[small] += 1.0
    inverse = 1.0 / shifted
    square = inverse * inverse
    series = (np.log(shifted) - 0.5 * inverse - square / 12.0
              + square * square / 120.0 - square ** 3 / 252.0)
    return correction + series


# --------------------------------------------------------------------------- #
# Likelihoods: each returns a per-step loss and its gradient in the output
# --------------------------------------------------------------------------- #
def _gaussian(output, target):
    mu = output[:, 0]
    activation = output[:, 1]
    sigma = _softplus(activation) + 1e-6
    standardised = (target - mu) / sigma
    loss = 0.5 * standardised ** 2 + np.log(sigma) + _HALF_LOG_2PI
    gradient = np.empty_like(output)
    gradient[:, 0] = -(target - mu) / sigma ** 2
    gradient[:, 1] = ((1.0 - standardised ** 2) * _sigmoid(activation)
                      / sigma)
    return loss, gradient


def _student(output, target, nu: float):
    """Unit scale: a free scale makes the t likelihood unbounded below."""
    offset = target - output[:, 0]
    constant = (math.lgamma(0.5 * (nu + 1.0)) - math.lgamma(0.5 * nu)
                - 0.5 * math.log(nu * math.pi))
    loss = constant + 0.5 * (nu + 1.0) * np.log1p(offset ** 2 / nu)
    gradient = np.empty_like(output)
    gradient[:, 0] = -(nu + 1.0) * offset / (nu + offset ** 2)
    return loss, gradient


def _negative_binomial(output, target):
    raw_mu = np.exp(np.clip(output[:, 0], -12.0, 12.0))
    raw_rate = np.exp(np.clip(output[:, 1], -6.0, 6.0))
    mu = raw_mu + 1e-8
    dispersion = raw_rate + 1e-4
    total = dispersion + mu
    loss = -(np.asarray(_gammaln(target + dispersion))
             - _gammaln(dispersion) - _gammaln(target + 1.0)
             + dispersion * (np.log(dispersion) - np.log(total))
             + target * (np.log(mu) - np.log(total)))
    d_mu = (dispersion + target) / total - target / mu
    d_rate = (-_digamma(target + dispersion) + _digamma(dispersion)
              - np.log(dispersion) + np.log(total)
              - 1.0 + (dispersion + target) / total)
    gradient = np.empty_like(output)
    gradient[:, 0] = d_mu * raw_mu
    gradient[:, 1] = d_rate * raw_rate
    return loss, gradient


def _pinball(output, target, taus):
    levels = np.asarray(taus, dtype=float)
    quantiles = np.empty_like(output)
    quantiles[:, 0] = output[:, 0]
    for index in range(1, quantiles.shape[1]):
        quantiles[:, index] = (quantiles[:, index - 1]
                               + _softplus(output[:, index]))
    error = target[:, None] - quantiles
    loss = np.where(error >= 0.0, levels * error, (levels - 1.0) * error)
    d_quantile = np.where(error >= 0.0, -levels, 1.0 - levels)
    d_quantile = d_quantile / levels.size
    gradient = np.empty_like(output)
    gradient[:, 0] = d_quantile.sum(axis=1)
    for index in range(1, output.shape[1]):
        tail = d_quantile[:, index:].sum(axis=1)
        gradient[:, index] = tail * _sigmoid(output[:, index])
    return loss.mean(axis=1), gradient


def _nll(output, target, variant: str, nu: float, taus):
    if variant in ("deepar", "gaussian_likelihood"):
        return _gaussian(output, target)
    if variant == "student_t":
        return _student(output, target, nu)
    if variant == "negbin_likelihood":
        return _negative_binomial(output, target)
    return _pinball(output, target, taus)


# --------------------------------------------------------------------------- #
# The recurrent cell
# --------------------------------------------------------------------------- #
class _Cell:
    """One tanh layer and one linear head, with hand-written backpropagation."""

    def __init__(self, hidden: int, outputs: int, rng: np.random.Generator):
        spread = 0.5 / np.sqrt(hidden)
        self.parameters = {
            "w_x": rng.normal(0.0, 0.6, (1, hidden)),
            "W_h": rng.normal(0.0, spread, (hidden, hidden)),
            "b_h": np.zeros(hidden),
            "W_o": rng.normal(0.0, spread, (hidden, outputs)),
            "b_o": np.zeros(outputs),
        }
        self.gradients = {key: np.zeros_like(value)
                          for key, value in self.parameters.items()}

    def forward(self, inputs: np.ndarray):
        p = self.parameters
        steps = inputs.shape[0]
        hidden = p["b_h"].size
        states = np.zeros((steps + 1, hidden))
        outputs = np.zeros((steps, p["b_o"].size))
        for step in range(steps):
            states[step + 1] = np.tanh(inputs[step] @ p["w_x"]
                                       + states[step] @ p["W_h"] + p["b_h"])
            outputs[step] = states[step + 1] @ p["W_o"] + p["b_o"]
        return states, outputs

    def backward(self, inputs: np.ndarray, states: np.ndarray,
                 outputs_grad: np.ndarray) -> None:
        p, g = self.parameters, self.gradients
        for key in g:
            g[key][...] = 0.0
        steps = inputs.shape[0]
        g["W_o"] = states[1:].T @ outputs_grad
        g["b_o"] = outputs_grad.sum(axis=0)
        carry = np.zeros(p["b_h"].size)
        for step in range(steps - 1, -1, -1):
            total = outputs_grad[step] @ p["W_o"].T + carry
            delta = total * (1.0 - states[step + 1] ** 2)
            g["w_x"] += inputs[step][:, None] * delta[None, :]
            g["W_h"] += states[step][:, None] @ delta[None, :]
            g["b_h"] += delta
            carry = delta @ p["W_h"].T


class _Adam:
    def __init__(self, parameters, gradients, rate: float, clip: float):
        self.parameters = parameters
        self.gradients = gradients
        self.rate = rate
        self.clip = clip
        self.first = {key: np.zeros_like(value) for key, value in
                      parameters.items()}
        self.second = {key: np.zeros_like(value) for key, value in
                       parameters.items()}
        self.count = 0

    def step(self) -> None:
        self.count += 1
        for key, value in self.parameters.items():
            gradient = np.clip(self.gradients[key], -self.clip, self.clip)
            self.first[key] = 0.9 * self.first[key] + 0.1 * gradient
            self.second[key] = 0.999 * self.second[key] + 0.001 * gradient ** 2
            corrected = self.first[key] / (1.0 - 0.9 ** self.count)
            scale = self.second[key] / (1.0 - 0.999 ** self.count)
            value -= self.rate * corrected / (np.sqrt(scale) + 1e-8)


# --------------------------------------------------------------------------- #
# The public model
# --------------------------------------------------------------------------- #
class DeepAR:
    """Autoregressive recurrent network with a likelihood head.

    Parameters
    ----------
    variant : str
        One of `VARIANTS`. `deepar` and `gaussian_likelihood` share the
        Gaussian head but forecast differently: `deepar` averages Monte Carlo
        trajectories, `gaussian_likelihood` recurses on the mean and builds an
        analytic interval. `student_t` thickens the tail, `negbin_likelihood`
        models counts, and `quantile_output` trains pinball losses directly.
    hidden : int
        Width of the recurrent state.
    horizon : int
        Default forecast length when `steps` is omitted.
    epochs, learning_rate, clip : int, float
        Full-batch Adam over the whole training series, with elementwise
        gradient clipping.
    nu : float
        Degrees of freedom for `student_t`; ignored elsewhere.
    taus : tuple
        Strictly increasing quantile levels, used by `quantile_output`.

    Notes
    -----
    `fit` stores `mean_`, `scale_`, `train_loss_` and `steps_`. Continuous
    variants standardise the series; `negbin_likelihood` keeps the counts on
    their own scale, because a count likelihood has no meaning on a
    standardised target. `predict` returns the median path, `sample` draws
    whole trajectories, and `interval` returns a central band. Sampling is
    reseeded on every call, so repeated calls return identical draws.
    """

    def __init__(self, variant: str = "deepar", *, hidden: int = 16,
                 horizon: int = 12, epochs: int = 300,
                 learning_rate: float = 0.003, clip: float = 5.0,
                 nu: float = 4.0, taus: tuple[float, ...] = TAUS,
                 seed: int = 0):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        for name, value in (("hidden", hidden), ("horizon", horizon),
                            ("epochs", epochs)):
            if not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (("learning_rate", learning_rate), ("clip", clip)):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if variant == "student_t" and (not np.isfinite(nu) or nu <= 2.0):
            raise ValueError("nu must exceed two for a finite variance")
        levels = tuple(float(value) for value in taus)
        if len(levels) < 3:
            raise ValueError("need at least three quantile levels")
        if any(not 0.0 < value < 1.0 for value in levels):
            raise ValueError("quantile levels must be in the open unit interval")
        pairs = zip(levels, levels[1:], strict=False)
        if any(later <= earlier for earlier, later in pairs):
            raise ValueError("quantile levels must be strictly increasing")
        if 0.5 not in levels:
            raise ValueError("the quantile grid must contain the median")
        self.variant = variant
        self.hidden = int(hidden)
        self.horizon = int(horizon)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.clip = float(clip)
        self.nu = float(nu)
        self.taus = levels
        self.seed = int(seed)
        self._fitted = False

    # -- shape checks ----------------------------------------------------- #
    @staticmethod
    def _series(y: np.ndarray) -> np.ndarray:
        values = np.asarray(y, dtype=float)
        if values.ndim != 1:
            raise ValueError("y must be one-dimensional")
        if values.size < 40:
            raise ValueError("y needs at least forty observations")
        if not np.all(np.isfinite(values)):
            raise ValueError("y must be finite")
        return values

    @property
    def _outputs(self) -> int:
        if self.variant == "quantile_output":
            return len(self.taus)
        if self.variant == "student_t":
            return 1
        return 2

    # -- fitting ---------------------------------------------------------- #
    def fit(self, y: np.ndarray) -> DeepAR:
        """Train the recurrent likelihood on one series."""
        values = self._series(y)
        counts = self.variant == "negbin_likelihood"
        if counts:
            if np.any(values < 0.0):
                raise ValueError("negative binomial targets must be counts")
            if np.any(np.abs(values - np.round(values)) > 1e-8):
                raise ValueError("negative binomial targets must be integers")
        spread = float(values.std())
        if not np.isfinite(spread) or spread <= 1e-12:
            raise ValueError("the series must have a non-zero spread")
        self.mean_, self.scale_ = (0.0, 1.0) if counts else (
            float(values.mean()), spread)
        self.input_mean_ = 0.0 if counts else self.mean_
        self.input_scale_ = spread
        inputs = (values - self.input_mean_) / self.input_scale_
        targets = values[1:] if counts else inputs[1:]
        self.steps_ = targets.size
        cell = _Cell(self.hidden, self._outputs,
                     np.random.default_rng(self.seed))
        optimiser = _Adam(cell.parameters, cell.gradients,
                          self.learning_rate, self.clip)
        loss = float("inf")
        for _ in range(self.epochs):
            states, outputs = cell.forward(inputs[:-1, None])
            step_loss, output_grad = _nll(
                outputs, targets, self.variant, self.nu, self.taus)
            loss = float(step_loss.mean())
            cell.backward(inputs[:-1, None], states,
                          output_grad / self.steps_)
            optimiser.step()
        self.cell_ = cell
        self.train_loss_ = loss
        self._fitted = True
        return self

    # -- the forecast recursion ------------------------------------------- #
    def _encode(self, y: np.ndarray) -> np.ndarray:
        values = self._series(y)
        scaled = (values - self.input_mean_) / self.input_scale_
        states, _ = self.cell_.forward(scaled[:-1, None])
        return states[-1]

    def _step(self, state: np.ndarray, previous: float):
        """`previous` lives in model space; counts need rescaling for the cell."""
        p = self.cell_.parameters
        feed = previous
        if self.variant == "negbin_likelihood":
            feed = (previous - self.input_mean_) / self.input_scale_
        state_next = np.tanh(np.array([feed]) @ p["w_x"]
                             + state @ p["W_h"] + p["b_h"])
        return state_next, state_next @ p["W_o"] + p["b_o"]

    def _analytic(self, previous: float, state: np.ndarray, steps: int):
        """Recurse on the conditional mean, accumulating a variance."""
        means = np.empty(steps)
        variances = np.empty(steps)
        total = 0.0
        for step in range(steps):
            state, output = self._step(state, previous)
            if self.variant == "negbin_likelihood":
                mu = np.exp(np.clip(output[0], -12.0, 12.0))
                dispersion = np.exp(np.clip(output[1], -6.0, 6.0))
                variance = mu + mu ** 2 / (dispersion + 1e-8)
            elif self.variant == "student_t":
                mu = output[0]
                variance = self.nu / (self.nu - 2.0)
            else:
                mu = output[0]
                variance = (_softplus(output[1]) + 1e-6) ** 2
            total += variance
            means[step] = mu
            variances[step] = total
            previous = mu
        return means, variances

    def _draw(self, output: np.ndarray, rng: np.random.Generator) -> float:
        if self.variant in ("deepar", "gaussian_likelihood"):
            sigma = _softplus(output[1]) + 1e-6
            return float(output[0] + sigma * rng.normal())
        if self.variant == "student_t":
            return float(output[0] + rng.standard_t(self.nu))
        if self.variant == "negbin_likelihood":
            mu = np.exp(np.clip(output[0], -12.0, 12.0))
            dispersion = np.exp(np.clip(output[1], -6.0, 6.0))
            probability = dispersion / (dispersion + mu + 1e-12)
            return float(rng.negative_binomial(dispersion, probability))
        return float(self._quantile_draw(output, rng))

    def _quantile_curve(self, output: np.ndarray) -> np.ndarray:
        quantiles = np.empty(len(self.taus))
        quantiles[0] = output[0]
        for index in range(1, len(self.taus)):
            quantiles[index] = quantiles[index - 1] + _softplus(output[index])
        return quantiles

    def _quantile_draw(self, output: np.ndarray, rng) -> float:
        levels = np.asarray(self.taus)
        return float(np.interp(rng.random(), levels,
                               self._quantile_curve(output)))

    # -- public prediction ------------------------------------------------ #
    def _prepare(self, y: np.ndarray, steps: int | None):
        horizon = self.horizon if steps is None else int(steps)
        if horizon < 1:
            raise ValueError("steps must be positive")
        values = self._series(y)
        state = self._encode(values)
        if self.variant == "negbin_likelihood":
            previous = float(values[-1])
        else:
            previous = float((values[-1] - self.mean_) / self.scale_)
        return values, state, previous, horizon

    def sample(self, y: np.ndarray, steps: int | None = None,
               n_samples: int = 400) -> np.ndarray:
        """Draw whole trajectories, feeding each draw back into the state."""
        if not self._fitted:
            raise RuntimeError("fit the model before sampling")
        if not isinstance(n_samples, (int, np.integer)) or n_samples < 1:
            raise ValueError("n_samples must be a positive integer")
        values, state, previous, horizon = self._prepare(y, steps)
        rng = np.random.default_rng(self.seed)
        paths = np.empty((n_samples, horizon))
        for index in range(n_samples):
            current_state = state
            current = previous
            for step in range(horizon):
                current_state, output = self._step(current_state, current)
                current = self._draw(output, rng)
                paths[index, step] = current
        return self.mean_ + self.scale_ * paths

    def predict(self, y: np.ndarray, steps: int | None = None) -> np.ndarray:
        """Median path in original units."""
        if not self._fitted:
            raise RuntimeError("fit the model before predicting")
        values, state, previous, horizon = self._prepare(y, steps)
        if self.variant == "deepar":
            return np.median(self.sample(values, horizon), axis=0)
        if self.variant == "quantile_output":
            path = np.empty(horizon)
            median = self.taus.index(0.5)
            for step in range(horizon):
                state, output = self._step(state, previous)
                previous = self._quantile_curve(output)[median]
                path[step] = previous
            return self.mean_ + self.scale_ * path
        means, _ = self._analytic(previous, state, horizon)
        return self.mean_ + self.scale_ * means

    def interval(self, y: np.ndarray, level: float = 0.8,
                 steps: int | None = None) -> np.ndarray:
        """Central `level` band of shape `(steps, 2)` in original units."""
        if not self._fitted:
            raise RuntimeError("fit the model before predicting")
        if not 0.0 < level < 1.0 or not np.isfinite(level):
            raise ValueError("level must lie strictly between zero and one")
        values, state, previous, horizon = self._prepare(y, steps)
        tail = 0.5 * (1.0 - level)
        if self.variant == "gaussian_likelihood":
            means, variances = self._analytic(previous, state, horizon)
            width = NormalDist().inv_cdf(1.0 - tail) * np.sqrt(variances)
            middle = self.mean_ + self.scale_ * means
            return np.column_stack([middle - width, middle + width])
        if self.variant == "quantile_output":
            lines = np.empty((horizon, len(self.taus)))
            for step in range(horizon):
                state, output = self._step(state, previous)
                lines[step] = self._quantile_curve(output)
                previous = lines[step, self.taus.index(0.5)]
            levels = np.asarray(self.taus)
            low = np.array([np.interp(tail, levels, row) for row in lines])
            high = np.array([np.interp(1.0 - tail, levels, row)
                             for row in lines])
            return np.column_stack([self.mean_ + self.scale_ * low,
                                    self.mean_ + self.scale_ * high])
        draws = self.sample(values, horizon)
        return np.column_stack([np.quantile(draws, tail, axis=0),
                                np.quantile(draws, 1.0 - tail, axis=0)])


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _seasonal(n: int, rng: np.random.Generator, outliers: bool = False
              ) -> np.ndarray:
    t = np.arange(n, dtype=float)
    values = (20.0 + 0.03 * t + 3.0 * np.sin(2.0 * np.pi * t / 12.0)
              + 1.2 * np.cos(2.0 * np.pi * t / 4.0)
              + rng.normal(0.0, 0.5, n))
    if outliers:
        values[::7] += 8.0
    return values


def _counts(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n, dtype=float)
    rate = np.exp(1.2 + 0.8 * np.sin(2.0 * np.pi * t / 7.0))
    return rng.poisson(rate).astype(float)


def _mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - forecast)))


def _coverage(actual: np.ndarray, band: np.ndarray) -> float:
    inside = (actual >= band[:, 0]) & (actual <= band[:, 1])
    return float(np.mean(inside))


def demo() -> None:
    split = 348
    rng = np.random.default_rng(0)
    values = _seasonal(360, rng)
    print("deepar. 348 points of trend, season and noise, 12 held out:")
    model = DeepAR("deepar", seed=0).fit(values[:split])
    median = model.predict(values[:split])
    gaussian = DeepAR("gaussian_likelihood", seed=0).fit(values[:split])
    mean_path = gaussian.predict(values[:split])
    seasonal = values[split - 12:split]
    naive = np.repeat(values[split - 1], 12)
    band = model.interval(values[:split], 0.8)
    analytic = gaussian.interval(values[:split], 0.8)
    assert median.shape == (12,)
    assert model.train_loss_ == gaussian.train_loss_
    assert _mae(values[split:], median) < _mae(values[split:], seasonal)
    assert _mae(values[split:], mean_path) < _mae(values[split:], seasonal)
    assert float(np.max(np.abs(median - mean_path))) > 0.0
    print(f"  {model.steps_} one-step transitions, hidden width {model.hidden}")
    print(f"  training NLL {model.train_loss_:.3f}; the two variants share it "
          "exactly")
    print(f"  Monte Carlo median MAE {_mae(values[split:], median):.4f}, "
          f"recursive mean {_mae(values[split:], mean_path):.4f}")
    print(f"  against {_mae(values[split:], seasonal):.4f} for seasonal naive "
          f"and {_mae(values[split:], naive):.4f} for the last value")
    print(f"  80% band: trajectories {float((band[:, 1] - band[:, 0]).mean()):.4f} "
          f"wide, analytic {float((analytic[:, 1] - analytic[:, 0]).mean()):.4f}")

    print("deepar. the band widens with the horizon:")
    widths = band[:, 1] - band[:, 0]
    assert widths[-1] > widths[0]
    assert band[:, 0].size == 12
    print(f"  first step {widths[0]:.4f}, last step {widths[-1]:.4f}; "
          f"coverage {_coverage(values[split:], band):.3f} on 12 points")

    print("negbin_likelihood. a weekly count series:")
    rng = np.random.default_rng(1)
    counts = _counts(360, rng)
    weekly = counts[split - 7:split - 7 + 12]
    poisson = DeepAR("negbin_likelihood", seed=0).fit(counts[:split])
    expected = poisson.predict(counts[:split])
    band = poisson.interval(counts[:split], 0.8)
    assert band[:, 0].min() >= 0.0
    assert _mae(counts[split:], expected) < _mae(counts[split:], weekly)
    print(f"  mean count {counts[:split].mean():.3f}, training NLL "
          f"{poisson.train_loss_:.3f}")
    print(f"  held-out MAE {_mae(counts[split:], expected):.4f} against "
          f"{_mae(counts[split:], weekly):.4f} for the same weekday last week")
    print(f"  80% band covers {_coverage(counts[split:], band):.3f} and never "
          f"goes below zero")

    print("student_t. the same series with injected spikes:")
    rng = np.random.default_rng(2)
    spiky = _seasonal(360, rng, outliers=True)
    robust = DeepAR("student_t", seed=0).fit(spiky[:split])
    normal = DeepAR("gaussian_likelihood", seed=0).fit(spiky[:split])
    robust_error = _mae(spiky[split:], robust.predict(spiky[:split]))
    normal_error = _mae(spiky[split:], normal.predict(spiky[:split]))
    assert robust_error < normal_error
    assert robust.interval(spiky[:split], 0.8)[:, 1].size == 12
    print(f"  held-out MAE {robust_error:.4f} for the t head against "
          f"{normal_error:.4f} for the Gaussian head")
    print(f"  training NLL {robust.train_loss_:.3f}; the t likelihood lives on "
          f"a unit scale")

    print("quantile_output. five pinball losses instead of a likelihood:")
    quantile = DeepAR("quantile_output", seed=0).fit(values[:split])
    middle = quantile.predict(values[:split])
    band = quantile.interval(values[:split], 0.8)
    assert np.all(band[:, 1] >= band[:, 0])
    assert _mae(values[split:], middle) < _mae(values[split:], naive)
    print(f"  held-out MAE {_mae(values[split:], middle):.4f}, band covers "
          f"{_coverage(values[split:], band):.3f} and is "
          f"{float((band[:, 1] - band[:, 0]).mean()):.4f} wide")
    print(f"  the pinball median trails the seasonal naive "
          f"({_mae(values[split:], seasonal):.4f}): five quantiles are less "
          "data-efficient than a shaped likelihood")

    print("  the pieces are what they claim:")
    assert _gammaln(np.array([5.0]))[0] > np.log(24.0) - 1e-9
    assert abs(_digamma(np.array([1.0]))[0] + 0.5772156649) < 1e-6
    shares = quantile.sample(values[:split], n_samples=7)
    assert shares.shape == (7, 12)
    print(f"  gammaln(5) {_gammaln(np.array([5.0]))[0]:.6f}, digamma(1) "
          f"{_digamma(np.array([1.0]))[0]:.6f}")
    print(f"  seven sampled trajectories have shape {shares.shape}")

    bad = [
        lambda: DeepAR("deepar_lite"),
        lambda: DeepAR("deepar", hidden=0),
        lambda: DeepAR("deepar", horizon=0),
        lambda: DeepAR("deepar", epochs=0),
        lambda: DeepAR("deepar", learning_rate=0.0),
        lambda: DeepAR("deepar", learning_rate=np.nan),
        lambda: DeepAR("deepar", clip=0.0),
        lambda: DeepAR("student_t", nu=2.0),
        lambda: DeepAR("student_t", nu=np.nan),
        lambda: DeepAR("quantile_output", taus=(0.5, 0.9)),
        lambda: DeepAR("quantile_output", taus=(0.0, 0.5, 0.9)),
        lambda: DeepAR("quantile_output", taus=(0.9, 0.5, 0.1)),
        lambda: DeepAR("quantile_output", taus=(0.1, 0.2, 0.3)),
        lambda: DeepAR("deepar").fit(np.zeros((60, 2))),
        lambda: DeepAR("deepar").fit(np.arange(30.0)),
        lambda: DeepAR("deepar").fit(np.r_[np.arange(60.0), np.nan]),
        lambda: DeepAR("deepar").fit(np.ones(60)),
        lambda: DeepAR("negbin_likelihood").fit(-np.ones(60)),
        lambda: DeepAR("negbin_likelihood", epochs=1).fit(
            np.abs(np.arange(60.0)) + 0.5),
        lambda: DeepAR("deepar").predict(values),
        lambda: DeepAR("deepar", epochs=1).fit(values).sample(values, steps=0),
        lambda: DeepAR("deepar", epochs=1).fit(values).sample(
            values, n_samples=0),
        lambda: DeepAR("deepar", epochs=1).fit(values).interval(
            values, level=1.0),
        lambda: DeepAR("deepar", epochs=1).fit(values).interval(
            values, level=0.0),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except (ValueError, RuntimeError):
            refused += 1
    print(f"{refused} malformed calls refused out of {len(bad)}: an unknown")
    print("  variant, a non-positive width, horizon, epoch count, learning")
    print("  rate or clip, a degrees-of-freedom value at or below two, fewer")
    print("  than three or non-increasing or out-of-range quantile levels")
    print("  without the median, a two-dimensional series, a short or")
    print("  non-finite series, a constant series, negative or fractional")
    print("  counts, prediction before fitting, a non-positive step count or")
    print("  sample count, and a central level outside the open unit interval")
    print("deepar demo passed")


if __name__ == "__main__":
    demo()
