"""DLinear — decomposition plus a linear map, and why that is hard to beat.

Chapter:   11-time-series
Companion: dlinear.md
Tier 2 · 4 variants: dlinear, nlinear, moving_avg_decomp, channel_independent
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("dlinear", "nlinear", "moving_avg_decomp", "channel_independent")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _check_frame(values: np.ndarray) -> tuple[np.ndarray, bool]:
    values = np.asarray(values, dtype=float)
    if values.ndim not in (1, 2):
        raise ValueError("the series must be one- or two-dimensional")
    if values.ndim == 1:
        return values[:, None], True
    return values, False


def _positive_int(value, name: str) -> int:
    if not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _moving_average(values: np.ndarray, kernel: int) -> np.ndarray:
    """Centred moving average along the last axis, edge-padded."""
    pad = kernel // 2
    padding = [(0, 0)] * (values.ndim - 1) + [(pad, pad)]
    padded = np.pad(values, padding, mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(padded, kernel, axis=-1)
    return view.mean(axis=-1)


def _decompose(values: np.ndarray, kernel: int
               ) -> tuple[np.ndarray, np.ndarray]:
    trend = _moving_average(values, kernel)
    return trend, values - trend


def _windows(values: np.ndarray, seq_len: int, horizon: int
             ) -> tuple[np.ndarray, np.ndarray]:
    count = values.shape[0] - seq_len - horizon + 1
    if count < 2:
        raise ValueError("need at least two training windows")
    starts = np.arange(count)
    left = values[starts[:, None] + np.arange(seq_len)[None, :]]
    right = values[starts[:, None] + seq_len + np.arange(horizon)[None, :]]
    return left, right


def _line_forecast(values: np.ndarray, horizon: int) -> np.ndarray:
    """Least-squares line through each row, evaluated at future indices."""
    length = values.shape[1]
    grid = np.arange(length, dtype=float)
    centred = grid - grid.mean()
    slope = values @ centred / float(centred @ centred)
    intercept = values.mean(axis=1)
    future = length + np.arange(horizon) - grid.mean()
    return intercept[:, None] + slope[:, None] * future[None, :]


class _Ridge:
    """Ridge regression with an unpenalised intercept."""

    def __init__(self, alpha: float):
        self.alpha = float(alpha)

    def fit(self, features: np.ndarray, targets: np.ndarray) -> _Ridge:
        features = np.asarray(features, dtype=float)
        targets = np.asarray(targets, dtype=float)
        if targets.ndim == 1:
            targets = targets[:, None]
        self.mean_ = features.mean(axis=0)
        scale = features.std(axis=0)
        self.scale_ = np.where(scale > 1e-12, scale, 1.0)
        design = np.column_stack(
            [np.ones(features.shape[0]), (features - self.mean_) / self.scale_])
        penalty = self.alpha * np.eye(design.shape[1])
        penalty[0, 0] = 0.0
        gram = design.T @ design + penalty
        moment = design.T @ targets
        try:
            self.coef_ = np.linalg.solve(gram, moment)
        except np.linalg.LinAlgError:
            self.coef_ = np.linalg.lstsq(gram, moment, rcond=None)[0]
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=float)
        design = np.column_stack(
            [np.ones(features.shape[0]), (features - self.mean_) / self.scale_])
        return design @ self.coef_


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
class DLinear:
    """Decomposition-linear forecaster with a ridge-fit linear head.

    Parameters
    ----------
    variant : str
        One of `VARIANTS`.
    seq_len : int, optional
        Look-back window `L`. Defaults to `max(3 * kernel, 2 * horizon + 8)`.
    horizon : int
        Forecast horizon `H`.
    kernel : int
        Odd moving-average window for the trend component.
    ridge : float
        Non-negative ridge penalty on standardised features; the intercept is
        never penalised.

    Notes
    -----
    `fit` accepts a series of shape `(n,)` or `(n, channels)`. `predict` takes
    the most recent `seq_len` observations (or more, in which case the last
    `seq_len` are used) and returns `(horizon,)` or `(horizon, channels)`.
    `decompose` exposes the moving-average split used by the decomposition
    variants.
    """

    def __init__(self, variant: str = "dlinear", *, seq_len: int | None = 48,
                 horizon: int = 12, kernel: int = 13, ridge: float = 1e-3):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        self.variant = variant
        self.seq_len = (None if seq_len is None
                        else _positive_int(seq_len, "seq_len"))
        self.horizon = _positive_int(horizon, "horizon")
        if not isinstance(kernel, (int, np.integer)) or kernel < 3:
            raise ValueError("kernel must be an odd integer of at least three")
        if kernel % 2 == 0:
            raise ValueError("kernel must be odd")
        if not np.isfinite(ridge) or ridge < 0.0:
            raise ValueError("ridge must be finite and non-negative")
        if self.seq_len is not None and kernel > self.seq_len:
            raise ValueError("kernel must not exceed seq_len")
        self.kernel = int(kernel)
        self.ridge = float(ridge)
        self._fitted = False

    # -- fitting ---------------------------------------------------------- #
    def fit(self, values: np.ndarray) -> DLinear:
        """Fit the linear heads on sliding windows of `values`."""
        matrix, single = _check_frame(values)
        if matrix.shape[0] < 8:
            raise ValueError("need at least eight observations")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("the series must be finite")
        self.input_ndim_ = 1 if single else 2
        self.n_channels_ = matrix.shape[1]
        length = (self.seq_len if self.seq_len is not None
                  else max(3 * self.kernel, 2 * self.horizon + 8))
        if self.kernel > length:
            raise ValueError("kernel must not exceed the look-back window")
        self.seq_len_ = int(length)
        left, right = _windows(matrix, self.seq_len_, self.horizon)
        self.n_windows_ = int(left.shape[0])
        channels = self.n_channels_

        if self.variant == "nlinear":
            self.nlinear_ = _Ridge(self.ridge)
            self.nlinear_.fit(*self._normalised(left, right))
        elif self.variant == "channel_independent":
            self.models_ = []
            for channel in range(channels):
                trend, seasonal = _decompose(left[:, :, channel], self.kernel)
                design = np.column_stack([trend, seasonal])
                self.models_.append(
                    _Ridge(self.ridge).fit(design, right[:, :, channel]))
        else:
            flat = left.transpose(0, 2, 1).reshape(-1, self.seq_len_)
            target = right.transpose(0, 2, 1).reshape(-1, self.horizon)
            trend, seasonal = _decompose(flat, self.kernel)
            if self.variant == "dlinear":
                design = np.column_stack([trend, seasonal])
                self.model_ = _Ridge(self.ridge).fit(design, target)
            else:
                self.trend_future_train_ = _line_forecast(trend, self.horizon)
                self.seasonal_model_ = _Ridge(self.ridge).fit(
                    seasonal, target - self.trend_future_train_)
        self._fitted = True
        self.train_loss_ = float(np.mean(
            (self._predict_windows(left) - right) ** 2))
        return self

    def _normalised(self, left: np.ndarray, right: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
        last = left[:, -1, :][:, None, :]
        features = (left - last).transpose(0, 2, 1).reshape(
            -1, self.seq_len_)
        targets = (right - last).transpose(0, 2, 1).reshape(
            -1, self.horizon)
        return features, targets

    # -- prediction ------------------------------------------------------- #
    def _predict_windows(self, left: np.ndarray) -> np.ndarray:
        count, _, channels = left.shape
        if self.variant == "nlinear":
            last = left[:, -1, :][:, None, :]
            features = (left - last).transpose(0, 2, 1).reshape(
                -1, self.seq_len_)
            forecast = self.nlinear_.predict(features)
            forecast = forecast.reshape(count, channels, self.horizon)
            return forecast.transpose(0, 2, 1) + last
        if self.variant == "channel_independent":
            forecast = np.empty((count, self.horizon, channels))
            for channel in range(channels):
                trend, seasonal = _decompose(left[:, :, channel], self.kernel)
                forecast[:, :, channel] = self.models_[channel].predict(
                    np.column_stack([trend, seasonal]))
            return forecast
        flat = left.transpose(0, 2, 1).reshape(-1, self.seq_len_)
        trend, seasonal = _decompose(flat, self.kernel)
        if self.variant == "dlinear":
            forecast = self.model_.predict(
                np.column_stack([trend, seasonal]))
        else:
            forecast = (_line_forecast(trend, self.horizon)
                        + self.seasonal_model_.predict(seasonal))
        return forecast.reshape(count, channels, self.horizon).transpose(0, 2, 1)

    def _require(self, values: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("fit the model before predicting")
        matrix, single = _check_frame(values)
        if single != (self.input_ndim_ == 1):
            raise ValueError("history dimensionality does not match the fit")
        if matrix.shape[1] != self.n_channels_:
            raise ValueError("history has the wrong number of channels")
        if matrix.shape[0] < self.seq_len_:
            raise ValueError("history is shorter than the look-back window")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("history must be finite")
        return matrix

    def predict(self, values: np.ndarray) -> np.ndarray:
        """Forecast `horizon` steps from the last `seq_len` observations."""
        history = self._require(values)[-self.seq_len_:]
        forecast = self._predict_windows(history[None, :, :])[0]
        if self.input_ndim_ == 1:
            return forecast[:, 0]
        return forecast

    def decompose(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the moving-average trend and seasonal residual of a history."""
        history = self._require(values)
        trend = _moving_average(history.T, self.kernel).T
        if self.input_ndim_ == 1:
            return trend[:, 0], (history - trend)[:, 0]
        return trend, history - trend


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _poly_series(n: int, rng: np.random.Generator) -> np.ndarray:
    time = np.arange(n, dtype=float)
    return (10.0 + 0.02 * time
            + 2.0 * np.sin(2.0 * np.pi * time / 24.0)
            + 0.8 * np.cos(4.0 * np.pi * time / 24.0)
            + rng.normal(0.0, 0.3, n))


def _rolling_error(model: DLinear, values: np.ndarray, start: int,
                   horizon: int, seq_len: int) -> float:
    errors = []
    for origin in range(start, values.size - horizon + 1):
        forecast = model.predict(values[origin - seq_len:origin])
        errors.append(np.abs(forecast - values[origin:origin + horizon]))
    return float(np.mean(np.concatenate(errors)))


def _naive_error(values: np.ndarray, start: int, horizon: int) -> float:
    errors = []
    for origin in range(start, values.size - horizon + 1):
        forecast = np.repeat(values[origin - 1], horizon)
        errors.append(np.abs(forecast - values[origin:origin + horizon]))
    return float(np.mean(np.concatenate(errors)))


def _seasonal_error(values: np.ndarray, start: int, horizon: int,
                    period: int) -> float:
    errors = []
    for origin in range(start, values.size - horizon + 1):
        index = origin - period + np.arange(horizon) % period
        errors.append(np.abs(values[index] - values[origin:origin + horizon]))
    return float(np.mean(np.concatenate(errors)))


def demo() -> None:
    rng = np.random.default_rng(0)
    print("dlinear. 720 points, linear trend and two harmonics, horizon 12:")
    values = _poly_series(720, rng)
    split = 600
    model = DLinear("dlinear", seq_len=48, horizon=12, kernel=13)
    model.fit(values[:split])
    error = _rolling_error(model, values, split, 12, 48)
    naive = _naive_error(values, split, 12)
    seasonal = _seasonal_error(values, split, 12, 24)
    assert error < naive
    assert error < seasonal
    assert model.n_windows_ == split - 48 - 12 + 1
    print(f"  {model.n_windows_} training windows, in-sample MSE "
          f"{model.train_loss_:.4f}")
    print(f"  rolling MAE {error:.4f} against {naive:.4f} for naive and "
          f"{seasonal:.4f} for seasonal naive")
    default = DLinear("dlinear", seq_len=None, kernel=13, horizon=12)
    assert default.seq_len is None
    assert default.fit(values).seq_len_ == 39

    print("  the decomposition is exact by construction:")
    trend, residual = model.decompose(values[:120])
    assert np.allclose(trend + residual, values[:120])
    ramp = 4.0 + 0.1 * np.arange(120)
    ramp_trend, _ = model.decompose(ramp)
    assert np.max(np.abs(ramp_trend[20:100] - ramp[20:100])) < 1e-12
    print(f"  trend plus residual reproduces the history; on a pure ramp the "
          f"interior trend error is "
          f"{np.max(np.abs(ramp_trend[20:100] - ramp[20:100])):.2e}")

    rng = np.random.default_rng(1)
    print("nlinear. a +1000 level break after training, horizon 12:")
    time = np.arange(700, dtype=float)
    base = (3.0 + 0.05 * time
            + 1.5 * np.sin(2.0 * np.pi * time / 12.0)
            + rng.normal(0.0, 0.2, time.size))
    shifted = base.copy()
    shifted[500:] += 1000.0
    plain = DLinear("dlinear", seq_len=48, horizon=12, kernel=13)
    plain.fit(base[:480])
    normalised = DLinear("nlinear", seq_len=48, horizon=12, kernel=13)
    normalised.fit(base[:480])
    history = base[452:500]
    plain_shift = plain.predict(history + 1000.0) - plain.predict(history)
    normal_shift = (normalised.predict(history + 1000.0)
                    - normalised.predict(history))
    plain_equivariance = float(np.max(np.abs(plain_shift - 1000.0)))
    normal_equivariance = float(np.max(np.abs(normal_shift - 1000.0)))
    assert normal_equivariance < 1e-10
    assert plain_equivariance > 1.0
    plain_error = _rolling_error(plain, shifted, 548, 12, 48)
    normalised_error = _rolling_error(normalised, shifted, 548, 12, 48)
    assert normalised_error < 0.3
    assert plain_error > 0.5
    print(f"  shift-equivariance error at +1000: {normal_equivariance:.2e} "
          f"for nlinear against {plain_equivariance:.4f} for dlinear")
    print(f"  rolling MAE one window after the break: {normalised_error:.4f} "
          f"for nlinear against {plain_error:.4f} for dlinear")

    rng = np.random.default_rng(2)
    print("moving_avg_decomp. a changing slope, horizon 12:")
    time = np.arange(720, dtype=float)
    trend = np.where(time < 360, 5.0 + 0.02 * time, 12.2 + 0.08 * (time - 360))
    values = (trend + 1.5 * np.sin(2.0 * np.pi * time / 30.0)
              + rng.normal(0.0, 0.2, time.size))
    line = DLinear("moving_avg_decomp", seq_len=60, horizon=12, kernel=25)
    line.fit(values[:600])
    error = _rolling_error(line, values, 600, 12, 60)
    naive = _naive_error(values, 600, 12)
    assert error < naive
    print(f"  rolling MAE {error:.4f} against {naive:.4f} for naive; the "
          f"trend head extrapolates a line instead of learning one")

    rng = np.random.default_rng(3)
    print("channel_independent. two channels with different scales and cycles:")
    time = np.arange(720, dtype=float)
    first = (5.0 + 4.0 * np.sin(2.0 * np.pi * time / 12.0)
             + rng.normal(0.0, 0.3, time.size))
    second = (0.5 + 0.4 * np.sin(2.0 * np.pi * time / 7.0)
              + rng.normal(0.0, 0.05, time.size))
    matrix = np.column_stack([first, second])
    pooled = DLinear("dlinear", seq_len=48, horizon=12, kernel=13)
    pooled.fit(matrix[:600])
    separate = DLinear("channel_independent", seq_len=48, horizon=12,
                       kernel=13)
    separate.fit(matrix[:600])
    pooled_errors = []
    separate_errors = []
    for origin in range(600, matrix.shape[0] - 12 + 1):
        history = matrix[origin - 48:origin]
        pooled_errors.append(np.abs(pooled.predict(history)
                                    - matrix[origin:origin + 12]))
        separate_errors.append(np.abs(separate.predict(history)
                                      - matrix[origin:origin + 12]))
    pooled_error = float(np.mean(np.concatenate(pooled_errors)))
    separate_error = float(np.mean(np.concatenate(separate_errors)))
    assert separate_error < pooled_error
    assert separate.predict(matrix[-48:]).shape == (12, 2)
    print(f"  two-channel MAE {separate_error:.4f} for separate heads against "
          f"{pooled_error:.4f} for one shared head")

    bad = [
        lambda: DLinear("dlinear_lite"),
        lambda: DLinear("dlinear", horizon=0),
        lambda: DLinear("dlinear", horizon=2.5),
        lambda: DLinear("dlinear", seq_len=0),
        lambda: DLinear("dlinear", kernel=4),
        lambda: DLinear("dlinear", kernel=1),
        lambda: DLinear("dlinear", kernel=13, seq_len=12),
        lambda: DLinear("dlinear", ridge=-1.0),
        lambda: DLinear("dlinear", ridge=np.nan),
        lambda: DLinear("dlinear").fit(np.zeros((20, 2, 2))),
        lambda: DLinear("dlinear").fit(np.arange(6.0)),
        lambda: DLinear("dlinear").fit(np.r_[np.zeros(20), np.nan]),
        lambda: DLinear("dlinear", seq_len=48).fit(np.arange(40.0)),
        lambda: DLinear("dlinear").predict(np.arange(48.0)),
        lambda: DLinear("dlinear").fit(np.arange(200.0)).predict(
            np.arange(10.0)),
        lambda: DLinear("dlinear").fit(np.arange(200.0)).predict(
            np.r_[np.arange(47.0), np.nan]),
        lambda: DLinear("dlinear").fit(np.arange(200.0)).predict(
            np.arange(96.0).reshape(48, 2)),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except (ValueError, RuntimeError):
            refused += 1
    print(f"{refused} malformed calls refused out of {len(bad)}: an unknown")
    print("  variant, a zero, fractional or negative horizon, a zero or")
    print("  fractional look-back, an even or too small kernel, a kernel")
    print("  longer than the window, a negative or non-finite ridge, a 3-d")
    print("  series, a series of fewer than eight points or of fewer than two")
    print("  windows, a non-finite series, prediction before fitting, a")
    print("  history that is too short or non-finite, and a history whose")
    print("  channel count disagrees with the fit")
    print("dlinear demo passed")


if __name__ == "__main__":
    demo()
