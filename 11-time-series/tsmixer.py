"""TSMixer — alternate mixing along time and across channels.

Chapter:   11-time-series
Companion: tsmixer.md
Tier 3 · 4 variants: tsmixer, tsmixer_ext, time_mix, feature_mix
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("tsmixer", "tsmixer_ext", "time_mix", "feature_mix")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _check_series(y: np.ndarray) -> np.ndarray:
    values = np.asarray(y, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError("y must be one- or two-dimensional")
    if values.shape[0] < 40:
        raise ValueError("y needs at least forty observations")
    if not np.all(np.isfinite(values)):
        raise ValueError("y must be finite")
    return values


def _layer_norm(x: np.ndarray):
    mean = x.mean(axis=-1, keepdims=True)
    centred = x - mean
    variance = (centred ** 2).mean(axis=-1, keepdims=True)
    inverse = 1.0 / np.sqrt(variance + 1e-5)
    return centred * inverse, (centred, inverse)


def _layer_norm_backward(gradient: np.ndarray, cache) -> np.ndarray:
    centred, inverse = cache
    width = gradient.shape[-1]
    scaled = (gradient * centred).sum(axis=-1, keepdims=True)
    total = gradient.sum(axis=-1, keepdims=True)
    return (inverse / width) * (
        width * gradient - total - centred * inverse ** 2 * scaled)


class _Adam:
    def __init__(self, parameters: dict[str, np.ndarray],
                 learning_rate: float):
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.first = {key: np.zeros_like(value) for key, value in
                      parameters.items()}
        self.second = {key: np.zeros_like(value) for key, value in
                       parameters.items()}
        self.step_count = 0

    def step(self, gradients: dict[str, np.ndarray]) -> None:
        self.step_count += 1
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        for key, value in self.parameters.items():
            gradient = gradients[key]
            self.first[key] = beta1 * self.first[key] + (1.0 - beta1) * gradient
            self.second[key] = (beta2 * self.second[key]
                                + (1.0 - beta2) * gradient ** 2)
            corrected1 = self.first[key] / (1.0 - beta1 ** self.step_count)
            corrected2 = self.second[key] / (1.0 - beta2 ** self.step_count)
            value -= (self.learning_rate * corrected1
                      / (np.sqrt(corrected2) + epsilon))


# --------------------------------------------------------------------------- #
# The mixer stack
# --------------------------------------------------------------------------- #
class _Mixer:
    """Alternating time-axis and channel-axis perceptrons with residuals."""

    def __init__(self, seq_len: int, channels: int, aux: int, horizon: int,
                 hidden: int, blocks: int, time_mix: bool, feature_mix: bool,
                 rng: np.random.Generator):
        self.seq_len = seq_len
        self.channels = channels
        self.aux = aux
        self.horizon = horizon
        self.blocks = blocks
        self.use_time = time_mix
        self.use_feature = feature_mix
        self.parameters: dict[str, np.ndarray] = {}
        time_spread = np.sqrt(2.0 / seq_len)
        feature_spread = np.sqrt(2.0 / (channels + aux))
        for index in range(blocks):
            if time_mix:
                self.parameters[f"T1_{index}"] = rng.normal(
                    0.0, time_spread, (seq_len, hidden))
                self.parameters[f"t1_{index}"] = np.zeros(hidden)
                self.parameters[f"T2_{index}"] = rng.normal(
                    0.0, 1.0 / np.sqrt(hidden), (hidden, seq_len))
                self.parameters[f"t2_{index}"] = np.zeros(seq_len)
            if feature_mix:
                self.parameters[f"F1_{index}"] = rng.normal(
                    0.0, feature_spread, (channels + aux, hidden))
                self.parameters[f"f1_{index}"] = np.zeros(hidden)
                self.parameters[f"F2_{index}"] = rng.normal(
                    0.0, 1.0 / np.sqrt(hidden), (hidden, channels))
                self.parameters[f"f2_{index}"] = np.zeros(channels)
        width = seq_len + (horizon * aux if aux else 0)
        self.parameters["Wo"] = rng.normal(0.0, np.sqrt(1.0 / width),
                                           (width, horizon))
        self.parameters["bo"] = np.zeros(horizon)
        self.gradients = {key: np.zeros_like(value) for key, value in
                          self.parameters.items()}

    def forward(self, x: np.ndarray, aux: np.ndarray | None,
                future: np.ndarray | None) -> tuple[np.ndarray, dict]:
        p = self.parameters
        cache = []
        current = x
        for index in range(self.blocks):
            entry = []
            if self.use_time:
                transposed = current.transpose(0, 2, 1)
                norm, norm_cache = _layer_norm(transposed)
                pre = norm @ p[f"T1_{index}"] + p[f"t1_{index}"]
                active = np.maximum(pre, 0.0)
                mixed = active @ p[f"T2_{index}"] + p[f"t2_{index}"]
                current = current + mixed.transpose(0, 2, 1)
                entry.append({"kind": "time", "norm": norm,
                              "cache": norm_cache, "pre": pre,
                              "active": active})
            if self.use_feature:
                norm, norm_cache = _layer_norm(current)
                joined = (np.concatenate([norm, aux], axis=2)
                          if self.aux else norm)
                pre = joined @ p[f"F1_{index}"] + p[f"f1_{index}"]
                active = np.maximum(pre, 0.0)
                mixed = active @ p[f"F2_{index}"] + p[f"f2_{index}"]
                current = current + mixed
                entry.append({"kind": "feature", "norm": norm,
                              "cache": norm_cache, "pre": pre,
                              "active": active, "joined": joined})
            cache.append(entry)
        flat = current.transpose(0, 2, 1).reshape(-1, self.seq_len)
        if self.aux:
            repeated = np.repeat(future.reshape(x.shape[0], -1),
                                 self.channels, axis=0)
            head = np.concatenate([flat, repeated], axis=1)
        else:
            head = flat
        forecast = (head @ p["Wo"] + p["bo"]).reshape(
            x.shape[0], self.channels, self.horizon)
        return forecast.transpose(0, 2, 1), {"blocks": cache, "head": head}

    def backward(self, gradient: np.ndarray, cache: dict) -> None:
        p = self.parameters
        g = self.gradients
        batch = gradient.shape[0]
        flat_grad = gradient.transpose(0, 2, 1).reshape(-1, self.horizon)
        g["Wo"] = cache["head"].T @ flat_grad
        g["bo"] = flat_grad.sum(axis=0)
        current_grad = (flat_grad @ p["Wo"][:self.seq_len].T).reshape(
            batch, self.channels, self.seq_len).transpose(0, 2, 1)
        for index in range(self.blocks - 1, -1, -1):
            for entry in reversed(cache["blocks"][index]):
                pre = entry["pre"]
                active = entry["active"]
                if entry["kind"] == "feature":
                    width = active.shape[-1]
                    g[f"F2_{index}"] = (active.reshape(-1, width).T
                                        @ current_grad.reshape(-1, self.channels))
                    g[f"f2_{index}"] = current_grad.sum(axis=(0, 1))
                    delta = (current_grad @ p[f"F2_{index}"].T) * (pre > 0.0)
                    joined = entry["joined"]
                    g[f"F1_{index}"] = (joined.reshape(-1, joined.shape[-1]).T
                                        @ delta.reshape(-1, width))
                    g[f"f1_{index}"] = delta.sum(axis=(0, 1))
                    normalised_grad = (delta @ p[f"F1_{index}"].T)[
                        ..., :self.channels]
                    current_grad = current_grad + _layer_norm_backward(
                        normalised_grad, entry["cache"])
                else:
                    transposed_grad = current_grad.transpose(0, 2, 1)
                    width = active.shape[-1]
                    g[f"T2_{index}"] = (active.reshape(-1, width).T
                                        @ transposed_grad.reshape(
                                            -1, self.seq_len))
                    g[f"t2_{index}"] = transposed_grad.sum(axis=(0, 1))
                    delta = (transposed_grad @ p[f"T2_{index}"].T) * (pre > 0.0)
                    g[f"T1_{index}"] = (entry["norm"].reshape(
                        -1, self.seq_len).T @ delta.reshape(-1, width))
                    g[f"t1_{index}"] = delta.sum(axis=(0, 1))
                    normalised_grad = delta @ p[f"T1_{index}"].T
                    current_grad = current_grad + _layer_norm_backward(
                        normalised_grad, entry["cache"]).transpose(0, 2, 1)


# --------------------------------------------------------------------------- #
# The public model
# --------------------------------------------------------------------------- #
class TSMixer:
    """All-MLP forecaster with alternating time and channel mixing.

    Parameters
    ----------
    variant : str
        One of `VARIANTS`. `time_mix` keeps only the time-axis perceptrons,
        `feature_mix` keeps only the channel-axis ones, `tsmixer` uses both,
        and `tsmixer_ext` also consumes exogenous features in the mixing
        branch and in the head.
    seq_len, horizon : int
        Look-back window and forecast horizon.
    blocks : int
        Number of mixing blocks.
    hidden : int
        Hidden width of each perceptron.
    epochs, learning_rate : int, float
        Full-batch Adam schedule.
    seed : int
        Seed for the weight initialisation.

    Notes
    -----
    `fit(y, exog=None)` standardises every channel of every window by its own
    mean and standard deviation. `tsmixer_ext` needs `exog` with `len(y)` rows
    and `predict` needs the `horizon` future rows. `fit` stores `train_loss_`,
    `n_windows_`, `channels_` and `aux_`.
    """

    def __init__(self, variant: str = "tsmixer", *, seq_len: int = 48,
                 horizon: int = 12, blocks: int = 2, hidden: int = 32,
                 epochs: int = 120, learning_rate: float = 0.01,
                 seed: int = 0):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        for name, value in (("seq_len", seq_len), ("horizon", horizon),
                            ("blocks", blocks), ("hidden", hidden),
                            ("epochs", epochs)):
            if not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive and finite")
        if not isinstance(seed, (int, np.integer)):
            raise ValueError("seed must be an integer")
        self.variant = variant
        self.seq_len = int(seq_len)
        self.horizon = int(horizon)
        self.blocks = int(blocks)
        self.hidden = int(hidden)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.seed = int(seed)
        self.use_time = variant != "feature_mix"
        self.use_feature = variant != "time_mix"
        self._fitted = False

    @staticmethod
    def _windows(series: np.ndarray, seq_len: int, horizon: int
                 ) -> tuple[np.ndarray, np.ndarray]:
        count = series.shape[0] - seq_len - horizon + 1
        if count < 8:
            raise ValueError("need at least eight training windows")
        starts = np.arange(count)
        left = series[starts[:, None] + np.arange(seq_len)[None, :], :]
        right = series[starts[:, None] + seq_len + np.arange(horizon)[None, :],
                       :]
        return left, right

    # -- fitting ---------------------------------------------------------- #
    def fit(self, y: np.ndarray, exog: np.ndarray | None = None) -> TSMixer:
        """Train on sliding windows of `y`."""
        series = _check_series(y)
        if self.variant == "tsmixer_ext" and exog is None:
            raise ValueError("tsmixer_ext needs exogenous features")
        if self.variant != "tsmixer_ext" and exog is not None:
            raise ValueError("exogenous features are only used by tsmixer_ext")
        aux_features = 0
        exog_windows = None
        future = None
        if self.variant == "tsmixer_ext":
            exog = np.asarray(exog, dtype=float)
            if exog.ndim != 2 or exog.shape[0] != series.shape[0]:
                raise ValueError("exog must be two-dimensional with len(y) rows")
            if not np.all(np.isfinite(exog)):
                raise ValueError("exog must be finite")
            self.exog_mean_ = exog.mean(axis=0)
            spread = exog.std(axis=0)
            self.exog_scale_ = np.where(spread > 1e-12, spread, 1.0)
            standardised = (exog - self.exog_mean_) / self.exog_scale_
            count = series.shape[0] - self.seq_len - self.horizon + 1
            starts = np.arange(count)
            exog_windows = standardised[
                starts[:, None] + np.arange(self.seq_len)[None, :], :]
            future = standardised[
                starts[:, None] + self.seq_len
                + np.arange(self.horizon)[None, :], :]
            aux_features = exog.shape[1]
        left, right = self._windows(series, self.seq_len, self.horizon)
        mean = left.mean(axis=1, keepdims=True)
        scale = left.std(axis=1, keepdims=True)
        scale = np.where(scale > 1e-8, scale, 1.0)
        inputs = (left - mean) / scale
        targets = (right - mean) / scale
        self.n_windows_ = int(inputs.shape[0])
        self.channels_ = series.shape[1]
        self.aux_ = aux_features
        mixer = _Mixer(self.seq_len, self.channels_, aux_features,
                       self.horizon, self.hidden, self.blocks, self.use_time,
                       self.use_feature, np.random.default_rng(self.seed))
        optimiser = _Adam(mixer.parameters, self.learning_rate)
        loss = float("inf")
        for _ in range(self.epochs):
            forecast, cache = mixer.forward(inputs, exog_windows, future)
            error = forecast - targets
            loss = float(np.mean(error ** 2))
            mixer.backward(2.0 * error / (error.size), cache)
            optimiser.step(mixer.gradients)
        self.mixer_ = mixer
        self.train_loss_ = loss
        self._fitted = True
        return self

    # -- prediction ------------------------------------------------------- #
    def predict(self, y: np.ndarray,
                exog: np.ndarray | None = None) -> np.ndarray:
        """Forecast `horizon` steps from the last `seq_len` observations."""
        if not self._fitted:
            raise RuntimeError("fit the model before predicting")
        series = _check_series(y)
        if series.shape[1] != self.channels_:
            raise ValueError("y has a different number of channels than fit")
        if series.shape[0] < self.seq_len + self.horizon:
            raise ValueError("history is too short for the fitted model")
        aux = None
        future = None
        if self.variant == "tsmixer_ext":
            if exog is None:
                raise ValueError("tsmixer_ext needs future exogenous features")
            exog = np.asarray(exog, dtype=float)
            if exog.ndim != 2 or exog.shape[1] != self.aux_:
                raise ValueError("exog must have the fitted number of columns")
            if exog.shape[0] < self.seq_len + self.horizon:
                raise ValueError("exog is too short for the fitted model")
            standardised = (exog - self.exog_mean_) / self.exog_scale_
            aux = standardised[-self.seq_len - self.horizon:-self.horizon]
            future = standardised[-self.horizon:]
            aux = aux[None, :, :]
            future = future[None, :, :]
        window = series[-self.seq_len:, :]
        mean = window.mean(axis=0, keepdims=True)
        spread = window.std(axis=0, keepdims=True)
        spread = np.where(spread > 1e-8, spread, 1.0)
        normalised = ((window - mean) / spread)[None, :, :]
        forecast, _ = self.mixer_.forward(normalised, aux, future)
        out = mean + spread * forecast[0]
        return out[:, 0] if out.shape[1] == 1 else out


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _multivariate(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n, dtype=float)
    base = 10.0 + 0.02 * t + 2.0 * np.sin(2.0 * np.pi * t / 12.0)
    base = base + rng.normal(0.0, 0.2, n)
    second = 0.5 * base + 1.5 * np.sin(2.0 * np.pi * t / 12.0 + 1.0)
    second = second + rng.normal(0.0, 0.2, n)
    third = -0.8 * base + 3.0 + rng.normal(0.0, 0.2, n)
    return np.column_stack([base, second, third])


def _mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - forecast)))


def demo() -> None:
    rng = np.random.default_rng(0)
    print("tsmixer. three channels, 360 points, 12 held out:")
    values = _multivariate(360, rng)
    split = 348
    model = TSMixer("tsmixer", seq_len=48, horizon=12, seed=0).fit(
        values[:split])
    forecast = model.predict(values[:split])
    naive = np.repeat(values[split - 1][None, :], 12, axis=0)
    assert forecast.shape == (12, 3)
    assert model.train_loss_ < 0.2
    assert _mae(values[split:], forecast) < _mae(values[split:], naive)
    print(f"  {model.n_windows_} windows, {model.channels_} channels, "
          f"{model.blocks} blocks of width {model.hidden}")
    print(f"  train MSE {model.train_loss_:.4f}, held-out MAE "
          f"{_mae(values[split:], forecast):.4f} against "
          f"{_mae(values[split:], naive):.4f} for the last-value forecast")

    print("time_mix. only the time-axis perceptrons:")
    time_only = TSMixer("time_mix", seq_len=48, horizon=12, seed=0)
    time_only.fit(values[:split])
    time_error = _mae(values[split:], time_only.predict(values[:split]))
    print(f"  train MSE {time_only.train_loss_:.4f}, held-out MAE "
          f"{time_error:.4f}")

    print("feature_mix. only the channel-axis perceptrons:")
    feature_only = TSMixer("feature_mix", seq_len=48, horizon=12, seed=0)
    feature_only.fit(values[:split])
    feature_error = _mae(values[split:], feature_only.predict(values[:split]))
    print(f"  train MSE {feature_only.train_loss_:.4f}, held-out MAE "
          f"{feature_error:.4f}")
    assert time_error < _mae(values[split:], naive)
    assert feature_error < _mae(values[split:], naive)

    print("  the variants differ only in which matrices they own:")
    both = set(model.mixer_.parameters)
    time_keys = set(time_only.mixer_.parameters)
    feature_keys = set(feature_only.mixer_.parameters)
    assert not any(key.startswith("F") for key in time_keys)
    assert not any(key.startswith("T") for key in feature_keys)
    assert time_keys | feature_keys == both
    print(f"  both {len(both)} arrays, time only {len(time_keys)}, feature "
          f"only {len(feature_keys)}; the head and the biases are shared")

    rng = np.random.default_rng(1)
    print("tsmixer_ext. an unpredictable driver that is known in advance:")
    t = np.arange(360, dtype=float)
    driver = rng.normal(0.0, 1.0, t.size)
    driven = (10.0 + 0.02 * t + 2.0 * np.sin(2.0 * np.pi * t / 12.0)
              + 3.0 * driver + rng.normal(0.0, 0.2, t.size))
    exog = driver[:, None]
    plain = TSMixer("tsmixer", seq_len=48, horizon=12, seed=0)
    plain.fit(driven[:split])
    augmented = TSMixer("tsmixer_ext", seq_len=48, horizon=12, seed=0)
    augmented.fit(driven[:split], exog[:split])
    plain_error = _mae(driven[split:], plain.predict(driven[:split]))
    augmented_error = _mae(driven[split:],
                           augmented.predict(driven[:split], exog[:split]))
    assert augmented_error < plain_error
    print(f"  held-out MAE {augmented_error:.4f} against {plain_error:.4f} "
          f"without the driver")
    print(f"  head width {augmented.mixer_.parameters['Wo'].shape[0]} = "
          f"{augmented.seq_len} look-back plus {augmented.horizon} driver "
          f"steps")

    print("  all three beat the last value, and one head serves every channel:")
    print(f"  time-only MAE {time_error:.4f}, feature-only "
          f"{feature_error:.4f}, both {_mae(values[split:], forecast):.4f}")
    print(f"  one head maps {model.seq_len} look-back steps to "
          f"{model.horizon} forecast steps for all {model.channels_} channels")

    bad = [
        lambda: TSMixer("tsmixer_lite"),
        lambda: TSMixer("tsmixer", seq_len=0),
        lambda: TSMixer("tsmixer", horizon=0),
        lambda: TSMixer("tsmixer", blocks=0),
        lambda: TSMixer("tsmixer", hidden=0),
        lambda: TSMixer("tsmixer", epochs=0),
        lambda: TSMixer("tsmixer", learning_rate=0.0),
        lambda: TSMixer("tsmixer", learning_rate=np.nan),
        lambda: TSMixer("tsmixer").fit(np.zeros((40, 2, 2))),
        lambda: TSMixer("tsmixer").fit(np.arange(30.0)),
        lambda: TSMixer("tsmixer").fit(np.r_[np.arange(60.0), np.nan]),
        lambda: TSMixer("tsmixer_ext").fit(values[:split]),
        lambda: TSMixer("tsmixer").fit(values[:split],
                                       np.ones((split, 1))),
        lambda: TSMixer("tsmixer").predict(values[:split]),
        lambda: TSMixer("tsmixer", seq_len=48, epochs=1).fit(
            values[:split]).predict(values[-20:]),
        lambda: TSMixer("tsmixer", seq_len=48, epochs=1).fit(
            values[:split]).predict(values[:split, :1]),
        lambda: TSMixer("tsmixer_ext", seq_len=48, epochs=1).fit(
            values[:split], np.ones((split, 1))).predict(values[:split]),
        lambda: TSMixer("tsmixer_ext", seq_len=48, epochs=1).fit(
            values[:split], np.ones((split, 2))).predict(
                values[:split], np.ones((split, 1))),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except (ValueError, RuntimeError):
            refused += 1
    print(f"{refused} malformed calls refused out of {len(bad)}: an unknown")
    print("  variant, a non-positive window, horizon, block count, hidden")
    print("  width or epoch count, a non-finite learning rate, a")
    print("  three-dimensional series, a short or non-finite series, missing")
    print("  exog for tsmixer_ext, exog passed to another variant, prediction")
    print("  before fitting, a history too short for the window plus horizon,")
    print("  a channel count that differs from the fitted one, tsmixer_ext")
    print("  prediction without exog, and an exog column count that differs")
    print("  from the fitted one")
    print("tsmixer demo passed")


if __name__ == "__main__":
    demo()
