"""PatchTST — patch the window, attend within it, predict channel by channel.

Chapter:   11-time-series
Companion: patchtst.md
Tier 3 · 4 variants: patchtst, channel_independent, supervised, self_supervised
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("patchtst", "channel_independent", "supervised", "self_supervised")


# --------------------------------------------------------------------------- #
# Small building blocks
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


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=-1, keepdims=True)


def _split_heads(x: np.ndarray, heads: int) -> np.ndarray:
    batch, length, dim = x.shape
    return x.reshape(batch, length, heads, dim // heads).transpose(0, 2, 1, 3)


def _merge_heads(x: np.ndarray, dim: int) -> np.ndarray:
    batch, heads, length, width = x.shape
    return x.transpose(0, 2, 1, 3).reshape(batch, length, dim)


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
# One transformer encoder layer
# --------------------------------------------------------------------------- #
class _Encoder:
    """Post-norm single-head-block transformer with multi-head attention."""

    def __init__(self, dim: int, heads: int, hidden: int,
                 rng: np.random.Generator):
        self.dim = dim
        self.heads = heads
        self.width = dim // heads
        spread = np.sqrt(2.0 / dim)
        self.parameters = {
            "Wq": rng.normal(0.0, spread, (dim, dim)),
            "bq": np.zeros(dim),
            "Wk": rng.normal(0.0, spread, (dim, dim)),
            "Wv": rng.normal(0.0, spread, (dim, dim)),
            "bv": np.zeros(dim),
            "Wo": rng.normal(0.0, spread, (dim, dim)),
            "bo": np.zeros(dim),
            "W1": rng.normal(0.0, spread, (dim, hidden)),
            "b1": np.zeros(hidden),
            "W2": rng.normal(0.0, 1.0 / np.sqrt(hidden), (hidden, dim)),
            "b2": np.zeros(dim),
        }
        self.gradients = {key: np.zeros_like(value) for key, value in
                          self.parameters.items()}

    def forward(self, x: np.ndarray):
        p = self.parameters
        normalised, norm_cache = _layer_norm(x)
        query = normalised @ p["Wq"] + p["bq"]
        key = normalised @ p["Wk"]
        value = normalised @ p["Wv"] + p["bv"]
        qh = _split_heads(query, self.heads)
        kh = _split_heads(key, self.heads)
        vh = _split_heads(value, self.heads)
        scores = qh @ kh.transpose(0, 1, 3, 2) / np.sqrt(self.width)
        attention = _softmax(scores)
        mixed = attention @ vh
        joined = _merge_heads(mixed, self.dim)
        attended = joined @ p["Wo"] + p["bo"]
        first = x + attended
        normalised2, norm2_cache = _layer_norm(first)
        hidden = normalised2 @ p["W1"] + p["b1"]
        active = np.maximum(hidden, 0.0)
        output = first + active @ p["W2"] + p["b2"]
        cache = (normalised, norm_cache, qh, kh, vh, attention, mixed, joined,
                 first, normalised2, norm2_cache, hidden, active)
        return output, cache

    def backward(self, gradient: np.ndarray, cache) -> np.ndarray:
        p = self.parameters
        (normalised, norm_cache, qh, kh, vh, attention, mixed, joined,
         first, normalised2, norm2_cache, hidden, active) = cache
        g = self.gradients
        batch, length, dim = gradient.shape
        # feed-forward branch of the second residual
        flat = gradient.reshape(-1, dim)
        g["W2"] = active.reshape(-1, active.shape[-1]).T @ flat
        g["b2"] = flat.sum(axis=0)
        hidden_grad = (flat @ p["W2"].T).reshape(batch, length, -1)
        hidden_grad = hidden_grad * (hidden > 0.0)
        wide = hidden_grad.reshape(-1, hidden_grad.shape[-1])
        g["W1"] = normalised2.reshape(-1, dim).T @ wide
        g["b1"] = hidden_grad.sum(axis=(0, 1))
        norm2_grad = hidden_grad @ p["W1"].T
        first_grad = gradient + _layer_norm_backward(norm2_grad, norm2_cache)
        # attention branch
        joined_grad = first_grad.reshape(-1, dim)
        g["Wo"] = joined.reshape(-1, dim).T @ joined_grad
        g["bo"] = joined_grad.sum(axis=0)
        mixed_grad = joined_grad @ p["Wo"].T
        mixed_grad = mixed_grad.reshape(batch, length, self.heads,
                                        self.width).transpose(0, 2, 1, 3)
        attention_grad = mixed_grad @ vh.transpose(0, 1, 3, 2)
        value_grad = attention.transpose(0, 1, 3, 2) @ mixed_grad
        centre = (attention_grad * attention).sum(axis=-1, keepdims=True)
        scores_grad = attention * (attention_grad - centre) / np.sqrt(self.width)
        query_grad = scores_grad @ kh
        key_grad = scores_grad.transpose(0, 1, 3, 2) @ qh
        query_grad = _merge_heads(query_grad, dim).reshape(-1, dim)
        key_grad = _merge_heads(key_grad, dim).reshape(-1, dim)
        value_grad = _merge_heads(value_grad, dim).reshape(-1, dim)
        g["Wq"] = normalised.reshape(-1, dim).T @ query_grad
        g["bq"] = query_grad.sum(axis=0)
        g["Wk"] = normalised.reshape(-1, dim).T @ key_grad
        g["Wv"] = normalised.reshape(-1, dim).T @ value_grad
        g["bv"] = value_grad.sum(axis=0)
        norm_grad = (query_grad @ p["Wq"].T
                     + key_grad @ p["Wk"].T
                     + value_grad @ p["Wv"].T).reshape(batch, length, dim)
        return first_grad + _layer_norm_backward(norm_grad, norm_cache)


# --------------------------------------------------------------------------- #
# Patching
# --------------------------------------------------------------------------- #
def _patch_count(seq_len: int, patch: int, stride: int) -> int:
    return (seq_len - patch) // stride + 1


def _patches(windows: np.ndarray, patch: int, stride: int) -> np.ndarray:
    count = windows.shape[0]
    number = _patch_count(windows.shape[1], patch, stride)
    starts = np.arange(number) * stride
    index = starts[:, None] + np.arange(patch)[None, :]
    return windows[:, index].reshape(count, number, patch)


def _windows(series: np.ndarray, seq_len: int, horizon: int
             ) -> tuple[np.ndarray, np.ndarray]:
    count = series.shape[0] - seq_len - horizon + 1
    if count < 8:
        raise ValueError("need at least eight training windows")
    starts = np.arange(count)
    left = series[starts[:, None] + np.arange(seq_len)[None, :], :]
    right = series[starts[:, None] + seq_len + np.arange(horizon)[None, :], :]
    return left, right


def _standardise(left: np.ndarray, right: np.ndarray):
    mean = left.mean(axis=1, keepdims=True)
    scale = left.std(axis=1, keepdims=True)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return (left - mean) / scale, (right - mean) / scale, mean, scale


# --------------------------------------------------------------------------- #
# The public model
# --------------------------------------------------------------------------- #
class PatchTST:
    """Patching transformer with channel independence and masked pretraining.

    Parameters
    ----------
    variant : str
        One of `VARIANTS`. `patchtst` trains from scratch on one channel;
        `channel_independent` does the same per channel with shared weights;
        `supervised` pretrains with masked patches and then fine-tunes every
        weight on the forecast; `self_supervised` pretrains and freezes the
        representation, fitting only the linear head.
    seq_len, horizon : int
        Look-back window and forecast horizon.
    patch, stride : int
        Patch length and step between consecutive patches.
    dim, heads, hidden : int
        Embedding width, attention heads and feed-forward width.
    epochs, pretrain_epochs : int
        Forecast training epochs and masked-patch pretraining epochs.
    learning_rate : float
        Adam step size.
    mask_ratio : float
        Fraction of patches hidden during pretraining.
    seed : int
        Seed for the weights and the mask draws.

    Notes
    -----
    `fit` stores `n_patches_`, `n_windows_`, `train_loss_`, `pretrain_loss_`
    and `channels_`. For the pretrained variants, `encoder_snapshot_` holds a
    copy of the encoder weights taken after pretraining, so the demo can show
    which variants moved them.
    """

    def __init__(self, variant: str = "patchtst", *, seq_len: int = 48,
                 horizon: int = 12, patch: int = 12, stride: int = 12,
                 dim: int = 16, heads: int = 2, hidden: int = 32,
                 epochs: int = 400, pretrain_epochs: int = 120,
                 learning_rate: float = 0.01, mask_ratio: float = 0.3,
                 seed: int = 0):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        for name, value in (("seq_len", seq_len), ("horizon", horizon),
                            ("patch", patch), ("stride", stride),
                            ("dim", dim), ("heads", heads), ("hidden", hidden),
                            ("epochs", epochs),
                            ("pretrain_epochs", pretrain_epochs)):
            if not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if patch > seq_len:
            raise ValueError("patch cannot exceed the look-back window")
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive and finite")
        if not 0.0 < mask_ratio < 1.0 or not np.isfinite(mask_ratio):
            raise ValueError("mask_ratio must lie strictly between zero and one")
        self.variant = variant
        self.seq_len = int(seq_len)
        self.horizon = int(horizon)
        self.patch = int(patch)
        self.stride = int(stride)
        self.dim = int(dim)
        self.heads = int(heads)
        self.hidden = int(hidden)
        self.epochs = int(epochs)
        self.pretrain_epochs = int(pretrain_epochs)
        self.learning_rate = float(learning_rate)
        self.mask_ratio = float(mask_ratio)
        self.seed = int(seed)
        self._fitted = False

    # -- pieces ----------------------------------------------------------- #
    def _build(self, channels: int) -> None:
        rng = np.random.default_rng(self.seed)
        self.n_patches_ = _patch_count(self.seq_len, self.patch, self.stride)
        self.channels_ = channels
        spread = np.sqrt(2.0 / self.patch)
        self.head_parameters = {
            "Wh": rng.normal(0.0, spread,
                             (self.horizon, self.n_patches_ * self.dim)),
            "bh": np.zeros(self.horizon),
        }
        self.reconstruction_parameters = {
            "Wr": rng.normal(0.0, spread, (self.patch, self.dim)),
            "br": np.zeros(self.patch),
        }
        self.embed_parameters = {
            "Wp": rng.normal(0.0, spread, (self.dim, self.patch)),
            "bp": np.zeros(self.dim),
            "pos": rng.normal(0.0, 0.02, (self.n_patches_, self.dim)),
        }
        self.encoder = _Encoder(self.dim, self.heads, self.hidden, rng)
        self._rng = np.random.default_rng(self.seed + 1)

    def _encode(self, patches: np.ndarray):
        p = self.embed_parameters
        embedded = patches @ p["Wp"].T + p["bp"] + p["pos"]
        output, cache = self.encoder.forward(embedded)
        return output, (patches, cache)

    def _encode_backward(self, gradient: np.ndarray, cache,
                         grads: dict[str, np.ndarray]) -> None:
        patches, encoder_cache = cache
        embedded_grad = self.encoder.backward(gradient, encoder_cache)
        grads["Wp"] = embedded_grad.reshape(-1, self.dim).T @ patches.reshape(
            -1, self.patch)
        grads["bp"] = embedded_grad.sum(axis=(0, 1))
        grads["pos"] = embedded_grad.sum(axis=0)

    # -- masked pretraining ----------------------------------------------- #
    def _pretrain(self, patches: np.ndarray) -> None:
        parameters = {**self.embed_parameters, **self.encoder.parameters,
                      **self.reconstruction_parameters}
        combined = _Adam(parameters, self.learning_rate)
        batch = patches.shape[0]
        count = self.n_patches_
        last = float("inf")
        for _ in range(self.pretrain_epochs):
            mask = self._rng.random((batch, count)) < self.mask_ratio
            empty = ~mask.any(axis=1)
            if empty.any():
                columns = self._rng.integers(0, count, size=int(empty.sum()))
                mask[empty, columns] = True
            visible = np.where(mask[:, :, None], 0.0, patches)
            encoded, cache = self._encode(visible)
            reconstructed = (encoded @ self.reconstruction_parameters["Wr"].T
                             + self.reconstruction_parameters["br"])
            error = reconstructed - patches
            last = float(np.mean((error[mask]) ** 2))
            scale = max(int(mask.sum()) * self.patch, 1)
            output_grad = 2.0 * error * mask[:, :, None] / scale
            grads = {}
            grads["Wr"] = output_grad.reshape(-1, self.patch).T @ encoded.reshape(
                -1, self.dim)
            grads["br"] = output_grad.sum(axis=(0, 1))
            encoded_grad = output_grad @ self.reconstruction_parameters["Wr"]
            self._encode_backward(encoded_grad, cache, grads)
            grads.update(self.encoder.gradients)
            combined.step(grads)
        self.pretrain_loss_ = last

    # -- forecast training ------------------------------------------------ #
    def _train(self, patches: np.ndarray, targets: np.ndarray,
               train_encoder: bool) -> float:
        parameters = {**self.head_parameters}
        if train_encoder:
            parameters = {**parameters, **self.embed_parameters,
                          **self.encoder.parameters}
        combined = _Adam(parameters, self.learning_rate)
        batch = patches.shape[0]
        loss = float("inf")
        for _ in range(self.epochs):
            encoded, cache = self._encode(patches)
            flat = encoded.reshape(batch, -1)
            forecast = flat @ self.head_parameters["Wh"].T \
                + self.head_parameters["bh"]
            loss = float(np.mean((forecast - targets) ** 2))
            output_grad = 2.0 * (forecast - targets) / (batch * self.horizon)
            grads = {}
            grads["Wh"] = output_grad.T @ flat
            grads["bh"] = output_grad.sum(axis=0)
            if train_encoder:
                encoded_grad = (output_grad @ self.head_parameters["Wh"]
                                ).reshape(batch, self.n_patches_, self.dim)
                self._encode_backward(encoded_grad, cache, grads)
                grads.update(self.encoder.gradients)
            combined.step(grads)
        return loss

    # -- fitting ---------------------------------------------------------- #
    def fit(self, y: np.ndarray) -> PatchTST:
        """Train on sliding windows of `y`."""
        series = _check_series(y)
        if series.shape[1] > 1 and self.variant != "channel_independent":
            raise ValueError("only channel_independent accepts several channels")
        left, right = _windows(series, self.seq_len, self.horizon)
        left, right, self.mean_, self.scale_ = _standardise(left, right)
        windows = left.transpose(0, 2, 1).reshape(-1, self.seq_len)
        targets = right.transpose(0, 2, 1).reshape(-1, self.horizon)
        self._build(series.shape[1])
        self.n_windows_ = int(windows.shape[0])
        patches = _patches(windows, self.patch, self.stride)
        self.pretrain_loss_ = float("nan")
        if self.variant in ("supervised", "self_supervised"):
            self._pretrain(patches)
            self.encoder_snapshot_ = {
                **{f"encoder.{key}": value.copy() for key, value in
                   self.encoder.parameters.items()},
                **{f"embed.{key}": value.copy() for key, value in
                   self.embed_parameters.items()},
            }
        self.train_loss_ = self._train(
            patches, targets, train_encoder=self.variant != "self_supervised")
        self._fitted = True
        return self

    # -- prediction ------------------------------------------------------- #
    def predict(self, y: np.ndarray) -> np.ndarray:
        """Forecast `horizon` steps from the last `seq_len` observations."""
        if not self._fitted:
            raise RuntimeError("fit the model before predicting")
        series = _check_series(y)
        if series.shape[1] != self.channels_:
            raise ValueError("y has a different number of channels than fit")
        if series.shape[0] < self.seq_len:
            raise ValueError("history is shorter than the look-back window")
        window = series[-self.seq_len:, :]
        mean = window.mean(axis=0, keepdims=True)
        spread = window.std(axis=0, keepdims=True)
        spread = np.where(spread > 1e-8, spread, 1.0)
        normalised = ((window - mean) / spread).T
        patches = _patches(normalised, self.patch, self.stride)
        encoded, _ = self._encode(patches)
        flat = encoded.reshape(-1, self.n_patches_ * self.dim)
        forecast = (flat @ self.head_parameters["Wh"].T
                    + self.head_parameters["bh"]).T
        out = mean + spread * forecast
        return out[:, 0] if out.shape[1] == 1 else out


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
    print("patchtst. 360 points, trend and two harmonics, 12 held out:")
    values = _series(360, rng)
    split = 348
    model = PatchTST("patchtst", seq_len=48, horizon=12, seed=0)
    model.fit(values[:split])
    forecast = model.predict(values[:split])
    naive = np.repeat(values[split - 1], 12)
    seasonal = values[split - 12:split]
    assert model.n_patches_ == 4
    assert model.train_loss_ < 0.2
    assert _mae(values[split:], forecast) < _mae(values[split:], seasonal)
    print(f"  {model.n_windows_} windows, {model.n_patches_} patches of "
          f"{model.patch} from a {model.seq_len}-step look-back")
    print(f"  train MSE {model.train_loss_:.4f}, held-out MAE "
          f"{_mae(values[split:], forecast):.4f} against "
          f"{_mae(values[split:], naive):.4f} for naive and "
          f"{_mae(values[split:], seasonal):.4f} for seasonal naive")

    print("channel_independent. two channels, one set of weights:")
    rng = np.random.default_rng(1)
    t = np.arange(360, dtype=float)
    first = _series(360, rng)
    second = 3.0 * first - 5.0 + rng.normal(0.0, 0.1, t.size)
    stacked = np.column_stack([first, second])
    multi = PatchTST("channel_independent", seq_len=48, horizon=12, seed=0)
    multi.fit(stacked[:split])
    output = multi.predict(stacked[:split])
    swapped = multi.predict(stacked[:split][:, ::-1])
    assert output.shape == (12, 2)
    assert float(np.max(np.abs(swapped[:, ::-1] - output))) < 1e-12
    error = float(np.mean(np.abs(output - stacked[split:])))
    print(f"  forecast shape {output.shape}, joint MAE {error:.4f}")
    print(f"  swapping the channels swaps the forecasts, maximum difference "
          f"{float(np.max(np.abs(swapped[:, ::-1] - output))):.1e}")

    print("supervised. masked pretraining, then every weight is fine-tuned:")
    fine = PatchTST("supervised", seq_len=48, horizon=12, seed=0)
    fine.fit(values[:split])
    moved = max(float(np.max(np.abs(fine.encoder.parameters[key]
                                    - fine.encoder_snapshot_[f"encoder.{key}"])))
                for key in fine.encoder.parameters)
    assert moved > 0.0
    fine_error = _mae(values[split:], fine.predict(values[:split]))
    print(f"  pretraining reconstruction MSE {fine.pretrain_loss_:.4f}, "
          f"forecast MSE {fine.train_loss_:.4f}")
    print(f"  held-out MAE {fine_error:.4f} against "
          f"{_mae(values[split:], forecast):.4f} for training from scratch")
    print(f"  fine-tuning moved the encoder by {moved:.4f}")

    print("self_supervised. pretrain, then freeze and fit only the head:")
    probe = PatchTST("self_supervised", seq_len=48, horizon=12, seed=0)
    probe.fit(values[:split])
    frozen = probe.encoder_snapshot_
    difference = max(float(np.max(np.abs(probe.encoder.parameters[key]
                                        - frozen[f"encoder.{key}"])))
                     for key in probe.encoder.parameters)
    assert difference == 0.0
    probe_error = _mae(values[split:], probe.predict(values[:split]))
    print(f"  reconstruction MSE {probe.pretrain_loss_:.4f}, frozen "
          f"representation difference {difference:.1e}")
    print(f"  linear probe held-out MAE {probe_error:.4f} against "
          f"{fine_error:.4f} for fine-tuning and "
          f"{_mae(values[split:], forecast):.4f} from scratch")

    print("  the patch count and the attention weights are what they claim:")
    assert _patch_count(48, 12, 6) == 7
    assert _patch_count(48, 12, 12) == 4
    weights = _softmax(np.array([[1.0, 1.0, 1.0]]))
    assert np.allclose(weights, 1.0 / 3.0)
    print(f"  strides 6 and 12 give {_patch_count(48, 12, 6)} and "
          f"{_patch_count(48, 12, 12)} patches; uniform scores average evenly")

    bad = [
        lambda: PatchTST("patchtst_lite"),
        lambda: PatchTST("patchtst", seq_len=0),
        lambda: PatchTST("patchtst", horizon=0),
        lambda: PatchTST("patchtst", patch=0),
        lambda: PatchTST("patchtst", stride=0),
        lambda: PatchTST("patchtst", dim=0),
        lambda: PatchTST("patchtst", heads=0),
        lambda: PatchTST("patchtst", hidden=0),
        lambda: PatchTST("patchtst", epochs=0),
        lambda: PatchTST("patchtst", pretrain_epochs=0),
        lambda: PatchTST("patchtst", patch=60),
        lambda: PatchTST("patchtst", dim=15, heads=2),
        lambda: PatchTST("patchtst", learning_rate=0.0),
        lambda: PatchTST("patchtst", learning_rate=np.nan),
        lambda: PatchTST("patchtst", mask_ratio=0.0),
        lambda: PatchTST("patchtst", mask_ratio=1.0),
        lambda: PatchTST("patchtst").fit(np.zeros((40, 2, 2))),
        lambda: PatchTST("patchtst").fit(np.arange(30.0)),
        lambda: PatchTST("patchtst").fit(np.r_[np.arange(60.0), np.nan]),
        lambda: PatchTST("patchtst").predict(values[:split]),
        lambda: PatchTST("patchtst", seq_len=48).fit(
            values[:split]).predict(np.arange(20.0)),
        lambda: PatchTST("channel_independent", seq_len=48).fit(
            stacked[:split]).predict(stacked[:split, :1]),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except (ValueError, RuntimeError):
            refused += 1
    print(f"{refused} malformed calls refused out of {len(bad)}: an unknown")
    print("  variant, a non-positive window, horizon, patch, stride, width,")
    print("  head count, hidden width, epoch count or pretraining epochs, a")
    print("  patch longer than the window, a width that is not divisible by")
    print("  the head count, a non-finite learning rate, a mask ratio outside")
    print("  the open unit interval, a three-dimensional series, a short or")
    print("  non-finite series, prediction before fitting, a short history,")
    print("  and a channel count that differs from the fitted one")
    print("patchtst demo passed")


if __name__ == "__main__":
    demo()
