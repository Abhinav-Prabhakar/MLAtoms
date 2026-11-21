"""TFT — variable selection, static context and interpretable attention.

Chapter:   11-time-series
Companion: tft.md
Tier 3 · 4 variants: tft, variable_selection, static_covariates,
interpretable_attention
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("tft", "variable_selection", "static_covariates",
            "interpretable_attention")

EPSILON = 1e-5


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
def _elu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0.0, x, np.expm1(np.minimum(x, 0.0)))


def _layer_norm(x: np.ndarray):
    mean = x.mean(axis=-1, keepdims=True)
    centred = x - mean
    variance = (centred ** 2).mean(axis=-1, keepdims=True)
    inverse = 1.0 / np.sqrt(variance + EPSILON)
    return centred * inverse, (centred, inverse)


def _layer_norm_backward(gradient: np.ndarray, cache) -> np.ndarray:
    centred, inverse = cache
    width = gradient.shape[-1]
    scaled = (gradient * centred).sum(axis=-1, keepdims=True)
    total = gradient.sum(axis=-1, keepdims=True)
    return (inverse / width) * (
        width * gradient - total - centred * inverse ** 2 * scaled)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - x.max(axis=axis, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=axis, keepdims=True)


def _softmax_backward(gradient: np.ndarray, weights: np.ndarray,
                      axis: int = -1) -> np.ndarray:
    projection = (gradient * weights).sum(axis=axis, keepdims=True)
    return weights * (gradient - projection)


def _glorot(rng: np.random.Generator, rows: int, columns: int,
            scale: float = 2.0) -> np.ndarray:
    return rng.normal(0.0, np.sqrt(scale / (rows + columns)), (rows, columns))


# --------------------------------------------------------------------------- #
# Gated residual network
# --------------------------------------------------------------------------- #
class _GRN:
    """ELU bottleneck, GLU gate, optional residual, layer norm.

    The static encoder has no residual: it maps a handful of summary features
    into the model width, so there is nothing to add.
    """

    def __init__(self, parameters: dict, prefix: str, dim: int, feedforward: int,
                 rng: np.random.Generator, context: bool,
                 in_dim: int | None = None, residual: bool = True):
        self.prefix = prefix
        self.parameters = parameters
        self.dim = dim
        self.in_dim = dim if in_dim is None else in_dim
        self.residual = residual
        self.context = context
        parameters[f"{prefix}.W1"] = _glorot(rng, self.in_dim, feedforward)
        parameters[f"{prefix}.b1"] = np.zeros(feedforward)
        parameters[f"{prefix}.W2"] = _glorot(rng, feedforward, dim)
        parameters[f"{prefix}.b2"] = np.zeros(dim)
        parameters[f"{prefix}.Wg"] = _glorot(rng, self.in_dim, dim)
        parameters[f"{prefix}.bg"] = np.zeros(dim)
        if context:
            parameters[f"{prefix}.Wc"] = _glorot(rng, dim, dim)

    def forward(self, x: np.ndarray, context: np.ndarray | None):
        p = self.parameters
        name = self.prefix
        pre = x @ p[f"{name}.W1"] + p[f"{name}.b1"]
        active = _elu(pre)
        gate_logits = x @ p[f"{name}.Wg"] + p[f"{name}.bg"]
        if self.context:
            gate_logits = gate_logits + context[:, None, :] @ p[
                f"{name}.Wc"]
        gate = 0.5 * (1.0 + np.tanh(0.5 * gate_logits))
        hidden = active @ p[f"{name}.W2"] + p[f"{name}.b2"]
        skip = gate * hidden if not self.residual else x + gate * hidden
        output, norm_cache = _layer_norm(skip)
        cache = (x, pre, active, gate, hidden, skip, norm_cache, context)
        return output, cache

    def backward(self, gradient: np.ndarray, cache,
                 grads: dict[str, np.ndarray]) -> tuple:
        p = self.parameters
        name = self.prefix
        x, pre, active, gate, hidden, _skip, norm_cache, context = cache
        d_skip = _layer_norm_backward(gradient, norm_cache)
        axes = tuple(range(len(x.shape) - 1))
        d_hidden = d_skip * gate
        d_gate = d_skip * hidden
        d_logits = d_gate * gate * (1.0 - gate)
        grads[f"{name}.W2"] = active.reshape(-1, active.shape[-1]).T @ \
            d_hidden.reshape(-1, hidden.shape[-1])
        grads[f"{name}.b2"] = d_hidden.sum(axis=axes)
        d_active = d_hidden @ p[f"{name}.W2"].T
        d_pre = np.where(pre > 0.0, d_active, d_active * np.exp(pre))
        grads[f"{name}.W1"] = x.reshape(-1, self.in_dim).T @ d_pre.reshape(
            -1, pre.shape[-1])
        grads[f"{name}.b1"] = d_pre.sum(axis=axes)
        grads[f"{name}.Wg"] = x.reshape(-1, self.in_dim).T @ d_logits.reshape(
            -1, self.dim)
        grads[f"{name}.bg"] = d_logits.sum(axis=axes)
        d_x = d_pre @ p[f"{name}.W1"].T + d_logits @ p[f"{name}.Wg"].T
        if self.residual:
            d_x = d_x + d_skip
        d_context = None
        if self.context:
            grads[f"{name}.Wc"] = context.T @ d_logits.sum(axis=1)
            d_context = (d_logits @ p[f"{name}.Wc"].T).sum(axis=1)
        return d_x, d_context


# --------------------------------------------------------------------------- #
# Variable selection
# --------------------------------------------------------------------------- #
class _Selector:
    """Project each variable, score it, and softmax the scores."""

    def __init__(self, parameters: dict, prefix: str, count: int, dim: int,
                 rng: np.random.Generator, learn: bool):
        self.prefix = prefix
        self.parameters = parameters
        self.count = count
        self.dim = dim
        self.learn = learn
        parameters[f"{prefix}.W"] = _glorot(rng, count, dim)
        parameters[f"{prefix}.score"] = rng.normal(
            0.0, 1.0 / np.sqrt(dim), dim)

    def forward(self, x: np.ndarray):
        p = self.parameters
        name = self.prefix
        projected = x[..., None] * p[f"{name}.W"]
        if self.learn:
            scores = projected @ p[f"{name}.score"]
            weights = _softmax(scores, axis=-1)
        else:
            weights = np.full(x.shape, 1.0 / self.count)
        output = (weights[..., None] * projected).sum(axis=-2)
        return output, (x, projected, weights)

    def backward(self, gradient: np.ndarray, cache,
                 grads: dict[str, np.ndarray]) -> np.ndarray:
        p = self.parameters
        name = self.prefix
        x, projected, weights = cache
        d_projected = weights[..., None] * gradient[..., None, :]
        if self.learn:
            d_weights = (projected * gradient[..., None, :]).sum(axis=-1)
            d_scores = _softmax_backward(d_weights, weights, axis=-1)
            d_projected = d_projected + d_scores[..., None] * p[
                f"{name}.score"]
            grads[f"{name}.score"] = (d_scores[..., None]
                                      * projected).sum(axis=(0, 1, 2))
        grads[f"{name}.W"] = (d_projected * x[..., None]).sum(axis=(0, 1))
        return (d_projected * p[f"{name}.W"]).sum(axis=-1)


# --------------------------------------------------------------------------- #
# Interpretable multi-head attention
# --------------------------------------------------------------------------- #
class _Attention:
    """Values shared across heads; the heads' outputs are averaged."""

    def __init__(self, parameters: dict, prefix: str, dim: int, heads: int,
                 rng: np.random.Generator):
        self.prefix = prefix
        self.parameters = parameters
        self.dim = dim
        self.heads = heads
        self.width = dim // heads
        parameters[f"{prefix}.Wq"] = _glorot(rng, dim, dim)
        parameters[f"{prefix}.Wk"] = _glorot(rng, dim, dim)
        parameters[f"{prefix}.Wv"] = _glorot(rng, dim, dim)
        parameters[f"{prefix}.Wo"] = _glorot(rng, self.width, dim)
        parameters[f"{prefix}.bq"] = np.zeros(dim)

    def _split(self, x: np.ndarray) -> np.ndarray:
        batch, steps, _ = x.shape
        return x.reshape(batch, steps, self.heads, self.width).transpose(
            0, 2, 1, 3)

    def forward(self, query: np.ndarray, memory: np.ndarray):
        p = self.parameters
        name = self.prefix
        left, left_cache = _layer_norm(query)
        right, right_cache = _layer_norm(memory)
        q = self._split(left @ p[f"{name}.Wq"] + p[f"{name}.bq"])
        k = self._split(right @ p[f"{name}.Wk"])
        v = self._split(right @ p[f"{name}.Wv"])
        scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(self.width)
        weights = _softmax(scores, axis=-1)
        heads = weights @ v
        joined = heads.mean(axis=1)
        output = joined @ p[f"{name}.Wo"]
        cache = (left, left_cache, right, right_cache, q, k, v, weights,
                 heads, joined)
        return output, cache

    def backward(self, gradient: np.ndarray, cache,
                 grads: dict[str, np.ndarray]) -> tuple:
        p = self.parameters
        name = self.prefix
        (left, left_cache, right, right_cache, q, k, v, weights, heads,
         joined) = cache
        batch, steps, _ = gradient.shape
        grads[f"{name}.Wo"] = (joined.reshape(-1, self.width).T
                               @ gradient.reshape(-1, self.dim))
        d_joined = gradient @ p[f"{name}.Wo"].T
        d_heads = np.repeat(d_joined[:, None], self.heads, axis=1) / self.heads
        d_weights = d_heads @ v.transpose(0, 1, 3, 2)
        d_v = weights.transpose(0, 1, 3, 2) @ d_heads
        d_scores = _softmax_backward(d_weights, weights, axis=-1) / np.sqrt(
            self.width)
        d_q = d_scores @ k
        d_k = d_scores.transpose(0, 1, 3, 2) @ q
        def merge(x):
            return x.transpose(0, 2, 1, 3).reshape(
                x.shape[0], x.shape[2], self.dim)
        d_q = merge(d_q)
        d_k = merge(d_k)
        d_v = merge(d_v)
        grads[f"{name}.Wq"] = left.reshape(-1, self.dim).T @ d_q.reshape(
            -1, self.dim)
        grads[f"{name}.bq"] = d_q.sum(axis=(0, 1))
        grads[f"{name}.Wk"] = right.reshape(-1, self.dim).T @ d_k.reshape(
            -1, self.dim)
        grads[f"{name}.Wv"] = right.reshape(-1, self.dim).T @ d_v.reshape(
            -1, self.dim)
        d_query = _layer_norm_backward(d_q @ p[f"{name}.Wq"].T, left_cache)
        d_memory = _layer_norm_backward(
            d_k @ p[f"{name}.Wk"].T + d_v @ p[f"{name}.Wv"].T, right_cache)
        return d_query, d_memory


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
class TFT:
    """Temporal Fusion Transformer in miniature.

    Parameters
    ----------
    variant : str
        One of `VARIANTS`. Each variant switches on one mechanism over a plain
        encoder-attention-decoder baseline, except `tft`, which has all three.
    seq_len, horizon : int
        Encoder window and decoder horizon.
    lags : int
        Number of lagged copies of the target used as past variables.
    period : int
        Seasonal period used to build the default future covariates.
    hidden, heads : int
        Width and head count; `hidden` must be divisible by `heads`.
    epochs, learning_rate : int, float
        Full-batch Adam schedule.
    quantiles : tuple
        Increasing quantile levels emitted at every horizon step.
    seed : int
        Seed for the initialisation.

    Notes
    -----
    `fit(y, future=None)` cuts sliding windows. The past variables are the
    target's own lags; the future variables default to sine, cosine and a
    scaled time index at the seasonal period; the static variables are the
    window's mean, standard deviation, last value and slope. `predict` returns
    the median path, `forecast_quantiles` the full quantile fan, and
    `attention` the head-averaged attention over the look-back.
    """

    def __init__(self, variant: str = "tft", *, seq_len: int = 48,
                 horizon: int = 12, lags: int = 3, period: int = 12,
                 hidden: int = 16, heads: int = 2, epochs: int = 200,
                 learning_rate: float = 0.01,
                 quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
                 seed: int = 0):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        for name, value in (("seq_len", seq_len), ("horizon", horizon),
                            ("lags", lags), ("period", period),
                            ("hidden", hidden), ("heads", heads),
                            ("epochs", epochs)):
            if not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if hidden % heads != 0:
            raise ValueError("hidden must be divisible by heads")
        if lags > seq_len:
            raise ValueError("lags must not exceed seq_len")
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive and finite")
        levels = tuple(float(value) for value in quantiles)
        if len(levels) < 2:
            raise ValueError("need at least two quantile levels")
        if any(not 0.0 < value < 1.0 for value in levels):
            raise ValueError("quantile levels must be in the open unit interval")
        pairs = zip(levels, levels[1:], strict=False)
        if any(later <= earlier for earlier, later in pairs):
            raise ValueError("quantile levels must be strictly increasing")
        if 0.5 not in levels:
            raise ValueError("the quantile grid must contain the median")
        self.variant = variant
        self.seq_len = int(seq_len)
        self.horizon = int(horizon)
        self.lags = int(lags)
        self.period = int(period)
        self.hidden = int(hidden)
        self.heads = int(heads)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.quantiles = levels
        self.seed = int(seed)
        self.use_selection = variant in ("tft", "variable_selection")
        self.use_static = variant in ("tft", "static_covariates")
        self.use_attention = variant in ("tft", "interpretable_attention")
        self._fitted = False

    # -- inputs ----------------------------------------------------------- #
    @staticmethod
    def _series(y: np.ndarray) -> np.ndarray:
        values = np.asarray(y, dtype=float)
        if values.ndim != 1:
            raise ValueError("y must be one-dimensional")
        if values.size < 60:
            raise ValueError("y needs at least sixty observations")
        if not np.all(np.isfinite(values)):
            raise ValueError("y must be finite")
        return values

    def _future(self, n: int, future: np.ndarray | None) -> np.ndarray:
        if future is None:
            time = np.arange(n, dtype=float)
            angle = 2.0 * np.pi * time / self.period
            return np.column_stack([np.sin(angle), np.cos(angle),
                                    time / max(n - 1, 1)])
        values = np.asarray(future, dtype=float)
        if values.ndim != 2 or values.shape[0] != n:
            raise ValueError(
                "future must be two-dimensional with len(y) + horizon rows")
        if values.shape[1] < 1:
            raise ValueError("future needs at least one column")
        if not np.all(np.isfinite(values)):
            raise ValueError("future must be finite")
        return values

    @staticmethod
    def _static(windows: np.ndarray) -> np.ndarray:
        """Mean, spread, last value and OLS slope of each window."""
        steps = np.arange(windows.shape[1], dtype=float)
        centred = steps - steps.mean()
        slope = ((windows - windows.mean(axis=1, keepdims=True)) * centred).sum(
            axis=1) / (centred ** 2).sum()
        return np.column_stack([windows.mean(axis=1), windows.std(axis=1),
                                windows[:, -1], slope])

    def _windows(self, scaled: np.ndarray, future: np.ndarray):
        starts = np.arange(self.lags - 1,
                           scaled.size - self.seq_len - self.horizon + 1)
        if starts.size < 8:
            raise ValueError("need at least eight training windows")
        past = np.stack([scaled[starts + offset - lag]
                         for offset in range(self.seq_len)
                         for lag in range(self.lags)], axis=1)
        past = past.reshape(starts.size, self.seq_len, self.lags)
        ahead = np.stack([future[starts + self.seq_len + offset]
                          for offset in range(self.horizon)], axis=1)
        static = self._static(np.stack(
            [scaled[starts + offset] for offset in range(self.seq_len)],
            axis=1))
        target = np.stack([scaled[starts + self.seq_len + offset]
                           for offset in range(self.horizon)], axis=1)
        return past, ahead, static, target, starts

    # -- graph ------------------------------------------------------------ #
    def _build(self):
        rng = np.random.default_rng(self.seed)
        p: dict[str, np.ndarray] = {}
        dim = self.hidden
        feedforward = 2 * dim
        self.static_grn = None
        if self.use_static:
            self.static_grn = _GRN(p, "static", dim, feedforward, rng, False,
                                   in_dim=4, residual=False)
        self.past_selector = _Selector(p, "past", self.lags, dim, rng,
                                       self.use_selection)
        self.future_selector = _Selector(p, "future", self.future_dim, dim, rng,
                                         self.use_selection)
        self.encoder = _GRN(p, "encoder", dim, feedforward, rng,
                            self.use_static)
        self.decoder = _GRN(p, "decoder", dim, feedforward, rng,
                            self.use_static)
        self.attention = None
        if self.use_attention:
            self.attention = _Attention(p, "attention", dim, self.heads, rng)
        self.head_grn = _GRN(p, "head", dim, feedforward, rng, self.use_static)
        p["position"] = rng.normal(0.0, 0.02, (self.horizon, dim))
        p["Wh"] = _glorot(rng, dim, len(self.quantiles))
        p["bh"] = np.zeros(len(self.quantiles))
        self.parameters = p

    # -- forward ---------------------------------------------------------- #
    def _forward(self, past, ahead, static):
        p = self.parameters
        if self.use_static:
            context, static_cache = self.static_grn.forward(static, None)
        else:
            context = np.zeros((past.shape[0], self.hidden))
            static_cache = None
        selected_past, past_cache = self.past_selector.forward(past)
        selected_ahead, future_cache = self.future_selector.forward(ahead)
        enc_input = selected_past + context[:, None, :]
        encoded, encoder_cache = self.encoder.forward(enc_input, context)
        dec_input = (selected_ahead + context[:, None, :]
                     + p["position"][None, :, :])
        decoded, decoder_cache = self.decoder.forward(dec_input, context)
        if self.use_attention:
            attention, attention_cache = self.attention.forward(decoded,
                                                                encoded)
        else:
            attention = np.repeat(encoded.mean(axis=1, keepdims=True),
                                  decoded.shape[1], axis=1)
            attention_cache = None
        fused, head_cache = self.head_grn.forward(decoded + attention, context)
        outputs = fused @ p["Wh"] + p["bh"]
        cache = (context, static_cache, past_cache, future_cache,
                 encoder_cache, decoder_cache, attention_cache, head_cache,
                 encoded, fused)
        return outputs, cache

    def _backward(self, gradient: np.ndarray, cache):
        p = self.parameters
        grads: dict[str, np.ndarray] = {}
        (context, static_cache, past_cache, future_cache, encoder_cache,
         decoder_cache, attention_cache, head_cache, encoded, fused) = cache
        dim = self.hidden
        grads["Wh"] = (fused.reshape(-1, dim).T
                       @ gradient.reshape(-1, gradient.shape[-1]))
        grads["bh"] = gradient.sum(axis=(0, 1))
        d_fused = gradient @ p["Wh"].T
        d_sum, d_context = self.head_grn.backward(d_fused, head_cache, grads)
        d_decoded = d_sum
        if self.use_attention:
            d_query, d_encoded = self.attention.backward(d_sum,
                                                         attention_cache, grads)
            d_decoded = d_decoded + d_query
        else:
            pooled = (d_sum / encoded.shape[1]).sum(axis=1, keepdims=True)
            d_encoded = np.repeat(pooled, encoded.shape[1], axis=1)
        d_dec_input, d_context_dec = self.decoder.backward(
            d_decoded, decoder_cache, grads)
        grads["position"] = d_dec_input.sum(axis=0)
        d_enc_input, d_context_enc = self.encoder.backward(
            d_encoded, encoder_cache, grads)
        self.past_selector.backward(d_enc_input, past_cache, grads)
        self.future_selector.backward(d_dec_input, future_cache, grads)
        if self.use_static:
            d_context = (d_context + d_context_dec + d_context_enc
                         + d_enc_input.sum(axis=1) + d_dec_input.sum(axis=1))
            self.static_grn.backward(d_context, static_cache, grads)
        return grads

    # -- training --------------------------------------------------------- #
    def fit(self, y: np.ndarray, future: np.ndarray | None = None) -> TFT:
        """Train on sliding windows of `y`."""
        values = self._series(y)
        self.mean_ = float(values.mean())
        self.scale_ = float(values.std())
        if self.scale_ <= 1e-12:
            raise ValueError("the series must have a non-zero spread")
        scaled = (values - self.mean_) / self.scale_
        covariates = self._future(values.size + self.horizon, future)
        self.future_dim = covariates.shape[1]
        past, ahead, static, target, _ = self._windows(scaled, covariates)
        self.static_mean_ = static.mean(axis=0)
        spread = static.std(axis=0)
        self.static_scale_ = np.where(spread > 1e-12, spread, 1.0)
        static = (static - self.static_mean_) / self.static_scale_
        self.static_ = static
        self.n_windows_ = int(past.shape[0])
        self._build()
        parameters = self.parameters
        first = {key: np.zeros_like(value) for key, value in
                 parameters.items()}
        second = {key: np.zeros_like(value) for key, value in
                  parameters.items()}
        loss = float("inf")
        for epoch in range(1, self.epochs + 1):
            outputs, cache = self._forward(past, ahead, static)
            loss, gradient = self._pinball(outputs, target)
            grads = self._backward(gradient / gradient.size, cache)
            for key, value in parameters.items():
                if key not in grads:
                    continue
                update = grads[key]
                first[key] = 0.9 * first[key] + 0.1 * update
                second[key] = 0.999 * second[key] + 0.001 * update ** 2
                corrected = first[key] / (1.0 - 0.9 ** epoch)
                scale = second[key] / (1.0 - 0.999 ** epoch)
                value -= self.learning_rate * corrected / (np.sqrt(scale) + 1e-8)
        self.train_loss_ = loss
        self.n_params_ = int(sum(value.size for value in parameters.values()))
        self._fitted = True
        return self

    def _pinball(self, outputs: np.ndarray, target: np.ndarray):
        levels = np.asarray(self.quantiles)
        error = target[..., None] - outputs
        loss = np.where(error >= 0.0, levels * error, (levels - 1.0) * error)
        gradient = np.where(error >= 0.0, -levels, 1.0 - levels)
        return float(loss.mean()), gradient

    # -- prediction ------------------------------------------------------- #
    def _prepare(self, y: np.ndarray, future: np.ndarray | None):
        values = self._series(y)
        if values.size < self.seq_len + self.lags:
            raise ValueError("history is shorter than the encoder window")
        scaled = (values - self.mean_) / self.scale_
        covariates = self._future(values.size + self.horizon, future)
        if covariates.shape[1] != self.future_dim:
            raise ValueError("future has a different number of columns than fit")
        past = np.stack([scaled[[-self.seq_len + offset - lag]]
                         for offset in range(self.seq_len)
                         for lag in range(self.lags)], axis=1)
        past = past.reshape(1, self.seq_len, self.lags)
        ahead = covariates[-self.horizon:][None, :, :]
        static = self._static(scaled[None, -self.seq_len:])
        static = (static - self.static_mean_) / self.static_scale_
        return past, ahead, static

    def forecast_quantiles(self, y: np.ndarray,
                           future: np.ndarray | None = None) -> np.ndarray:
        """Quantile fan of shape `(horizon, len(quantiles))`."""
        if not self._fitted:
            raise RuntimeError("fit the model before predicting")
        past, ahead, static = self._prepare(y, future)
        outputs, _ = self._forward(past, ahead, static)
        return self.mean_ + self.scale_ * outputs[0]

    def predict(self, y: np.ndarray, future: np.ndarray | None = None
                ) -> np.ndarray:
        """Median path in original units."""
        fan = self.forecast_quantiles(y, future)
        return fan[:, self.quantiles.index(0.5)]

    def interval(self, y: np.ndarray, level: float = 0.8,
                 future: np.ndarray | None = None) -> np.ndarray:
        """Central `level` band, interpolated on the quantile grid."""
        if not 0.0 < level < 1.0 or not np.isfinite(level):
            raise ValueError("level must lie strictly between zero and one")
        fan = self.forecast_quantiles(y, future)
        tail = 0.5 * (1.0 - level)
        levels = np.asarray(self.quantiles)
        low = np.array([np.interp(tail, levels, row) for row in fan])
        high = np.array([np.interp(1.0 - tail, levels, row) for row in fan])
        return np.column_stack([low, high])

    def _encode(self, y: np.ndarray, future: np.ndarray | None):
        """Recompute the encoder stack and keep the selector caches."""
        past, ahead, static = self._prepare(y, future)
        p = self.parameters
        if self.use_static:
            context, _ = self.static_grn.forward(static, None)
        else:
            context = np.zeros((1, self.hidden))
        selected_past, past_cache = self.past_selector.forward(past)
        selected_ahead, future_cache = self.future_selector.forward(ahead)
        encoded, _ = self.encoder.forward(
            selected_past + context[:, None, :], context)
        decoded, _ = self.decoder.forward(
            selected_ahead + context[:, None, :] + p["position"][None], context)
        return context, past_cache, future_cache, encoded, decoded

    def attention_weights(self, y: np.ndarray,
                          future: np.ndarray | None = None) -> np.ndarray:
        """Head-averaged attention of shape `(horizon, seq_len)`."""
        if not self._fitted:
            raise RuntimeError("fit the model before predicting")
        if not self.use_attention:
            raise ValueError(f"{self.variant} does not use attention")
        _, _, _, encoded, decoded = self._encode(y, future)
        _, cache = self.attention.forward(decoded, encoded)
        return cache[7][0].mean(axis=0)

    def variable_weights(self, y: np.ndarray,
                         future: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Average softmax weight per past lag and per future covariate."""
        if not self._fitted:
            raise RuntimeError("fit the model before predicting")
        if not self.use_selection:
            raise ValueError(f"{self.variant} does not select variables")
        _, past_cache, future_cache, _, _ = self._encode(y, future)
        return {"past": past_cache[2][0].mean(axis=0),
                "future": future_cache[2][0].mean(axis=0)}


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _seasonal(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    return (20.0 + 0.03 * t + 3.0 * np.sin(2.0 * np.pi * t / 12.0)
            + 1.2 * np.cos(2.0 * np.pi * t / 4.0) + rng.normal(0.0, 0.5, n))


def _mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - forecast)))


def demo() -> None:
    split = 228
    values = _seasonal(240, 0)
    actual = values[split:]
    seasonal = _mae(actual, values[split - 12:split])
    flat = _mae(actual, np.repeat(values[split - 1], 12))
    print("tft. 228 points of trend, season and noise, 12 held out:")
    models = {}
    for variant in VARIANTS:
        models[variant] = TFT(variant, seq_len=24, hidden=8, epochs=150,
                              seed=0).fit(values[:split])
    errors = {name: _mae(actual, model.predict(values[:split]))
              for name, model in models.items()}
    full = models["tft"]
    print(f"  {full.n_windows_} training windows of 24 steps, {full.n_params_} "
          f"parameters in the full model")
    for variant in VARIANTS:
        model = models[variant]
        print(f"    {variant:<23s} pinball {model.train_loss_:.4f}   "
              f"held-out MAE {errors[variant]:.4f}")
    print(f"  for reference: seasonal naive {seasonal:.4f}, last value "
          f"{flat:.4f}")

    print("tft. removing any one mechanism costs accuracy on this series:")
    assert errors["tft"] < seasonal
    assert errors["tft"] < errors["variable_selection"]
    assert errors["tft"] < errors["static_covariates"]
    assert errors["tft"] < errors["interpretable_attention"]
    assert all(error < flat for error in errors.values())
    assert all(np.isfinite(model.train_loss_) for model in models.values())
    print(f"  the full model improves on seasonal naive by "
          f"{seasonal - errors['tft']:.4f}")
    print(f"  variable selection alone {errors['variable_selection']:.4f}, "
          f"static context alone {errors['static_covariates']:.4f},")
    print(f"  attention alone {errors['interpretable_attention']:.4f}; one "
          "draw from one seed, so read the")
    print("  ordering as an illustration rather than a law")

    print("tft. the selection layer says which lags it reads:")
    weights = full.variable_weights(values[:split])
    past, ahead = weights["past"], weights["future"]
    assert past.shape == (3,) and ahead.shape == (3,)
    assert abs(float(past.sum()) - 1.0) < 1e-9
    assert abs(float(ahead.sum()) - 1.0) < 1e-9
    assert float(past.max()) > float(past.min()) + 0.05
    print(f"  lag weights {np.round(past, 3).tolist()} -- the softmax picks "
          f"lag {int(past.argmax()) + 1} first")
    print(f"  covariate weights {np.round(ahead, 3).tolist()} for sine, "
          "cosine and scaled time")

    print("tft. attention is nearly uniform on this series:")
    attention = full.attention_weights(values[:split])
    assert attention.shape == (12, 24)
    assert np.allclose(attention.sum(axis=1), 1.0)
    assert float(attention.max()) > 1.0 / 24.0
    positions = (attention * np.arange(24)).sum(axis=1)
    drift = float(positions.max() - positions.min())
    entropy = float(-(attention * np.log(attention + 1e-12)).sum(axis=1).mean())
    assert drift < 1.0
    assert float(np.log(24.0)) - entropy < 0.05
    print(f"  the largest single weight is {float(attention.max()):.3f} where "
          f"a flat row would be {1.0 / 24.0:.3f}")
    print(f"  mean entropy {entropy:.4f} against {float(np.log(24.0)):.4f} "
          f"for a flat row")
    print(f"  the mean encoder position moves only {drift:.3f} steps across "
          "the horizon")

    print("tft. the static encoder summarises each window into four numbers:")
    assert full.static_.shape == (full.n_windows_, 4)
    assert abs(float(full.static_.mean())) < 1e-9
    assert np.allclose(full.static_.std(axis=0), 1.0)
    assert np.all(full.static_scale_ > 0.0)
    named = ", ".join(f"{value:.2f}" for value in full.static_mean_)
    print(f"  average raw features {named}: window mean, spread, last value,")
    print("  and slope")

    print("tft. three quantiles, and the band they imply:")
    fan = full.forecast_quantiles(values[:split])
    band = full.interval(values[:split], 0.8)
    narrow = full.interval(values[:split], 0.5)
    assert fan.shape == (12, 3)
    assert np.all(np.diff(fan, axis=1) >= -1e-12)
    assert np.allclose(fan[:, 1], full.predict(values[:split]))
    assert np.allclose(band[:, 0], fan[:, 0])
    assert np.allclose(band[:, 1], fan[:, 2])
    assert float((narrow[:, 1] - narrow[:, 0]).mean()) < float(
        (band[:, 1] - band[:, 0]).mean())
    coverage = float(np.mean((actual >= fan[:, 0]) & (actual <= fan[:, 1 + 1])))
    assert coverage > 0.5
    width = float((band[:, 1] - band[:, 0]).mean())
    print("  the fan is monotone, its middle column is `predict`")
    print(f"  and the 80% band averages {width:.4f}")
    print(f"  the 50% band is narrower and the 80% band covers "
          f"{coverage:.3f} of twelve points")

    print("  the pieces are what they claim:")
    context = full.static_grn
    assert context.in_dim == 4 and not context.residual
    assert full.encoder.in_dim == 8 and full.encoder.residual
    assert full.attention.width == 4 and full.attention.heads == 2
    share = full.attention.forward(
        np.zeros((1, 12, 8)) + 0.1, np.zeros((1, 24, 8)) + 0.1)[1][7]
    assert np.allclose(share.sum(axis=-1), 1.0)
    print(f"  the static GRN maps 4 features to {context.dim} without a "
          f"residual skip")
    print(f"  {full.attention.heads} heads of width {full.attention.width} "
          "share one value projection")
    print("  and every attention row sums to one")

    bad = [
        lambda: TFT("tft_lite"),
        lambda: TFT("tft", seq_len=0),
        lambda: TFT("tft", horizon=0),
        lambda: TFT("tft", lags=0),
        lambda: TFT("tft", period=0),
        lambda: TFT("tft", hidden=0),
        lambda: TFT("tft", heads=0),
        lambda: TFT("tft", epochs=0),
        lambda: TFT("tft", hidden=8, heads=3),
        lambda: TFT("tft", seq_len=4, lags=6),
        lambda: TFT("tft", learning_rate=0.0),
        lambda: TFT("tft", learning_rate=np.nan),
        lambda: TFT("tft", quantiles=(0.5,)),
        lambda: TFT("tft", quantiles=(0.0, 0.5, 0.9)),
        lambda: TFT("tft", quantiles=(0.9, 0.5, 0.1)),
        lambda: TFT("tft", quantiles=(0.1, 0.2, 0.3)),
        lambda: TFT("tft", epochs=1).fit(np.zeros((80, 2))),
        lambda: TFT("tft", epochs=1).fit(np.arange(40.0)),
        lambda: TFT("tft", epochs=1).fit(np.r_[np.arange(80.0), np.nan]),
        lambda: TFT("tft", epochs=1).fit(np.ones(80)),
        lambda: TFT("tft", epochs=1, seq_len=60).fit(_seasonal(80, 0)),
        lambda: TFT("tft", epochs=1).fit(
            _seasonal(120, 0), np.zeros((120, 2))),
        lambda: TFT("tft", epochs=1).fit(
            _seasonal(120, 0), np.zeros((132, 0))),
        lambda: TFT("tft", epochs=1).fit(
            _seasonal(120, 0), np.full((132, 2), np.inf)),
        lambda: TFT("tft", epochs=1).predict(_seasonal(120, 0)),
        lambda: full.attention_weights(values[:split], np.zeros((240, 2))),
        lambda: models["static_covariates"].attention_weights(values[:split]),
        lambda: models["static_covariates"].variable_weights(values[:split]),
        lambda: full.interval(values[:split], 1.0),
        lambda: full.interval(values[:split], 0.0),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except (ValueError, RuntimeError):
            refused += 1
    print(f"{refused} malformed calls refused out of {len(bad)}: an unknown")
    print("  variant, a non-positive window, horizon, lag, period, width,")
    print("  head count, epoch count or learning rate, a width that is not a")
    print("  multiple of the head count, more lags than steps, fewer than two")
    print("  or out-of-range or non-increasing quantile levels without the")
    print("  median, a two-dimensional or short or non-finite or constant")
    print("  series, too few windows, covariates with the wrong row or column")
    print("  count or a non-finite entry, prediction before fitting, an")
    print("  attention or selection query on a variant that has neither, and")
    print("  a central level outside the open unit interval")
    print("tft demo passed")


if __name__ == "__main__":
    demo()
