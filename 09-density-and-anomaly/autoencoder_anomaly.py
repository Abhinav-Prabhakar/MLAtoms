r"""Autoencoder anomaly detection — what the network cannot rebuild.

Chapter:   09-density-and-anomaly
Companion: autoencoder_anomaly.md
Tier 2 · 5 variants: ae, vae, lstm_ae, denoising_ae, thresholding

    encoder:  z = f_theta(x),      decoder:  y = g_phi(z)                 (1)
    score:    e(x) = (1/d) ||x - g_phi(f_theta(x))||^2                    (2)
    denoise:  train on  x_tilde = x + sigma eps,  target x                (3)
    vae:      L = E_q ||x - y||^2 + beta KL(q(z|x) || N(0, I))            (4)
    lstm:     z = (h_T, c_T),  y_t = W_o h_t,  h_0 = h_T                  (5)
    threshold: s(x) = sum_j r_j^2 / v_j  ~  chi^2_d                       (6)

An autoencoder learns a manifold, not a density. It is trained to reproduce its own
input through a narrow bottleneck, so what it learns is the set of directions the
data actually use; a point that needs directions the bottleneck cannot represent
comes back distorted, and the distortion is the anomaly score. That is the whole
idea, and it has one large consequence: the score measures *distance from the
learned manifold*, not improbability. A point in the middle of a hole in the data
reconstructs perfectly if the hole is inside the manifold's span, and a point on a
sparse but real part of the data reconstructs badly. Every failure below follows
from that one sentence.

The five variants are four ways to make the manifold harder to cheat and one way to
read the residual:

- `ae` is the plain bottleneck. Its weak point is that the reconstruction target is
  the input itself, so with enough capacity it can learn the identity and
  reconstruct anomalies as well as anything else.
- `denoising_ae` removes that escape. Training on a corrupted input with the clean
  one as the target forces the network to learn the structure that survives the
  corruption — the manifold — because copying is no longer an available strategy.
- `vae` puts a probability model on the latent space, so the score has two terms:
  the reconstruction, and the KL divergence of the inferred posterior from the
  prior. The second term is what stops a far-away input from being assigned an
  arbitrarily remote latent code.
- `lstm_ae` is the same bottleneck for sequences: the encoder reads the window and
  the decoder regenerates it from the final hidden state. Parameter sharing across
  time is what makes it able to represent a *phase* rather than a list of values.
- `thresholding` keeps the plain autoencoder and changes only the score, dividing
  each feature's squared residual by that feature's training residual variance. The
  sum is then a chi-square statistic, and a quiet feature's small absolute
  residual finally counts for as much as a loud one's.

Everything here is trained with hand-written backpropagation and Adam, numpy only.
The demos are small enough that this is honest rather than heroic: a 2-dimensional
bottleneck on 600 points converges in a few hundred steps, and the point of the
atom is the score and its calibration, not the optimiser.
"""

from __future__ import annotations

import math

import numpy as np

VARIANTS = ("ae", "vae", "lstm_ae", "denoising_ae", "thresholding")
# --------------------------------------------------------------------------- #
# Small dense network, hand-written forward and backward
# --------------------------------------------------------------------------- #
def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def _mlp(params: list, X: np.ndarray) -> tuple:
    """Forward through [W1, b1, W2, b2, ...]: tanh hidden, linear output."""
    layers = len(params) // 2
    pres, posts = [], [X]
    for i in range(layers):
        z = X @ params[2 * i].T + params[2 * i + 1]
        pres.append(z)
        X = np.tanh(z) if i < layers - 1 else z
        posts.append(X)
    return X, (pres, posts)


def _mlp_back(params: list, cache: tuple, delta: np.ndarray) -> list:
    """Gradients of a scalar loss given dL/d(output)."""
    pres, posts = cache
    layers = len(params) // 2
    grads = [None] * (2 * layers)
    for i in range(layers - 1, -1, -1):
        grads[2 * i] = delta.T @ posts[i]
        grads[2 * i + 1] = delta.sum(axis=0)
        if i:
            delta = (delta @ params[2 * i]) * (1.0 - posts[i] ** 2)
    return grads


def _mlp_input(params: list, cache: tuple, delta: np.ndarray) -> np.ndarray:
    """dL/d(input) for the same loss _mlp_back differentiates."""
    posts = cache[1]
    for i in range(len(params) // 2 - 1, 0, -1):
        delta = (delta @ params[2 * i]) * (1.0 - posts[i] ** 2)
    return delta @ params[0]


def _init_mlp(rng: np.random.Generator, sizes: list) -> list:
    """Glorot-uniform weights and zero biases, in the flat list _mlp expects."""
    params = []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:], strict=True):
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        params.append(rng.uniform(-limit, limit, size=(fan_out, fan_in)))
        params.append(np.zeros(fan_out))
    return params


class _Adam:
    """Adam on a flat list of arrays; deterministic given the gradients."""

    def __init__(self, params: list, lr: float) -> None:
        self.p = params
        self.lr = lr
        self.m = [np.zeros_like(a) for a in params]
        self.v = [np.zeros_like(a) for a in params]
        self.t = 0

    def step(self, grads: list, lr: float | None = None) -> None:
        rate = self.lr if lr is None else lr
        self.t += 1
        c1 = 1.0 - 0.9 ** self.t
        c2 = 1.0 - 0.999 ** self.t
        for i, g in enumerate(grads):
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
            self.p[i] -= rate * (self.m[i] / c1) / (np.sqrt(self.v[i] / c2) + 1e-8)


# --------------------------------------------------------------------------- #
# LSTM, batched over sequences
# --------------------------------------------------------------------------- #
def _lstm_forward(X: np.ndarray, params: list, h0: np.ndarray,
                  c0: np.ndarray) -> tuple:
    """Run an LSTM over (m, T, dx); gates ordered input, forget, output, cell."""
    W, U, b = params
    h, c = h0, c0
    width = h.shape[1]
    cache, outs = [], []
    for t in range(X.shape[1]):
        z = X[:, t] @ W.T + h @ U.T + b
        i = _sigmoid(z[:, :width])
        f = _sigmoid(z[:, width:2 * width])
        o = _sigmoid(z[:, 2 * width:3 * width])
        g = np.tanh(z[:, 3 * width:])
        c_prev, h_prev = c, h
        c = f * c_prev + i * g
        tc = np.tanh(c)
        h = o * tc
        cache.append((X[:, t], h_prev, c_prev, i, f, o, g, tc))
        outs.append(h)
    return h, c, np.stack(outs, axis=1), cache


def _lstm_backward(cache: list, params: list, dh_seq: np.ndarray,
                   dc_last: np.ndarray) -> tuple:
    """BPTT through _lstm_forward; returns dx, dh0, dc0 and the parameter grads."""
    W, U, b = params
    width = W.shape[0] // 4
    dW = np.zeros_like(W)
    dU = np.zeros_like(U)
    db = np.zeros_like(b)
    steps = len(cache)
    dx = np.zeros((dh_seq.shape[0], steps, W.shape[1]))
    dh_next = np.zeros((dh_seq.shape[0], width))
    dc_next = dc_last
    for t in range(steps - 1, -1, -1):
        x_t, h_prev, c_prev, i, f, o, g, tc = cache[t]
        dh = dh_seq[:, t] + dh_next
        do = dh * tc
        dc = dc_next + dh * o * (1.0 - tc ** 2)
        dz = np.concatenate([
            dc * g * i * (1.0 - i),
            dc * c_prev * f * (1.0 - f),
            do * o * (1.0 - o),
            dc * i * (1.0 - g ** 2),
        ], axis=1)
        dW += dz.T @ x_t
        dU += dz.T @ h_prev
        db += dz.sum(axis=0)
        dx[:, t] = dz @ W
        dh_next = dz @ U
        dc_next = dc * f
    return dx, dh_next, dc_next, (dW, dU, db)


# --------------------------------------------------------------------------- #
# The detector
# --------------------------------------------------------------------------- #
class AutoencoderAnomaly:
    """Reconstruction-error detector with five variants.

    Parameters
    ----------
    variant : {"ae", "vae", "lstm_ae", "denoising_ae", "thresholding"}
        Which model or which score. `thresholding` fits the plain autoencoder and
        changes only the score, to the standardised residual sum of (6).
    latent : int
        Bottleneck width. The manifold dimension the model can represent.
    hidden : int
        Width of the single hidden layer in encoder and decoder.
    epochs, lr, batch_size : int, float, int
        Adam schedule. Full-batch when `batch_size` exceeds the sample size.
    corruption : float
        Standard deviation of the Gaussian noise added to the *input* during
        `denoising_ae` training. The target is always the clean input.
    beta : float
        Weight on the KL term of (4). 1.0 is the exact ELBO, and on data that
        lives on a thin manifold it collapses the encoder to the prior: the KL
        is charged per latent dimension while the reconstruction gain is bounded
        by the data variance, so past a threshold the constant solution wins.
        The default 0.1 is a beta-VAE weight, not the ELBO.
    contamination : float or None
        Flag budget for `predict`. `None` uses the chi-square threshold of (6) for
        `thresholding`, and the `alpha` quantile of the training scores otherwise.
    alpha : float
        Nominal tail probability when `contamination` is None.
    seed : int
        Controls initialisation, minibatch order and the VAE's reparameterisation.

    Attributes
    ----------
    scores_ : ndarray
        Training score per row.
    threshold_ : float
    recon_ : ndarray
        Training reconstruction, same shape as `X` after `fit`.
    residual_ : ndarray or None
        Per-feature residual standard deviations, for `thresholding`.
    n_features_ : int
    loss_history_ : list of float
        Mean training loss per epoch, for diagnostics.
    """

    def __init__(self, *, variant: str = "ae", latent: int = 2, hidden: int = 16,
                 epochs: int = 300, lr: float = 0.02, batch_size: int = 64,
                 corruption: float = 0.3, beta: float = 0.1,
                 contamination: float | None = 0.1, alpha: float = 0.05,
                 seed: int = 0) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
        if latent < 1 or hidden < 1:
            raise ValueError("latent and hidden must be positive")
        if epochs < 1 or batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if not 0.0 < lr < 1.0:
            raise ValueError("lr must lie in (0, 1)")
        if corruption < 0.0 or beta <= 0.0:
            raise ValueError("corruption must be non-negative and beta positive")
        if contamination is not None and not 0.0 < contamination < 1.0:
            raise ValueError("contamination must lie in (0, 1) or be None")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        self.variant = variant
        self.latent = int(latent)
        self.hidden = int(hidden)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.corruption = float(corruption)
        self.beta = float(beta)
        self.contamination = contamination
        self.alpha = float(alpha)
        self.seed = int(seed)

    # -- fitting ----------------------------------------------------------- #
    def fit(self, X: np.ndarray) -> AutoencoderAnomaly:
        X = np.asarray(X, dtype=float)
        if self.variant == "lstm_ae":
            if X.ndim != 3:
                raise ValueError("lstm_ae needs X of shape (n, steps, features)")
            self.n_, self.n_steps_, self.n_features_ = X.shape
            if self.n_ < 3 or self.n_steps_ < 2 or self.n_features_ < 1:
                raise ValueError("need at least three sequences of two steps")
        else:
            if X.ndim == 1:
                X = X[:, None]
            if X.ndim != 2:
                raise ValueError("X must be one- or two-dimensional")
            self.n_, self.n_features_ = X.shape
            if self.n_ < 3 or self.n_features_ < 1:
                raise ValueError("need at least three points and one feature")
        self.data_ = X
        self.loss_history_ = []
        rng = np.random.default_rng(self.seed)

        if self.variant == "lstm_ae":
            self._fit_lstm(rng)
        elif self.variant == "vae":
            self._fit_vae(rng)
        else:
            noisy = self.variant == "denoising_ae"
            self._fit_ae(rng, noisy)

        self.recon_ = self._reconstruct(X)
        if self.variant == "thresholding":
            error = (X - self.recon_) ** 2
            self.residual_ = np.sqrt(error.mean(axis=0)) + 1e-12
            self.scores_ = self.score_samples(X)
        else:
            self.residual_ = None
            self.scores_ = self.score_samples(X)
        self.threshold_ = self._threshold()
        return self

    def _fit_ae(self, rng: np.random.Generator, noisy: bool) -> None:
        d, k, h = self.n_features_, self.latent, self.hidden
        enc = _init_mlp(rng, [d, h, k])
        dec = _init_mlp(rng, [k, h, d])
        opt = _Adam(enc + dec, self.lr)
        X = self.data_
        for _ in range(self.epochs):
            total = 0.0
            for index in self._batches(rng, len(X)):
                clean = X[index]
                seen = clean
                if noisy:
                    seen = clean + self.corruption * rng.standard_normal(clean.shape)
                code, enc_cache = _mlp(enc, seen)
                out, dec_cache = _mlp(dec, code)
                diff = out - clean
                total += float(np.sum(diff ** 2))
                delta_out = 2.0 * diff / (len(index) * d)
                d_code = _mlp_input(dec, dec_cache, delta_out)
                grads = _mlp_back(enc, enc_cache, d_code)
                grads += _mlp_back(dec, dec_cache, delta_out)
                opt.step(grads)
            self.loss_history_.append(total / (len(X) * d))
        self.enc_, self.dec_ = enc, dec

    def _fit_vae(self, rng: np.random.Generator) -> None:
        d, k, h = self.n_features_, self.latent, self.hidden
        enc = _init_mlp(rng, [d, h, 2 * k])
        dec = _init_mlp(rng, [k, h, d])
        opt = _Adam(enc + dec, self.lr)
        X = self.data_
        for _ in range(self.epochs):
            total = 0.0
            for index in self._batches(rng, len(X)):
                clean = X[index]
                m = len(index)
                head, enc_cache = _mlp(enc, clean)
                mu = head[:, :k]
                logvar = np.clip(head[:, k:], -8.0, 8.0)
                sd = np.exp(0.5 * logvar)
                eps = rng.standard_normal(mu.shape)
                code = mu + sd * eps
                out, dec_cache = _mlp(dec, code)
                diff = out - clean
                kl = 0.5 * np.sum(mu ** 2 + np.exp(logvar) - 1.0 - logvar, axis=1)
                total += float(np.sum(diff ** 2)) + self.beta * float(kl.sum())
                delta_out = 2.0 * diff / m
                d_code = _mlp_input(dec, dec_cache, delta_out)
                d_mu = d_code + self.beta * mu / m
                d_logvar = (d_code * eps * 0.5 * sd
                            + self.beta * 0.5 * (np.exp(logvar) - 1.0) / m)
                delta_head = np.concatenate([d_mu, d_logvar], axis=1)
                grads = _mlp_back(enc, enc_cache, delta_head)
                grads += _mlp_back(dec, dec_cache, delta_out)
                opt.step(grads)
            self.loss_history_.append(total / len(X))
        self.enc_, self.dec_ = enc, dec

    def _fit_lstm(self, rng: np.random.Generator) -> None:
        d, h, steps = self.n_features_, self.hidden, self.n_steps_
        enc = [rng.normal(scale=0.3, size=(4 * h, d)),
               rng.normal(scale=0.3, size=(4 * h, h)),
               np.zeros(4 * h)]
        dec = [np.zeros((4 * h, 0)),
               rng.normal(scale=0.3, size=(4 * h, h)),
               np.zeros(4 * h)]
        out_w = rng.normal(scale=0.3, size=(d, h))
        out_b = np.zeros(d)
        opt = _Adam(enc + dec + [out_w, out_b], self.lr)
        X = self.data_
        for _ in range(self.epochs):
            total = 0.0
            for index in self._batches(rng, len(X)):
                clean = X[index]
                m = len(index)
                zeros = np.zeros((m, steps, 0))
                h0 = np.zeros((m, h))
                h_t, c_t, _, enc_cache = _lstm_forward(clean, enc, h0, h0)
                _, _, states, dec_cache = _lstm_forward(
                    zeros, dec, h_t, c_t)
                out = states @ out_w.T + out_b
                diff = out - clean
                total += float(np.sum(diff ** 2))
                delta_out = 2.0 * diff / (m * steps * d)
                d_out_w = np.einsum("mtd,mth->dh", delta_out, states)
                d_out_b = delta_out.sum(axis=(0, 1))
                d_states = delta_out @ out_w
                _, dh0, dc0, d_dec = _lstm_backward(
                    dec_cache, dec, d_states, np.zeros((m, h)))
                d_enc_seq = np.zeros((m, steps, h))
                d_enc_seq[:, -1] = dh0
                _, _, _, d_enc = _lstm_backward(enc_cache, enc, d_enc_seq, dc0)
                opt.step(list(d_enc) + list(d_dec) + [d_out_w, d_out_b])
            self.loss_history_.append(total / (len(X) * steps * d))
        self.enc_, self.dec_ = enc, dec
        self.out_w_, self.out_b_ = out_w, out_b

    def _batches(self, rng: np.random.Generator, n: int):
        order = rng.permutation(n)
        size = min(self.batch_size, n)
        for start in range(0, n, size):
            yield order[start:start + size]

    # -- scoring ----------------------------------------------------------- #
    def _reconstruct(self, X: np.ndarray) -> np.ndarray:
        if self.variant == "lstm_ae":
            h0 = np.zeros((len(X), self.hidden))
            h_t, c_t, _, _ = _lstm_forward(X, self.enc_, h0, h0)
            _, _, states, _ = _lstm_forward(
                np.zeros((len(X), X.shape[1], 0)), self.dec_, h_t, c_t)
            return states @ self.out_w_.T + self.out_b_
        if self.variant == "vae":
            head, _ = _mlp(self.enc_, X)
            code = head[:, :self.latent]
        else:
            code, _ = _mlp(self.enc_, X)
        out, _ = _mlp(self.dec_, code)
        return out

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        X = self._check(X)
        recon = self._reconstruct(X)
        if self.variant == "lstm_ae":
            return ((X - recon) ** 2).mean(axis=(1, 2))
        if self.variant == "thresholding":
            return (((X - recon) ** 2) / self.residual_ ** 2).sum(axis=1)
        if self.variant == "vae":
            head, _ = _mlp(self.enc_, X)
            mu = head[:, :self.latent]
            logvar = np.clip(head[:, self.latent:], -8.0, 8.0)
            kl = 0.5 * np.sum(mu ** 2 + np.exp(logvar) - 1.0 - logvar, axis=1)
            return ((X - recon) ** 2).sum(axis=1) + self.beta * kl
        return ((X - recon) ** 2).mean(axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "threshold_"):
            raise RuntimeError("fit before predicting")
        return np.where(self.score_samples(X) > self.threshold_, -1, 1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).predict(X)

    def _threshold(self) -> float:
        if self.contamination is not None:
            return float(np.quantile(self.scores_, 1.0 - self.contamination))
        if self.variant == "thresholding":
            return float(_chi2_ppf(self.alpha, self.n_features_))
        raise ValueError("contamination=None is only calibrated for the "
                         "thresholding variant; pass a contamination budget")

    def _check(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "n_features_"):
            raise RuntimeError("fit before scoring")
        X = np.asarray(X, dtype=float)
        if self.variant == "lstm_ae":
            if X.ndim != 3 or X.shape[1:] != (self.n_steps_, self.n_features_):
                raise ValueError(f"expected shape (m, {self.n_steps_}, "
                                 f"{self.n_features_})")
            return X
        if X.ndim == 1:
            X = X[:, None]
        if X.ndim != 2 or X.shape[1] != self.n_features_:
            raise ValueError(f"expected shape (m, {self.n_features_})")
        return X


def _chi2_ppf(alpha: float, dof: int) -> float:
    """Wilson-Hilferty: the chi-square quantile, accurate to a fraction of a
    percent for dof above a handful and exact in the limit.
    """
    z = _norm_ppf(1.0 - alpha)
    factor = 1.0 - 2.0 / (9.0 * dof) + z * math.sqrt(2.0 / (9.0 * dof))
    return dof * factor ** 3


def _norm_ppf(q: float) -> float:
    """Acklam's rational approximation, good to 1e-9 over (0, 1)."""
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    low, high = 0.02425, 1.0 - 0.02425
    if q < low:
        t = math.sqrt(-2.0 * math.log(q))
        return (((((c[0] * t + c[1]) * t + c[2]) * t + c[3]) * t + c[4]) * t
                + c[5]) / ((((d[0] * t + d[1]) * t + d[2]) * t + d[3]) * t + 1.0)
    if q > high:
        t = math.sqrt(-2.0 * math.log(1.0 - q))
        return -(((((c[0] * t + c[1]) * t + c[2]) * t + c[3]) * t + c[4]) * t
                 + c[5]) / ((((d[0] * t + d[1]) * t + d[2]) * t + d[3]) * t + 1.0)
    t = q - 0.5
    r = t * t
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r
            + a[5]) * t / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r
                            + b[4]) * r + 1.0)


# --------------------------------------------------------------------------- #
# Demos
# --------------------------------------------------------------------------- #
def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC with ties averaged: P(positive scores above a negative)."""
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    for value in np.unique(scores):
        tied = scores == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    positive = labels.astype(bool)
    n_pos = int(positive.sum())
    n_neg = len(labels) - n_pos
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2)
                 / (n_pos * n_neg))


def _circle(rng: np.random.Generator, n: int, radius: float = 0.5,
            noise: float = 0.02) -> np.ndarray:
    """Points on a circle of the given radius, with isotropic noise."""
    angles = rng.uniform(0.0, 2.0 * np.pi, n)
    points = radius * np.column_stack([np.cos(angles), np.sin(angles)])
    return points + noise * rng.normal(size=(n, 2))


def _ring(rng: np.random.Generator, n: int, radius: float, dim: int,
          basis: np.ndarray, noise: float = 0.02) -> np.ndarray:
    """A circle of the given radius living in the first two columns of `basis`."""
    angles = rng.uniform(0.0, 2.0 * np.pi, n)
    points = np.zeros((n, dim))
    points[:, 0] = radius * np.cos(angles)
    points[:, 1] = radius * np.sin(angles)
    return points @ basis.T + noise * rng.normal(size=(n, dim))


def _demo_width() -> None:
    """The bottleneck has to be as wide as the manifold, and no wider."""
    rng = np.random.default_rng(0)
    circle = _circle(rng, 400)

    fits = {}
    for latent in (1, 2, 3):
        model = AutoencoderAnomaly(latent=latent, epochs=600, seed=0).fit(circle)
        fits[latent] = float(np.mean((model.recon_ - circle) ** 2))

    probes = np.random.default_rng(1).uniform(0.0, 2.0 * np.pi, 30)
    unit = np.column_stack([np.cos(probes), np.sin(probes)])
    hole = 0.1 * unit + 0.02 * np.random.default_rng(2).normal(size=(30, 2))
    shell = 0.9 * unit + 0.02 * np.random.default_rng(3).normal(size=(30, 2))

    labels = np.r_[np.zeros(400, bool), np.ones(30, bool)]
    scores = {}
    for latent in (1, 2):
        model = AutoencoderAnomaly(latent=latent, epochs=600, seed=0).fit(circle)
        for name, points in (("hole", hole), ("shell", shell)):
            joined = np.r_[model.scores_, model.score_samples(points)]
            scores[latent, name] = _auc(joined, labels)

    assert fits[2] < 0.01 * fits[1]
    assert fits[3] < 3.0 * fits[2]
    assert scores[1, "hole"] < 0.2
    assert scores[2, "hole"] > 0.9
    assert scores[1, "shell"] > 0.7
    assert scores[2, "shell"] > 0.95

    print("a circle of radius 0.5 in the plane, four hundred points:")
    for latent, mse in fits.items():
        print(f"  bottleneck {latent}: reconstruction mse {mse:.2e}")
    print("  a two-dimensional bottleneck is the manifold's own dimension, and "
          "it is four orders of magnitude better than one; a third dimension "
          "buys nothing.")
    print("AUC against thirty points in the hole (radius 0.1) and on a shell "
          "(radius 0.9):")
    for latent in (1, 2):
        print(f"  bottleneck {latent}: hole {scores[latent, 'hole']:.3f}   "
              f"shell {scores[latent, 'shell']:.3f}")
    print(f"  Too narrow, the model is a prototype: the hole is closer to the "
          f"one-dimensional code than the circle is, so it scores {scores[1, 'hole']:.3f} "
          f"-- ranked below every real point. At the right width the hole scores "
          f"{scores[2, 'hole']:.3f}.")


def _demo_hole() -> None:
    """The interior of the learned span is reconstructed perfectly."""
    rng = np.random.default_rng(7)
    dim = 8
    basis, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    ring = _ring(rng, 500, 0.5, dim, basis)
    hole = _ring(rng, 30, 0.1, dim, basis)
    shell = _ring(rng, 30, 1.1, dim, basis)
    labels = np.r_[np.zeros(500, bool), np.ones(30, bool)]

    rows = {}
    for variant in ("ae", "denoising_ae"):
        for latent in (2, 4, 8):
            model = AutoencoderAnomaly(variant=variant, latent=latent, hidden=32,
                                       epochs=600, seed=0).fit(ring)
            rows[variant, latent] = (
                float(model.scores_.mean()),
                float(model.score_samples(hole).mean()),
                _auc(np.r_[model.scores_, model.score_samples(hole)], labels),
                _auc(np.r_[model.scores_, model.score_samples(shell)], labels),
            )

    assert all(0.4 < rows["ae", k][2] < 0.8 for k in (2, 4, 8))
    assert all(rows["denoising_ae", k][2] < 0.1 for k in (2, 4, 8))
    assert all(row[3] > 0.99 for row in rows.values())
    assert rows["denoising_ae", 8][0] > 50.0 * rows["ae", 8][0]

    print("the same construction in eight dimensions, five hundred points on a "
          "circle of radius 0.5 in a random plane:")
    print("  variant        latent   train score   hole score   AUC hole   AUC shell")
    for (variant, latent), (train, hole_score, auc_hole, auc_shell) in rows.items():
        print(f"  {variant:<13s} {latent:>6d}   {train:11.2e}   "
              f"{hole_score:10.2e}   {auc_hole:8.3f}   {auc_shell:9.3f}")
    print(f"  Both variants catch the shell at AUC {rows['ae', 2][3]:.3f}. The hole "
          f"is a different matter: the plain autoencoder sits at chance, and the "
          f"denoiser is far below it. Predicting the clean input from a noisy one "
          f"learns a conditional mean, and a conditional mean is a shrinkage "
          f"operator -- it pulls the centre of the span towards itself, so the "
          f"hole reconstructs better than the data.")


def _demo_vae() -> None:
    """Beta trades reconstruction against the prior, and the KL is exact."""
    rng = np.random.default_rng(0)
    circle = _circle(rng, 400)

    rows = {}
    for beta in (0.03, 0.1, 0.3):
        model = AutoencoderAnomaly(variant="vae", latent=2, epochs=500,
                                   beta=beta, seed=0).fit(circle)
        mu = _mlp(model.enc_, circle)[0][:, :2]
        rows[beta] = (float(np.mean((model.recon_ - circle) ** 2)),
                      float(np.linalg.norm(mu, axis=1).mean()))

    assert rows[0.03][1] > rows[0.1][1] > rows[0.3][1]
    assert rows[0.03][0] < rows[0.3][0] / 20.0
    assert rows[0.03][1] > 1.0
    assert rows[0.3][1] < 0.05

    model = AutoencoderAnomaly(variant="vae", latent=1, beta=1.0, epochs=1,
                               seed=0).fit(np.zeros((4, 1)))
    probe = np.zeros((1, 1))
    cases = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.5),
             (0.0, math.log(0.25), 0.5 * (0.25 - 1.0 - math.log(0.25))))
    exact = []
    for mu, logvar, closed in cases:
        model.enc_ = [np.zeros((2, 1)), np.zeros(2), np.zeros((2, 2)),
                      np.array([mu, logvar])]
        model.dec_ = [np.zeros((2, 1)), np.zeros(2), np.zeros((1, 2)),
                      np.zeros(1)]
        value = float(model.score_samples(probe)[0])
        assert abs(value - closed) < 1e-12
        exact.append((mu, logvar, value, closed))

    print("a beta-vae on the same circle, five hundred epochs:")
    for beta, (mse, norm) in rows.items():
        print(f"  beta {beta:<5}: reconstruction mse {mse:.2e}   "
              f"mean |mu| {norm:.3f}")
    print(f"  At beta 1.0 -- the exact ELBO -- the encoder collapses to the "
          f"prior: the KL is charged per latent dimension while the gain is "
          f"bounded by the data variance, so past beta about 0.25 the constant "
          f"solution wins and the latent mean falls to {rows[0.3][1]:.3f}.")
    print("The KL term of (4) against its closed form for a one-dimensional "
          "Gaussian posterior, at beta 1:")
    for mu, logvar, value, closed in exact:
        print(f"  mu {mu:4.1f}  log var {logvar:7.3f}  "
              f"KL {value:.6f}  expected {closed:.6f}")
    print("  With no reconstruction term the score is the KL itself, so this "
          "checks the formula and not just its shape.")


def _demo_residual() -> None:
    """Dividing by each feature's own scale is what finds a quiet outlier."""
    rng = np.random.default_rng(3)
    scale = np.array([10.0, 0.1, 1.0])
    train = rng.normal(size=(600, 3)) * scale
    fresh = rng.normal(size=(600, 3)) * scale
    probe = np.array([[0.0, 0.5, 0.0]])

    rows = {}
    for variant in ("ae", "thresholding"):
        model = AutoencoderAnomaly(variant=variant, latent=2, hidden=8,
                                   epochs=500, contamination=0.05,
                                   seed=0).fit(train)
        score = float(model.score_samples(probe)[0])
        rows[variant] = (score, float((model.scores_ < score).mean()),
                         float(model.threshold_),
                         float((model.predict(fresh) == -1).mean()))

    model = AutoencoderAnomaly(variant="thresholding", latent=2, hidden=8,
                               epochs=500, contamination=None, alpha=0.05,
                               seed=0).fit(train)
    chi = float(model.threshold_)
    rms = np.sqrt(((train - model._reconstruct(train)) ** 2).mean(axis=0))
    in_sample = float(np.mean(model.predict(train) == -1))

    assert rows["ae"][1] < 0.5
    assert rows["thresholding"][1] > 0.99
    assert rows["thresholding"][0] > 20.0
    assert abs(chi - _chi2_ppf(0.05, 3)) < 1e-9
    assert abs(float(model.scores_.mean()) - 3.0) < 1e-9
    assert 0.03 < in_sample < 0.07

    print("three features with standard deviations 10, 0.1 and 1, and one "
          "probe that is five sigma out on the quiet one:")
    print("  variant        score   percentile   threshold   fresh flags")
    for variant, (score, pct, thr, rate) in rows.items():
        print(f"  {variant:<13s} {score:6.3f}   {pct:10.3f}   {thr:9.3f}   "
              f"{rate:11.1%}")
    print(f"  Per-feature residual rms {rms.round(3)} against data sds "
          f"{scale.round(3)}: the network reproduces the loud feature to 2% of "
          f"its spread and the quiet one to 97%, because that is what a "
          f"squared-error loss on the raw scale asks for.")
    print(f"  The unweighted score ranks the probe at the "
          f"{rows['ae'][1]:.1%} percentile; standardising each residual by its "
          f"own rms puts it at {rows['thresholding'][1]:.1%}. The in-sample "
          f"mean of the standardised score is exactly $d = 3$, and the "
          f"chi-square threshold at alpha 0.05 is {chi:.3f} (exact 7.815), "
          f"which flags {in_sample:.1%} of the training set. On fresh normal "
          f"data it flags {rows['thresholding'][3]:.1%}: the residual scales "
          f"were estimated in sample, so the reference law is too narrow.")


def _demo_sequence() -> None:
    """A shared recurrent decoder learns the phase, not the samples."""
    rng = np.random.default_rng(11)
    steps = 24
    n = 240
    grid = np.arange(steps)

    def sine(phase: float, cycles: float = 1.0) -> np.ndarray:
        return np.sin(2.0 * np.pi * cycles * grid / steps + phase)

    train = np.stack([sine(p) for p in rng.uniform(0.0, 2.0 * np.pi, n)])
    train = train[:, :, None] + 0.02 * rng.normal(size=(n, steps, 1))
    unseen = np.stack([sine(p) for p in rng.uniform(0.0, 2.0 * np.pi, 60)])
    unseen = unseen[:, :, None] + 0.02 * rng.normal(size=(60, steps, 1))
    spike = train[:20].copy()
    spike[:, 12, 0] += 0.8
    fast = np.stack([sine(p, cycles=2.0) for p in rng.uniform(0.0, 2.0 * np.pi, 20)])
    fast = fast[:, :, None] + 0.02 * rng.normal(size=(20, steps, 1))

    model = AutoencoderAnomaly(variant="lstm_ae", hidden=12, epochs=60,
                               batch_size=64, seed=0).fit(train)
    scores = model.score_samples(train)
    labels = np.r_[np.zeros(n, bool), np.ones(20, bool)]
    auc_spike = _auc(np.r_[scores, model.score_samples(spike)], labels)
    auc_fast = _auc(np.r_[scores, model.score_samples(fast)], labels)
    unseen_scores = model.score_samples(unseen)
    threshold = float(np.quantile(scores, 0.95))
    unseen_rate = float(np.mean(unseen_scores > threshold))
    mean_auc = _auc(np.abs(np.r_[train.mean(axis=(1, 2)), fast.mean(axis=(1, 2))]),
                    labels)

    assert auc_spike > 0.98
    assert auc_fast > 0.98
    assert unseen_rate < 0.06
    assert mean_auc < 0.6

    print("two hundred and forty sine waves of random phase, twenty-four "
          "samples each, one latent state:")
    print(f"  loss {model.loss_history_[0]:.4f} -> {model.loss_history_[-1]:.6f}")
    print(f"  clean {scores.mean():.2e}   unseen phase {unseen_scores.mean():.2e}   "
          f"spike {model.score_samples(spike).mean():.2e}   "
          f"double frequency {model.score_samples(fast).mean():.2e}")
    print(f"  AUC spike {auc_spike:.3f}   AUC double frequency {auc_fast:.3f}   "
          f"unseen phases above the 95th percentile: {unseen_rate:.1%}")
    print(f"  The sixty unseen phases score like the training phases, because the "
          f"model learned the family and not the sample. A detector built on the "
          f"window mean gets AUC {mean_auc:.3f} on the double-frequency windows: "
          f"a whole number of extra cycles leaves the mean where it was.")


def _demo_broken() -> None:
    """Every malformed call is refused rather than silently scored."""
    circle = np.random.default_rng(0).normal(size=(40, 2))
    calls = [
        lambda: AutoencoderAnomaly(variant="nope"),
        lambda: AutoencoderAnomaly(latent=0),
        lambda: AutoencoderAnomaly(hidden=0),
        lambda: AutoencoderAnomaly(epochs=0),
        lambda: AutoencoderAnomaly(batch_size=0),
        lambda: AutoencoderAnomaly(lr=0.0),
        lambda: AutoencoderAnomaly(lr=1.5),
        lambda: AutoencoderAnomaly(corruption=-1.0),
        lambda: AutoencoderAnomaly(beta=0.0),
        lambda: AutoencoderAnomaly(contamination=0.0),
        lambda: AutoencoderAnomaly(contamination=1.0),
        lambda: AutoencoderAnomaly(alpha=0.0),
        lambda: AutoencoderAnomaly().fit(np.zeros((2, 3))),
        lambda: AutoencoderAnomaly().fit(np.zeros((10, 0))),
        lambda: AutoencoderAnomaly().fit(np.zeros((4, 2, 2))),
        lambda: AutoencoderAnomaly(variant="lstm_ae").fit(np.zeros((10, 2))),
        lambda: AutoencoderAnomaly(variant="lstm_ae").fit(np.zeros((2, 4, 1))),
        lambda: AutoencoderAnomaly().fit(circle).score_samples(np.zeros((3, 5))),
        lambda: AutoencoderAnomaly().predict(circle),
        lambda: AutoencoderAnomaly(contamination=None).fit(circle),
    ]
    for call in calls:
        try:
            call()
        except (ValueError, RuntimeError):
            continue
        raise AssertionError("expected a refusal")
    print(f"{len(calls)} malformed calls refused: bad arguments, too few rows, "
          f"wrong rank, wrong width, scoring before fitting, and a "
          f"chi-square threshold asked of a variant that has no null law.")


def main() -> None:
    _demo_width()
    _demo_hole()
    _demo_vae()
    _demo_residual()
    _demo_sequence()
    _demo_broken()
    print("autoencoder anomaly demo passed")


if __name__ == "__main__":
    main()
