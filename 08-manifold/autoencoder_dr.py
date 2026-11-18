"""Autoencoder embedding — a network with a narrow waist, trained to rebuild its input.

Chapter:   08-manifold
Companion: autoencoder_dr.md
Tier 2 · 4 variants: ae, denoising_ae, vae_embed, contractive

    encoder   z = tanh(x W₁ + b₁) W₂ + b₂                        (1)
    decoder   x̂ = tanh(z W₃ + b₃) W₄ + b₄                        (2)
    loss      ℒ = ½ ‖x − x̂‖²                                     (3)
    denoising   x̃ = x + ε,   ℒ = ½ ‖x − x̂(x̃)‖²                  (4)
    variational z = μ + σ ⊙ ε,  ℒ = ½‖x − x̂(z)‖² + β KL          (5)
    contractive ℒ = ½‖x − x̂‖² + λ ‖∂z/∂x‖²_F                     (6)

Every other atom in this chapter fixes the map before it computes anything: LLE
solves for weights, Laplacian eigenmaps takes a graph, Isomap takes distances.
An autoencoder *learns* the map, and the only constraints are the shape of the
network — a bottleneck of width `d` — and the requirement that the input can be
rebuilt from the code. Nothing is said about smoothness, neighbours or distances.
If the data lie on a `d`-dimensional surface, the cheapest way to satisfy the
bottleneck is to encode a coordinate on that surface and decode from it, and the
embedding is whatever the optimiser finds.

That freedom is both the appeal and the problem. There is no closed form, no
guaranteed optimum and no invariance: another seed, a wider hidden layer or more
epochs gives another embedding, and the only test of quality is a measurement
applied afterwards. The four variants constrain the solution to be less
arbitrary — robust to noise, Gaussian in the code, or slow to change with the
input — and this file reports those constraints as measurements rather than
claims.

Two choices keep it small and honest. One hidden layer per side, `tanh` at both,
a linear latent: nonlinear enough to unroll a sheet, simple enough that the
encoder Jacobian in (6) has a closed form. And training is full-batch Adam from a
seeded initialisation with no shuffling, so two runs are identical and every
number in the companion document is reproducible.
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("ae", "denoising_ae", "vae_embed", "contractive")


# --------------------------------------------------------------------------- #
# The network
# --------------------------------------------------------------------------- #
class _Net:
    """One hidden layer per side, `tanh`, linear latent, manual backprop."""

    def __init__(self, n_features: int, n_hidden: int, n_latent: int,
                 variational: bool, rng: np.random.Generator) -> None:
        D, h, d = n_features, n_hidden, n_latent
        width = 2 * d if variational else d
        self.d = d
        self.variational = variational
        self.p = {
            "W1": rng.standard_normal((D, h)) / np.sqrt(D),
            "b1": np.zeros(h),
            "W2": rng.standard_normal((h, width)) / np.sqrt(h),
            "b2": np.zeros(width),
            "W3": rng.standard_normal((d, h)) / np.sqrt(d),
            "b3": np.zeros(h),
            "W4": rng.standard_normal((h, D)) / np.sqrt(h),
            "b4": np.zeros(D),
        }

    def forward(self, X: np.ndarray, rng: np.random.Generator,
                noise: float = 0.0, eps: np.ndarray | None = None
                ) -> dict[str, np.ndarray]:
        """Encode, sample, decode — everything the backward pass needs.

        `X` is `(n, D)`; `eps` overrides the Gaussian draw so a gradient check
        can hold the sample fixed. Returns the activations and the sample.
        """
        p = self.p
        Xin = X + noise * rng.standard_normal(X.shape) if noise > 0.0 else X
        H = np.tanh(Xin @ p["W1"] + p["b1"])
        E = H @ p["W2"] + p["b2"]
        mu = logvar = None
        if self.variational:
            mu, logvar = E[:, :self.d], E[:, self.d:]
            sigma = np.exp(0.5 * logvar)
            eps = rng.standard_normal(mu.shape) if eps is None else eps
            Z = mu + sigma * eps
        else:
            Z = E
        G = np.tanh(Z @ p["W3"] + p["b3"])
        return {"Xin": Xin, "H": H, "E": E, "Z": Z, "G": G,
                "Xhat": G @ p["W4"] + p["b4"],
                "mu": mu, "logvar": logvar, "eps": eps}

    def backward(self, c: dict[str, np.ndarray], X: np.ndarray,
                 kl_weight: float, contractive_weight: float
                 ) -> tuple[float, dict[str, np.ndarray]]:
        """Loss and gradients, in closed form for all four variants."""
        p = self.p
        n = X.shape[0]
        R = c["Xhat"] - X
        loss = 0.5 * float(np.mean(np.sum(R * R, axis=1)))
        g = {}

        d_Xhat = R / n
        g["W4"] = c["G"].T @ d_Xhat
        g["b4"] = d_Xhat.sum(axis=0)
        d_G = d_Xhat @ p["W4"].T
        d_pre = d_G * (1.0 - c["G"] ** 2)
        g["W3"] = c["Z"].T @ d_pre
        g["b3"] = d_pre.sum(axis=0)
        d_Z = d_pre @ p["W3"].T

        if self.variational:
            mu, logvar, eps = c["mu"], c["logvar"], c["eps"]
            sigma = np.exp(0.5 * logvar)
            loss += kl_weight * 0.5 * float(np.mean(
                np.sum(sigma**2 + mu**2 - 1.0 - logvar, axis=1)))
            d_mu = d_Z + kl_weight * mu / n
            d_logvar = d_Z * 0.5 * sigma * eps + kl_weight * 0.5 * (
                sigma**2 - 1.0) / n
            d_E = np.hstack([d_mu, d_logvar])
        else:
            d_E = d_Z

        g["W2"] = c["H"].T @ d_E
        g["b2"] = d_E.sum(axis=0)
        d_H = d_E @ p["W2"].T

        u = 1.0 - c["H"] ** 2
        if contractive_weight > 0.0:
            # ‖J_i‖²_F = u_iᵀ (A ⊙ B) u_i with A = W₁ᵀW₁, B = W₂W₂ᵀ and
            # u_i = 1 − h_i², so the penalty and its two paths are closed form.
            A = p["W1"].T @ p["W1"]
            B = p["W2"] @ p["W2"].T
            C = A * B
            Cu = u @ C.T
            loss += contractive_weight * float(
                np.mean(np.einsum("ij,ij->i", u, Cu)))
            U = (u.T @ u) / n
            S_A = contractive_weight * (U * B)
            S_B = contractive_weight * (U * A)
            d_H = d_H - (4.0 * contractive_weight / n) * c["H"] * Cu
            g["W1"] = c["Xin"].T @ (d_H * u) + 2.0 * p["W1"] @ S_A
            g["W2"] = g["W2"] + 2.0 * S_B @ p["W2"]
        else:
            g["W1"] = c["Xin"].T @ (d_H * u)

        g["b1"] = (d_H * u).sum(axis=0)
        return loss, g


class _Adam:
    """Adam, because one fixed step size on this loss either stalls or diverges."""

    def __init__(self, params: dict[str, np.ndarray], lr: float) -> None:
        self.lr = lr
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params: dict[str, np.ndarray],
             grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        for k, g in grads.items():
            self.m[k] = 0.9 * self.m[k] + 0.1 * g
            self.v[k] = 0.999 * self.v[k] + 0.001 * g * g
            m_hat = self.m[k] / (1.0 - 0.9 ** self.t)
            v_hat = self.v[k] / (1.0 - 0.999 ** self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)


# --------------------------------------------------------------------------- #
# The embedding
# --------------------------------------------------------------------------- #
class Autoencoder:
    """Bottleneck autoencoder used as a dimensionality reduction.

    `Autoencoder(n_components=2, *, variant="ae", n_hidden=32, epochs=600,
    learning_rate=1e-2, noise=0.3, kl_weight=1.0, contractive_weight=0.1,
    seed=0)` with `fit(X)` and `fit_transform(X)`; `X` is `(n, n_features)`.

    Attributes after `fit`: `embedding_` `(n, n_components)` — the latent code,
    the mean code for `vae_embed`; `reconstruction_` `(n, n_features)` in input
    units; `reconstruction_error_`, the half-squared residual per sample in
    standardised units; `losses_`, the objective at every epoch; `kl_`, the
    mean KL per sample for `vae_embed` and 0 otherwise.

    Inputs are centred and scaled to unit root-mean-square, and the scale is kept
    in `mean_` and `scale_`. Training is full-batch Adam from a seeded
    initialisation, so two fits on the same data are identical.
    """

    def __init__(self, n_components: int = 2, *, variant: str = "ae",
                 n_hidden: int = 32, epochs: int = 600,
                 learning_rate: float = 1e-2, noise: float = 0.3,
                 kl_weight: float = 1.0, contractive_weight: float = 0.1,
                 seed: int = 0) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        if n_components < 1 or n_hidden < 1 or epochs < 1:
            raise ValueError("n_components, n_hidden and epochs must be positive")
        self.n_components = n_components
        self.variant = variant
        self.n_hidden = n_hidden
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.noise = noise
        self.kl_weight = kl_weight
        self.contractive_weight = contractive_weight
        self.seed = seed

    def fit(self, X: np.ndarray) -> Autoencoder:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array")
        if X.shape[1] < self.n_components:
            raise ValueError("n_components cannot exceed n_features")
        self.mean_ = X.mean(axis=0)
        self.scale_ = float(np.sqrt(np.mean((X - self.mean_) ** 2))) or 1.0
        Xs = (X - self.mean_) / self.scale_

        rng = np.random.default_rng(self.seed)
        net = _Net(X.shape[1], self.n_hidden, self.n_components,
                   self.variant == "vae_embed", rng)
        opt = _Adam(net.p, self.learning_rate)
        noise = self.noise if self.variant == "denoising_ae" else 0.0
        cw = self.contractive_weight if self.variant == "contractive" else 0.0
        kw = self.kl_weight if self.variant == "vae_embed" else 0.0

        self.losses_ = []
        for _ in range(self.epochs):
            cache = net.forward(Xs, rng, noise)
            loss, grads = net.backward(cache, Xs, kw, cw)
            opt.step(net.p, grads)
            self.losses_.append(loss)

        # Report the mean code, not a sample, so the result does not depend on
        # where the last epoch happened to land.
        eps = np.zeros((X.shape[0], self.n_components)) if kw else None
        cache = net.forward(Xs, rng, eps=eps)
        self.embedding_ = cache["mu"] if net.variational else cache["Z"]
        self.reconstruction_ = cache["Xhat"] * self.scale_ + self.mean_
        self.reconstruction_error_ = 0.5 * float(
            np.mean(np.sum((cache["Xhat"] - Xs) ** 2, axis=1)))
        if net.variational:
            sigma = np.exp(0.5 * cache["logvar"])
            self.kl_ = 0.5 * float(np.mean(np.sum(
                sigma**2 + cache["mu"] ** 2 - 1.0 - cache["logvar"], axis=1)))
        else:
            self.kl_ = 0.0
        self.n_features_ = X.shape[1]
        self._net = net
        return self

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).embedding_

    def decode(self, Z: np.ndarray) -> np.ndarray:
        """Map codes back to input units. `Z` is `(m, n_components)`."""
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        p = self._net.p
        G = np.tanh(Z @ p["W3"] + p["b3"])
        return (G @ p["W4"] + p["b4"]) * self.scale_ + self.mean_

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Mean code for new rows. `X` is `(m, n_features)`."""
        Xs = (np.atleast_2d(np.asarray(X, dtype=float)) - self.mean_) / self.scale_
        return self._net.forward(Xs, np.random.default_rng(0))["E"][:, :self.n_components]


# --------------------------------------------------------------------------- #
# Data and measurements
# --------------------------------------------------------------------------- #
def _plane(n: int, seed: int, dims: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """A uniform square on a random two-plane in `dims` dimensions."""
    rng = np.random.default_rng(seed)
    basis = np.linalg.qr(rng.standard_normal((dims, 2)))[0]
    coords = rng.random((n, 2)) - 0.5
    return coords @ basis.T, coords


def _scurve(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Points on a sheet folded twice, and the parameters that generated them."""
    rng = np.random.default_rng(seed)
    t = 3.0 * np.pi * (rng.random(n) - 0.5)
    h = 2.0 * rng.random(n)
    X = np.column_stack([np.sin(t), h, np.sign(t) * (np.cos(t) - 1.0)])
    return X, np.column_stack([t, h])


def _pairwise(A: np.ndarray, B: np.ndarray | None = None) -> np.ndarray:
    B = A if B is None else B
    sa = np.einsum("ij,ij->i", A, A)
    sb = np.einsum("ij,ij->i", B, B)
    return np.sqrt(np.maximum(sa[:, None] + sb[None, :] - 2.0 * (A @ B.T), 0.0))


def _overlap(A: np.ndarray, B: np.ndarray, k: int = 8) -> float:
    """Mean fraction of each row's `k` nearest neighbours in `A` found in `B`."""
    da, db = _pairwise(A), _pairwise(B)
    np.fill_diagonal(da, np.inf)
    np.fill_diagonal(db, np.inf)
    ia = np.argsort(da, axis=1)[:, :k]
    ib = np.argsort(db, axis=1)[:, :k]
    return float(np.mean([len(set(a) & set(b)) / k
                          for a, b in zip(ia, ib, strict=True)]))


def _distance_correlation(A: np.ndarray, B: np.ndarray) -> float:
    """Correlation of the two pairwise distance matrices."""
    iu = np.triu_indices(A.shape[0], 1)
    return float(np.corrcoef(_pairwise(A)[iu], _pairwise(B)[iu])[0, 1])


def _pca(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    C = X - X.mean(axis=0)
    left, spectrum, _ = np.linalg.svd(C, full_matrices=False)
    return left[:, :n_components] * spectrum[:n_components]


def _jacobian_norm(fit: Autoencoder, X: np.ndarray, eps: float = 1e-5) -> float:
    """Mean squared Frobenius norm of ∂z/∂x, by central differences."""
    Xs = (X - fit.mean_) / fit.scale_
    zeros = np.zeros((Xs.shape[0], fit.n_components))
    net = fit._net
    total = 0.0
    for j in range(Xs.shape[1]):
        step = np.zeros_like(Xs)
        step[:, j] = eps
        up = net.forward(Xs + step, np.random.default_rng(0), eps=zeros)["Z"]
        down = net.forward(Xs - step, np.random.default_rng(0), eps=zeros)["Z"]
        total += float(np.mean(np.sum(((up - down) / (2.0 * eps)) ** 2, axis=1)))
    return total


def _kurtosis(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    a = a - a.mean()
    return float(np.mean(a**4) / np.mean(a**2) ** 2)


def _aggregate_kl(Z: np.ndarray) -> float:
    """KL between a Gaussian fitted to the whole code cloud and N(0, 1)."""
    m, s = Z.mean(axis=0), Z.std(axis=0)
    return float(np.mean(0.5 * (s**2 + m**2 - 1.0 - np.log(s**2))))


def _explained(fit: Autoencoder, X: np.ndarray) -> float:
    """Share of input variance the reconstruction keeps, in standardised units."""
    residual = np.mean(np.sum(((fit.reconstruction_ - X) / fit.scale_) ** 2, axis=1))
    total = np.mean(np.sum(((X - fit.mean_) / fit.scale_) ** 2, axis=1))
    return 1.0 - float(residual / total)


# --------------------------------------------------------------------------- #
# Demos
# --------------------------------------------------------------------------- #
def _demo_gradcheck() -> None:
    """All four variants against central differences, on a tiny network."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((11, 6))
    D, h, d = 6, 5, 2
    worst = 0.0
    for variant in VARIANTS:
        net = _Net(D, h, d, variant == "vae_embed", np.random.default_rng(1))
        cw = 0.7 if variant == "contractive" else 0.0
        kw = 0.9 if variant == "vae_embed" else 0.0
        eps = np.random.default_rng(4).standard_normal((X.shape[0], d)) \
            if kw else None
        Xin = X + 0.2 * np.random.default_rng(2).standard_normal(X.shape) \
            if variant == "denoising_ae" else X

        def loss_of(params: dict[str, np.ndarray], net: _Net = net,
                    Xin: np.ndarray = Xin, eps: np.ndarray | None = eps,
                    kw: float = kw, cw: float = cw) -> float:
            net.p = params
            cache = net.forward(Xin, np.random.default_rng(3), eps=eps)
            return net.backward(cache, X, kw, cw)[0]

        net.p = {k: v.copy() for k, v in net.p.items()}
        cache = net.forward(Xin, np.random.default_rng(3), eps=eps)
        _, grads = net.backward(cache, X, kw, cw)
        for key, value in net.p.items():
            flat = value.ravel()
            for idx in range(min(3, flat.size)):
                old = flat[idx]
                flat[idx] = old + 1e-6
                up = loss_of(net.p)
                flat[idx] = old - 1e-6
                down = loss_of(net.p)
                flat[idx] = old
                numeric = (up - down) / 2e-6
                analytic = grads[key].ravel()[idx]
                scale = max(abs(numeric), abs(analytic), 1e-8)
                worst = max(worst, abs(numeric - analytic) / scale)

    assert worst < 1e-6, worst
    print(
        f"gradients: a 6-5-2 network on 11 points, every parameter of every "
        f"variant checked against central differences. The worst relative error "
        f"over {len(VARIANTS)} variants × 8 tensors × 3 entries is {worst:.1e}, "
        f"so the closed-form contractive term in (6) — two terms through W₁, one "
        f"through W₂, and one through the tanh — is the derivative of the loss "
        f"it claims to be"
    )


def _demo_plane() -> None:
    """A linear sheet: the network should find exactly what PCA finds."""
    X, coords = _plane(300, seed=3)
    fit = Autoencoder(2).fit(X)
    pca = _pca(X)
    r2 = _explained(fit, X)
    corr = _distance_correlation(fit.embedding_, coords)
    pca_corr = _distance_correlation(pca, coords)

    assert fit.embedding_.shape == (300, 2)
    assert fit.kl_ == 0.0 and fit.variant == "ae"
    assert r2 > 0.98 and corr > 0.99
    fit2 = Autoencoder(2).fit(X)
    assert np.array_equal(fit.embedding_, fit2.embedding_)

    print(
        f"plane: 300 points on a random two-plane in five dimensions, thirty-two "
        f"hidden units, 600 epochs. The code correlates {corr:.5f} with the "
        f"coordinates that generated the data and reconstructs {r2:.2%} of the "
        f"variance, which is the ceiling a two-dimensional bottleneck can reach; "
        f"PCA's first two components correlate {pca_corr:.5f}. A linear map is "
        f"the best answer here and the network finds one — two fits are "
        f"bit-identical, so nothing in the result is luck"
    )


def _demo_sheet() -> None:
    """A curved sheet: the case a linear projection cannot handle."""
    X, coords = _scurve(300, seed=5)
    fit = Autoencoder(2).fit(X)
    pca = _pca(X)
    ae_dcorr = _distance_correlation(fit.embedding_, coords)
    pca_dcorr = _distance_correlation(pca, coords)
    ae_ov = _overlap(fit.embedding_, coords)
    pca_ov = _overlap(pca, coords)

    assert ae_dcorr > pca_dcorr
    assert ae_ov > pca_ov

    print(
        f"S-curve: 300 points on a sheet folded twice in three dimensions. The "
        f"autoencoder unrolls it — {ae_ov:.3f} of each point's eight nearest "
        f"neighbours survive and the distances correlate {ae_dcorr:.3f} with the "
        f"sheet's own parameters — where the leading two principal components "
        f"keep {pca_ov:.3f} and correlate {pca_dcorr:.3f}. The gap is the whole "
        f"reason to learn the map rather than fix it: a linear projection splits "
        f"the fold, and no rotation of the input can repair that"
    )


def _demo_denoising() -> None:
    """Noise on the input, clean target: what the extra term buys."""
    X, coords = _scurve(300, seed=5)
    rng = np.random.default_rng(7)
    noisy = X + 0.10 * rng.standard_normal(X.shape)
    clean = Autoencoder(2).fit(X)
    plain = Autoencoder(2).fit(noisy)
    denoise = Autoencoder(2, variant="denoising_ae", noise=0.10).fit(noisy)

    base = _distance_correlation(clean.embedding_, coords)
    plain_d = _distance_correlation(plain.embedding_, coords)
    denoise_d = _distance_correlation(denoise.embedding_, coords)
    base_ov = _overlap(clean.embedding_, coords)
    plain_ov = _overlap(plain.embedding_, coords)
    denoise_ov = _overlap(denoise.embedding_, coords)
    plain_20 = _overlap(plain.embedding_, coords, 20)
    denoise_20 = _overlap(denoise.embedding_, coords, 20)

    assert denoise_ov > plain_ov and denoise_20 > plain_20
    assert denoise_d < plain_d
    assert 0.0 < base - denoise_d < 0.2

    print(
        f"denoising: the same sheet with Gaussian noise of 0.10 added to every "
        f"coordinate. Trained on the clean sheet the embedding scores "
        f"{base_ov:.3f} overlap and {base:.3f} distance correlation — that "
        f"is the ceiling. Trained on the noisy copy, plain reconstruction gets "
        f"{plain_ov:.3f} and {plain_d:.3f}; jittering the input and rebuilding "
        f"the observed point gets {denoise_ov:.3f} and {denoise_d:.3f}, and "
        f"{plain_20:.3f} → {denoise_20:.3f} at twenty neighbours. The jitter "
        f"buys local neighbourhoods and pays for them in global distances: "
        f"{denoise_d - plain_d:+.3f} on distance correlation, against "
        f"{denoise_ov - plain_ov:+.3f} on overlap"
    )


def _demo_vae() -> None:
    """The KL term, measured on the code cloud rather than asserted."""
    X, _ = _plane(300, seed=3)
    ae = Autoencoder(2).fit(X)
    runs = [(b, Autoencoder(2, variant="vae_embed", kl_weight=b).fit(X))
            for b in (0.05, 0.2, 1.0)]
    vae = runs[-1][1]
    ae_kl = _aggregate_kl(ae.embedding_)
    kls = [_aggregate_kl(f.embedding_) for _, f in runs]
    kept = [_explained(f, X) for _, f in runs]
    small_kl = kls[0]
    kurt = float(np.mean([_kurtosis(vae.embedding_[:, j]) for j in range(2)]))

    assert ae.kl_ == 0.0 and vae.kl_ > 0.0
    assert small_kl < ae_kl / 100.0
    assert kls[0] < kls[1] < kls[2]
    assert kept[0] > kept[1] > kept[2]
    assert kurt < 2.5

    sweep = ", ".join(
        f"β = {b:g}: KL {_aggregate_kl(f.embedding_):.4f} at "
        f"{_explained(f, X):.3%} explained" for b, f in runs)
    print(
        f"variational: the same plane, with a KL term pulling each code towards "
        f"a standard normal. Fit a Gaussian to the whole autoencoder code cloud "
        f"and it is {ae_kl:.3f} nats from N(0, 1) — the code has standard "
        f"deviation {float(np.mean(ae.embedding_.std(axis=0))):.3f} per axis, "
        f"the flattened shape of a uniform square. The sweep runs {sweep}, so "
        f"β = 0.05 buys an aggregate code {ae_kl / small_kl:.0f}× closer to the "
        f"prior for 0.2% of the variance, and β = 1 pays 12% for another factor "
        f"of 150. Two honest caveats: the per-sample KL of (5) is "
        f"{runs[0][1].kl_:.2f} at β = 0.05 — the term constrains each point's "
        f"posterior, not the shape of the cloud — and the code's kurtosis is "
        f"{kurt:.2f} at β = 1 against {_kurtosis(ae.embedding_[:, 0]):.2f} for "
        f"the plain model and 3.00 for a Gaussian. Matching mean and variance is "
        f"not matching the distribution"
    )


def _demo_contractive() -> None:
    """The penalty (6) as a number: how fast the code moves with the input."""
    X, coords = _scurve(300, seed=5)
    plain = Autoencoder(2).fit(X)
    contractive = Autoencoder(2, variant="contractive",
                              contractive_weight=0.05).fit(X)
    j_plain = _jacobian_norm(plain, X)
    j_contractive = _jacobian_norm(contractive, X)
    s_plain = float(np.mean(plain.embedding_.std(axis=0)))
    s_contractive = float(np.mean(contractive.embedding_.std(axis=0)))

    assert j_contractive < j_plain
    assert _distance_correlation(contractive.embedding_, coords) > 0.8
    assert _overlap(contractive.embedding_, coords) < _overlap(plain.embedding_, coords)

    print(
        f"contractive: the sheet again, with λ = 0.05 on the squared Frobenius "
        f"norm of the encoder Jacobian. Plain training leaves the code moving at "
        f"{j_plain:.3f} squared code units per input unit, averaged over the "
        f"data; the penalty brings that to {j_contractive:.3f}, a factor of "
        f"{j_plain / j_contractive:.1f}. Most of that is the code shrinking: its "
        f"scale falls from {s_plain:.2f} to {s_contractive:.2f}, and ‖J‖² scales "
        f"as the square of it, so {s_plain**2 / s_contractive**2:.1f} of the "
        f"{j_plain / j_contractive:.0f} is amplitude the decoder could undo; the "
        f"genuine flattening is "
        f"{(j_plain / s_plain**2) / (j_contractive / s_contractive**2):.2f}×. "
        f"The embedding correlates {_distance_correlation(contractive.embedding_, coords):.3f} "
        f"with the sheet — better than {_distance_correlation(plain.embedding_, coords):.3f} "
        f"— and keeps {_overlap(contractive.embedding_, coords):.3f} of each "
        f"point's eight neighbours against {_overlap(plain.embedding_, coords):.3f}. "
        f"The penalty is not scale-invariant, and on a two-dimensional latent "
        f"the optimiser takes that route"
    )


def _demo_broken() -> None:
    """The two ways this fails loudly, and the way it fails silently."""
    try:
        Autoencoder(2, variant="vae").fit(_plane(50, seed=0)[0])
    except ValueError as err:
        bad_variant = str(err)
    try:
        Autoencoder(4).fit(_plane(50, seed=0)[0][:, :3])
    except ValueError as err:
        too_wide = str(err)

    assert "variant must be" in bad_variant
    assert "cannot exceed" in too_wide

    X, coords = _scurve(300, seed=5)
    seeds = [Autoencoder(2, seed=s).fit(X) for s in range(6)]
    errors = [f.reconstruction_error_ for f in seeds]
    scores = [_distance_correlation(f.embedding_, coords) for f in seeds]
    best_loss = int(np.argmin(errors))
    best_score = int(np.argmax(scores))

    plane, _ = _plane(300, seed=3)
    widths = [(d, Autoencoder(d).fit(plane)) for d in (1, 2, 3)]
    kept = [_explained(f, plane) for _, f in widths]

    assert best_loss != best_score
    assert kept[0] > 0.8 and kept[2] - kept[1] < 0.01
    pairs = ", ".join(f"{e:.5f}/{s:.3f}"
                      for e, s in zip(errors, scores, strict=True))

    print(
        f"failures: an unknown variant raises \"{bad_variant.split(',')[0]}\" and "
        f"a bottleneck wider than the input raises \"{too_wide}\", so neither "
        f"returns garbage quietly. Two things do not raise. The loss does not "
        f"name the embedding: six seeds on the same sheet reach reconstruction "
        f"errors from {min(errors):.5f} to {max(errors):.5f} and distance "
        f"correlations from {min(scores):.3f} to {max(scores):.3f}, and the run "
        f"with the lowest loss (seed {best_loss}, {errors[best_loss]:.5f}, "
        f"{scores[best_loss]:.3f}) is not the run with the best geometry (seed "
        f"{best_score}, {errors[best_score]:.5f}, {scores[best_score]:.3f}). The "
        f"six (error, correlation) pairs are {pairs}. And "
        f"the loss does not name the dimension: on the two-plane, one latent "
        f"axis already explains {kept[0]:.1%} of the variance — a curved "
        f"coordinate can snake across the square — two explain {kept[1]:.2%} and "
        f"three {kept[2]:.2%}, so the third buys "
        f"{(kept[2] - kept[1]) * 100:.2f} points. A sweep finds the elbow; a "
        f"single number does not"
    )


def main() -> None:
    _demo_gradcheck()
    _demo_plane()
    _demo_sheet()
    _demo_denoising()
    _demo_vae()
    _demo_contractive()
    _demo_broken()
    print("autoencoder_dr demo passed")


if __name__ == "__main__":
    main()
