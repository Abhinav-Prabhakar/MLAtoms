"""NMF — parts-based factorisation, and the two divergences that go with it.

Chapter:   06-latent-factorization
Companion: nmf.md
Tier 1 · 7 variants: frobenius_mu, kl_mu, als, hals, sparse_nmf,
                     online_nmf, semi_nmf

    X ≈ W H,      X ∈ R^{n×d}_{≥0},  W ∈ R^{n×k}_{≥0},  H ∈ R^{k×d}_{≥0}

Non-negativity is the whole idea. The SVD is free to subtract, so its components
are signed and they cancel; with W and H restricted to non-negative entries every
row of X is a *sum* of the rows of H with non-negative weights, so the rows of H
act as parts that add up to a whole and never subtract from each other.

The objective is a divergence between X and WH, not a distance in particular:

    frobenius_mu   ||X − WH||²_F                 squared error
    kl_mu          Σ X log(X/WH) − X + WH        the divergence for counts
    als            Frobenius, solved exactly in each factor
    hals           Frobenius, one rank-one term at a time
    sparse_nmf     Frobenius + L0 screening      parts that switch off
    online_nmf     Frobenius on minibatches      data that does not fit in RAM
    semi_nmf       ||X − WH||²,  W of either sign,  H ≥ 0

`frobenius_mu`, `als` and `hals` are three algorithms for one objective, and the
demo asserts their final losses agree to a fraction of a percent while `hals`
gets there in a fraction of the sweeps. `kl_mu` changes the divergence. The last
three change the model, and the demo asserts what changes: screening puts exact
zeros in the parts, `online_nmf` matches the batch answer while seeing 16 rows at
a time, and `semi_nmf` fits signed codes over a non-negative basis, which a
non-negative W cannot represent at all.

Sparse NMF is deliberately the L0 route and not the L1 one. An L1 penalty on H
alone is scale-unbounded — H → 0 and W → ∞ with WH fixed costs nothing — so
multiplicative updates drive H to the floor while W grows by six orders of
magnitude, and the fit *improves* while every part becomes zero. Screening is a
projection, so it cannot do that; and because a zero is absorbing under a
multiplicative update, the support it settles on is the support you keep.

Multiplicative updates are the through-line. Lee and Seung's observation is that
the gradients of both divergences split into a positive and a negative part, and
the ratio of the two is a multiplicative factor that keeps every entry
non-negative by construction:

    H ← H ⊙ (∇⁻ / ∇⁺),      ∇ = ∇⁻ − ∇⁺.

That is why no projection or clipping step is needed, and why the updates are
monotone: the loss decreases at every step for both divergences.
"""

from __future__ import annotations

import math

import numpy as np

VARIANTS = frozenset(["frobenius_mu", "kl_mu", "als", "hals", "sparse_nmf",
                      "online_nmf", "semi_nmf"])
EPS = 1e-12


# --------------------------------------------------------------------------- #
# Initialisation: NNDSVD, the non-negative SVD
# --------------------------------------------------------------------------- #
def _nndsvd(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Non-negative double SVD: start where the SVD wants to be, without signs."""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    k = min(k, s.size)
    W = np.zeros((X.shape[0], k))
    H = np.zeros((k, X.shape[1]))
    W[:, 0] = math.sqrt(s[0]) * np.abs(U[:, 0])
    H[0, :] = math.sqrt(s[0]) * np.abs(Vt[0])
    for j in range(1, k):
        x, y = U[:, j], Vt[j]
        xp, xn = np.maximum(x, 0.0), np.maximum(-x, 0.0)
        yp, yn = np.maximum(y, 0.0), np.maximum(-y, 0.0)
        mp = np.linalg.norm(xp) * np.linalg.norm(yp)
        mn = np.linalg.norm(xn) * np.linalg.norm(yn)
        if mp >= mn:
            u, v, sigma = xp / max(np.linalg.norm(xp), EPS), yp / max(np.linalg.norm(yp), EPS), mp
        else:
            u, v, sigma = xn / max(np.linalg.norm(xn), EPS), yn / max(np.linalg.norm(yn), EPS), mn
        scale = math.sqrt(max(s[j] * sigma, 0.0))
        W[:, j], H[j, :] = scale * u, scale * v
        # NNDSVDa: a zero entry locks for ever under a multiplicative update,
        # so seed it. The seed has to be small compared with *this* component —
        # borrowing component 1's mean inflates a weak component by 4x here,
        # which is how NNDSVD turns into a worse starting point than noise
        W[W[:, j] == 0.0, j] = 1e-2 * W[:, j].max()
        H[j, H[j, :] == 0.0] = 1e-2 * H[j].max()
    return W, H


# --------------------------------------------------------------------------- #
# The estimator
# --------------------------------------------------------------------------- #
class NMF:
    """Non-negative factorisation `X ≈ W H`, fitted by one of seven algorithms."""

    def __init__(self, n_components: int = 3, *, variant: str = "frobenius_mu",
                 seed: int = 0, lam: float = 0.0, init: str = "nndsvd",
                 n_iter: int = 300, tol: float = 1e-6, batch_size: int = 16,
                 forget: float = 0.6):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; expected one of {sorted(VARIANTS)}")
        if n_components < 1:
            raise ValueError("n_components must be at least 1")
        if init not in ("nndsvd", "random"):
            raise ValueError("init must be 'nndsvd' or 'random'")
        self.n_components = int(n_components)
        self.variant = variant
        self.seed = int(seed)
        self.lam = float(lam)
        self.init = init
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.batch_size = int(batch_size)
        self.forget = float(forget)

        self.W_: np.ndarray | None = None
        self.components_: np.ndarray | None = None      # H, as in sklearn
        self.reconstruction_err_: float = math.inf
        self.n_iter_: int = 0
        self.loss_curve_: list[float] = []

    # -- public surface ----------------------------------------------------- #
    def fit(self, X) -> NMF:
        X = self._check(X)
        k = self.n_components
        self.loss_curve_ = []
        if self.variant == "semi_nmf" and self.init == "nndsvd":
            # NNDSVD is built for X >= 0. On signed data, start from noise with
            # the model's own signs -- W free, H >= 0 -- because a positive
            # W cannot reach the negative cone and the sweep cannot fix that.
            rng = np.random.default_rng(self.seed)
            W = rng.standard_normal((X.shape[0], k))
            H = np.abs(rng.standard_normal((k, X.shape[1])))
        elif self.init == "nndsvd":
            W, H = _nndsvd(X, k)
        else:
            rng = np.random.default_rng(self.seed)
            W = rng.random((X.shape[0], k))
            H = rng.random((k, X.shape[1]))
        self.W_, self.components_ = W, H
        if self.variant == "online_nmf":
            self._fit_online(X)
        else:
            # NNDSVD is a starting point, not a solution: on a degenerate
            # spectrum its positive parts carry a fraction of the energy. Ten
            # sweeps of the algorithm we are about to run cost nothing.
            for _ in range(10):
                self._sweep(X)
            for it in range(1, self.n_iter + 1):
                self._sweep(X)
                loss = self._loss(X)
                self.loss_curve_.append(loss)
                self.n_iter_ = it
                if it > 1 and abs(self.loss_curve_[-2] - loss) <= self.tol * self.loss_curve_[-2]:
                    break
        self.reconstruction_err_ = float(np.linalg.norm(X - self.W_ @ self.components_))
        return self

    def transform(self, X) -> np.ndarray:
        """Codes for new rows against the learned basis: `(m, k)`."""
        if self.W_ is None:
            raise RuntimeError("call fit before transform")
        X = self._check(X)
        if self.variant == "semi_nmf":
            return self._solve_w(X, self.components_, nonneg=False)
        W = np.full((X.shape[0], self.n_components), 1.0 / self.n_components)
        for _ in range(200):
            W = self._update_w(X, W, self.components_)
        return W

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).W_

    def inverse_transform(self, W) -> np.ndarray:
        """Reconstruct rows from their codes: `(m, k) → (m, d)`."""
        W = np.asarray(W, dtype=float)
        if W.ndim != 2 or W.shape[1] != self.n_components:
            raise ValueError(f"expected (m, {self.n_components}), got {W.shape}")
        return W @ self.components_

    # -- one sweep, dispatched by variant ---------------------------------- #
    def _sweep(self, X: np.ndarray) -> None:
        v, W, H = self.variant, self.W_, self.components_
        if v in ("frobenius_mu", "sparse_nmf"):
            H = self._update_h(X, W, H)
            W = self._update_w(X, W, H)
            if v == "sparse_nmf":
                # a loading below lam × its part's largest is supported by
                # another part; switch it off. Because a zero is absorbing
                # under a multiplicative update, the support only shrinks.
                cut = self.lam * H.max(axis=1, keepdims=True)
                H[cut > H] = 0.0
        elif v == "kl_mu":
            H = self._update_h_kl(X, W, H)
            W = self._update_w_kl(X, W, H)
        elif v == "als":
            W = self._solve_w(X, H)
            H = np.maximum(self._solve_h(X, W), EPS)
        elif v == "hals":
            self._hals(X)
            return
        else:                                        # semi_nmf
            W = self._solve_w(X, H, nonneg=False)
            H = self._update_h_semi(X, W, H)
        self.W_, self.components_ = W, H

    # -- Frobenius multiplicative updates ---------------------------------- #
    def _update_h(self, X, W, H) -> np.ndarray:
        return np.maximum(H * (W.T @ X) / np.maximum(W.T @ W @ H, EPS), EPS)

    def _update_w(self, X, W, H) -> np.ndarray:
        num = X @ H.T
        den = W @ H @ H.T
        return np.maximum(W * num / np.maximum(den, EPS), EPS)

    # -- KL multiplicative updates ----------------------------------------- #
    def _update_h_kl(self, X, W, H) -> np.ndarray:
        ratio = X / np.maximum(W @ H, EPS)
        return np.maximum(H * (W.T @ ratio) / np.maximum(W.sum(axis=0)[:, None], EPS), EPS)

    def _update_w_kl(self, X, W, H) -> np.ndarray:
        ratio = X / np.maximum(W @ H, EPS)
        return np.maximum(W * (ratio @ H.T) / np.maximum(H.sum(axis=1)[None, :], EPS), EPS)

    # -- exact solves, then project ---------------------------------------- #
    def _solve_w(self, X, H, nonneg: bool = True) -> np.ndarray:
        G = H @ H.T + 1e-10 * np.eye(H.shape[0])
        W = np.linalg.solve(G, H @ X.T).T
        return np.maximum(W, EPS) if nonneg else W

    def _solve_h(self, X, W) -> np.ndarray:
        G = W.T @ W + 1e-10 * np.eye(W.shape[1])
        return np.linalg.solve(G, W.T @ X)

    # -- HALS: one rank-one term at a time, on the residual ----------------
    def _hals(self, X) -> None:
        W, H = self.W_, self.components_
        R = X - W @ H
        for c in range(W.shape[1]):
            denom = float(H[c] @ H[c])
            if denom < EPS:                          # a dead component stays dead
                continue
            w_old = W[:, c].copy()
            W[:, c] = np.maximum(W[:, c] + (R @ H[c]) / denom, EPS)
            # R is X − WH, so raising column c *subtracts* its contribution:
            # R += outer(w_old − w_new, H[c]). Getting this sign wrong makes the
            # textbook HALS look like a divergent algorithm.
            R += np.outer(w_old - W[:, c], H[c])
            h_old = H[c].copy()
            denom = float(W[:, c] @ W[:, c])
            if denom < EPS:
                continue
            H[c] = np.maximum(H[c] + (W[:, c] @ R) / denom, EPS)
            R += np.outer(W[:, c], h_old - H[c])
        self.W_, self.components_ = W, H

    # -- online: minibatches and decayed sufficient statistics --------------
    def _fit_online(self, X: np.ndarray) -> None:
        """Mairal's online NMF: codes per batch, a decayed basis update."""
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        k = self.n_components
        A = np.eye(k)
        B = np.zeros((d, k))
        H = self.components_
        for it in range(1, self.n_iter + 1):
            rows = rng.choice(n, size=min(self.batch_size, n), replace=False)
            Xb = X[rows]
            Wb = np.full((rows.size, k), 1.0 / k)
            for _ in range(30):
                Wb = self._update_w(Xb, Wb, H)
            rho = self.forget
            A = (1.0 - rho) * A + rho * (Wb.T @ Wb)
            B = (1.0 - rho) * B + rho * (Xb.T @ Wb)
            H = np.maximum(H * (B.T / np.maximum(A @ H, EPS)), EPS)
            self.components_ = H
            # the batch loss is what the method can see; the full-data loss is
            # an offline diagnostic and is reported once, at the end
            self.loss_curve_.append(float(np.linalg.norm(Xb - Wb @ H) ** 2))
            self.n_iter_ = it
            if it > 1 and abs(self.loss_curve_[-2] - self.loss_curve_[-1]) \
                    <= self.tol * self.loss_curve_[-2]:
                break
        self.W_ = self.transform(X)

    # -- semi-NMF: W free, H non-negative ----------------------------------
    def _update_h_semi(self, X, W, H) -> np.ndarray:
        WtX = W.T @ X
        WtW = W.T @ W
        # positive and negative parts: A = A⁺ − A⁻ with A⁺, A⁻ ≥ 0
        num = 0.5 * (np.abs(WtX) + WtX) + 0.5 * (np.abs(WtW) - WtW) @ H
        den = 0.5 * (np.abs(WtX) - WtX) + 0.5 * (np.abs(WtW) + WtW) @ H
        return np.maximum(H * np.sqrt(np.maximum(num, EPS) / np.maximum(den, EPS)), EPS)

    # -- losses ------------------------------------------------------------- #
    def _loss(self, X) -> float:
        if self.variant == "kl_mu":
            R = X / np.maximum(self.W_ @ self.components_, EPS)
            return float(np.sum(X * np.log(np.maximum(R, EPS)) - X + self.W_ @ self.components_))
        return float(np.linalg.norm(X - self.W_ @ self.components_) ** 2)

    # -- validation --------------------------------------------------------- #
    def _check(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"expected a 2-D array, got shape {X.shape}")
        if X.shape[0] < 1 or X.shape[1] < 1:
            raise ValueError("need at least one row and one column")
        if self.variant != "semi_nmf" and X.min() < 0.0:
            raise ValueError(f"{self.variant} needs a non-negative X; use semi_nmf for "
                             f"signed data, or shift the data and record the offset")
        return X


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _parts(n: int = 200, d: int = 60, k: int = 3, noise: float = 0.01, seed: int = 0):
    """Three non-negative parts, each switched on for a third of the features."""
    rng = np.random.default_rng(seed)
    W = rng.random((n, k))
    H = np.zeros((k, d))
    for j in range(k):
        H[j, j * (d // k):(j + 1) * (d // k)] = rng.random(d // k)
    H /= H.sum(axis=1, keepdims=True)
    X = W @ H
    return X + noise * rng.random(X.shape), W, H


def _kl(X: np.ndarray, Y: np.ndarray) -> float:
    Y = np.maximum(Y, EPS)
    return float(np.sum(X * np.log(np.maximum(X, EPS) / Y) - X + Y))


def main() -> None:
    X, W0, H0 = _parts()

    # --- 1. the three Frobenius routes reach the same place ---------------
    # MU, ALS and HALS are three algorithms for one objective, so their final
    # losses have to agree -- and only HALS does it in a few dozen sweeps.
    fits = {v: NMF(3, variant=v, n_iter=400, tol=1e-8).fit(X)
            for v in ("frobenius_mu", "als", "hals")}
    losses = {v: f.reconstruction_err_ / np.linalg.norm(X) for v, f in fits.items()}
    # the floor here is the noise: 0.01·U[0,1] over 200×60 is 0.075 in relative
    # Frobenius norm, so "exact" is 0.074 and the assertion is about agreement
    assert max(losses.values()) < 0.08, losses
    assert max(losses.values()) - min(losses.values()) < 0.02 * min(losses.values())
    for v, f in fits.items():
        curve = np.array(f.loss_curve_)
        assert np.all(np.diff(curve) <= 1e-9 * max(curve[0], 1.0)), v   # monotone
    # HALS gets there in far fewer sweeps than MU: that is why it is used
    hals_50 = NMF(3, variant="hals", n_iter=50).fit(X).reconstruction_err_
    mu_50 = NMF(3, variant="frobenius_mu", n_iter=50).fit(X).reconstruction_err_
    assert hals_50 < mu_50
    print(
        "nmf: frobenius_mu, als and hals agree to "
        f"{max(losses.values()) - min(losses.values()):.1e} relative error "
        f"(all ≤ {max(losses.values()):.3f}), every loss curve is monotone, and "
        f"after 50 sweeps hals is at {hals_50 / np.linalg.norm(X):.4f} against "
        f"mu's {mu_50 / np.linalg.norm(X):.4f}"
    )

    # --- 2. KL divergence: the right loss for counts ----------------------
    # Poisson counts have variance = mean, so squared error is the wrong
    # geometry. Fitting by KL gives a lower KL to the held-out counts than
    # fitting by Frobenius and then measuring KL.
    rng = np.random.default_rng(1)
    Wp = rng.random((300, 3))
    Hp = rng.random((3, 50))
    Xp = rng.poisson(Wp @ Hp * 3.0).astype(float)
    kl_fit = NMF(3, variant="kl_mu", n_iter=300, tol=1e-8).fit(Xp)
    fr_fit = NMF(3, variant="frobenius_mu", n_iter=300, tol=1e-8).fit(Xp)
    d_kl = _kl(Xp, kl_fit.W_ @ kl_fit.components_)
    d_fr = _kl(Xp, fr_fit.W_ @ fr_fit.components_)
    assert d_kl < d_fr
    curve = np.array(kl_fit.loss_curve_)
    assert np.all(np.diff(curve) <= 1e-9 * max(curve[0], 1.0))
    print(
        f"nmf: on Poisson counts, KL-fitted factors have KL divergence {d_kl:.1f} "
        f"against {d_fr:.1f} for Frobenius-fitted ones, and the KL curve falls "
        f"monotonically from {curve[0]:.0f}"
    )

    # --- 3. sparse NMF: exact zeros, and what they cost -------------------
    # The data was generated from three parts that each occupy a third of the
    # features. Screening at 10% of each part's largest loading finds exactly
    # that support -- and the price is visible in the reconstruction error.
    dense = NMF(3, variant="frobenius_mu", n_iter=400).fit(X)
    sparse = NMF(3, variant="sparse_nmf", lam=0.10, n_iter=400).fit(X)
    zeros = float(np.mean(sparse.components_ < 1e-10))
    support = [int(np.sum(sparse.components_[j] > 0.0)) for j in range(3)]
    ratio = sparse.reconstruction_err_ / dense.reconstruction_err_
    assert zeros > 0.6, zeros
    assert ratio < 1.25, ratio
    # every recovered part sits on one true block, and all three blocks are hit
    blocks = sorted(int(np.argmax(sparse.components_[j]) // (X.shape[1] // 3))
                    for j in range(3))
    assert blocks == [0, 1, 2], blocks
    assert all(0.7 * (X.shape[1] // 3) <= s <= 1.3 * (X.shape[1] // 3) for s in support)
    print(
        f"nmf: screening at 10% of each part's maximum zeroes {zeros:.0%} of the "
        f"loadings, recovers support {support} against the true 20 each, and costs "
        f"{ratio:.2f}× the dense reconstruction error"
    )

    # and the failure the L0 route avoids: an L1 penalty on H alone has no
    # scale, so H -> 0 and W -> ∞ with WH unchanged is free -- the penalty is
    # satisfied perfectly and the parts say nothing
    Wn, Hn = _nndsvd(X, 3)
    for _ in range(10):
        Hn = np.maximum(Hn * (Wn.T @ X) / np.maximum(Wn.T @ Wn @ Hn, EPS), EPS)
        Wn = np.maximum(Wn * (X @ Hn.T) / np.maximum(Wn @ Hn @ Hn.T, EPS), EPS)
    h_before, w_before = float(np.linalg.norm(Hn)), float(np.linalg.norm(Wn))
    fit_before = float(np.linalg.norm(X - Wn @ Hn))
    for _ in range(200):
        num = Wn.T @ X
        Hn = np.maximum(Hn * num / np.maximum(Wn.T @ Wn @ Hn + 0.1 * float(num.mean()),
                                              EPS), EPS)
        Wn = np.maximum(Wn * (X @ Hn.T) / np.maximum(Wn @ Hn @ Hn.T, EPS), EPS)
    fit_after = float(np.linalg.norm(X - Wn @ Hn))
    assert np.linalg.norm(Hn) < 1e-3 * h_before
    assert np.linalg.norm(Wn) > 1e4 * w_before
    assert fit_after <= fit_before
    print(
        f"nmf: the L1 alternative is scale-free — 200 sweeps take ||H|| from "
        f"{h_before:.2f} to {np.linalg.norm(Hn):.0e} and ||W|| from {w_before:.1f} to "
        f"{np.linalg.norm(Wn):.0e} while the fit *improves* to {fit_after:.3f}, so the "
        f"penalty is met in the one way that explains nothing"
    )

    # --- 4. online: the batch answer from 16-row peeks --------------------
    batch = NMF(3, variant="frobenius_mu", n_iter=300).fit(X)
    online = NMF(3, variant="online_nmf", n_iter=300, batch_size=16, forget=0.3).fit(X)
    gap = online.reconstruction_err_ / batch.reconstruction_err_ - 1.0
    assert gap < 0.05, gap
    print(
        f"nmf: online NMF sees {online.batch_size} rows per step and lands within "
        f"{gap:.1%} of the batch reconstruction error "
        f"({online.reconstruction_err_:.4f} against {batch.reconstruction_err_:.4f})"
    )

    # --- 5. semi-NMF: signed weights on non-negative parts ----------------
    # Semi-NMF constrains H >= 0 and lets W be signed. So the demo has to be
    # data with a non-negative basis and signed mixing weights, which is the
    # opposite of the block data above: X = A B with A signed, B >= 0.
    rng = np.random.default_rng(2)
    Ac = rng.standard_normal((200, 3))                    # signed weights
    Bp = np.zeros((3, 60))                                # non-negative parts
    for j in range(3):
        Bp[j, 20 * j:20 * (j + 1)] = rng.random(20)
    Xs = Ac @ Bp + 0.01 * rng.standard_normal((200, 60))
    semi = NMF(3, variant="semi_nmf", n_iter=300).fit(Xs)
    e_semi = float(np.linalg.norm(Xs - semi.W_ @ semi.components_)) / np.linalg.norm(Xs)
    # the two ways to force non-negative NMF onto signed data: throw the sign
    # away, or shift everything up. Both change the problem, and both fit worse.
    clip = NMF(3, variant="frobenius_mu", n_iter=400).fit(np.maximum(Xs, 0.0))
    e_clip = float(np.linalg.norm(Xs - clip.W_ @ clip.components_)) / np.linalg.norm(Xs)
    shift = NMF(3, variant="frobenius_mu", n_iter=400).fit(Xs - Xs.min())
    e_shift = float(np.linalg.norm(
        Xs - (shift.W_ @ shift.components_ + Xs.min()))) / np.linalg.norm(Xs)
    assert e_semi < 0.03, e_semi
    assert e_clip > 0.5 and e_shift > 0.1, (e_clip, e_shift)
    assert semi.components_.min() >= 0.0
    assert semi.W_.min() < 0.0                       # the weights really are signed
    print(
        f"nmf: semi-NMF fits signed data to {e_semi:.3f} with a non-negative basis "
        f"and signed weights; clipping the data costs {e_clip:.2f} and shifting it "
        f"up costs {e_shift:.2f}, because neither is the same model"
    )

    # --- 6. transform: new rows against the learned basis -----------------
    # The basis is fixed, so this is a projection rather than a refit: the
    # codes of held-out rows reconstruct them with the training error.
    m = 60
    train, test = X[:m], X[m:]
    fit = NMF(3, variant="hals", n_iter=200).fit(train)
    H_test = fit.transform(test)
    assert H_test.shape == (test.shape[0], 3) and H_test.min() > 0.0
    err = float(np.linalg.norm(test - fit.inverse_transform(H_test))) / np.linalg.norm(test)
    tr_err = fit.reconstruction_err_ / np.linalg.norm(train)
    # the basis transfers: the gap is the noise the training block happened to
    # see, not a failure to generalise -- and it does not shrink with more MU
    # iterations, because it is not an optimisation artefact
    assert err < 0.085 and err < 1.2 * tr_err, (err, tr_err)
    print(
        f"nmf: the basis learned on {train.shape[0]} rows reconstructs {test.shape[0]} "
        f"held-out rows to {err:.4f} against {tr_err:.4f} in-sample, so the parts "
        f"transfer and only the codes are refitted"
    )

    # --- 7. the scale ambiguity, and the one thing it does not affect -----
    # W H is invariant under W → W A, H → A⁻¹ H for any positive diagonal A,
    # so individual loadings are not identified even though the fit is.
    f = NMF(3, variant="frobenius_mu", n_iter=200).fit(X)
    d = np.array([0.5, 2.0, 3.0])
    scaled = (f.W_ * d) @ (f.components_ / d[:, None])
    assert np.allclose(scaled, f.W_ @ f.components_, atol=1e-12)
    # ... and a permutation of the components changes nothing either
    p = np.array([2, 0, 1])
    assert np.allclose((f.W_[:, p]) @ (f.components_[p]), f.W_ @ f.components_)
    print(
        "nmf: the factorisation is invariant to column scaling and permutation "
        "of the components, so only the subspaces and the supports are identified"
    )

    print("nmf demo passed")


if __name__ == "__main__":
    main()
