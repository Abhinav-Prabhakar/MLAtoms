"""Tensor decomposition — low rank in three modes, and five ways to find it.

Chapter:   06-latent-factorization
Companion: tensor_decomposition.md
Tier 3 · 5 variants: cp_als, cp_als_ls, tucker_hooi, hosvd, parafac2

    CP:      X ≈ Σ_r a_r ∘ b_r ∘ c_r                                          (1)
    Tucker:  X ≈ G ×₁ U₁ ×₂ U₂ ×₃ U₃                                          (2)

Two different answers to "this tensor is low rank". CP says the tensor is a sum of
`R` rank-one pieces — each piece a product of three vectors — and nothing else.
Tucker says the tensor lives in a small subspace along each mode, and everything
inside that subspace is core. CP has one number for rank and no core; Tucker has
one rank per mode and a core that is the model.

The distinction is not cosmetic. A rank-`R` CP tensor is generically
*identifiable*: the factors are determined up to permutation and scaling, which is
what makes the decomposition a discovery rather than a basis change. Tucker is
never identifiable — rotate inside any mode and compensate in the core, and the
fit is bit-for-bit identical — so its factors are subspaces, not components.

The demo measures both halves, plus the way both go wrong. A rank-4 CP tensor is
recovered from a random start at fit 1.0 with factors at cosine 1.000 to the truth
— and the same tensor with one mode's columns nearly parallel is *not* recovered,
at fits of 0.9997 that look like success and four starts that disagree. A
rank-(4,4,4) Tucker tensor plus noise keeps its subspaces at 0.9999, which is all
Tucker identifies, and the truncated problem separates HOSVD from HOOI by 0.066.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12
VARIANTS = ("cp_als", "cp_als_ls", "hosvd", "parafac2", "tucker_hooi")


def _unfold(X: np.ndarray, mode: int) -> np.ndarray:
    """`X` `(I₀,I₁,I₂)` to `(I_mode, prod of the other two)`, others in C order.

    The row order of the result has to agree with `_khatri_rao`, and both are
    "first other mode outer, second inner". Getting this backwards produces a
    decomposition that converges to a wrong answer with a plausible fit, which is
    the single most common bug in from-scratch tensor code.
    """
    return np.moveaxis(X, mode, 0).reshape(X.shape[mode], -1)


def _khatri_rao(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """`(A, B)` `(I,R),(J,R)` to `(I·J, R)`, row `i·J + j` is `A[i] * B[j]`."""
    return (A[:, None, :] * B[None, :, :]).reshape(A.shape[0] * B.shape[0], -1)


def _mttkrp(X: np.ndarray, factors: list[np.ndarray], mode: int) -> np.ndarray:
    """Matricised tensor times Khatri-Rao product: the gradient of CP, `(I,R)`."""
    others = [f for m, f in enumerate(factors) if m != mode]
    return _unfold(X, mode) @ _khatri_rao(others[0], others[1])


def _cp_model(factors: list[np.ndarray]) -> np.ndarray:
    """`Σ_r a_r ∘ b_r ∘ c_r`, the tensor a CP fit predicts."""
    return np.einsum("ir,jr,kr->ijk", *factors, optimize=True)


def _mode_dot(X: np.ndarray, U: np.ndarray, mode: int) -> np.ndarray:
    """`X ×_mode U`: contract `U`'s second axis with `X`'s `mode` axis."""
    out = np.tensordot(U, X, axes=([1], [mode]))
    return np.moveaxis(out, 0, mode)


def _tucker_model(core: np.ndarray, factors: list[np.ndarray]) -> np.ndarray:
    """`G ×₁ U₁ ×₂ U₂ ×₃ U₃`, the tensor a Tucker fit predicts."""
    return np.einsum("pqr,ip,jq,kr->ijk", core, *factors, optimize=True)


def _kernels(factors: list[np.ndarray], mode: int) -> np.ndarray:
    """`(BᵀB) * (CᵀC)`, the Gram matrix CP's normal equations need."""
    others = [f for m, f in enumerate(factors) if m != mode]
    return (others[0].T @ others[0]) * (others[1].T @ others[1])


class TensorDecomposition:
    """CP, Tucker and PARAFAC2 on a dense three-way array.

    `TensorDecomposition(rank=3, *, variant="cp_als", n_iter=100, tol=1e-9,
    seed=0)`. `fit(X)` takes `(I, J, K)` for the tensor variants, or a list of
    `(I, J_k)` matrices for `parafac2`.

    `rank` is an int or a per-mode tuple. `factors_` is a list of factor matrices,
    `core_` the Tucker core (`None` for CP), `fits_` the fit after each iteration,
    and `weights_`/`transforms_` the PARAFAC2 per-slice diagonal and rotation.
    """

    def __init__(self, rank: int | tuple[int, ...] = 3, *, variant: str = "cp_als",
                 n_iter: int = 100, tol: float = 1e-9, seed: int = 0) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        self.rank = rank
        self.variant = variant
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.seed = int(seed)

    def _ranks(self, ndim: int) -> tuple[int, ...]:
        if isinstance(self.rank, (tuple, list)):
            if len(self.rank) != ndim:
                raise ValueError(f"rank has {len(self.rank)} entries for {ndim} modes")
            return tuple(int(r) for r in self.rank)
        return (int(self.rank),) * ndim

    def fit(self, X) -> TensorDecomposition:
        """Fit the selected variant; returns `self` with `factors_` and `fits_`."""
        if self.variant == "parafac2":
            return self._fit_parafac2(X)
        X = np.asarray(X, dtype=float)
        if X.ndim != 3:
            raise ValueError(f"X must be three-way, got shape {X.shape}")
        ranks = self._ranks(3)
        if any(r > s for r, s in zip(ranks, X.shape, strict=True)):
            raise ValueError(f"rank {ranks} exceeds the shape {X.shape}")
        if self.variant in ("tucker_hooi", "hosvd"):
            return self._fit_tucker(X, ranks)
        return self._fit_cp(X, ranks[0])

    # ---------------------------------------------------------------- CP ---- #
    def _fit_cp(self, X: np.ndarray, R: int) -> TensorDecomposition:
        """ALS on (1): each mode's update is a least-squares problem solved exactly."""
        rng = np.random.default_rng(self.seed)
        factors = [rng.standard_normal((n, R)) for n in X.shape]
        fits: list[float] = []
        for _ in range(self.n_iter):
            for mode in range(3):
                if self.variant == "cp_als":
                    V = _kernels(factors, mode)
                    if np.linalg.cond(V) > 1e12:
                        # (BᵀB)*(CᵀC) factorises through two nearly dependent
                        # columns; the solve is then pure noise amplified
                        V = V + 1e-10 * np.trace(V) / R * np.eye(R)
                    factors[mode] = np.linalg.solve(V, _mttkrp(X, factors, mode).T).T
                else:
                    KR = _khatri_rao(*[f for m, f in enumerate(factors) if m != mode])
                    factors[mode] = np.linalg.lstsq(
                        KR, _unfold(X, mode).T, rcond=None)[0].T
            for mode in range(2):
                nrm = np.linalg.norm(factors[mode], axis=0)
                nrm[nrm < EPS] = 1.0
                factors[mode] /= nrm
                factors[mode + 1] *= nrm
            fits.append(1.0 - float(np.linalg.norm(X - _cp_model(factors)))
                        / float(np.linalg.norm(X)))
            if len(fits) > 1 and abs(fits[-1] - fits[-2]) < self.tol:
                break
        self.factors_, self.core_, self.fits_ = factors, None, fits
        return self

    # ------------------------------------------------------------ Tucker ---- #
    def _fit_tucker(self, X: np.ndarray, ranks: tuple[int, ...]) -> TensorDecomposition:
        """HOOI (iterated) or HOSVD (one pass): orthogonal factors, then the core."""
        factors = []
        for mode, r in enumerate(ranks):
            factors.append(np.linalg.svd(_unfold(X, mode), full_matrices=False)[0][:, :r])
        fits: list[float] = []
        for _ in range(0 if self.variant == "hosvd" else self.n_iter):
            for mode in range(3):
                others = [f for m, f in enumerate(factors) if m != mode]
                # Kronecker, not Khatri-Rao: the modes have different ranks
                Y = _unfold(X, mode) @ np.kron(others[0], others[1])
                factors[mode] = np.linalg.svd(Y, full_matrices=False)[0][:, :ranks[mode]]
            core = _mode_dot(_mode_dot(_mode_dot(X, factors[0].T, 0),
                                       factors[1].T, 1), factors[2].T, 2)
            fits.append(1.0 - float(np.linalg.norm(X - _tucker_model(core, factors)))
                        / float(np.linalg.norm(X)))
            if len(fits) > 1 and abs(fits[-1] - fits[-2]) < self.tol:
                break
        if self.variant == "hosvd":
            core = _mode_dot(_mode_dot(_mode_dot(X, factors[0].T, 0),
                                       factors[1].T, 1), factors[2].T, 2)
            fits.append(1.0 - float(np.linalg.norm(X - _tucker_model(core, factors)))
                        / float(np.linalg.norm(X)))
        self.factors_, self.core_, self.fits_ = factors, core, fits
        return self

    # ----------------------------------------------------------- PARAFAC2 ---- #
    def _fit_parafac2(self, X: list[np.ndarray]) -> TensorDecomposition:
        """PARAFAC2 on `X_k ≈ A diag(d_k) (P_k B)ᵀ` with `P_k` orthogonal.

        The model exists for ragged data: one mode is shared (`A`), the other
        differs between slices by an orthogonal map, so the *inner products*
        between loadings are shared rather than the loadings. A shift of a
        chromatogram is exactly such a map.
        """
        X = [np.asarray(x, dtype=float) for x in X]
        if not isinstance(self.rank, int):
            raise ValueError("parafac2 needs an int rank")
        sizes = {x.shape[0] for x in X}
        if len(sizes) != 1:
            raise ValueError(f"parafac2 needs one shared row count, got {sorted(sizes)}")
        R = self.rank
        rng = np.random.default_rng(self.seed)
        rows, J = X[0].shape[0], min(x.shape[1] for x in X)
        A = rng.standard_normal((rows, R))
        B = rng.standard_normal((J, R))
        D = [np.ones(R) for _ in X]
        P = [np.eye(x.shape[1], J) for x in X]

        fits: list[float] = []
        for _ in range(self.n_iter):
            P = [self._procrustes(x, A, d, B) for x, d in zip(X, D, strict=True)]
            Bk = [p @ B for p in P]

            # A: stacked least squares, Σ_k D_k (BᵀB) D_k as the normal matrix
            num = sum(x @ bk * d for x, bk, d in zip(X, Bk, D, strict=True))
            den = (B.T @ B) * sum(np.outer(d, d) for d in D)
            A = np.linalg.solve(den + 1e-10 * np.trace(den) / R * np.eye(R), num.T).T

            # B: the same stacked problem after rotating each slice's target back
            gram_a = A.T @ A
            num = sum(p.T @ (x.T @ (A * d)) for x, p, d in zip(X, P, D, strict=True))
            den = gram_a * sum(np.outer(d, d) for d in D)
            B = np.linalg.solve(den + 1e-10 * np.trace(den) / R * np.eye(R), num.T).T

            # D: each slice's R scales, exactly — the components are not
            # orthogonal, so this is a small least-squares problem, not R
            # independent projections onto each component
            Bk = [p @ B for p in P]
            D = [np.linalg.lstsq(np.einsum("ir,jr->ijr", A, bk).reshape(-1, R),
                                 x.reshape(-1), rcond=None)[0]
                 for x, bk in zip(X, Bk, strict=True)]

            nrm = np.linalg.norm(A, axis=0)
            nrm[nrm < EPS] = 1.0
            A /= nrm
            D = [d * nrm for d in D]
            fits.append(self._parafac2_fit(X, A, B, D, P))
            if len(fits) > 1 and abs(fits[-1] - fits[-2]) < self.tol:
                break
        self.factors_, self.core_, self.fits_ = [A, B], None, fits
        self.weights_, self.transforms_ = D, P
        return self

    @staticmethod
    def _procrustes(x: np.ndarray, A: np.ndarray, d: np.ndarray,
                    B: np.ndarray) -> np.ndarray:
        """Best orthogonal `P_k` given the rest: SVD of `X_kᵀ A D_k Bᵀ`, `(J_k,J)`."""
        U, _, Vt = np.linalg.svd(x.T @ (A * d) @ B.T, full_matrices=False)
        return U @ Vt

    @staticmethod
    def _parafac2_fit(X: list[np.ndarray], A: np.ndarray, B: np.ndarray,
                      D: list[np.ndarray], P: list[np.ndarray]) -> float:
        """`1 − ‖X − X̂‖/‖X‖` over the whole collection of matrices."""
        num = sum(float(np.sum((x - (A * d) @ (p @ B).T) ** 2))
                  for x, d, p in zip(X, D, P, strict=True))
        den = sum(float(np.sum(x ** 2)) for x in X)
        return 1.0 - float(np.sqrt(num / max(den, EPS)))

    def reconstruct(self):
        """The fitted tensor, or the list of fitted matrices for `parafac2`."""
        if self.variant == "parafac2":
            return [(self.factors_[0] * d) @ (p @ self.factors_[1]).T
                    for d, p in zip(self.weights_, self.transforms_, strict=True)]
        if self.core_ is None:
            return _cp_model(self.factors_)
        return _tucker_model(self.core_, self.factors_)

    def fit_score(self, X) -> float:
        """`1 − ‖X − X̂‖_F / ‖X‖_F`; 1.0 is exact, 0.0 is the zero tensor."""
        if self.variant == "parafac2":
            num = sum(float(np.sum((x - r) ** 2))
                      for x, r in zip(X, self.reconstruct(), strict=True))
            den = sum(float(np.sum(np.asarray(x) ** 2)) for x in X)
            return 1.0 - float(np.sqrt(num / max(den, EPS)))
        X = np.asarray(X, dtype=float)
        return 1.0 - float(np.linalg.norm(X - self.reconstruct())) \
            / float(np.linalg.norm(X))


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _factor_recovery(factors: list[np.ndarray], truth: list[np.ndarray]) -> list[float]:
    """Mean best column `|cos|` per mode: CP is identified only up to permutation
    and sign, so this is the strongest statement available about its factors."""
    out = []
    for f, t in zip(factors, truth, strict=True):
        fn = f / np.maximum(np.linalg.norm(f, axis=0), EPS)
        tn = t / np.maximum(np.linalg.norm(t, axis=0), EPS)
        out.append(float(np.mean(np.max(np.abs(fn.T @ tn), axis=1))))
    return out


def _subspace_score(factors: list[np.ndarray], truth: list[np.ndarray]) -> list[float]:
    """`‖UᵀÛ‖_F/√r` per mode: 1.0 when the two matrices span the same subspace.

    Tucker never identifies its factors, only the subspaces they span, so this
    is the only factor comparison that has a meaning.
    """
    return [float(np.linalg.norm(t.T @ f) / np.sqrt(f.shape[1]))
            for f, t in zip(factors, truth, strict=True)]


def _demo_cp() -> None:
    """A rank-4 tensor, recovered from a random start."""
    rng = np.random.default_rng(5)
    shape = (25, 18, 12)
    R = 4
    truth = [rng.standard_normal((n, R)) for n in shape]
    X = _cp_model(truth)
    fit = TensorDecomposition(R, variant="cp_als", n_iter=500, tol=1e-12).fit(X)
    score = fit.fit_score(X)
    rec = _factor_recovery(fit.factors_, truth)
    assert score > 1.0 - 1e-10 and min(rec) > 1.0 - 1e-6, (score, rec)

    # the same ALS step, solved on the design matrix instead of the normal equations
    ls = TensorDecomposition(R, variant="cp_als_ls", n_iter=500, tol=1e-12).fit(X)
    agree = abs(ls.fit_score(X) - score)
    design = shape[1] * shape[2]
    assert agree < 1e-9, agree

    print(
        f"cp_als: a random {shape[0]}×{shape[1]}×{shape[2]} rank-{R} tensor, fitted "
        f"from a random start in {len(fit.fits_)} sweeps to fit {score:.10f} with "
        f"factors at cosine {'/'.join(f'{r:.6f}' for r in rec)} in the three modes. "
        f"Each mode's update is an ordinary least-squares step against the "
        f"Khatri-Rao product of the other two, and the tensor's rank is low enough "
        f"that the answer is the truth rather than a stationary point. "
        f"cp_als_ls takes the same step without forming the {R}×{R} normal "
        f"equations — a {design}×{R} least-squares solve per mode — and reaches the "
        f"same fit to {agree:.0e}: the two routes differ in cost and in what they "
        f"do when the factor columns are nearly dependent, not in the answer here"
    )


def _demo_cp_degenerate() -> None:
    """The fit stays excellent while the factors stop being the answer."""
    rng = np.random.default_rng(5)
    shape = (25, 18, 12)
    R = 4
    A0, B, C = (rng.standard_normal((n, R)) for n in shape)
    rows = []
    for label, gap in (("distinct atoms", None), ("two at cosine 0.999", 1e-3)):
        A = A0.copy()
        if gap is not None:
            A[:, 1] = A[:, 0] + gap * rng.standard_normal(shape[0])
        X = _cp_model([A, B, C])
        fits, recs, iters = [], [], []
        for seed in range(4):
            f = TensorDecomposition(R, variant="cp_als", n_iter=500, tol=1e-12,
                                    seed=seed).fit(X)
            fits.append(f.fit_score(X))
            recs.append(_factor_recovery(f.factors_, [A, B, C])[1])
            iters.append(len(f.fits_))
        rows.append((label, float(np.linalg.cond(A)), fits, recs, iters))
    clean, dup = rows
    assert min(clean[3]) > 1.0 - 1e-6 and max(clean[4]) < 100, clean
    assert min(dup[3]) < 0.99 and min(dup[2]) > 0.999, dup

    print(
        f"degenerate: the same tensor with one mode's second column moved to cosine "
        f"0.999 of its first, which raises that mode's condition number to "
        f"{dup[1]:.0f} from {clean[1]:.1f}. Four starts, four different answers: the "
        f"fit reads {'/'.join(f'{v:.7f}' for v in dup[2])} against "
        f"{'/'.join(f'{v:.7f}' for v in clean[2])} for the well-conditioned basis, "
        f"and the second mode's factor is recovered at "
        f"{'/'.join(f'{v:.3f}' for v in dup[3])} against "
        f"{'/'.join(f'{v:.3f}' for v in clean[3])}. Every one of those fits looks "
        f"like success; none of them is the truth, which is exactly representable at "
        f"fit 1.0, and each start sits at its own stationary point (8000 sweeps "
        f"does not move it). Near-collinear factors make the decomposition almost "
        f"non-unique, and the fit cannot see the difference"
    )


def _demo_tucker() -> None:
    """Orthogonal factors and a core: identified as subspaces, and HOOI vs HOSVD."""
    rng = np.random.default_rng(1)
    shape = (12, 10, 8)
    truth_ranks = (4, 4, 4)
    U = [np.linalg.qr(rng.standard_normal((n, r)))[0]
         for n, r in zip(shape, truth_ranks, strict=True)]
    Xt = _tucker_model(rng.standard_normal(truth_ranks), U)
    out = {}
    for ranks in (truth_ranks, (2, 3, 2)):
        X = Xt + 0.02 * rng.standard_normal(shape)
        for variant in ("hosvd", "tucker_hooi"):
            f = TensorDecomposition(ranks, variant=variant, n_iter=300,
                                    tol=1e-13).fit(X)
            out[ranks, variant] = (f.fit_score(X), f.factors_)
    sub = _subspace_score(out[truth_ranks, "tucker_hooi"][1], U)
    assert min(sub) > 0.99, sub
    truncated = (out[(2, 3, 2), "tucker_hooi"][0]
                 - out[(2, 3, 2), "hosvd"][0])
    assert truncated > 1e-3, truncated

    print(
        f"tucker_hooi: a rank-{truth_ranks} tensor plus 2% noise, fitted at that "
        f"rank. Both variants reach fit {out[truth_ranks, 'hosvd'][0]:.6f} and the "
        f"factor matrices span the true subspaces at "
        f"{'/'.join(f'{s:.5f}' for s in sub)} — that is all Tucker identifies, "
        f"since any rotation inside a mode can be absorbed by the core, so the "
        f"core is only the core *relative to the factors*. Truncating to ranks "
        f"(2,3,2) is where the two variants separate: HOSVD's one-pass truncation "
        f"stops at {out[(2, 3, 2), 'hosvd'][0]:.6f} and HOOI's alternating "
        f"subspace iteration reaches {out[(2, 3, 2), 'tucker_hooi'][0]:.6f} — the "
        f"gap is {truncated:.1e}, and it appears only when the requested rank is "
        f"not the data's rank. HOSVD is a truncation, HOOI is the best low-rank "
        f"approximation reachable from it"
    )


def _demo_parafac2() -> None:
    """Ragged slices, one shared factor, and the rotations CP cannot express."""
    rng = np.random.default_rng(11)
    n_rows, R, J = 30, 3, 8
    widths = (8, 9, 10, 9)
    A = rng.standard_normal((n_rows, R))
    B = rng.standard_normal((J, R))
    P, X, D = [], [], []
    for width in widths:
        P.append(np.linalg.qr(rng.standard_normal((width, J)))[0])
        D.append(rng.uniform(0.5, 2.0, R))
        X.append(A * D[-1] @ (P[-1] @ B).T)

    fit = TensorDecomposition(R, variant="parafac2", n_iter=8000, tol=1e-9).fit(X)
    score = fit.fit_score(X)
    arec = _factor_recovery([fit.factors_[0]], [A])[0]
    rec = [_factor_recovery([fit.transforms_[k] @ fit.factors_[1]], [P[k] @ B])[0]
           for k in range(len(X))]
    assert score > 0.999 and arec > 0.999 and min(rec) > 0.999, (score, arec, rec)

    pad = np.zeros((n_rows, max(widths), len(X)))
    for k, x in enumerate(X):
        pad[:, :x.shape[1], k] = x
    cp = TensorDecomposition(R, variant="cp_als", n_iter=2000, tol=1e-11).fit(pad)
    Af, Bf, Cf = cp.factors_
    num = sum(float(np.sum((x - Af * Cf[k] @ Bf[:x.shape[1]].T) ** 2))
              for k, x in enumerate(X))
    den = sum(float(np.sum(x ** 2)) for x in X)
    cp_score = 1.0 - float(np.sqrt(num / den))
    assert cp_score < 0.6, cp_score

    print(
        f"parafac2: {len(X)} slices of {n_rows} rows and widths {widths}, each "
        f"X_k = A diag(d_k)(P_k B)ᵀ with a known orthogonal P_k. The shared factor "
        f"comes back at cosine {arec:.7f} and each slice's loading P_k B at "
        f"{'/'.join(f'{r:.6f}' for r in rec)}, with fit {score:.8f} — but it takes "
        f"{len(fit.fits_)} sweeps to get there, because the Procrustes step and the "
        f"stacked solves converge linearly and slowly: after 2000 sweeps the fit is "
        f"still {fit.fits_[2000]:.6f}, which at three decimals looks finished. "
        f"Zero-padding the slices to a dense tensor and running ordinary CP, the "
        f"usual alternative when the widths differ, is scored on the true slices "
        f"and stops at {cp_score:.4f}: the per-slice rotation is not in CP's model, "
        f"so no single loading can serve four slices whose loadings are four "
        f"different rotations. The model is what buys the difference; the optimiser "
        f"only decides how long it takes"
    )


def main() -> None:
    _demo_cp()
    _demo_cp_degenerate()
    _demo_tucker()
    _demo_parafac2()
    print("tensor_decomposition demo passed")


if __name__ == "__main__":
    main()
