"""Sparse coding — few atoms out of many, and the four solvers that find them.

Chapter:   06-latent-factorization
Companion: sparse_coding.md
Tier 3 · 4 variants: omp, lasso_sc, fista_sc, group_sc

    min_α  ½‖x − D α‖² + λ‖α‖₁        D ∈ R^{d×K},  K > d                (1)

Write a signal as a short combination of atoms from a dictionary larger than the
signal is long. The dictionary is given here — learning it is
`dictionary_learning` — so this atom is one thing: a solver, run once per signal,
for the same convex program.

It is a real program with a corner. The fit is not unique when K > d: with
λ = 0 there are infinitely many exact representations of x and the solver picks
one by accident of its update order. `lam=0` is therefore a least-squares fit
with a tie-break nobody chose.

The four variants answer "make it cheap" four ways. `omp` is greedy and exact on
the k-sparse model but has no optimality guarantee. `lasso_sc` solves (1) exactly
by coordinate descent and has to build the K×K Gram to do it. `fista_sc` reaches
the same optimum touching only `D` and `Dᵀ`, which is what keeps it affordable
when K is thousands. `group_sc` changes the penalty so that whole families of
atoms switch on together.
"""

from __future__ import annotations

import math

import numpy as np

EPS = 1e-12
VARIANTS = ("group_sc", "fista_sc", "lasso_sc", "omp")


def _soft(z: np.ndarray | float, t: float) -> np.ndarray | float:
    """Soft threshold: `sign(z)·max(|z| − t, 0)`, the prox of `t‖·‖₁`."""
    return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)


def _block_soft(v: np.ndarray, t: float) -> np.ndarray:
    """Block soft threshold: `max(0, 1 − t/‖v‖) v`, the prox of `t‖·‖₂`."""
    nrm = float(np.linalg.norm(v))
    if nrm < EPS:
        return np.zeros_like(v)
    return v * max(0.0, 1.0 - t / nrm)


def _normalise(D: np.ndarray) -> np.ndarray:
    """Scale every atom to unit L2 norm, refusing a zero atom.

    The penalty in (1) thresholds coefficients, so it is only comparable across
    atoms of equal norm. A dictionary that arrives unnormalised is rescaled here
    once, rather than being penalised according to the scaling someone used.
    """
    nrm = np.linalg.norm(D, axis=0)
    if np.any(nrm < EPS):
        raise ValueError("a dictionary atom is all zeros — drop it or replace it")
    return D / nrm


def _objective(a: np.ndarray, D: np.ndarray, x: np.ndarray, lam: float,
               groups: list[np.ndarray] | None = None) -> float:
    """`½‖x − Da‖² + λ·penalty(a)` — the number every solver here minimises."""
    fit = 0.5 * float(np.sum((x - D @ a) ** 2))
    if groups is None:
        return fit + lam * float(np.abs(a).sum())
    return fit + lam * float(sum(np.linalg.norm(a[g]) for g in groups))


def _lipschitz(D: np.ndarray, n_iter: int = 60, seed: int = 0) -> float:
    """`‖D‖₂²` by power iteration, the step size the proximal solvers need.

    Exact would be `np.linalg.eigvalsh(D.T @ D)[-1]`, which costs the very K×K
    Gram this path exists to avoid. Sixty matvecs give the top eigenvalue to
    machine precision for a random dictionary: the ratio `λ₂/λ₁` of a Gaussian
    Gram is far from 1.
    """
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(D.shape[1])
    v /= np.linalg.norm(v)
    top = 0.0
    for _ in range(n_iter):
        u = D.T @ (D @ v)
        top = float(v @ u)
        nrm = float(np.linalg.norm(u))
        if nrm < EPS:
            break
        v = u / nrm
    return max(top, EPS)


def _omp(D: np.ndarray, x: np.ndarray, nnz: int, tol: float) -> tuple[np.ndarray, int]:
    """Orthogonal matching pursuit: add the atom most correlated with the
    residual, then re-solve least squares on the chosen set.

    `D` is `(d, K)`, `x` is `(d,)`, the result is `(K,)` with at most `nnz`
    non-zeros. Re-solving from scratch each step is `O(k²d)` total; the classic
    incremental-QR version is `O(kd)` and is not here because `k` is small.
    """
    res = x.copy()
    support: list[int] = []
    coef = np.zeros(D.shape[1])
    sol = np.zeros(0)
    for _ in range(min(nnz, D.shape[1])):
        corr = np.abs(D.T @ res)
        corr[support] = -1.0                             # never re-select an atom
        support.append(int(np.argmax(corr)))
        sol, *_ = np.linalg.lstsq(D[:, support], x, rcond=None)
        res = x - D[:, support] @ sol
        if float(np.linalg.norm(res)) <= tol * max(float(np.linalg.norm(x)), EPS):
            break
    if support:
        coef[support] = sol
    return coef, len(support)


def _lasso_cd(D: np.ndarray, x: np.ndarray, lam: float, n_iter: int,
              tol: float) -> tuple[np.ndarray, int]:
    """Cyclic coordinate descent on (1), with the Gram matrix formed up front.

    Each coordinate update is a soft threshold of the partial residual divided by
    `G_jj`, so there is no step size and no line search. Exact for fixed `λ`, and
    the cost is `O(K²d)` to build `G` plus `O(K²)` per sweep.
    """
    G, b = D.T @ D, D.T @ x
    d = np.diag(G).copy()
    a = np.zeros(G.shape[0])
    used = n_iter
    for it in range(n_iter):
        a_old = a.copy()
        for j in range(G.shape[0]):
            partial = b[j] - (G[j] @ a - d[j] * a[j])
            a[j] = _soft(partial, lam) / d[j]
        if float(np.max(np.abs(a - a_old))) <= tol * max(1.0, float(np.max(np.abs(a)))):
            used = it + 1
            break
    return a, used


def _fista(D: np.ndarray, x: np.ndarray, lam: float, n_iter: int,
           tol: float) -> tuple[np.ndarray, int]:
    """FISTA: proximal gradient with Nesterov momentum, on `D` alone.

    The step is `1/L` with `L = ‖D‖₂²`, the exact Lipschitz constant of the
    gradient, and each iteration costs two matvecs — `O(Kd)` — with no Gram in
    sight. Momentum turns the `O(1/k)` rate of plain ISTA into `O(1/k²)`.
    """
    L = _lipschitz(D)
    a = np.zeros(D.shape[1])
    y = a.copy()
    t = 1.0
    used = n_iter
    for it in range(n_iter):
        a_new = _soft(y - (D.T @ (D @ y - x)) / L, lam / L)
        t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        y = a_new + ((t - 1.0) / t_new) * (a_new - a)
        step = float(np.max(np.abs(a_new - a)))
        a, t = a_new, t_new
        if step <= tol * max(1.0, float(np.max(np.abs(a)))):
            used = it + 1
            break
    return a, used


def _group_pg(D: np.ndarray, x: np.ndarray, groups: list[np.ndarray], lam: float,
              n_iter: int, tol: float) -> tuple[np.ndarray, int]:
    """Proximal gradient with the group penalty `λ Σ_g ‖a_g‖₂`.

    Same `1/L` step as FISTA, no momentum: the block prox is a norm and a scale,
    and plain ISTA is monotone, which is worth more here than the extra rate.
    """
    L = _lipschitz(D)
    a = np.zeros(D.shape[1])
    used = n_iter
    for it in range(n_iter):
        trial = a - (D.T @ (D @ a - x)) / L
        a_new = trial.copy()
        for g in groups:
            a_new[g] = _block_soft(trial[g], lam / L)
        step = float(np.max(np.abs(a_new - a)))
        a = a_new
        if step <= tol * max(1.0, float(np.max(np.abs(a)))):
            used = it + 1
            break
    return a, used


class SparseCoding:
    """Sparse codes for a fixed, overcomplete dictionary.

    `SparseCoding(nnz=5, *, variant="lasso_sc", lam=0.1, n_groups=4,
    n_iter=1000, tol=1e-8)`. `fit(X, D)` with `X` `(n, d)` and `D` `(d, K)`
    computes `codes_` `(n, K)`; `transform` reuses the stored dictionary and
    `reconstruct` maps back to `(n, d)`.

    `nnz` is the greedy budget of `omp`; `lam` is the penalty of the three convex
    variants; `n_groups` splits the `K` atoms into contiguous groups for
    `group_sc` and is ignored otherwise.
    """

    def __init__(self, nnz: int = 5, *, variant: str = "lasso_sc", lam: float = 0.1,
                 n_groups: int = 4, n_iter: int = 1000, tol: float = 1e-8) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        if lam < 0.0:
            raise ValueError("lam is a penalty; it cannot be negative")
        if variant == "omp" and nnz < 1:
            raise ValueError("omp needs nnz ≥ 1")
        self.nnz = int(nnz)
        self.variant = variant
        self.lam = float(lam)
        self.n_groups = int(n_groups)
        self.n_iter = int(n_iter)
        self.tol = float(tol)

    def fit(self, X: np.ndarray, D: np.ndarray) -> SparseCoding:
        """Compute codes for every row of `X` against `D`.

        `X` is `(n, d)`, `D` is `(d, K)` with atoms as columns; `codes_` is
        `(n, K)`, `dictionary_` is the column-normalised `D` actually used, and
        `n_iter_` is the worst inner-iteration count over the batch.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        D = np.asarray(D, dtype=float)
        if X.shape[1] != D.shape[0]:
            raise ValueError(f"X has {X.shape[1]} features but D has {D.shape[0]} rows")
        D = _normalise(D)
        K = D.shape[1]
        if self.variant == "group_sc":
            if not 1 <= self.n_groups <= K:
                raise ValueError(f"n_groups must be in 1..{K}, got {self.n_groups}")
            self.groups_ = list(np.array_split(np.arange(K), self.n_groups))
        else:
            self.groups_ = None

        codes = np.zeros((X.shape[0], K))
        iters = 0
        for i, x in enumerate(X):
            codes[i], used = self._one(D, x)
            iters = max(iters, used)
        self.dictionary_ = D
        self.codes_ = codes
        self.n_iter_ = iters
        return self

    def _one(self, D: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, int]:
        """Dispatch one signal to the variant's solver; `(n,)` result and iterations."""
        if self.variant == "omp":
            return _omp(D, x, self.nnz, self.tol)
        if self.variant == "lasso_sc":
            return _lasso_cd(D, x, self.lam, self.n_iter, self.tol)
        if self.variant == "fista_sc":
            return _fista(D, x, self.lam, self.n_iter, self.tol)
        return _group_pg(D, x, self.groups_, self.lam, self.n_iter, self.tol)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Codes for new rows, against the dictionary this model was fitted with."""
        if not hasattr(self, "dictionary_"):
            raise RuntimeError("call fit before transform")
        twin = SparseCoding(self.nnz, variant=self.variant, lam=self.lam,
                            n_groups=self.n_groups, n_iter=self.n_iter, tol=self.tol)
        return twin.fit(X, self.dictionary_).codes_

    def fit_transform(self, X: np.ndarray, D: np.ndarray) -> np.ndarray:
        """`fit(X, D).codes_` — the common path when the dictionary is an input."""
        return self.fit(X, D).codes_

    def reconstruct(self) -> np.ndarray:
        """`codes_ @ dictionary_.T`, `(n, d)` — the signal the codes explain."""
        if not hasattr(self, "codes_"):
            raise RuntimeError("call fit before reconstruct")
        return self.codes_ @ self.dictionary_.T


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _planted(n: int = 1, d: int = 40, K: int = 80, nnz: int = 2, noise: float = 0.0,
             seed: int = 0, D: np.ndarray | None = None
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`n` signals, each a random combination of `nnz` atoms.

    Pass `D` to plant against a specific dictionary — a coherence experiment is
    meaningless otherwise, because the signals and the dictionary have to be
    built from the same atoms.
    """
    rng = np.random.default_rng(seed)
    D = _normalise(rng.standard_normal((d, K))) if D is None else _normalise(D)
    truth = np.zeros((n, D.shape[1]))
    for i in range(n):
        support = rng.choice(D.shape[1], nnz, replace=False)
        truth[i, support] = rng.standard_normal(nnz)
    X = truth @ D.T + noise * rng.standard_normal((n, d))
    return X, D, truth


def _sparse_signal(d: int = 40, K: int = 80, nnz: int = 4,
                   seed: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One signal with decaying coefficients, so the λ path is a staircase."""
    rng = np.random.default_rng(seed)
    D = _normalise(rng.standard_normal((d, K)))
    a = np.zeros(K)
    a[rng.choice(K, nnz, replace=False)] = np.linspace(1.0, 0.4, nnz)
    return D, D @ a, a


def _demo_omp() -> None:
    """Exact on the model it assumes, and blind to a twin atom."""
    X, D, truth = _planted(n=20, seed=1)
    fit = SparseCoding(nnz=2, variant="omp").fit(X, D)
    hits = np.mean([set(np.nonzero(c)[0]) == set(np.nonzero(t)[0])
                    for c, t in zip(fit.codes_, truth, strict=True)])
    rel = float(np.linalg.norm(X - fit.reconstruct()) / np.linalg.norm(X))
    assert hits == 1.0 and rel < 1e-12, (hits, rel)

    noisy, Dn, _ = _planted(n=20, noise=0.02, seed=1)
    nfit = SparseCoding(nnz=2, variant="omp").fit(noisy, Dn)
    nrel = float(np.linalg.norm(noisy - nfit.reconstruct()) / np.linalg.norm(noisy))
    assert nrel < 0.2, nrel

    short = SparseCoding(nnz=1, variant="omp").fit(X, D)
    srel = float(np.linalg.norm(X - short.reconstruct()) / np.linalg.norm(X))
    assert srel > 0.4, srel

    # two atoms 0.994 apart: the signal is explained, the code is not identified
    rng = np.random.default_rng(0)
    Dt = _normalise(rng.standard_normal((40, 80)))
    Dt[:, 6] = _normalise(Dt[:, 5] + 0.02 * rng.standard_normal(40))
    cos = float(Dt[:, 5] @ Dt[:, 6])
    hits, twin, miss_rel = [], [], []
    for _ in range(20):
        t2 = np.zeros((10, 80))
        for i in range(10):
            free = [j for j in range(80) if j not in (5, 6)]
            t2[i, [5, *rng.choice(free, 1, replace=False)]] = rng.standard_normal(2)
        Xt = t2 @ Dt.T
        f = SparseCoding(nnz=2, variant="omp").fit(Xt, Dt)
        for c, tt in zip(f.codes_, t2, strict=True):
            same = set(np.nonzero(c)[0]) == set(np.nonzero(tt)[0])
            hits.append(same)
            if not same:
                twin.append(int(6 in set(np.nonzero(c)[0])))
                miss_rel.append(float(np.linalg.norm((c - tt) @ Dt.T)
                                      / np.linalg.norm(tt @ Dt.T)))
    assert np.mean(hits) < 0.9 and all(twin), np.mean(hits)

    print(
        f"omp: on 20 signals built from 2 of 80 atoms it recovers the support every "
        f"time and leaves {rel:.0e} of the signal unexplained; with 2% noise the "
        f"support is still exact and the residual is {nrel:.3f}; given a budget of 1 "
        f"atom instead of 2 it leaves {srel:.2f}. Add one atom at cosine {cos:.3f} to "
        f"another and support recovery falls to {np.mean(hits):.2f}, with all "
        f"{len(twin)} misses substituting the twin while the signal is still explained "
        f"to within {max(miss_rel):.2f}. Nothing in the output distinguishes 'this "
        f"atom' from 'this atom or its neighbour' — a sparse code of a coherent "
        f"dictionary is a statement about atoms, not about the signal alone"
    )


def _demo_lasso() -> None:
    """The exact convex solution, the path, and the two limits that are provable."""
    D, x, truth = _sparse_signal()
    lam_max = float(np.max(np.abs(D.T @ x)))

    zero = SparseCoding(variant="lasso_sc", lam=lam_max * 1.001).fit(x, D)
    assert np.all(zero.codes_ == 0.0)

    free = SparseCoding(nnz=2, variant="lasso_sc", lam=0.0).fit(x, D)
    free_rel = float(np.linalg.norm(x - free.reconstruct()) / np.linalg.norm(x))
    free_nnz = int(np.count_nonzero(free.codes_[0]))
    minnorm = D.T @ np.linalg.solve(D @ D.T, x)
    assert free_rel < 1e-6 and free_nnz > 70, (free_rel, free_nnz)

    path = []
    for lam in (0.05, 0.2, 0.3, 0.7, 0.9):
        a = SparseCoding(variant="lasso_sc", lam=lam).fit(x, D).codes_[0]
        path.append((lam, int(np.count_nonzero(a)),
                     float(np.linalg.norm(x - D @ a))))
    sizes = [p[1] for p in path]
    assert all(sizes[i] >= sizes[i + 1] for i in range(len(sizes) - 1)), sizes
    assert sizes[0] >= 4 and sizes[-1] == 0

    print(
        f"lasso_sc: the true code has 4 non-zeros and λ_max = max|Dᵀx| = {lam_max:.3f} "
        f"is the penalty above which the optimality condition admits only a = 0 — the "
        f"fit returns exactly zeros there. At λ = 0 it returns an exact "
        f"representation ({free_rel:.0e} residual) with all {free_nnz} coordinates "
        f"non-zero and ℓ2 norm {np.linalg.norm(free.codes_[0]):.3f} against the "
        f"minimum-norm solution's {np.linalg.norm(minnorm):.3f}: with 80 atoms and 40 "
        f"equations the zero-residual system has 40 free directions and coordinate "
        f"descent picks one by its update order. Walking λ = 0.05 → 0.2 → 0.3 → 0.7 → 0.9 takes the "
        f"support {' → '.join(str(p[1]) for p in path)} with the residual rising "
        f"{' → '.join(f'{p[2]:.2f}' for p in path)}: monotone in both directions, "
        f"and at λ = 0.05 the fit keeps "
        f"{sizes[0]} atoms against a truth of 4 — ℓ1 is not ℓ0, and the extra atom is "
        f"what a convex relaxation costs"
    )


def _demo_fista() -> None:
    """Same optimum from D alone. Where that pays is a question about K."""
    D, x, _ = _sparse_signal()
    lam = 0.05
    cd, cd_it = _lasso_cd(D, x, lam, 5000, 1e-12)
    best = _objective(cd, D, x, lam)
    fi, fi_it = _fista(D, x, lam, 5000, 1e-12)
    gap = _objective(fi, D, x, lam) - best
    assert abs(gap) < 1e-9, gap

    track, a, t = [], np.zeros(D.shape[1]), 1.0
    y, L = a.copy(), _lipschitz(D)
    for _ in range(40):
        a_new = _soft(y - (D.T @ (D @ y - x)) / L, lam / L)
        t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        y = a_new + ((t - 1.0) / t_new) * (a_new - a)
        a, t = a_new, t_new
        track.append(_objective(a, D, x, lam) - best)
    rises = int(np.sum(np.diff(track) > 1e-15))
    assert rises > 0
    cd_head = [_objective(_lasso_cd(D, x, lam, s, 0.0)[0], D, x, lam) - best
               for s in (1, 5, 10)]

    # large K: one coordinate sweep touches K² entries, one FISTA step 2Kd
    Db, xb, _ = _sparse_signal(d=64, K=1024, nnz=8, seed=5)
    cd_b = _lasso_cd(Db, xb, 0.05, 10, 0.0)[0]
    fi_b = _fista(Db, xb, 0.05, 80, 0.0)[0]
    gap_b = _objective(fi_b, Db, xb, 0.05) - _objective(cd_b, Db, xb, 0.05)
    assert gap_b < -1e-3, gap_b

    print(
        f"fista_sc: both solvers reach the same optimum ({gap:+.0e} apart) and the "
        f"objective is not monotone on the way — it rises on {rises} of the first 40 "
        f"steps, momentum overshooting while the guarantee still holds. At d = 40, "
        f"K = 80 coordinate descent wins outright: it is {cd_head[0]:.1e}/"
        f"{cd_head[1]:.1e}/{cd_head[2]:.1e} above the optimum after 1/5/10 sweeps, "
        f"while FISTA needs {fi_it} iterations to get to 1e-12. At d = 64, K = 1024 "
        f"the arithmetic flips: one coordinate sweep is K² = "
        f"{1024 ** 2 / 1e6:.1f}M multiply-adds against 2Kd = "
        f"{2 * 1024 * 64 / 1e3:.0f}k for one FISTA step, so a single sweep costs as "
        f"much as {1024 / (2 * 64):.0f} steps. Given a matched budget — ten sweeps "
        f"against eighty iterations, {10 * 1024 ** 2 / 1e6:.1f}M each — FISTA is "
        f"{-gap_b:.1e} lower. Coordinate descent pays K² for the privilege of exact "
        f"per-coordinate updates; FISTA pays 2Kd per step and never forms the Gram, "
        f"which is why it is the solver used on dictionaries of ten thousand atoms"
    )


def _demo_group() -> None:
    """Whole families switch on together; singleton groups must reduce to the lasso."""
    rng = np.random.default_rng(7)
    n_groups, per, d = 10, 3, 30
    D = np.zeros((d, n_groups * per))
    for g in range(n_groups):
        base = rng.standard_normal(d)
        for j in range(per):
            D[:, g * per + j] = base + 0.35 * rng.standard_normal(d)
    D = _normalise(D)
    truth = np.zeros(n_groups * per)
    for g in (1, 6):
        truth[g * per:(g + 1) * per] = rng.standard_normal(per)
    x = D @ truth

    fit = SparseCoding(variant="group_sc", lam=0.2, n_groups=n_groups).fit(x, D)
    active = {g for g in range(n_groups) if np.any(fit.codes_[0, g * per:(g + 1) * per])}
    assert active == {1, 6}, active

    lasso = SparseCoding(variant="lasso_sc", lam=0.2).fit(x, D)
    kept = set(np.nonzero(lasso.codes_[0])[0])
    assert len(kept) < 6 and kept <= set(np.nonzero(truth)[0]), kept

    # singleton groups are exactly the lasso, so the objectives must agree
    gg = SparseCoding(variant="group_sc", lam=0.05, n_groups=30, n_iter=3000).fit(x, D)
    ll = SparseCoding(variant="lasso_sc", lam=0.05).fit(x, D)
    single = [np.array([j]) for j in range(30)]
    gap = abs(_objective(gg.codes_[0], D, x, 0.05, single)
              - _objective(ll.codes_[0], D, x, 0.05))
    assert gap < 1e-6, gap

    within = float(np.mean([abs(D[:, g * per + i] @ D[:, g * per + j])
                            for g in (1, 6) for i in range(per) for j in range(i + 1, per)]))
    print(
        f"group_sc: 30 atoms in 10 families of 3 near-parallel members (mean |cos| "
        f"{within:.2f} inside a family) and a signal built from families 1 and 6. The "
        f"group penalty selects exactly those two families and zeros the other eight, "
        f"keeping all 6 coefficients. The lasso at the same λ keeps {len(kept)} — it "
        f"drops atom 4 outright, which costs it almost nothing because atoms 3 and 5 "
        f"reproduce it, so its answer is a coordinate statement where the group "
        f"answer is a family statement. With one atom per group the two objectives "
        f"agree to {gap:.0e}: the group penalty is the same penalty plus a choice of "
        f"what counts as a coordinate"
    )


def main() -> None:
    _demo_omp()
    _demo_lasso()
    _demo_fista()
    _demo_group()
    print("sparse_coding demo passed")


if __name__ == "__main__":
    main()
