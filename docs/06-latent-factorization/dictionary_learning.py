"""Dictionary learning — turning sparse coding into a method that learns its basis.

Chapter:   06-latent-factorization
Companion: dictionary_learning.md
Tier 3 · 4 variants: ksvd, mod, online_dl, sparse_dict

    min_{D, A}  ‖X − A Dᵀ‖_F²     subject to  ‖a_i‖₀ ≤ s,  ‖d_j‖ = 1       (1)

Sparse coding assumes the dictionary. This atom learns it, and (1) is the same
problem the whole chapter keeps meeting with one extra block: the factorisation
is non-convex, and the only practical algorithm is alternating minimisation —
code given the dictionary (`sparse_coding`), then dictionary given the codes.

The reason it works at all is that each half is easy. Given `D`, the codes are
the sparse coding problem, solved per row. Given the codes, the dictionary is a
least-squares problem with a closed form — which is `mod`. The interesting
variants differ in how they spend the second half: `mod` solves it in one matrix
inverse, `ksvd` updates one atom at a time with an SVD that also revises that
atom's coefficients, `online_dl` never forms the full problem at all, and
`sparse_dict` constrains each atom to be sparse in a fixed base.

Non-convexity is not a footnote. A random initialisation, a different seed and a
different number of atoms all give different dictionaries with nearly the same
reconstruction error, and the atoms come out in no particular order, each one
sign-flipped at random. Any comparison between two learned dictionaries has to
be up to that group, which is what the demo's matching does and what
`atom_recovery` is for.
"""

from __future__ import annotations

import numpy as np
from sparse_coding import _lasso_cd, _normalise  # sibling module: same folder, same sys.path

EPS = 1e-12
VARIANTS = ("ksvd", "mod", "online_dl", "sparse_dict")


def _codes(X: np.ndarray, D: np.ndarray, nnz: int, lam: float) -> np.ndarray:
    """Sparse codes for every row of `X`, `(n, K)`, by orthogonal matching pursuit.

    `nnz` is the per-row budget. Greedy rather than convex because dictionary
    learning runs this step a hundred times and the outer loop only needs codes
    that are good, not optimal: a solver accurate to 1e-12 inside an alternation
    that moves the dictionary by 1e-1 is wasted work.
    """
    from sparse_coding import SparseCoding

    return SparseCoding(nnz=nnz, variant="omp").fit(X, D).codes_


def _rescale(D: np.ndarray, A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit columns for `D`, with `A` scaled to keep `A Dᵀ` unchanged.

    The constraint `‖d_j‖ = 1` in (1) is what stops the trivial solution where
    every atom grows and every coefficient shrinks; enforcing it by rescaling
    both factors is exact, whereas rescaling only `D` changes the fit.
    """
    nrm = np.linalg.norm(D, axis=0)
    keep = nrm > EPS
    if not np.all(keep):
        nrm = np.where(keep, nrm, 1.0)
    return D / nrm, A * nrm


class DictionaryLearning:
    """Learn an overcomplete dictionary and its sparse codes.

    `DictionaryLearning(n_components=20, *, variant="ksvd", nnz=3, n_iter=12,
    lam=0.1, batch=128, gamma=0.1, base=None, seed=0)`. `fit(X)` takes `(n, d)`
    and produces `dictionary_` `(d, K)`, `codes_` `(n, K)` and `rmse_`, the
    reconstruction error after each outer iteration.

    `nnz` is the coding budget, `n_iter` the number of alternations, `batch` and
    `lam` are used by `online_dl`, and `gamma` is the sparsity penalty on the
    atoms of `sparse_dict` in its base `base` `(d, P)` (identity by default).
    """

    def __init__(self, n_components: int = 20, *, variant: str = "ksvd",
                 nnz: int = 3, n_iter: int = 12, lam: float = 0.1,
                 batch: int = 128, gamma: float = 0.1,
                 base: np.ndarray | None = None, seed: int = 0) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        if n_components < 1 or nnz < 1:
            raise ValueError("n_components and nnz must be at least 1")
        if gamma < 0.0:
            raise ValueError("gamma is a penalty; it cannot be negative")
        self.n_components = int(n_components)
        self.variant = variant
        self.nnz = int(nnz)
        self.n_iter = int(n_iter)
        self.lam = float(lam)
        self.batch = int(batch)
        self.gamma = float(gamma)
        self.base = None if base is None else np.asarray(base, dtype=float)
        self.seed = int(seed)

    def fit(self, X: np.ndarray) -> DictionaryLearning:
        """Alternate coding and dictionary updates, `n_iter` times.

        Initialised from random unit atoms. `dictionary_` is `(d, K)` with unit
        columns, `codes_` `(n, K)` with at most `nnz` non-zeros per row, and
        `rmse_` the reconstruction error of the codes that were used to produce
        each dictionary.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be (n, d), got shape {X.shape}")
        n, d = X.shape
        rng = np.random.default_rng(self.seed)
        if n >= self.n_components:
            # initialise on real signals, not on noise: a Gaussian dictionary has
            # no relation to the data's structure, so the first coding pass spends
            # its budget fitting the atoms it was given
            D = _normalise(X[rng.choice(n, self.n_components, replace=False)].T)
        else:
            D = _normalise(rng.standard_normal((d, self.n_components)))
        base = np.eye(d) if self.base is None else self.base
        if base.shape[0] != d:
            raise ValueError(f"base has {base.shape[0]} rows but X has {d} columns")
        C = np.linalg.lstsq(base, D, rcond=None)[0] if self.variant == "sparse_dict" else None
        if C is not None:
            D = _normalise(base @ C)

        A = _codes(X, D, self.nnz, self.lam)
        self.rmse_ = [self._error(X, A, D)]
        for _ in range(self.n_iter):
            if self.variant == "online_dl":
                D, A = self._online(X, D, rng)
            else:
                if self.variant == "mod":
                    D, A = _rescale(*self._mod(X, A))
                elif self.variant == "ksvd":
                    D = self._ksvd(X, A, D)
                else:
                    D, A, C = self._sparse_dict(X, A, D, C, base)
                A = _codes(X, D, self.nnz, self.lam)
            self.rmse_.append(self._error(X, A, D))
        self.dictionary_ = D
        self.codes_ = A
        self.base_ = base
        return self

    @staticmethod
    def _error(X: np.ndarray, A: np.ndarray, D: np.ndarray) -> float:
        """Root-mean-square reconstruction error per entry — comparable across variants."""
        return float(np.sqrt(np.mean((X - A @ D.T) ** 2)))

    def _mod(self, X: np.ndarray, A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Method of optimal directions: `Dᵀ = (AᵀA)⁻¹AᵀX`, one linear solve.

        The ridge on `AᵀA` is not decoration: an atom used by no row makes that
        matrix singular, and the ridge turns "undefined" into "stay where you
        are" for exactly those directions.
        """
        G = A.T @ A
        D = np.linalg.solve(G + 1e-8 * np.eye(G.shape[0]), A.T @ X).T
        return D, A

    def _ksvd(self, X: np.ndarray, A: np.ndarray, D: np.ndarray) -> np.ndarray:
        """K-SVD: one atom at a time, by the rank-one SVD of its own residual.

        For atom `j`, restricting to the rows that use it, the residual without
        its contribution is `E`, and the best `(d_j, a_j)` pair is the leading
        singular pair of `E`. Both factors update at once, which is why K-SVD
        converges in fewer passes than MOD at the same cost per atom.
        """
        for j in range(D.shape[1]):
            rows = np.nonzero(A[:, j])[0]
            if rows.size == 0:
                D[:, j], A[:, j] = self._revive(X, A, D)
                continue
            R = X - A @ D.T + np.outer(A[:, j], D[:, j])
            U, s, Vt = np.linalg.svd(R[rows], full_matrices=False)
            D[:, j] = Vt[0]
            A[rows, j] = s[0] * U[:, 0]
        return D

    @staticmethod
    def _revive(X: np.ndarray, A: np.ndarray, D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Give a dead atom the worst-reconstructed signal, unit normed."""
        resid = X - A @ D.T
        worst = int(np.argmax(np.sum(resid ** 2, axis=1)))
        vec = resid[worst]
        nrm = float(np.linalg.norm(vec))
        if nrm < EPS:
            return D[:, 0], np.zeros(A.shape[0])
        coef = np.zeros(A.shape[0])
        coef[worst] = nrm
        return vec / nrm, coef

    def _online(self, X: np.ndarray, D: np.ndarray,
                rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Online dictionary learning: accumulate `A = Σααᵀ`, `B = Σxαᵀ`, update.

        One pass over the data per outer iteration, one dictionary update per
        batch. The two accumulators are all the state the update needs, so the
        memory is `O(K² + dK)` regardless of how many signals arrive — which is
        the entire point of the variant.
        """
        K = D.shape[1]
        A_acc = np.zeros((K, K))
        B_acc = np.zeros((X.shape[1], K))
        codes = np.zeros((X.shape[0], K))
        order = rng.permutation(X.shape[0])
        for start in range(0, X.shape[0], self.batch):
            idx = order[start:start + self.batch]
            Xb = X[idx]
            Ab = _codes(Xb, D, self.nnz, self.lam)
            codes[idx] = Ab
            A_acc += Ab.T @ Ab
            B_acc += Xb.T @ Ab
            for j in range(K):
                if A_acc[j, j] < 1e-9:
                    D[:, j], _ = self._revive(Xb, Ab, D)
                    continue
                col = A_acc[:, j].copy()
                col[j] = 0.0
                D[:, j] = (B_acc[:, j] - D @ col) / A_acc[j, j]
                nrm = float(np.linalg.norm(D[:, j]))
                if nrm > EPS:
                    D[:, j] /= nrm
        return D, codes

    def _sparse_dict(self, X: np.ndarray, A: np.ndarray, D: np.ndarray,
                     C: np.ndarray, base: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Double sparsity: `D = base · C` with `C` sparse, so atoms stay sparse.

        The atom update is the same restricted least-squares problem as K-SVD,
        with the SVD replaced by a lasso in the base — the second factor is
        constrained rather than solved, which is the price and the point.
        """
        G = base.T @ base
        for j in range(D.shape[1]):
            aj = A[:, j]
            w = float(aj @ aj)
            if w < EPS:
                continue
            R = X - A @ D.T + np.outer(aj, D[:, j])
            b = base.T @ (R.T @ aj) / w
            c = _lasso_cd(G, b, self.gamma, 200, 1e-10)[0]
            nrm = float(np.linalg.norm(base @ c))
            if nrm < 1e-8:
                continue
            C[:, j] = c / nrm
            A[:, j] = aj * nrm
            D[:, j] = base @ C[:, j]
        return D, A, C


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _planted(n: int = 600, d: int = 20, K: int = 40, nnz: int = 3,
             seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`n` signals from `nnz` atoms of a random dictionary with unit columns."""
    rng = np.random.default_rng(seed)
    D = _normalise(rng.standard_normal((d, K)))
    A = np.zeros((n, K))
    for i in range(n):
        A[i, rng.choice(K, nnz, replace=False)] = rng.standard_normal(nnz)
    return A @ D.T, D, A


def atom_recovery(learned: np.ndarray, truth: np.ndarray) -> float:
    """Mean `max_j |cos|` of each learned atom with the true dictionary.

    Correct up to permutation and sign, which is the only identification a
    dictionary has: nothing in (1) orders the atoms or fixes their signs.
    """
    G = np.abs(_normalise(learned).T @ _normalise(truth))
    return float(np.mean(np.max(G, axis=1)))


def _demo_ksvd() -> None:
    """It learns the atoms, not just a basis that fits."""
    X, Dtrue, _ = _planted(n=500)
    fit = DictionaryLearning(40, variant="ksvd", nnz=3, n_iter=12).fit(X)
    rmse = fit.rmse_
    rec = atom_recovery(fit.dictionary_, Dtrue)
    assert rmse[-1] < 0.6 * rmse[0], rmse
    assert rec > 0.85, rec

    alive = int(np.sum(np.any(fit.codes_ != 0.0, axis=0)))
    print(
        f"ksvd: 500 signals of 20 dimensions, built from 3 of 40 unknown atoms. "
        f"Initialised on 40 of those signals, the error falls {rmse[0]:.3f} → "
        f"{rmse[3]:.3f} → {rmse[-1]:.3f} over {len(rmse) - 1} alternations, and the "
        f"learned atoms match the true ones at mean cosine {rec:.3f} — sign and order "
        f"are arbitrary, so that is the strongest statement available. {alive} of 40 "
        f"atoms are used by at least one signal: the coding budget, not the atom "
        f"count, decides how many survive"
    )


def _demo_mod_vs_ksvd() -> None:
    """The famous variant does not win, and neither does its rival: it is a draw."""
    rows = []
    for seed in (10, 11, 12):
        X, _, _ = _planted(n=300, K=30, seed=seed)
        mod = DictionaryLearning(30, variant="mod", nnz=3, n_iter=6).fit(X)
        ksvd = DictionaryLearning(30, variant="ksvd", nnz=3, n_iter=6).fit(X)
        rows.append((mod.rmse_[-1], ksvd.rmse_[-1]))
    wins = sum(m < k for m, k in rows)
    assert 0 < wins < len(rows), rows
    mean_mod = float(np.mean([m for m, _ in rows]))
    mean_ksvd = float(np.mean([k for _, k in rows]))
    assert abs(mean_mod - mean_ksvd) < 0.01, (mean_mod, mean_ksvd)

    print(
        f"mod: one (AᵀA)⁻¹AᵀX against K-SVD's 30 rank-one SVDs, on three problems "
        f"(300 signals, 6 alternations). MOD reads "
        f"{'/'.join(f'{m:.3f}' for m, _ in rows)} and K-SVD "
        f"{'/'.join(f'{k:.3f}' for _, k in rows)} — MOD wins {wins} of 3 and the "
        f"means are {mean_mod:.3f} and {mean_ksvd:.3f}. Given fixed codes MOD is the "
        f"exact joint least-squares solution while K-SVD is one Gauss-Seidel sweep, "
        f"and the coefficient update K-SVD is known for is thrown away as soon as "
        f"the alternation re-codes. Neither dominates: the difference is inside the "
        f"spread the initialisation produces, which is what a non-convex objective "
        f"does to a comparison between two stationary-point algorithms"
    )


def _demo_online() -> None:
    """One pass per epoch, bounded state, nearly the same dictionary."""
    X, Dtrue, _ = _planted(n=800, seed=2)
    batch = DictionaryLearning(40, variant="online_dl", nnz=3, batch=100,
                               n_iter=8).fit(X)
    batchfit = DictionaryLearning(40, variant="ksvd", nnz=3, n_iter=8).fit(X)
    gap = batch.rmse_[-1] - batchfit.rmse_[-1]
    assert gap < 0.05, gap
    rec = atom_recovery(batch.dictionary_, Dtrue)
    assert rec > 0.8, rec

    state = (40 * 40 + 20 * 40) * 8 / 1e3
    print(
        f"online_dl: {X.shape[0]} signals in batches of 100, one dictionary update per "
        f"batch, and no code matrix retained — the state is the 40×40 accumulator A "
        f"and the 20×40 B, {state:.0f} kB against the {X.shape[0] * 40 * 8 / 1e3:.0f} kB "
        f"the batch methods hold. It reaches {batch.rmse_[-1]:.4f} against "
        f"{batchfit.rmse_[-1]:.4f} for K-SVD, at mean atom cosine {rec:.3f} to the "
        f"truth — better here, because 64 batch updates beat 8 full passes with the "
        f"same total data, but those updates are progressively stale: the "
        f"accumulators are unweighted, so the first batches dominate forever"
    )


def _demo_sparse_dict() -> None:
    """Atoms written in a base, and what the constraint costs."""
    X, _, _ = _planted(seed=3)
    ladder = []
    for gamma in (0.0, 0.15):
        f = DictionaryLearning(40, variant="sparse_dict", nnz=3, gamma=gamma,
                               n_iter=6).fit(X)
        per = np.count_nonzero(f.dictionary_, axis=0)
        ladder.append((float(np.mean(per)), int(np.max(per)), f.rmse_[-1]))
    assert ladder[0][0] == 20.0, ladder[0]
    assert ladder[1][0] < 7.0 and ladder[1][2] > ladder[0][2]

    mod = DictionaryLearning(40, variant="mod", nnz=3, n_iter=6).fit(X)
    print(
        f"sparse_dict: atoms written in a fixed base, D = ΦC with C penalised. At "
        f"γ = 0 the atoms are dense ({ladder[0][0]:.0f} of 20 base coefficients) and "
        f"MOD on the same data reaches {mod.rmse_[-1]:.4f} against {ladder[0][2]:.4f} "
        f"here — the same model, reached by coordinate rather than joint updates. At "
        f"γ = 0.15 the atoms keep {ladder[1][0]:.1f} coefficients (at most "
        f"{ladder[1][1]}) and the error rises to {ladder[1][2]:.4f}: "
        f"{20 / ladder[1][0]:.0f}× fewer parameters per atom for "
        f"{(ladder[1][2] / ladder[0][2] - 1) * 100:.0f}% more reconstruction error. "
        f"That ratio is the entire argument for double sparsity — the atoms become "
        f"readable, and what you pay is fit"
    )


def main() -> None:
    _demo_ksvd()
    _demo_mod_vs_ksvd()
    _demo_online()
    _demo_sparse_dict()
    print("dictionary_learning demo passed")


if __name__ == "__main__":
    main()
