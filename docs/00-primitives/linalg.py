"""Linear algebra — 11 factorisations every other atom is secretly using.

Chapter:   00-primitives
Companion: linalg.md
Tier 1 · 11 variants: lu, qr, cholesky, svd, truncated_svd, randomized_svd, eigh,
power_iteration, woodbury, solve_triangular, condition_number

`numpy.linalg` is used for the heavy lifting where a hand-written version would
only add bugs (SVD, eigh). LU, QR, Cholesky, power iteration and the Woodbury
identity are implemented explicitly because their structure is the lesson.
"""

from __future__ import annotations

import numpy as np


def lu(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PA = LU by Doolittle elimination with partial pivoting.

    Returns (P, L, U) with A = P^T L U. Partial pivoting is not optional: without
    it the factorisation is unstable even for well-conditioned matrices.
    """
    A = np.asarray(A, dtype=float).copy()
    n = A.shape[0]
    P = np.eye(n)
    L = np.eye(n)
    for k in range(n - 1):
        p = k + np.argmax(np.abs(A[k:, k]))
        if abs(A[p, k]) < 1e-14:
            raise ValueError("matrix is singular to working precision")
        if p != k:
            A[[k, p]] = A[[p, k]]
            P[[k, p]] = P[[p, k]]
            L[[k, p], :k] = L[[p, k], :k]
        L[k + 1 :, k] = A[k + 1 :, k] / A[k, k]
        A[k + 1 :, k:] -= np.outer(L[k + 1 :, k], A[k, k:])
    return P, L, np.triu(A)


def qr(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A = QR by Householder reflections. Q is orthogonal, R upper triangular."""
    A = np.asarray(A, dtype=float).copy()
    m, n = A.shape
    Q = np.eye(m)
    for k in range(min(m - 1, n)):
        x = A[k:, k]
        e = np.zeros_like(x)
        e[0] = np.sign(x[0]) if x[0] != 0 else 1.0
        v = x + e * np.linalg.norm(x)
        v = v / np.linalg.norm(v)
        A[k:, k:] -= 2.0 * np.outer(v, v @ A[k:, k:])
        Q[:, k:] -= 2.0 * np.outer(Q[:, k:] @ v, v)
    return Q[:, :n], A[:n, :]


def cholesky(A: np.ndarray) -> np.ndarray:
    """A = LL^T for symmetric positive definite A. Half the cost of LU and it
    fails exactly when A is not PD — a useful property, not a limitation."""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    L = np.zeros((n, n))
    for j in range(n):
        s = A[j, j] - L[j, :j] @ L[j, :j]
        if s <= 0:
            raise ValueError("matrix is not positive definite")
        L[j, j] = np.sqrt(s)
        L[j + 1 :, j] = (A[j + 1 :, j] - L[j + 1 :, :j] @ L[j, :j]) / L[j, j]
    return L


def svd(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A = U S V^T, singular values descending."""
    return np.linalg.svd(A, full_matrices=False)


def truncated_svd(A: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Best rank-k approximation in Frobenius and spectral norm (Eckart-Young).
    Computed densely here; `randomized_svd` is the scalable route."""
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    return U[:, :k], S[:k], Vt[:k]


def randomized_svd(
    A: np.ndarray, k: int, oversample: int = 10, iters: int = 2,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomised range finder (Halko-Martinsson-Tropp).

    Cost is O(nm·k) instead of O(nm·min(n,m)); the price is a small approximation
    error that power iteration drives down geometrically.
    """
    rng = np.random.default_rng(rng)
    m, n = A.shape
    p = min(k + oversample, min(m, n))
    Q = rng.normal(size=(n, p))
    Y = A @ Q
    for _ in range(iters):
        Q, _ = np.linalg.qr(Y)
        Z, _ = np.linalg.qr(A.T @ Q)
        Y = A @ Z
    Q, _ = np.linalg.qr(Y)
    B = Q.T @ A
    Ub, S, Vt = np.linalg.svd(B, full_matrices=False)
    return Q @ Ub[:, :k], S[:k], Vt[:k]


def eigh(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric eigendecomposition A = Q Λ Q^T, eigenvalues ascending."""
    return np.linalg.eigh(A)


def power_iteration(
    A: np.ndarray, iters: int = 500, tol: float = 1e-10
) -> tuple[float, np.ndarray]:
    """Dominant eigenpair of a symmetric matrix by repeated multiplication.

    Converges at rate |λ₂/λ₁|, which is the basis of PageRank, spectral clustering
    and every "iterate until it stops changing" vector in this repository.
    """
    A = np.asarray(A, dtype=float)
    v = np.ones(A.shape[0])
    v /= np.linalg.norm(v)
    for _ in range(iters):
        w = A @ v
        nw = np.linalg.norm(w)
        if nw == 0:
            break
        v_new = w / nw
        if min(np.linalg.norm(v_new - v), np.linalg.norm(v_new + v)) < tol:
            v = v_new
            break
        v = v_new
    return float(v @ A @ v), v


def woodbury(
    A: np.ndarray, U: np.ndarray, C: np.ndarray, V: np.ndarray
) -> np.ndarray:
    """(A + U C V)^{-1} without inverting the big matrix:

    (A + UCV)^{-1} = A^{-1} - A^{-1}U (C^{-1} + V A^{-1} U)^{-1} V A^{-1}

    The inner inverse is k×k. This is why low-rank updates (LoRA, Kalman updates,
    Sherman-Morrison) are cheap.
    """
    Ainv = np.linalg.inv(A)
    k = C.shape[0]
    inner = np.linalg.inv(np.linalg.inv(C) + V @ Ainv @ U) if k else np.zeros((0, 0))
    return Ainv - Ainv @ U @ inner @ V @ Ainv


def solve_triangular(L: np.ndarray, b: np.ndarray, lower: bool = True) -> np.ndarray:
    """Forward/back substitution: O(n²) instead of the O(n³) of a generic solve."""
    L = np.asarray(L, dtype=float)
    b = np.asarray(b, dtype=float)
    n = L.shape[0]
    x = np.zeros_like(b)
    idx = range(n) if lower else range(n - 1, -1, -1)
    for i in idx:
        s = b[i] - (L[i, :i] @ x[:i] if lower else L[i, i + 1 :] @ x[i + 1 :])
        x[i] = s / L[i, i]
    return x


def condition_number(A: np.ndarray, p: float = 2.0) -> float:
    """κ(A) = ‖A‖‖A⁻¹‖: how many digits a solve can cost you."""
    return float(np.linalg.cond(A, p=p))


def main() -> None:
    rng = np.random.default_rng(0)
    A = rng.normal(size=(7, 7))

    # 1. PA = LU and the factors have the right shape.
    P, L, U = lu(A)
    assert np.allclose(P.T @ L @ U, A)
    assert np.allclose(np.tril(L), L) and np.allclose(np.triu(U), U)
    assert np.allclose(np.diag(L), 1.0)
    print("lu: PA = LU")

    # 2. A = QR with an orthonormal Q.
    Q, R = qr(A)
    assert np.allclose(Q @ R, A)
    assert np.allclose(Q.T @ Q, np.eye(7), atol=1e-12)
    assert np.allclose(np.tril(R, -1), 0.0)
    print("qr: A = QR, Q^T Q = I")

    # 3. Cholesky on an SPD matrix, and it rejects a non-PD one.
    S = A @ A.T + 7 * np.eye(7)
    Lc = cholesky(S)
    assert np.allclose(Lc @ Lc.T, S)
    try:
        cholesky(A + A.T)
    except ValueError:
        pass
    else:
        raise AssertionError("cholesky should reject an indefinite matrix")
    print("cholesky: A = LL^T, rejects non-PD")

    # 4. Rank-k truncation is optimal and the randomised version nearly matches it.
    low = rng.normal(size=(60, 5)) @ rng.normal(size=(5, 40))  # rank 5
    U1, S1, V1 = truncated_svd(low, 5)
    assert np.allclose((U1 * S1) @ V1, low, atol=1e-10)
    Ur, Sr, Vr = randomized_svd(low, 5, rng=rng)
    err_exact = np.linalg.norm(low - (U1 * S1) @ V1)
    err_rand = np.linalg.norm(low - (Ur * Sr) @ Vr)
    assert err_rand < 1e-6 * max(np.linalg.norm(low), 1.0), (err_exact, err_rand)
    print(f"svd: exact rank-5 error {err_exact:.1e}, randomised {err_rand:.1e}")

    # 5. Power iteration finds the dominant eigenpair.
    M = rng.normal(size=(30, 30))
    M = M @ M.T
    lam, v = power_iteration(M)
    assert np.allclose(M @ v, lam * v, atol=1e-6)
    assert abs(lam - np.linalg.eigvalsh(M).max()) < 1e-6
    print(f"power iteration: lambda = {lam:.4f}")

    # 6. Woodbury equals the direct inverse, at a fraction of the dimension.
    n, k = 200, 3
    A0 = np.eye(n) * 2.0
    U0 = rng.normal(size=(n, k))
    V0 = rng.normal(size=(k, n))
    C0 = np.eye(k)
    direct = np.linalg.inv(A0 + U0 @ C0 @ V0)
    assert np.allclose(woodbury(A0, U0, C0, V0), direct, atol=1e-10)
    print("woodbury: matches direct inverse for a rank-3 update")

    # 7. Triangular solve matches the generic one.
    b = rng.normal(size=7)
    assert np.allclose(solve_triangular(Lc, b), np.linalg.solve(Lc, b))
    assert np.allclose(solve_triangular(Lc.T, b, lower=False), np.linalg.solve(Lc.T, b))
    print("triangular solve ok")

    # 8. Conditioning: the Hilbert matrix is the canonical ill-conditioned beast.
    H = 1.0 / (np.arange(1, 13)[:, None] + np.arange(1, 13)[None, :] - 1.0)
    assert condition_number(np.eye(12)) == 1.0
    assert condition_number(H) > 1e10
    print(f"condition number: I = 1, Hilbert(12) = {condition_number(H):.1e}")
    print("linalg demo passed")


if __name__ == "__main__":
    main()
