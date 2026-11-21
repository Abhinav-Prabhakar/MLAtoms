"""MMD and HSIC — kernel two-sample and independence tests.

Chapter:   03-kernels-and-instance
Companion: mmd.md
Tier 3 · 5 variants: quadratic, linear_time, mmd_rbf, hsic, permutation_test
"""

from __future__ import annotations

from functools import partial

import numpy as np

VARIANTS = ("quadratic", "linear_time", "mmd_rbf", "hsic", "permutation_test")
KERNELS = ("linear", "rbf", "laplacian")
EPS = 1e-300


# --------------------------------------------------------------------------- #
# Kernel blocks and the bandwidth
# --------------------------------------------------------------------------- #
def _sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances, shape `(len(A), len(B))`."""
    aa = np.einsum("ij,ij->i", A, A)[:, None]
    bb = np.einsum("ij,ij->i", B, B)[None, :]
    return np.maximum(aa + bb - 2.0 * (A @ B.T), 0.0)


def _gram(A: np.ndarray, B: np.ndarray, kernel: str, gamma: float | None
          ) -> np.ndarray:
    """Kernel block `K[i, j] = k(A[i], B[j])`, shape `(len(A), len(B))`.

    `gamma` is ignored by the linear kernel and required by the other two.
    """
    if kernel == "linear":
        return A @ B.T
    if gamma is None:
        raise ValueError("gamma is required for a non-linear kernel")
    if kernel == "rbf":
        return np.exp(-gamma * _sqdist(A, B))
    distances = np.abs(A[:, None, :] - B[None, :, :]).sum(axis=2)
    return np.exp(-gamma * distances)


def _median_gamma(A: np.ndarray, B: np.ndarray) -> float:
    """Bandwidth `1 / (2 · median distance)` from the pooled sample.

    The factor two puts the kernel value at the median distance at
    `exp(-1/2) = 0.607`, halfway between a step function and a straight line.
    """
    pooled = np.vstack([A, B])
    squared = _sqdist(pooled, pooled)
    upper = squared[np.triu_indices(pooled.shape[0], k=1)]
    positive = upper[upper > 0.0]
    if positive.size == 0:
        raise ValueError("all pooled points coincide; no bandwidth exists")
    return 1.0 / (2.0 * float(np.median(positive)))


def _resolve_gamma(A: np.ndarray, B: np.ndarray, kernel: str,
                   gamma: float | None) -> float | None:
    """Validate an explicit bandwidth or fall back to the median heuristic."""
    if kernel == "linear":
        return None
    if gamma is None:
        return _median_gamma(A, B)
    gamma = float(gamma)
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("gamma must be positive and finite")
    return gamma


# --------------------------------------------------------------------------- #
# The quadratic and linear-time estimators
# --------------------------------------------------------------------------- #
def _as_sample(A, name: str) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if A.shape[1] == 0:
        raise ValueError(f"{name} must have at least one column")
    if not np.all(np.isfinite(A)):
        raise ValueError(f"{name} must be finite")
    return A


def mmd2(X: np.ndarray, Y: np.ndarray, *, kernel: str = "rbf",
         gamma: float | None = None, unbiased: bool = False) -> float:
    """Squared maximum mean discrepancy between two samples.

    `X` is `(m, d)` and `Y` is `(n, d)`; the result is a non-negative float for
    the V-statistic and may be slightly negative for the U-statistic. With a
    linear kernel it equals `‖mean(X) − mean(Y)‖²` exactly.
    """
    X, Y = _as_sample(X, "X"), _as_sample(Y, "Y")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same width")
    if kernel not in KERNELS:
        raise ValueError(f"unknown kernel {kernel!r}")
    gamma = _resolve_gamma(X, Y, kernel, gamma)
    kxx = _gram(X, X, kernel, gamma)
    kyy = _gram(Y, Y, kernel, gamma)
    kxy = _gram(X, Y, kernel, gamma)
    m, n = X.shape[0], Y.shape[0]
    if not unbiased:
        return float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean())
    if m < 2 or n < 2:
        raise ValueError("the unbiased estimator needs two rows per sample")
    within_x = (kxx.sum() - np.trace(kxx)) / (m * (m - 1))
    within_y = (kyy.sum() - np.trace(kyy)) / (n * (n - 1))
    return float(within_x + within_y - 2.0 * kxy.mean())


def _block_size(m: int, n: int, block: int | None) -> int:
    if block is None:
        return max(2, int(np.sqrt(min(m, n))))
    block = int(block)
    if block < 2:
        raise ValueError("block must be at least 2")
    return block


def linear_time_mmd2(X: np.ndarray, Y: np.ndarray, *, kernel: str = "rbf",
                     gamma: float | None = None,
                     block: int | None = None) -> float:
    """Average of the unbiased estimate over contiguous blocks.

    Cost is `O(n · block)` instead of `O(n²)`; the default block is `⌊√n⌋`,
    which is the bias/variance tradeoff from Gretton et al. (2012), giving
    `O(n^1.5)`. A fixed block size gives a strictly linear estimator.
    """
    X, Y = _as_sample(X, "X"), _as_sample(Y, "Y")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same width")
    if kernel not in KERNELS:
        raise ValueError(f"unknown kernel {kernel!r}")
    gamma = _resolve_gamma(X, Y, kernel, gamma)
    m, n = X.shape[0], Y.shape[0]
    size = _block_size(m, n, block)
    count = min(m, n) // size
    if count < 1:
        raise ValueError("samples are shorter than one block")
    total = 0.0
    for start in range(0, count * size, size):
        xb = X[start:start + size]
        yb = Y[start:start + size]
        total += mmd2(xb, yb, kernel=kernel, gamma=gamma, unbiased=True)
    return float(total / count)


# --------------------------------------------------------------------------- #
# HSIC
# --------------------------------------------------------------------------- #
def _centre(K: np.ndarray) -> np.ndarray:
    """`HKH` for `H = I − 11ᵀ/n`, without forming `H`."""
    return (K - K.mean(axis=0, keepdims=True) - K.mean(axis=1, keepdims=True)
            + K.mean())


def hsic(X: np.ndarray, Y: np.ndarray, *, kernel: str = "rbf",
         gamma: float | None = None, gamma_y: float | None = None,
         normalize: bool = True) -> float:
    """Hilbert–Schmidt independence criterion for paired samples.

    `X` and `Y` are `(n, d)` and `(n, e)` with the same `n` rows. Returns the
    biased HSIC `tr(KHLH)/n²`, or its cosine normalisation in `[0, 1]`.
    """
    X, Y = _as_sample(X, "X"), _as_sample(Y, "Y")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("HSIC needs paired samples of equal length")
    if kernel not in KERNELS:
        raise ValueError(f"unknown kernel {kernel!r}")
    gx = _resolve_gamma(X, X, kernel, gamma)
    gy = _resolve_gamma(Y, Y, kernel, gamma_y)
    kc = _centre(_gram(X, X, kernel, gx))
    lc = _centre(_gram(Y, Y, kernel, gy))
    value = float(np.mean(kc * lc))
    if not normalize:
        return value
    denominator = float(np.sqrt(np.mean(kc * kc) * np.mean(lc * lc)))
    return value / denominator if denominator > EPS else 0.0


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def permutation_test(statistic, first: np.ndarray, second: np.ndarray, *,
                     mode: str = "pool", n_perm: int = 200,
                     seed: int = 0) -> dict:
    """Null distribution of `statistic(first, second)` by permutation.

    `mode="pool"` pools both samples and re-splits at the original sizes, which
    is the two-sample null. `mode="rows"` permutes the rows of `second` against
    a fixed `first`, which is the paired/independence null. The p-value is
    `(1 + #{null ≥ observed}) / (1 + n_perm)`, so it is never zero.
    """
    if mode not in ("pool", "rows"):
        raise ValueError("mode must be 'pool' or 'rows'")
    if n_perm < 1:
        raise ValueError("n_perm must be at least 1")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    first = _as_sample(first, "first")
    second = _as_sample(second, "second")
    if first.shape[1] != second.shape[1]:
        raise ValueError("first and second must have the same width")
    observed = float(statistic(first, second))
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    if mode == "pool":
        pooled = np.vstack([first, second])
        cut = first.shape[0]
        for i in range(n_perm):
            order = rng.permutation(pooled.shape[0])
            null[i] = statistic(pooled[order[:cut]], pooled[order[cut:]])
    else:
        if first.shape[0] != second.shape[0]:
            raise ValueError("mode='rows' needs samples of equal length")
        for i in range(n_perm):
            null[i] = statistic(first, second[rng.permutation(second.shape[0])])
    exceed = int(np.sum(null >= observed))
    return {"statistic": observed, "null": null,
            "p_value": (1.0 + exceed) / (1.0 + n_perm), "n_perm": n_perm}


def witness(X: np.ndarray, Y: np.ndarray, *, kernel: str = "rbf",
            gamma: float | None = None) -> np.ndarray:
    """Witness function on the pooled sample, shape `(m + n,)`.

    `f(z) = mean_i k(X_i, z) − mean_j k(Y_j, z)`; it is the function whose
    Hilbert-space norm is the MMD. The difference of its class means is the
    biased statistic exactly.
    """
    X, Y = _as_sample(X, "X"), _as_sample(Y, "Y")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same width")
    gamma = _resolve_gamma(X, Y, kernel, gamma)
    pooled = np.vstack([X, Y])
    return (_gram(pooled, X, kernel, gamma).mean(axis=1)
            - _gram(pooled, Y, kernel, gamma).mean(axis=1))


# --------------------------------------------------------------------------- #
# Variant router
# --------------------------------------------------------------------------- #
def mmd(X: np.ndarray, Y: np.ndarray, *, variant: str = "quadratic",
        kernel: str = "rbf", gamma: float | None = None,
        unbiased: bool = False, block: int | None = None,
        normalize: bool = True, n_perm: int = 0, seed: int = 0) -> dict:
    """Run one variant of the two-sample or independence test.

    Returns a dict with `variant`, `statistic`, `kernel`, `gamma`, `gamma_y`,
    `block`, `witness`, `p_value` and `null`; keys a variant does not use are
    `None`. The bandwidth is resolved once from the observed data and then held
    fixed across permutations — recomputing it inside the null would make each
    permutation use a different kernel.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    if kernel not in KERNELS:
        raise ValueError(f"unknown kernel {kernel!r}")
    X, Y = _as_sample(X, "X"), _as_sample(Y, "Y")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same width")
    if variant == "mmd_rbf" and kernel != "rbf":
        raise ValueError("mmd_rbf is the RBF instantiation; pass kernel='rbf'")
    if variant == "hsic" and X.shape[0] != Y.shape[0]:
        raise ValueError("the independence test needs paired samples")
    if n_perm < 0:
        raise ValueError("n_perm must be non-negative")
    if variant == "permutation_test" and n_perm < 1:
        raise ValueError("this variant exists to calibrate; set n_perm >= 1")

    if variant == "hsic":
        gamma_x = _resolve_gamma(X, X, kernel, gamma)
        gamma_y = _resolve_gamma(Y, Y, kernel, gamma)
        statistic = partial(hsic, kernel=kernel, gamma=gamma_x,
                            gamma_y=gamma_y, normalize=normalize)
        mode, gamma_value = "rows", gamma_x
    else:
        gamma_value = _resolve_gamma(X, Y, kernel, gamma)
        gamma_y = None
        if variant == "linear_time":
            statistic = partial(linear_time_mmd2, kernel=kernel,
                                gamma=gamma_value, block=block)
        else:
            statistic = partial(mmd2, kernel=kernel, gamma=gamma_value,
                                unbiased=unbiased)
        mode = "pool"

    result = {"variant": variant, "kernel": kernel, "gamma": gamma_value,
              "gamma_y": gamma_y, "witness": None,
              "block": None, "p_value": None, "null": None}
    if variant == "linear_time":
        result["block"] = _block_size(X.shape[0], Y.shape[0], block)
    if variant == "mmd_rbf":
        result["witness"] = witness(X, Y, kernel=kernel, gamma=gamma_value)
    if n_perm > 0:
        tested = permutation_test(statistic, X, Y, mode=mode, n_perm=n_perm,
                                  seed=seed)
        result.update(tested)
        result["variant"] = variant
    else:
        result["statistic"] = float(statistic(X, Y))
    return result


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def demo() -> None:
    rng = np.random.default_rng(0)
    same_a = rng.normal(0.0, 1.0, (80, 2))
    same_b = rng.normal(0.0, 1.0, (80, 2))
    shifted = rng.normal(0.0, 1.0, (80, 2)) + np.array([0.8, 0.0])

    linear = mmd2(same_a, shifted, kernel="linear")
    closed = float(np.sum((same_a.mean(axis=0) - shifted.mean(axis=0)) ** 2))
    print("a linear kernel has a closed form: MMD^2 is the squared distance")
    print(f"between the means, {linear:.6f} against {closed:.6f} from the")
    print("two sample means")
    assert np.isclose(linear, closed, rtol=1e-12, atol=1e-12)
    assert abs(mmd2(same_a, same_a, kernel="linear")) < 1e-12

    assert abs(mmd2(same_a, same_a)) < 1e-12
    print("the biased estimator is exactly zero when the two samples are the")
    print("same rows, but not when they are independent draws from one")
    print("distribution: the diagonal terms bias it up by (1/m + 1/n)(1 - kappa)")
    replicates = 200
    biased, unbiased = [], []
    for _ in range(replicates):
        left = rng.normal(0.0, 1.0, (40, 2))
        right = rng.normal(0.0, 1.0, (40, 2))
        biased.append(mmd2(left, right))
        unbiased.append(mmd2(left, right, unbiased=True))
    biased_mean = float(np.mean(biased))
    unbiased_mean = float(np.mean(unbiased))
    unbiased_se = float(np.std(unbiased, ddof=1) / np.sqrt(replicates))
    print(f"  mean biased    {biased_mean:+.5f}  at 2/m = 0.05 times 1 - kappa")
    print(f"  mean unbiased  {unbiased_mean:+.5f} ± {unbiased_se:.5f}  centred")
    assert biased_mean > 5.0 * unbiased_se
    assert abs(unbiased_mean) < 4.0 * unbiased_se

    null_run = mmd(same_a, same_b, n_perm=200, seed=1)
    shift_run = mmd(same_a, shifted, n_perm=200, seed=1)
    print("permuting the pooled sample gives a null for the same statistic:")
    print(f"  same distribution   MMD^2 {null_run['statistic']:.5f}  "
          f"p {null_run['p_value']:.3f}")
    print(f"  mean shifted by 0.8 MMD^2 {shift_run['statistic']:.5f}  "
          f"p {shift_run['p_value']:.3f}")
    assert null_run["p_value"] > 0.05
    assert shift_run["p_value"] < 0.01
    assert shift_run["statistic"] > null_run["statistic"]
    for run in (null_run, shift_run):
        expected = (1.0 + np.sum(run["null"] >= run["statistic"])) / 201.0
        assert np.isclose(run["p_value"], expected)

    print("with the bandwidth fixed, blocking trades variance for a factor of")
    print("sqrt(n) in cost; the gap to the quadratic estimate closes with n:")
    pilot = rng.normal(0.0, 1.0, (200, 2))
    gamma = _median_gamma(pilot, pilot + np.array([0.5, 0.0]))
    gaps = []
    for size in (160, 640, 2560):
        left = rng.normal(0.0, 1.0, (size, 2))
        right = rng.normal(0.0, 1.0, (size, 2)) + np.array([0.5, 0.0])
        slow = mmd2(left, right, gamma=gamma, unbiased=True)
        fast = linear_time_mmd2(left, right, gamma=gamma)
        run = mmd(left, right, variant="linear_time", gamma=gamma)
        gaps.append(abs(fast - slow))
        print(f"  n {size:4d}  quadratic {slow:+.5f}  blocked {fast:+.5f}  "
              f"block {run['block']:2d}  gap {gaps[-1]:.5f}")
        assert run["block"] == max(2, int(np.sqrt(size)))
    assert all(a > b for a, b in zip(gaps[:-1], gaps[1:], strict=True))
    assert gaps[-1] < 0.005
    print("  O(n * block) per estimate: sqrt(n) blocks of sqrt(n) points")

    print("the median heuristic puts the kernel value at the median distance")
    print("at exp(-1/2), independent of the data scale:")
    spread = rng.normal(0.0, 5.0, (60, 3))
    run = mmd(spread, spread + 1.0, variant="mmd_rbf", n_perm=100, seed=2)
    pooled = np.vstack([spread, spread + 1.0])
    distances = _sqdist(pooled, pooled)[np.triu_indices(120, k=1)]
    median = float(np.median(distances[distances > 0.0]))
    assert np.isclose(np.exp(-run["gamma"] * median), np.exp(-0.5))
    print(f"  gamma {run['gamma']:.6f}  median d^2 {median:.4f}  "
          f"k(median) {np.exp(-run['gamma'] * median):.4f}")

    w = run["witness"]
    gap = float(w[:60].mean() - w[60:].mean())
    print(f"the witness gap equals the statistic: {gap:.6f} against "
          f"{run['statistic']:.6f}")
    assert np.isclose(gap, run["statistic"], atol=1e-12)
    assert w.shape == (120,)

    print("HSIC asks whether two variables are independent, not whether their")
    print("distributions match:")
    x = rng.normal(0.0, 1.0, (150, 1))
    noise = rng.normal(0.0, 1.0, (150, 1))
    dependent = x ** 2 + 0.2 * noise
    free = mmd(x, noise, variant="hsic", n_perm=200, seed=3)
    tied = mmd(x, dependent, variant="hsic", n_perm=200, seed=3)
    print(f"  independent   nHSIC {free['statistic']:.5f}  "
          f"p {free['p_value']:.3f}")
    print(f"  y = x^2 + noise nHSIC {tied['statistic']:.5f}  "
          f"p {tied['p_value']:.3f}")
    assert free["p_value"] > 0.2
    assert tied["p_value"] < 0.01
    assert tied["statistic"] > free["statistic"]
    assert 0.0 <= free["statistic"] <= 1.0
    assert np.isclose(hsic(x, x, gamma=free["gamma"],
                           gamma_y=free["gamma"]), 1.0)
    manual = np.mean(_centre(_gram(x, x, "rbf", _median_gamma(x, x)))
                     * _centre(_gram(noise, noise, "rbf",
                                     _median_gamma(noise, noise))))
    assert np.isclose(hsic(x, noise, normalize=False), manual)

    raw = mmd2(same_a, shifted, unbiased=True)
    held = mmd(same_a, shifted, variant="permutation_test", unbiased=True,
               n_perm=300, seed=4)
    print(f"the permutation variant reports its null of {held['n_perm']} draws,")
    print(f"observed {held['statistic']:.6f} against {raw:.6f} for the raw call")
    assert np.isclose(held["statistic"], raw)

    bad = [
        lambda: mmd(same_a, shifted, variant="energy"),
        lambda: mmd(same_a, shifted, kernel="cosine"),
        lambda: mmd(same_a, shifted, variant="mmd_rbf", kernel="linear"),
        lambda: mmd(same_a, same_b[:, :1]),
        lambda: mmd(same_a, shifted[:40], variant="hsic"),
        lambda: mmd(same_a, shifted, variant="permutation_test"),
        lambda: mmd(same_a, shifted, gamma=0.0),
        lambda: mmd(same_a, shifted, n_perm=-1),
        lambda: mmd(same_a, shifted, n_perm=10, seed=-1),
        lambda: mmd(same_a.ravel(), shifted),
        lambda: mmd(np.full((10, 2), np.nan), shifted),
        lambda: mmd2(same_a[:1], shifted[:1], unbiased=True),
        lambda: mmd2(same_a, shifted, kernel="poly"),
        lambda: linear_time_mmd2(same_a[:3], shifted[:3], block=8),
        lambda: linear_time_mmd2(same_a, shifted, block=1),
        lambda: permutation_test(mmd2, same_a, shifted, mode="sideways"),
        lambda: permutation_test(mmd2, same_a, shifted, n_perm=0),
        lambda: permutation_test(mmd2, same_a, shifted[:40], mode="rows"),
        lambda: hsic(same_a, same_b[:40]),
        lambda: hsic(same_a, shifted, kernel="poly"),
        lambda: _median_gamma(np.zeros((4, 2)), np.zeros((4, 2))),
        lambda: _gram(same_a, shifted, "rbf", None),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except ValueError:
            refused += 1
    print(f"{refused} malformed calls refused out of {len(bad)}: an unknown")
    print("  variant or kernel, an RBF-only variant with another kernel, a")
    print("  width mismatch, an unpaired independence test, a calibration")
    print("  variant with no permutations, a zero bandwidth, a negative")
    print("  permutation count or seed, a 1-d sample, a non-finite entry, an")
    print("  unbiased estimate on one row, a block larger than the sample, a")
    print("  block of one, an unknown permutation mode, unequal paired rows,")
    print("  a constant pooled sample, and a missing kernel bandwidth")
    print("mmd demo passed")


if __name__ == "__main__":
    demo()
