"""Copulas — dependence on its own, with the margins stripped off.

Chapter:   05-probabilistic
Companion: copula.md
Tier 3 · 6 variants: gaussian, t, clayton, gumbel, frank, vine

    F(x₁,…,x_d) = C(F₁(x₁),…,F_d(x_d))

Sklar's theorem: every joint distribution factorises into its margins and a
copula, and the copula is unique on the support of the margins. The copula is
the dependence, and nothing else.

    C(u₁,…,u_d) = P(U₁ ≤ u₁, …, U_d ≤ u_d),   U_j = F_j(X_j) ∈ (0,1)

Uniform margins by construction, so the copula carries no location, no scale,
no skew — only how the coordinates move together. Two consequences:

    margins      free: any F_j, plugged in afterwards, or fitted separately
    dependence   one object C, and the variants are different C

The variants are chosen for what they say about the *tails*:

    gaussian     no tail dependence at all, for any |ρ| < 1
    t            symmetric tail dependence, controlled by df and ρ
    clayton      lower-tail dependence only: crashes together, rallies apart
    gumbel       upper-tail dependence only: rallies together, crashes apart
    frank        neither tail: dependence concentrated in the body
    vine         a product of pair copulas, so the tails can differ per pair

The demo is built on exact statements rather than on plausibility: the
grounding axioms C(u,1)=u and C(u,0)=0, the density as the mixed derivative of
the CDF, the density integrating to one, uniform margins by Kolmogorov–Smirnov,
Kendall's tau three ways (analytic, quadrature, empirical), the recovered
parameter from a rank-transformed fit, and the tail-dependence limits against
their closed forms.

Everything is numpy and `math`: the normal CDF comes from `math.erfc`, the
Student-t CDF from a continued fraction for the incomplete beta, and the
gumbel/clayton samplers from the Marshall–Olkin frailty representations.
"""

from __future__ import annotations

import math

import numpy as np

VARIANTS = frozenset(["gaussian", "t", "clayton", "gumbel", "frank", "vine"])

LOG2PI = math.log(2.0 * math.pi)
SQRT2 = math.sqrt(2.0)
EPS = 1e-12
DEFAULT_PARAM = {"gaussian": 0.5, "clayton": 1.0, "gumbel": 1.5, "frank": 2.0}


# --------------------------------------------------------------------------- #
# Numerics: Φ, Φ⁻¹, the Student-t CDF, and two samplers
# --------------------------------------------------------------------------- #
def _Phi(x) -> np.ndarray:
    """Standard normal CDF, through libm's correctly-rounded `erfc`."""
    x = np.asarray(x, dtype=float)
    out = np.empty(x.shape)
    flat, res = x.ravel(), out.ravel()
    for i, v in enumerate(flat):
        res[i] = 0.5 * math.erfc(-v / SQRT2)
    return out


_A = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
      1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
_B = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
      6.680131188771972e01, -1.328068155288572e01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
      -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
      3.754408661907416e00)


def _Phi_inv(u) -> np.ndarray:
    """Acklam's rational approximation, then two Newton steps on `_Phi`."""
    # the ends of the interval are clipped: Φ⁻¹(0) is −∞ and the rational form
    # evaluates it as 0·∞, so no representation is lost by stopping at 1e−300
    u = np.clip(np.asarray(u, dtype=float), 1e-300, 1.0 - 1e-16)
    x = np.zeros_like(u)
    lo, hi = u < 0.02425, u > 0.97575
    mid = ~(lo | hi)
    if lo.any():
        q = np.sqrt(-2.0 * np.log(u[lo]))
        x[lo] = (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
                ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)
    if hi.any():
        q = np.sqrt(-2.0 * np.log(1.0 - u[hi]))
        x[hi] = -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
                 ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)
    if mid.any():
        q = u[mid] - 0.5
        r = q * q
        x[mid] = (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q / \
                 (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0)
    for _ in range(2):                     # 1e-9 is not enough for a sampler
        err = np.clip(_Phi(x) - u, -0.5, 0.5)
        x = x - err / np.maximum(np.exp(-0.5 * x * x - 0.5 * LOG2PI), 1e-300)
    return x


def _betacf(a: float, b: float, x: np.ndarray) -> np.ndarray:
    """Continued fraction for the incomplete beta, by modified Lentz."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    tiny = 1e-300
    c = np.ones_like(x)
    d = 1.0 / np.where(np.abs(1.0 - qab * x / qap) < tiny, tiny, 1.0 - qab * x / qap)
    h = d.copy()
    for m in range(1, 400):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)),
                   -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1.0 + aa * d
            d = 1.0 / np.where(np.abs(d) < tiny, tiny, d)
            c = 1.0 + aa / c
            c = np.where(np.abs(c) < tiny, tiny, c)
            h = h * d * c
        if np.all(np.abs(d * c - 1.0) < 1e-15):
            break
    return h


def _betainc(a: float, b: float, x: np.ndarray) -> np.ndarray:
    """Regularised incomplete beta `I_x(a,b)`, one branch per half."""
    x = np.clip(np.asarray(x, dtype=float), 1e-300, 1.0 - 1e-16)
    front = np.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                   + a * np.log(x) + b * np.log1p(-x))
    out = np.empty_like(x)
    small = x < (a + 1.0) / (a + b + 2.0)
    if np.any(small):
        out[small] = front[small] * _betacf(a, b, x[small]) / a
    if np.any(~small):
        out[~small] = 1.0 - front[~small] * _betacf(b, a, 1.0 - x[~small]) / b
    return out


def _t_cdf(x, df: float) -> np.ndarray:
    """`P(T ≤ x)` for a Student-t, from the symmetric incomplete beta."""
    x = np.asarray(x, dtype=float)
    p = 0.5 * _betainc(0.5 * df, 0.5, df / (df + x * x))
    return np.where(x > 0.0, 1.0 - p, p)


def _t_logpdf(x, df: float) -> np.ndarray:
    return (math.lgamma(0.5 * (df + 1.0)) - math.lgamma(0.5 * df)
            - 0.5 * math.log(df * math.pi) - 0.5 * (df + 1.0) * np.log1p(x * x / df))


def _t_ppf(u, df: float) -> np.ndarray:
    """Newton on `_t_cdf`, started from the better of two guesses.

    In the body the normal quantile is within 30% and Newton fixes it. In the
    tail it is hopeless once `df` is small, so the start switches to the t's own
    asymptotic, `P(T > x) ~ K x^{−ν}`, which is exact to leading order.
    """
    u = np.clip(np.asarray(u, dtype=float), 1e-15, 1.0 - 1e-15)
    log_k = (math.lgamma(0.5 * (df + 1.0)) - math.lgamma(0.5 * df)
             - 0.5 * math.log(df * math.pi) + 0.5 * (df - 1.0) * math.log(df))
    tail = np.exp((log_k - np.log(np.minimum(u, 1.0 - u))) / df)
    x = np.where(u < 0.01, -tail, np.where(u > 0.99, tail, _Phi_inv(u)))
    # the CDF carries about 1e-13 of absolute noise, so the steps stall there;
    # 1e-11 in `x` is far below anything a density can feel
    for _ in range(40):
        step = np.clip((_t_cdf(x, df) - u) / np.exp(_t_logpdf(x, df)), -5.0, 5.0)
        x = x - step
        if np.max(np.abs(step)) < 1e-11:
            break
    return x


def _gamma(rng, shape: float, size: int) -> np.ndarray:
    """Marsaglia–Tsang, with the `shape < 1` boost. `np.random` has no gamma."""
    shape = float(shape)
    if shape < 1.0:
        return _gamma(rng, shape + 1.0, size) * rng.random(size) ** (1.0 / shape)
    d = shape - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)
    out = np.empty(size)
    todo = np.ones(size, dtype=bool)
    while todo.any():
        x = rng.standard_normal(int(todo.sum()))
        v = (1.0 + c * x) ** 3
        ok = v > 0.0
        u = rng.random(x.size)
        acc = ok & (np.log(np.maximum(u, 1e-300))
                    < 0.5 * x * x + d - d * v + d * np.log(np.where(ok, v, 1.0)))
        idx = np.flatnonzero(todo)[acc]
        out[idx] = d * v[acc]
        todo[idx] = False
    return out


def _stable(rng, alpha: float, size: int) -> np.ndarray:
    """Kanter's positive α-stable with Laplace transform `exp(-s^α)`."""
    phi = rng.uniform(0.0, math.pi, size)
    w = rng.exponential(1.0, size)
    return (np.sin(alpha * phi) / np.sin(phi) ** (1.0 / alpha)
            * (np.sin((1.0 - alpha) * phi) / w) ** ((1.0 - alpha) / alpha))


def _bvn(h, k, rho: float, nodes: int = 200) -> np.ndarray:
    """Bivariate normal CDF, Drezner's integral, 200-point Gauss–Legendre.

        Φ₂(h,k;ρ) = Φ(h)Φ(k) + (1/2π)∫₀^{asin ρ} exp(−(h²−2hk sinθ+k²)/2cos²θ) dθ
    """
    h = np.atleast_1d(np.asarray(h, dtype=float)).ravel()
    k = np.atleast_1d(np.asarray(k, dtype=float)).ravel()
    if rho == 0.0:
        return _Phi(h) * _Phi(k)
    a = math.asin(min(max(rho, -0.999999), 0.999999))
    x, w = np.polynomial.legendre.leggauss(nodes)
    th = 0.5 * a * (x + 1.0)
    c2 = np.cos(th) ** 2
    q = (np.outer(h * h, np.ones_like(th)) - 2.0 * np.outer(h * k, np.sin(th))
         + np.outer(k * k, np.ones_like(th)))
    return _Phi(h) * _Phi(k) + (np.exp(-q / (2.0 * c2)) @ (0.5 * a * w)) / (2.0 * math.pi)


# --------------------------------------------------------------------------- #
# The h-functions: h(u|v) = ∂C(u,v)/∂v, and their inverses
# --------------------------------------------------------------------------- #
def _h(u, v, family: str, param: float) -> np.ndarray:
    """`∂C(u,v)/∂v` — the conditional CDF of `u` given `v`."""
    u, v = np.clip(u, EPS, 1.0 - EPS), np.clip(v, EPS, 1.0 - EPS)
    if family == "gaussian":
        return _Phi((_Phi_inv(u) - param * _Phi_inv(v)) / math.sqrt(1.0 - param * param))
    if family == "clayton":
        return v ** (-param - 1.0) * (u ** (-param) + v ** (-param) - 1.0) ** (-(1.0 + param) / param)
    if family == "gumbel":
        a, b = -np.log(u), -np.log(v)
        s = (a ** param + b ** param) ** (1.0 / param)
        return np.exp(-s) * s ** (1.0 - param) * b ** (param - 1.0) / v
    if family == "frank":
        a = np.exp(-param * u) - 1.0
        b = np.exp(-param * v) - 1.0
        return a * np.exp(-param * v) / ((math.exp(-param) - 1.0) + a * b)
    raise ValueError(f"no h-function for {family!r}")


def _h_inv(w, v, family: str, param: float) -> np.ndarray:
    """Inverse of `_h` in its first argument: `h(u|v) = w` solved for `u`."""
    w = np.clip(w, EPS, 1.0 - EPS)
    if family == "gaussian":
        return _Phi(_Phi_inv(w) * math.sqrt(1.0 - param * param) + param * _Phi_inv(v))
    if family == "clayton":
        a = (w * np.clip(v, EPS, 1.0) ** (param + 1.0)) ** (-param / (1.0 + param))
        return np.clip(a - np.clip(v, EPS, 1.0) ** (-param) + 1.0, EPS, None) ** (-1.0 / param)
    if family == "frank":
        b = np.exp(-param * np.clip(v, EPS, 1.0)) - 1.0
        s = w * (math.exp(-param) - 1.0) / (np.exp(-param * np.clip(v, EPS, 1.0)) - w * b)
        return np.clip(-np.log1p(s) / param, EPS, 1.0 - EPS)
    if family == "gumbel":
        lo = np.full_like(w, EPS)                      # h is increasing in u
        hi = np.full_like(w, 1.0 - EPS)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            up = _h(mid, v, family, param) < w
            lo, hi = np.where(up, mid, lo), np.where(up, hi, mid)
        return 0.5 * (lo + hi)
    raise ValueError(f"no inverse h-function for {family!r}")


def _pair_logpdf(u, v, family: str, param: float) -> np.ndarray:
    """`log c(u,v)` for one bivariate pair copula."""
    u, v = np.clip(u, EPS, 1.0 - EPS), np.clip(v, EPS, 1.0 - EPS)
    if family == "gaussian":
        z1, z2 = _Phi_inv(u), _Phi_inv(v)
        r = param
        return (-0.5 * math.log(1.0 - r * r)
                - (z1 * z1 - 2.0 * r * z1 * z2 + z2 * z2) / (2.0 * (1.0 - r * r))
                + 0.5 * (z1 * z1 + z2 * z2))
    if family == "clayton":
        return (math.log(1.0 + param) - (param + 1.0) * (np.log(u) + np.log(v))
                - (2.0 + 1.0 / param) * np.log(u ** (-param) + v ** (-param) - 1.0))
    if family == "gumbel":
        a, b = -np.log(u), -np.log(v)
        s = (a ** param + b ** param) ** (1.0 / param)
        return (-s - np.log(u) - np.log(v) + (param - 1.0) * (np.log(a) + np.log(b))
                + (1.0 / param - 2.0) * np.log(a ** param + b ** param) + np.log(s + param - 1.0))
    if family == "frank":
        b = math.exp(-param) - 1.0
        g = (np.exp(-param * u) - 1.0) * (np.exp(-param * v) - 1.0) / b
        return math.log(-param / b) - param * (u + v) - 2.0 * np.log1p(g)
    raise ValueError(f"no pair density for {family!r}")


# --------------------------------------------------------------------------- #
# Ranks and Kendall's tau
# --------------------------------------------------------------------------- #
def pseudo_observations(X) -> np.ndarray:
    """`rank/(n+1)` per column — the margins removed, nothing else changed."""
    X = np.asarray(X, dtype=float)
    X = X.reshape(-1, 1) if X.ndim == 1 else X
    n = X.shape[0]
    if n < 2:
        raise ValueError("need at least two rows")
    r = np.argsort(np.argsort(X, axis=0, kind="stable"), axis=0, kind="stable") + 1.0
    return r / (n + 1.0)


def _inversions(ranks: np.ndarray) -> int:
    """Fenwick tree, O(n log n) — a discordant pair is one inversion."""
    n = ranks.size
    bit = np.zeros(n + 1, dtype=np.int64)
    inv = 0
    for seen, rank in enumerate(ranks):
        s, k = 0, int(rank)
        while k > 0:
            s += bit[k]
            k -= k & -k
        inv += seen - s
        k = int(rank)
        while k <= n:
            bit[k] += 1
            k += k & -k
    return int(inv)


def _kendall_pair(x, y) -> float:
    """Kendall's tau, ties-free, by counting inversions after sorting on `x`."""
    n = np.asarray(x).size
    order = np.argsort(np.asarray(x), kind="stable")
    ranks = np.argsort(np.argsort(np.asarray(y)[order], kind="stable"), kind="stable") + 1
    return 1.0 - 4.0 * _inversions(ranks) / (n * (n - 1.0))


def kendall_tau(U) -> np.ndarray:
    """The pairwise Kendall tau matrix of pseudo-observations `U`."""
    U = np.asarray(U, dtype=float)
    d = U.shape[1]
    T = np.eye(d)
    for i in range(d):
        for j in range(i + 1, d):
            T[i, j] = T[j, i] = _kendall_pair(U[:, i], U[:, j])
    return T


def _tau_from_param(family: str, param: float) -> float:
    if family in ("gaussian", "t"):
        return 2.0 / math.pi * math.asin(param)
    if family == "clayton":
        return param / (param + 2.0)
    if family == "gumbel":
        return 1.0 - 1.0 / param
    if family == "frank":
        if abs(param) < 1e-9:
            return 0.0                                   # independence, θ → 0
        x, w = np.polynomial.legendre.leggauss(200)      # Debye D₁(θ)
        t = 0.5 * param * (x + 1.0)
        # the ½·θ is the Jacobian of [−1,1] → [0,θ]; dropping it scales D₁ by θ
        d1 = 0.5 * param * np.sum(w * np.where(np.abs(t) < 1e-12, 1.0, t / np.expm1(t)))
        return 1.0 - 4.0 / param + 4.0 / param ** 2 * d1
    raise ValueError(f"no tau for {family!r}")


def _param_from_tau(family: str, tau: float) -> float:
    if family in ("gaussian", "t"):
        return math.sin(0.5 * math.pi * tau)
    if family == "clayton":
        if tau <= 0.0:
            raise ValueError("clayton only models positive dependence; tau must be > 0")
        return 2.0 * tau / (1.0 - tau)
    if family == "gumbel":
        if tau <= 0.0:
            raise ValueError("gumbel only models positive dependence; tau must be > 0")
        return 1.0 / (1.0 - tau)
    if family == "frank":
        lo, hi = -80.0, 80.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if _tau_from_param("frank", mid) < tau else (lo, mid)
        return 0.5 * (lo + hi)
    raise ValueError(f"no parameter inversion for {family!r}")


# --------------------------------------------------------------------------- #
# The copula
# --------------------------------------------------------------------------- #
class Copula:
    """One dependence structure, six families, arbitrary margins.

    `gaussian`, `t`, `clayton`, `gumbel`, `frank` are one-parameter families
    with `d` exchangeable coordinates. `vine` is a 3-dimensional C-vine built
    from three pair copulas, each of which may be a different family.
    """

    def __init__(self, variant: str = "gaussian", *, d: int = 2, theta: float | None = None,
                 rho: float | np.ndarray | None = None, df: float = 4.0,
                 families=None, seed: int = 0):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; expected one of {sorted(VARIANTS)}")
        if d < 2:
            raise ValueError("a copula needs at least two dimensions")
        self.variant = variant
        self.d = int(d)
        self.theta = float(DEFAULT_PARAM.get(variant, 1.0) if theta is None else theta)
        self.rho = 0.5 if rho is None else rho
        self.df = float(df)
        self.families = tuple(families) if families is not None else ("gumbel", "clayton", "gaussian")
        self.seed = int(seed)
        self._check()
        self.pairs_ = self._init_pairs() if variant == "vine" else []
        self.tau_ = None

    # -- construction ------------------------------------------------------- #
    def _check(self) -> None:
        v = self.variant
        if v in ("gaussian", "t"):
            r = np.asarray(self.rho, dtype=float)
            if r.ndim == 2:
                if r.shape != (self.d, self.d) or not np.allclose(r, r.T):
                    raise ValueError("a correlation matrix must be d×d and symmetric")
            elif not -1.0 < float(r) < 1.0:
                raise ValueError("rho must lie in (-1, 1); a copula is a copula")
            if v == "t" and self.df <= 0.0:
                raise ValueError("df must be positive")
        elif v == "clayton" and self.theta <= 0.0:
            raise ValueError("clayton needs theta > 0")
        elif v == "gumbel" and self.theta < 1.0:
            raise ValueError("gumbel needs theta >= 1 (theta = 1 is independence)")
        elif v == "frank" and abs(self.theta) < 1e-9:
            raise ValueError("frank needs theta != 0")
        elif v == "vine" and self.d != 3:
            raise ValueError("the vine variant is implemented for d = 3: one tree per level")

    def _init_pairs(self) -> list[tuple[int, int, str, float]]:
        if len(self.families) != 3:
            raise ValueError("a 3-D C-vine has three pair copulas: (1,2), (1,3), (2,3|1)")
        out = []
        for (i, j), item in zip([(0, 1), (0, 2), (1, 2)], self.families, strict=True):
            fam, param = (item, None) if isinstance(item, str) else item
            if fam not in DEFAULT_PARAM:
                raise ValueError(f"unsupported pair family {fam!r}")
            out.append((i, j, fam, float(DEFAULT_PARAM[fam] if param is None else param)))
        return out

    def _corr(self) -> np.ndarray:
        r = np.asarray(self.rho, dtype=float)
        if r.ndim == 2:
            return r
        R = np.full((self.d, self.d), float(r))
        np.fill_diagonal(R, 1.0)
        return R

    def _matrix(self, U) -> np.ndarray:
        U = np.asarray(U, dtype=float)
        U = U.reshape(1, -1) if U.ndim == 1 else U
        if U.ndim != 2 or U.shape[1] != self.d:
            raise ValueError(f"expected shape (m, {self.d}), got {U.shape}")
        if np.any(U < 0.0) or np.any(U > 1.0):
            raise ValueError("copula arguments must lie in [0, 1]; use Gaussian margins upstream")
        return U

    # -- sampling ----------------------------------------------------------- #
    def sample(self, n: int, rng=None) -> np.ndarray:
        """`n` draws from the copula, as an `(n, d)` array of uniforms."""
        rng = np.random.default_rng(self.seed) if rng is None else rng
        n = int(n)
        if n <= 0:
            raise ValueError("n must be positive")
        v, d = self.variant, self.d
        if v in ("gaussian", "t"):
            L = np.linalg.cholesky(self._corr())
            z = rng.standard_normal((n, d)) @ L.T
            if v == "gaussian":
                return np.clip(_Phi(z), EPS, 1.0 - EPS)
            w = 2.0 * _gamma(rng, 0.5 * self.df, n)     # χ²_ν = 2·Gamma(ν/2, 1)
            return np.clip(_t_cdf(z * np.sqrt(self.df / w)[:, None], self.df), EPS, 1.0 - EPS)
        if v == "clayton":
            # Marshall–Olkin: V ~ Gamma(1/θ), E_j ~ Exp(1), U_j = (1+E_j/V)^(−1/θ)
            w = _gamma(rng, 1.0 / self.theta, n)
            e = rng.exponential(1.0, (n, d))
            return np.clip((1.0 + e / w[:, None]) ** (-1.0 / self.theta), EPS, 1.0 - EPS)
        if v == "gumbel":
            # Marshall–Olkin: S ~ Stable(1/θ), E_j ~ Exp(1), U_j = exp(−(E_j/S)^(1/θ))
            s = _stable(rng, 1.0 / self.theta, n)
            e = rng.exponential(1.0, (n, d))
            return np.clip(np.exp(-(e / s[:, None]) ** (1.0 / self.theta)), EPS, 1.0 - EPS)
        if v == "frank":
            if d != 2:
                raise NotImplementedError(
                    "frank sampling uses the closed-form conditional, which is a "
                    "bivariate object; the d-variate Frank needs its discrete "
                    "Marshall–Olkin representation"
                )
            w = rng.random((n, 2))
            u1, u2 = w[:, 0], w[:, 1]
            a1 = np.exp(-self.theta * u1) - 1.0
            s = u2 * (math.exp(-self.theta) - 1.0) / (np.exp(-self.theta * u1) - u2 * a1)
            return np.clip(np.column_stack([u1, -np.log1p(s) / self.theta]), EPS, 1.0 - EPS)
        return self._sample_vine(n, rng)

    def _sample_vine(self, n: int, rng) -> np.ndarray:
        """C-vine on root 1: u₁ = w₁, u₂ = h₁₂⁻¹(w₂|u₁), then u₃ through two h's."""
        (_, _, f12, p12), (_, _, f13, p13), (_, _, f23, p23) = self.pairs_
        w = rng.random((n, 3))
        u1 = w[:, 0]
        u2 = _h_inv(w[:, 1], u1, f12, p12)
        mid = _h_inv(_h_inv(w[:, 2], _h(u2, u1, f12, p12), f23, p23), u1, f13, p13)
        return np.clip(np.column_stack([u1, u2, mid]), EPS, 1.0 - EPS)

    # -- the CDF ------------------------------------------------------------ #
    def cdf(self, U) -> np.ndarray:
        """`C(u)`. Closed form except for the Gaussian's bivariate integral."""
        U = self._matrix(U)
        if self.variant == "t":
            raise NotImplementedError(
                "the multivariate-t CDF has no closed form; integrate the density "
                "or use pymvnorm. The *density* is closed form and is implemented"
            )
        if self.variant == "vine":
            raise NotImplementedError(
                "a vine's CDF needs a d−1 dimensional integral that no pair copula "
                "gives you; sample, or use the density"
            )
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            if self.variant == "gaussian":
                if self.d != 2:
                    raise NotImplementedError(
                        "the d-variate normal CDF needs Genz' reduction; this file "
                        "implements the bivariate Drezner integral"
                    )
                z = _Phi_inv(U)
                return _bvn(z[:, 0], z[:, 1], float(np.asarray(self.rho).ravel()[0]))
            if self.variant == "clayton":
                s = np.sum(U ** (-self.theta), axis=1) - self.d + 1.0
                return s ** (-1.0 / self.theta)
            if self.variant == "gumbel":
                a = -np.log(U)
                return np.exp(-np.sum(a ** self.theta, axis=1) ** (1.0 / self.theta))
            b = (math.exp(-self.theta) - 1.0) ** (self.d - 1)
            a = np.exp(-self.theta * U) - 1.0
            return -np.log1p(np.prod(a, axis=1) / b) / self.theta

    # -- the density -------------------------------------------------------- #
    def logpdf(self, U) -> np.ndarray:
        """`log c(u)`: the part of the joint density that is not the margins."""
        # a density needs the open interval: clayton's is infinite on the corner
        U = np.clip(self._matrix(U), EPS, 1.0 - EPS)
        if self.variant == "vine":
            (i12, j12, f12, p12), (i13, j13, f13, p13), (i23, j23, f23, p23) = self.pairs_
            u1, u2, u3 = U[:, i12], U[:, j12], U[:, j13]
            return (_pair_logpdf(u1, u2, f12, p12) + _pair_logpdf(u1, u3, f13, p13)
                    + _pair_logpdf(_h(u2, u1, f12, p12), _h(u3, u1, f13, p13), f23, p23))
        if self.variant == "gaussian":
            z = _Phi_inv(U)
            R = self._corr()
            a = np.linalg.inv(R) - np.eye(self.d)
            return -0.5 * float(np.linalg.slogdet(R)[1]) - 0.5 * np.einsum("ij,jk,ik->i", z, a, z)
        if self.variant == "t":
            df, R = self.df, self._corr()
            x = np.column_stack([_t_ppf(U[:, j], df) for j in range(self.d)])
            q = np.einsum("ij,jk,ik->i", x, np.linalg.inv(R), x)
            log_mvt = (math.lgamma(0.5 * (df + self.d)) - math.lgamma(0.5 * df)
                       - 0.5 * self.d * math.log(df * math.pi)
                       - 0.5 * float(np.linalg.slogdet(R)[1])
                       - 0.5 * (df + self.d) * np.log1p(q / df))
            return log_mvt - _t_logpdf(x, df).sum(axis=1)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            if self.variant == "clayton":
                th, d = self.theta, self.d
                s = np.sum(U ** (-th), axis=1) - d + 1.0
                const = sum(math.log(1.0 + i * th) for i in range(d))
                return (const - (th + 1.0) * np.log(U).sum(axis=1)
                        - (d + 1.0 / th) * np.log(s))
        if self.variant == "gumbel" and self.d != 2:
            raise NotImplementedError("the Gumbel density here is bivariate")
        # gumbel and frank are the bivariate pair densities the vine also uses,
        # so they are written once, in `_pair_logpdf`
        return _pair_logpdf(U[:, 0], U[:, 1], self.variant, self.theta)

    # -- fitting ------------------------------------------------------------ #
    def fit(self, X) -> Copula:
        """Rank-transform the data, then invert Kendall's tau into the parameter."""
        U = pseudo_observations(X)
        if U.shape[1] != self.d:
            raise ValueError(f"expected {self.d} columns, got {U.shape[1]}")
        self.tau_ = kendall_tau(U)
        if self.variant == "vine":
            self._fit_vine(U)
            return self
        tri = self.tau_[np.triu_indices(self.d, 1)]
        tau = float(tri.mean())
        if self.variant in ("gaussian", "t"):
            self.rho = _param_from_tau(self.variant, tau)
            if self.variant == "t":
                self.df = self._fit_df(U)
        else:
            self.theta = _param_from_tau(self.variant, tau)
        return self

    def _fit_vine(self, U) -> None:
        """Sequential estimation: tree 1 from the data, tree 2 from h-transforms."""
        u1 = U[:, 0]
        pairs, pseudo = [], []
        for j in (1, 2):
            fam = self.pairs_[j - 1][2]
            param = _param_from_tau(fam, float(self.tau_[0, j]))
            pairs.append((0, j, fam, param))
            pseudo.append(_h(U[:, j], u1, fam, param))
        fam = self.pairs_[2][2]
        pairs.append((1, 2, fam, _param_from_tau(fam, _kendall_pair(*pseudo))))
        self.pairs_ = pairs

    def _fit_df(self, U) -> float:
        """Profile the degrees of freedom, with `rho` already fixed by tau.

        The profile is flat: `nu` and `rho` are competing explanations for the
        same tail mass, and `rho` is pinned by tau before `nu` is chosen. Two
        thousand rows decide between the grid points as well as twenty.
        """
        grid = (1.0, 2.0, 3.0, 5.0, 6.0, 8.0, 12.0, 20.0, 40.0)
        step = max(1, U.shape[0] // 2000)
        return max(grid, key=lambda v: self._loglik_at(U[::step], v))

    def _loglik_at(self, U, df: float) -> float:
        old, self.df = self.df, df
        try:
            return float(self.logpdf(U).sum())
        finally:
            self.df = old

    # -- the summary numbers ------------------------------------------------ #
    def tau(self) -> np.ndarray:
        """Analytic Kendall's tau, as a `d×d` matrix."""
        if self.variant == "vine":
            raise NotImplementedError(
                "a vine's tau is not one number: its pairwise taus come from the "
                "pair copulas and the tree structure together, so read `tau_`"
            )
        if self.variant in ("gaussian", "t"):
            R = self._corr()
            return 2.0 / math.pi * np.arcsin(np.clip(R, -1.0, 1.0))
        return np.full((self.d, self.d), _tau_from_param(self.variant, self.theta)) \
            + (1.0 - _tau_from_param(self.variant, self.theta)) * np.eye(self.d)

    def tail(self) -> dict[str, float]:
        """Lower and upper tail dependence, `λ = lim_{u→0} C(u,u)/u` and its mirror."""
        if self.variant == "vine":
            raise NotImplementedError(
                "a vine's joint tail dependence is set by the first tree, but no "
                "single formula covers mixed families; read the pair parameters"
            )
        v = self.variant
        if v == "gaussian":
            lam = 0.0 if abs(float(np.asarray(self.rho).ravel()[0])) < 1.0 else 1.0
            return {"lower": lam, "upper": lam}
        if v == "t":
            r = float(np.asarray(self.rho).ravel()[0])
            lam = 2.0 * float(_t_cdf(np.array([-math.sqrt((self.df + 1.0) * (1.0 - r) / (1.0 + r))]),
                                     self.df + 1.0)[0])
            return {"lower": lam, "upper": lam}
        if v == "clayton":
            return {"lower": 2.0 ** (-1.0 / self.theta), "upper": 0.0}
        if v == "gumbel":
            return {"lower": 0.0, "upper": 2.0 - 2.0 ** (1.0 / self.theta)}
        return {"lower": 0.0, "upper": 0.0}


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _grid2(nodes: int):
    """Gauss–Legendre nodes and weights on (0,1)², as points and a weight matrix."""
    x, w = np.polynomial.legendre.leggauss(nodes)
    u, ww = 0.5 * (x + 1.0), 0.5 * w
    U1, U2 = np.meshgrid(u, u, indexing="ij")
    return np.column_stack([U1.ravel(), U2.ravel()]), np.outer(ww, ww)


def _ks(U: np.ndarray) -> float:
    """Largest gap between the empirical CDF of a column and the identity."""
    n = U.shape[0]
    return float(max(np.max(np.abs(np.sort(U[:, j]) - np.arange(1, n + 1) / n))
                     for j in range(U.shape[1])))


def main() -> None:
    rng = np.random.default_rng(0)
    n = 20000
    u = 0.37

    # --- 1. the grounding axioms, on every family with a closed-form CDF ---
    # C(u,1) = u is what makes a copula a copula: the other coordinates set to
    # one must return the margin you left free. C(u,0) = 0 is the same statement
    # at the bottom, and it is where the families that blow up at the corner
    # have to be checked rather than assumed.
    specs = {"gaussian": {"rho": 0.6}, "clayton": {"theta": 2.0},
             "gumbel": {"theta": 1.5}, "frank": {"theta": 3.0}}
    for variant, kw in specs.items():
        c = Copula(variant, **kw)
        assert abs(c.cdf([[u, 1.0]])[0] - u) < 1e-12
        assert abs(c.cdf([[1.0, u]])[0] - u) < 1e-12
        assert c.cdf([[u, 0.0]])[0] < 1e-12
        assert c.cdf([[0.0, u]])[0] < 1e-12
        assert abs(c.cdf([[1.0, 1.0]])[0] - 1.0) < 1e-12
    # and once more in three dimensions, where the exponents are different
    for variant, kw in (("clayton", {"theta": 2.0}), ("gumbel", {"theta": 1.5}),
                        ("frank", {"theta": 3.0})):
        c = Copula(variant, d=3, **kw)
        assert abs(c.cdf([[u, 1.0, 1.0]])[0] - u) < 1e-12
        assert abs(c.cdf([[1.0, u, 1.0]])[0] - u) < 1e-12
        assert abs(c.cdf([[1.0, 1.0, 1.0]])[0] - 1.0) < 1e-12
    print(
        f"copula: all four closed-form families satisfy C(u,1) = u, C(u,0) = 0 "
        f"and C(1,…,1) = 1 at u = {u}, in two and three dimensions"
    )

    # --- 2. the density is the mixed derivative of the CDF -----------------
    # This is the only check that pins down the normalising constants: a
    # forgotten (1−ρ²)^(−1/2) leaves every identity above intact.
    h = 1e-4
    a, b = 0.37, 0.61
    dens = {}
    for variant, kw in specs.items():
        c = Copula(variant, **kw)
        cross = (c.cdf([[a + h, b + h]])[0] - c.cdf([[a + h, b - h]])[0]
                 - c.cdf([[a - h, b + h]])[0] + c.cdf([[a - h, b - h]])[0]) / (4 * h * h)
        exact = math.exp(float(c.logpdf([[a, b]])[0]))
        dens[variant] = (cross, exact)
        assert abs(cross / exact - 1.0) < 1e-5, (variant, cross, exact)

    # --- 3. and it integrates to one --------------------------------------
    # Gauss–Legendre is spectrally accurate on the smooth families and only
    # algebraic on clayton and gumbel, whose densities diverge at the corner.
    pts, W = _grid2(200)
    integ = {}
    for variant, kw in specs.items():
        c = Copula(variant, **kw)
        d = np.exp(c.logpdf(pts)).reshape(W.shape)
        integ[variant] = float(np.sum(d * W))
        assert abs(integ[variant] - 1.0) < 1e-3, (variant, integ[variant])
    print(
        f"copula: c(u) reproduces ∂²C/∂u∂v to "
        f"{max(abs(x / y - 1) for x, y in dens.values()):.1e} at ({a}, {b}), and "
        f"∫∫c = 1 to {max(abs(v - 1) for v in integ.values()):.1e} on a 200² "
        f"Gauss–Legendre grid"
    )

    # --- 4. samples: uniform margins, and calibrated joint probabilities ---
    # A sampler that is wrong in the dependence still passes the margin test;
    # it is the joint CDF at an interior point that catches it.
    models = {"gaussian": Copula("gaussian", rho=0.6),
              "t": Copula("t", rho=0.6, df=5.0),
              "clayton": Copula("clayton", theta=2.0),
              "gumbel": Copula("gumbel", theta=1.5),
              "frank": Copula("frank", theta=3.0),
              "vine": Copula("vine", d=3, families=(("gumbel", 1.7), ("clayton", 1.2),
                                                   ("gaussian", 0.5)))}
    draws = {k: m.sample(n, rng) for k, m in models.items()}
    lim = 1.95 / math.sqrt(n)                     # 0.999 quantile of the KS law
    for name, U in draws.items():
        assert _ks(U) < lim, (name, _ks(U))
        assert U.min() > 0.0 and U.max() < 1.0
    joint = {}
    for name in ("gaussian", "clayton", "gumbel", "frank"):
        U = draws[name]
        emp = float(np.mean((U[:, 0] <= 0.3) & (U[:, 1] <= 0.6)))
        model = float(models[name].cdf([[0.3, 0.6]])[0])
        sigma = math.sqrt(model * (1.0 - model) / n)
        assert abs(emp - model) < 4.0 * sigma, (name, emp, model)
        joint[name] = abs(emp - model) / sigma
    print(
        f"copula: all six samplers pass a uniform-margin KS test at n = {n} "
        f"(worst gap {max(_ks(U) for U in draws.values()):.4f} against the "
        f"{lim:.4f} threshold), and the four closed-form families hit their own "
        f"joint CDF at (0.3, 0.6) within {max(joint.values()):.1f}σ"
    )

    # --- 5. Kendall's tau, three ways -------------------------------------
    # Analytic (from the parameter), by quadrature (4∫∫C dc − 1, which knows
    # nothing about the parameter), and empirical (from the draws). Three
    # independent routes to the same number is the strongest statement here.
    tau_rows = {}
    sigma_tau = math.sqrt(2.0 * (2.0 * n + 5.0) / (9.0 * n * (n - 1.0)))
    for name in ("gaussian", "clayton", "gumbel", "frank"):
        m = models[name]
        quad = 4.0 * float(np.sum(m.cdf(pts) * np.exp(m.logpdf(pts)) * W.ravel())) - 1.0
        ana = float(m.tau()[0, 1])
        emp = _kendall_pair(draws[name][:, 0], draws[name][:, 1])
        # the same 200² grid as above, so gumbel and clayton are still limited
        # by the corner singularity rather than by the identity
        assert abs(quad - ana) < 1e-4, (name, quad, ana)
        assert abs(emp - ana) < 4.0 * sigma_tau, (name, emp, ana, sigma_tau)
        tau_rows[name] = (ana, quad, emp)
    # the t copula has a closed-form tau and no closed-form CDF, so only two of
    # the three routes are open to it
    t_ana = float(models["t"].tau()[0, 1])
    t_emp = _kendall_pair(draws["t"][:, 0], draws["t"][:, 1])
    assert abs(t_emp - t_ana) < 4.0 * sigma_tau, (t_emp, t_ana)
    # the vine's pairwise taus come back out of its first tree
    V = draws["vine"]
    vine_tau = kendall_tau(V)
    for j, param in ((1, 1.7), (2, 1.2)):
        assert abs(vine_tau[0, j] - _tau_from_param(models["vine"].pairs_[j - 1][2], param)) < 0.02
        assert models["vine"].pairs_[j - 1][3] == param
    print(
        "copula: Kendall's tau analytic / quadrature / empirical agree — "
        + ", ".join(f"{k} {a:.4f}/{q:.4f}/{e:.4f}" for k, (a, q, e) in tau_rows.items())
        + f"; the t copula's {t_ana:.4f} has no quadrature route (no CDF) and comes "
        f"back at {t_emp:.4f} from {n} draws"
    )

    # --- 6. fit: rank transform, tau inversion, and margin invariance -----
    # The fit never sees the values, only their ranks, so any increasing map of
    # a column has to leave the estimate unchanged -- exactly, not to rounding.
    gen = {"gaussian": {"rho": 0.7}, "t": {"rho": 0.7, "df": 6.0},
           "clayton": {"theta": 3.0}, "gumbel": {"theta": 2.0}, "frank": {"theta": 4.0}}
    fitted = {}
    for variant, kw in gen.items():
        true = Copula(variant, **kw)
        Z = true.sample(4000, rng)                    # one draw, then the margins
        X = np.column_stack([np.exp(Z[:, 0]), Z[:, 1] ** 3])
        fit = Copula(variant).fit(X)
        fitted[variant] = fit
        assert fit.tau_ is not None
        # a different monotone map, and a different seed, must not move tau
        X2 = np.column_stack([X[:, 0] ** 2, np.log(X[:, 1])])
        again = Copula(variant).fit(X2)
        assert abs(again.tau_[0, 1] - fit.tau_[0, 1]) < 1e-12
    assert abs(fitted["gaussian"].rho - 0.7) < 0.03
    assert abs(fitted["clayton"].theta - 3.0) < 0.35
    assert abs(fitted["gumbel"].theta - 2.0) < 0.12
    assert abs(fitted["frank"].theta - 4.0) < 0.55
    assert fitted["t"].df in (5.0, 6.0)                   # 6 is the truth
    # the fitted parameter reproduces the data's tau by construction
    for fit in fitted.values():
        assert abs(float(fit.tau()[0, 1]) - fit.tau_[0, 1]) < 2e-3

    def _param_name(f: Copula) -> str:
        return "rho" if f.variant in ("gaussian", "t") else "theta"

    print(
        "copula: a rank-transformed fit recovers "
        + ", ".join(f"{k} {_param_name(f)} = "
                    f"{float(np.asarray(getattr(f, _param_name(f))).ravel()[0]):.3f}"
                    for k, f in fitted.items())
        + f"; t also picks df = {fitted['t'].df:.0f} by profile likelihood"
    )

    # --- 7. tail dependence: the limit, against the closed form -----------
    # λ_U = lim (C(u,u) − u²)/(1−u). Written that way the two numbers near one
    # cancel once, inside a product, rather than twice across the expression.
    def limit(c: Copula, side: str, eps: float) -> float:
        if side == "L":
            return float(c.cdf([[eps, eps]])[0]) / eps
        s = 1.0 - eps
        return (float(c.cdf([[s, s]])[0]) - s * s) / eps

    # clayton and gumbel are exact: at eps = 1e-8 the ratio is already the
    # limit, which is the closed form and nothing else
    assert abs(limit(models["clayton"], "L", 1e-8) - 2.0 ** (-0.5)) < 1e-6
    assert abs(limit(models["gumbel"], "U", 1e-8) - (2.0 - 2.0 ** (2.0 / 3.0))) < 1e-6
    # and the tail they do not have is zero, approached at a rate they set
    assert limit(models["clayton"], "U", 1e-8) < 1e-6
    assert limit(models["gumbel"], "L", 1e-8) < 1e-4
    # gaussian and frank have no tail dependence at all, and converge to it
    # slowly: the ratio is still 4e-3 (gaussian) and 3e-8 (frank) at eps = 1e-8,
    # so it has to be seen shrinking rather than asserted to be zero
    g_seq = [limit(models["gaussian"], "U", e) for e in (1e-4, 1e-6, 1e-8)]
    f_seq = [limit(models["frank"], "L", e) for e in (1e-4, 1e-6, 1e-8)]
    assert g_seq[0] > g_seq[1] > g_seq[2] > 0.0
    assert f_seq[0] > f_seq[1] > f_seq[2] > 0.0
    # the t copula's λ is positive, and at a finite u the ratio C(u,u)/u is
    # still above it: its CDF has no closed form, but the chi-square mixture
    # does. Conditional on W ~ χ²_ν the pair is Gaussian, so
    #     C(u,u) = ∫₀^∞ Φ₂(z√(w/ν), z√(w/ν); ρ) χ²_ν(w) dw,   z = t⁻¹_ν(u),
    # and that is a deterministic sequence converging to λ from above.
    lam_t = models["t"].tail()["upper"]
    rho_t, df_t = float(models["t"].rho), models["t"].df

    def t_ratio(u: float) -> float:
        z = float(_t_ppf(np.array([u]), df_t)[0])
        x, w = np.polynomial.legendre.leggauss(400)
        W = 40.0 * (x + 1.0)                    # χ²₅ has essentially no mass past 80
        dens = np.exp((0.5 * df_t - 1.0) * np.log(W) - 0.5 * W
                      - 0.5 * df_t * math.log(2.0) - math.lgamma(0.5 * df_t))
        s = z * np.sqrt(W / df_t)
        return float(np.sum(_bvn(s, s, rho_t) * dens * 40.0 * w)) / u

    t_seq = [t_ratio(u) for u in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)]
    assert all(a > b for a, b in zip(t_seq, t_seq[1:], strict=False))
    assert abs(t_seq[-1] - lam_t) < 5e-3
    print(
        f"copula: tail dependence — clayton lower {limit(models['clayton'], 'L', 1e-8):.6f} "
        f"= 2^(-1/θ) and gumbel upper {limit(models['gumbel'], 'U', 1e-8):.6f} = 2-2^(1/θ) "
        f"exactly; gaussian and frank decay to 0 ({g_seq[0]:.1e} → {g_seq[2]:.1e} and "
        f"{f_seq[0]:.1e} → {f_seq[2]:.1e}); the t's C(u,u)/u falls "
        f"{t_seq[0]:.3f} → {t_seq[-1]:.3f} towards its limit {lam_t:.3f}"
    )

    # --- 8. the vine: h∘h⁻¹, and a marginal that has to be a pair copula --
    # Multiplying three pair densities is only a copula if each h is a proper
    # conditional CDF; the sharp test is that integrating the 3-D density over
    # u₃ returns the (1,2) pair density, not just that the margins look flat.
    vine = models["vine"]
    for fam, param in (("gaussian", 0.5), ("clayton", 1.2), ("gumbel", 1.7), ("frank", 2.0)):
        uu = rng.uniform(0.05, 0.95, 2000)
        vv = rng.uniform(0.05, 0.95, 2000)
        assert np.max(np.abs(_h_inv(_h(uu, vv, fam, param), vv, fam, param) - uu)) < 1e-10
        grid = np.linspace(1e-9, 1.0 - 1e-9, 2001)
        hh = _h(grid, np.full_like(grid, 0.42), fam, param)
        assert np.all(np.diff(hh) > 0.0) and hh[0] < 1e-6 and hh[-1] > 1.0 - 1e-6
    x, w = np.polynomial.legendre.leggauss(400)
    u3 = 0.5 * (x + 1.0)
    W3 = 0.5 * w
    err = 0.0
    for p1, p2 in ((0.3, 0.7), (0.55, 0.2), (0.8, 0.45)):
        cube = np.column_stack([np.full_like(u3, p1), np.full_like(u3, p2), u3])
        got = float(np.sum(np.exp(vine.logpdf(cube)) * W3))
        err = max(err, abs(got / math.exp(
            _pair_logpdf(np.array([p1]), np.array([p2]), "gumbel", 1.7)[0]) - 1.0))
    assert err < 1e-6, err
    # and the second tree is what makes (2,3) depend on each other at all
    assert abs(vine_tau[1, 2]) > abs(vine_tau[0, 1]) - 0.1
    print(
        f"copula: the 3-D vine's h-functions invert to 1e-10, integrating its "
        f"density over u₃ returns the (1,2) Gumbel pair density to {err:.1e}, and "
        f"its pairwise taus are {vine_tau[0, 1]:.3f}, {vine_tau[0, 2]:.3f}, "
        f"{vine_tau[1, 2]:.3f} against pair parameters 1.7, 1.2, 0.5"
    )

    # --- 9. what each variant refuses, and why ----------------------------
    for call, why in ((lambda: Copula("t").cdf([[0.5, 0.5]]), "multivariate-t CDF"),
                      (lambda: Copula("vine", d=3).cdf([[0.5, 0.5, 0.5]]), "vine CDF"),
                      (lambda: Copula("vine", d=3).tail(), "vine tails"),
                      (lambda: Copula("gaussian", d=3).cdf([[0.5, 0.5, 0.5]]),
                       "trivariate normal CDF")):
        try:
            call()
        except NotImplementedError:
            continue
        raise AssertionError(f"{why} should not be silently available")
    print(
        "copula: the t and vine CDFs, the vine's tail limit and the trivariate "
        "normal CDF raise rather than return a number, because no closed form has "
        "ever been written for them"
    )

    print("copula demo passed")


if __name__ == "__main__":
    main()
