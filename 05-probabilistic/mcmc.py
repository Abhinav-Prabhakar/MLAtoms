"""MCMC — sampling the posterior when the algebra fails.

Chapter:   05-probabilistic
Companion: mcmc.md
Tier 2 · 7 variants: metropolis_hastings, gibbs, hmc, nuts, slice_sampling,
adaptive_metropolis, diagnostics

A Markov chain whose stationary distribution is the target. Every variant below
is a different transition kernel K(x -> x') with ∫ π(x) K(x, x') dx = π(x'):

    metropolis_hastings   propose, then accept with min(1, π(x')/π(x))
    gibbs                 accept everything, cycle through full conditionals
    hmc                   Hamiltonian dynamics + a Metropolis correction
    nuts                  HMC whose trajectory length is chosen by u-turns
    slice_sampling        sample uniformly under the density along a line
    adaptive_metropolis   tune the proposal during burn-in, then freeze it
    diagnostics           R-hat, ESS and the acceptance rate — not a kernel

The last one is included because a sampler without diagnostics is a random
number generator with a story attached.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import numpy as np

VARIANTS = frozenset(
    ["metropolis_hastings", "gibbs", "hmc", "nuts", "slice_sampling",
     "adaptive_metropolis", "diagnostics"]
)

LogProb = Callable[[np.ndarray], float]
Grad = Callable[[np.ndarray], np.ndarray]


def _vec(x: np.ndarray | float) -> np.ndarray:
    return np.atleast_1d(np.asarray(x, dtype=float))


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def _acov(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Autocovariance of one chain for lags 0..max_lag, by FFT."""
    x = np.asarray(x, dtype=float)
    n = x.size
    x = x - x.mean()
    size = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(x, size)
    raw = np.fft.irfft(f * np.conj(f), size)[: max_lag + 1]
    # unbiased: divide by (n - lag) rather than n
    return raw / (n - np.arange(max_lag + 1))


def autocorrelation(chains: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """Mean autocorrelation across chains, lags 0..max_lag, shape `(max_lag+1,)`."""
    chains = np.asarray(chains, dtype=float)
    if chains.ndim == 1:
        chains = chains[None, :]
    _, n = chains.shape
    max_lag = min(n // 2, max_lag if max_lag is not None else n // 2)
    rho = np.mean([_acov(c, max_lag) for c in chains], axis=0)
    return rho / rho[0]


def ess(chains: np.ndarray) -> float:
    """Effective sample size, Geyer's initial monotone positive pair sum.

    The naive `n / (1 + 2 Σ ρ_t)` is unstable because the tail of the
    autocorrelation is estimated by noise. Geyer's trick: sum *pairs* `ρ_2k +
    ρ_2k+1` and stop at the first non-positive pair, which is a truncation that
    stays asymptotically consistent.
    """
    chains = np.asarray(chains, dtype=float)
    if chains.ndim == 1:
        chains = chains[None, :]
    m, n = chains.shape
    rho = autocorrelation(chains)
    total = 0.0
    lag = 0
    while lag + 1 < rho.size:
        pair = rho[lag] + rho[lag + 1]
        if pair <= 0.0:
            break
        total += pair
        lag += 2
    # geometric ergodicity says tau >= ... nothing: an anti-correlated chain has
    # tau < 1 and would report more information than independent draws. Cap it.
    tau = max(-1.0 + 2.0 * total, 1.0)
    return float(min(m * n / tau, m * n))


def rhat(chains: np.ndarray) -> float:
    """Gelman-Rubin potential scale reduction, split-free form.

    `sqrt(((n-1)/n W + B/n) / W)`: between-chain variance adds to within-chain
    variance when the chains disagree, so 1.0 means "indistinguishable" and
    anything above ~1.01 means the chains have not mixed.
    """
    chains = np.asarray(chains, dtype=float)
    if chains.ndim == 1:
        chains = chains[None, :]
    m, n = chains.shape
    if m < 2:
        raise ValueError("rhat needs at least two chains")
    W = float(np.mean(chains.var(axis=1, ddof=1)))
    B = float(n * np.var(chains.mean(axis=1), ddof=1))
    var_hat = (n - 1) / n * W + B / n
    return float(np.sqrt(var_hat / W))


def diagnose(chains: np.ndarray) -> dict[str, float]:
    """The three numbers to look at before believing any posterior summary."""
    chains = np.asarray(chains, dtype=float)
    if chains.ndim == 1:
        chains = chains[None, :]
    pooled = chains.reshape(-1)
    return {
        "rhat": rhat(chains) if chains.shape[0] > 1 else float("nan"),
        "ess": ess(chains),
        "mean": float(pooled.mean()),
        "sd": float(pooled.std(ddof=1)),
        "mcse": float(pooled.std(ddof=1) / np.sqrt(ess(chains))),
    }


# --------------------------------------------------------------------------- #
# Sampler
# --------------------------------------------------------------------------- #
class _Node(NamedTuple):
    """One subtree of the NUTS recursion: both endpoints plus their gradients."""

    xm: np.ndarray
    pm: np.ndarray
    gm: np.ndarray
    xp: np.ndarray
    pp: np.ndarray
    gp: np.ndarray
    lp_m: float
    lp_p: float
    n: int
    s: int
    alpha: float
    n_alpha: int


class MCMC:
    """One class, six kernels, one set of diagnostics.

    Parameters
    ----------
    log_prob : callable, optional
        Unnormalised log density, `x (d,) -> float`. No normalising constant is
        needed, which is the entire point: MCMC only ever compares densities.
        Only `gibbs` may pass `None`, when it is given the full conditionals.
    variant : str
        One of VARIANTS.
    grad : callable, optional
        Gradient of `log_prob`. Required by `hmc` and `nuts`; without it the
        gradient is taken by central differences, which is correct but costs
        `2d` density evaluations per leapfrog step.
    scale : float
        Random-walk proposal standard deviation for the Metropolis variants.
        The asymptotic optimum for a Gaussian target is `2.38/sqrt(d)` times its
        standard deviation — that is what `adaptive_metropolis` discovers.
    n_leapfrog : int
        Leapfrog steps per `hmc` trajectory. Trajectory *length* is what matters
        (`n_leapfrog * step_size`), and a length near one period of the target
        is the sweet spot.
    step_size : float
        Leapfrog step for `hmc` and the initial step for `nuts` (which tunes it).
    mass : array_like, optional
        Diagonal metric. `mass = 1/σ²` per coordinate is the ideal choice; a
        wrong mass turns a sphere into a cigar and HMC into a slow sampler.
    conditionals : list of callables, optional
        `gibbs` only: `f(x, rng) -> float`, a sampler for coordinate `j` given
        the rest of `x`. If omitted, `log_conditionals` is slice-sampled.
    log_conditionals : list of callables, optional
        `gibbs` only: `g(x) -> float`, the log full conditional of coordinate
        `j` at the current `x`.
    width : float
        Slice-sampling window; `width=1.0` is a sane default in the units of the
        target. Too small and the sampler crawls, too large and it wastes work.
    n_warmup : int
        Steps discarded before collection. `adaptive_metropolis` and `nuts` adapt
        during this window and freeze afterwards; the others ignore it.
    target_accept : float
        Adaptation target. `0.44` in one dimension and `0.234` above it follows
        from the optimal-scaling theory of random-walk Metropolis.
    max_depth : int
        `nuts` recursion cap: at most `2**max_depth` leapfrog steps per draw.

    Notes
    -----
    `sample` always returns a chain shaped `(n_samples, d)`; the costs are
    recorded as `n_eval_` (density calls) and `n_grad_` (gradient calls), which
    is how the demo compares kernels fairly — effective samples per evaluation,
    not per iteration.
    """

    def __init__(
        self,
        log_prob: LogProb,
        *,
        variant: str = "metropolis_hastings",
        grad: Grad | None = None,
        scale: float = 1.0,
        n_leapfrog: int = 16,
        step_size: float = 0.25,
        mass: np.ndarray | None = None,
        conditionals: list[Callable] | None = None,
        log_conditionals: list[Callable] | None = None,
        width: float = 1.0,
        n_warmup: int = 0,
        target_accept: float | None = None,
        max_depth: int = 10,
    ):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        self.variant = variant
        self._log_prob = log_prob
        self._grad_fn = grad
        self.scale = float(scale)
        self.n_leapfrog = int(n_leapfrog)
        self.step_size = float(step_size)
        self.mass = None if mass is None else np.asarray(mass, dtype=float)
        self.conditionals = conditionals
        self.log_conditionals = log_conditionals
        self.width = float(width)
        self.n_warmup = int(n_warmup)
        self.target_accept = target_accept
        self.max_depth = int(max_depth)

    # -- counting wrappers: every comparison of kernels needs a price ------- #
    def _lp(self, x: np.ndarray) -> float:
        self.n_eval_ += 1
        return float(self._log_prob(x))

    def _grad(self, x: np.ndarray) -> np.ndarray:
        if self._grad_fn is not None:
            self.n_grad_ += 1
            return _vec(self._grad_fn(x))
        self.n_eval_ += 2 * x.size
        h = 1e-6
        g = np.empty(x.size)
        for i in range(x.size):
            e = np.zeros(x.size)
            e[i] = h
            g[i] = (self._log_prob(x + e) - self._log_prob(x - e)) / (2.0 * h)
        return g

    def _target_accept(self, d: int) -> float:
        """Optimal acceptance depends on the kernel, not just the dimension.

        `0.44 / 0.234` is the random-walk Metropolis result (0.234 is the
        high-dimensional limit for `2.38/sqrt(d)` scaling). Hamiltonian methods
        can afford — and need — a much stricter target near 0.8, because a
        rejected trajectory has already paid for every leapfrog step.
        """
        if self.target_accept is not None:
            return float(self.target_accept)
        if self.variant in ("hmc", "nuts"):
            return 0.8
        return 0.44 if d == 1 else 0.234

    # ------------------------------------------------------------------ #
    def sample(
        self, n_samples: int, x0: np.ndarray | float, *, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """Run the kernel and return `(n_samples, d)`.

        Sets `accept_rate_`, `chain_`, `log_prob_`, `n_eval_`, `n_grad_`, and for
        the adaptive variants `scale_`, `step_size_` and `trace_`.
        """
        rng = np.random.default_rng(0) if rng is None else rng
        x0 = _vec(x0)
        self.n_eval_ = 0
        self.n_grad_ = 0
        self.accept_rate_ = 0.0
        self.trace_: dict[str, list] = {"proposal_scale": [], "accept": [], "depth": []}
        self.depth_: list[int] = []

        if self.variant == "gibbs":
            chain = self._gibbs(n_samples, x0, rng)
        elif self.variant in ("hmc", "nuts"):
            chain = self._hamiltonian(n_samples, x0, rng)
        elif self.variant == "slice_sampling":
            chain = self._slice(n_samples, x0, rng)
        else:
            chain = self._random_walk(n_samples, x0, rng)

        self.chain_ = chain
        self.log_prob_ = (
            np.array([self._log_prob(x) for x in chain])
            if self._log_prob is not None else np.empty(0)
        )
        return chain

    # -- random-walk Metropolis ----------------------------------------- #
    def _random_walk(self, n: int, x0: np.ndarray, rng) -> np.ndarray:
        x = x0.copy()
        d = x.size
        lp = self._lp(x)
        scale = self.scale
        adapt = self.variant == "adaptive_metropolis"
        target = self._target_accept(d)
        out = np.empty((n, d))
        accepted = 0
        total = self.n_warmup + n
        for i in range(total):
            prop = x + scale * rng.standard_normal(d)
            lpp = self._lp(prop)
            ok = np.log(rng.random()) < lpp - lp
            if ok:
                x, lp = prop, lpp
                accepted += 1
            if adapt and i < self.n_warmup:
                # Robbins-Monro on log(scale): the step is proportional to the
                # error in the *running* acceptance rate, and the gain decays as
                # i^-0.6 so the scale settles instead of oscillating forever.
                rate = accepted / (i + 1)
                scale = float(np.exp(np.log(scale) + (i + 1) ** -0.6 * (rate - target)))
                self.trace_["proposal_scale"].append(scale)
            if i >= self.n_warmup:
                out[i - self.n_warmup] = x
                self.trace_["accept"].append(float(ok))
        self.scale_ = scale
        self.accept_rate_ = float(np.mean(self.trace_["accept"]))
        return out

    # -- Gibbs ----------------------------------------------------------- #
    def _gibbs(self, n: int, x0: np.ndarray, rng) -> np.ndarray:
        x = x0.copy()
        d = x.size
        samplers = self.conditionals
        if samplers is None:
            if self.log_conditionals is None:
                raise ValueError("gibbs needs `conditionals` or `log_conditionals`")
            samplers = [self._slice_conditional(j) for j in range(d)]
        if len(samplers) != d:
            raise ValueError("one conditional per coordinate")
        out = np.empty((n, d))
        for i in range(self.n_warmup + n):
            for j, draw in enumerate(samplers):
                x[j] = draw(x, rng)
            if i >= self.n_warmup:
                out[i - self.n_warmup] = x
        # a Gibbs sweep accepts every proposal by construction
        self.accept_rate_ = 1.0
        return out

    def _slice_conditional(self, j: int) -> Callable:
        logc = self.log_conditionals[j]

        def draw(x: np.ndarray, rng) -> float:
            def f(v: float) -> float:
                probe = x.copy()
                probe[j] = v
                return float(logc(probe))

            return self._slice_1d(f, float(x[j]), rng)

        return draw

    # -- slice sampling -------------------------------------------------- #
    def _slice_1d(self, logp: Callable, x0: float, rng: np.random.Generator) -> float:
        """Neal's stepping-out and shrinkage on one coordinate.

        The level `y = log p(x0) - Exp(1)` defines the slice; the window is
        grown until both ends fall outside it, then shrunk towards `x0` until a
        point above the level is found. Every returned point is uniform on the
        slice, which is exactly detailed balance.
        """
        y = float(logp(x0)) - float(rng.exponential())
        left = x0 - self.width * float(rng.random())
        right = left + self.width
        j = int(np.floor(20 * rng.random()))
        k = 20 - 1 - j
        while j > 0 and float(logp(left)) > y:
            left -= self.width
            j -= 1
        while k > 0 and float(logp(right)) > y:
            right += self.width
            k -= 1
        for _ in range(1000):
            x1 = left + (right - left) * float(rng.random())
            if float(logp(x1)) > y:
                return x1
            if x1 > x0:
                right = x1
            else:
                left = x1
        return x0

    def _slice(self, n: int, x0: np.ndarray, rng) -> np.ndarray:
        x = x0.copy()
        d = x.size
        out = np.empty((n, d))
        for i in range(self.n_warmup + n):
            for j in range(d):
                def f(v: float, j: int = j) -> float:
                    probe = x.copy()
                    probe[j] = v
                    return self._lp(probe)

                x[j] = self._slice_1d(f, float(x[j]), rng)
            if i >= self.n_warmup:
                out[i - self.n_warmup] = x
        self.accept_rate_ = 1.0
        return out

    # -- Hamiltonian Monte Carlo ---------------------------------------- #
    def _leapfrog(self, x, p, g, eps):
        """One symplectic step in momentum form.

        `p ~ N(0, M)` and `K = ½ pᵀM⁻¹p`. Updating the *momentum* and dividing
        by the mass inside the position update is what keeps a diagonal metric
        from silently rescaling the dynamics: updating a velocity as if it were
        a momentum is off by a factor of `M`, which shows up as energy that is
        not conserved on an anisotropic target.
        """
        m = self.mass if self.mass is not None else np.ones(x.size)
        p = p + 0.5 * eps * g
        x = x + eps * p / m
        lp = self._lp(x)
        g = self._grad(x)
        p = p + 0.5 * eps * g
        return x, p, g, lp

    def _kinetic(self, p: np.ndarray) -> float:
        m = self.mass if self.mass is not None else np.ones(p.size)
        return 0.5 * float(np.sum(p * p / m))

    def _hamiltonian(self, n: int, x0: np.ndarray, rng) -> np.ndarray:
        if self.variant == "nuts":
            return self._nuts(n, x0, rng)
        x = x0.copy()
        d = x.size
        lp = self._lp(x)
        g = self._grad(x)
        out = np.empty((n, d))
        errors = []
        accepted = 0
        total = self.n_warmup + n
        for i in range(total):
            m = self.mass if self.mass is not None else np.ones(d)
            p0 = rng.standard_normal(d) * np.sqrt(m)
            h0 = -lp + self._kinetic(p0)
            xn, vn, gn, lpn = x, p0, g, lp
            for _ in range(self.n_leapfrog):
                xn, vn, gn, lpn = self._leapfrog(xn, vn, gn, self.step_size)
            h1 = -lpn + self._kinetic(vn)
            errors.append(abs(h1 - h0))
            if np.log(rng.random()) < h0 - h1:
                x, lp, g = xn, lpn, gn
                accepted += 1
            if i >= self.n_warmup:
                out[i - self.n_warmup] = x
        self.accept_rate_ = accepted / total
        self.energy_error_ = float(np.mean(errors))
        return out

    def _find_step(self, x: np.ndarray, lp: float, g: np.ndarray, rng) -> float:
        """Heuristic initial step (Hoffman & Gelman, Algorithm 4).

        Dual averaging converges from almost anywhere, but "almost" is doing
        work: start at a step where a single leapfrog still accepts about half
        the time, and the adaptation only has to refine. Doubling/halving until
        the acceptance crosses 0.5 is one gradient per trial.
        """
        m = self.mass if self.mass is not None else np.ones(x.size)
        p0 = rng.standard_normal(x.size) * np.sqrt(m)
        H0 = -lp + self._kinetic(p0)
        eps = 1.0
        direction = 1
        for trial in range(60):
            x1, p1, _, lp1 = self._leapfrog(x, p0, g, direction * eps)
            H1 = -lp1 + self._kinetic(p1)
            better = np.log(0.5) < (H0 - H1)
            if trial == 0:
                direction = 1 if better else -1
            elif better != (direction == 1):
                break
            eps *= 2.0**direction
        return float(np.clip(eps, 1e-6, 1e3))

    def _nuts(self, n: int, x0: np.ndarray, rng) -> np.ndarray:
        """No-U-Turn sampler with dual-averaged step size (Hoffman & Gelman).

        The tree doubles until either endpoint turns back towards the other; the
        proposal is drawn from the subtree whose size is uniform over the
        visited states, which is what makes the doubling unbiased.
        """
        x = x0.copy()
        d = x.size
        lp = self._lp(x)
        g = self._grad(x)
        out = np.empty((n, d))
        gamma, t0, kappa = 0.05, 10.0, 0.75
        delta = self._target_accept(d)

        eps0 = self._find_step(x, lp, g, rng)
        mu = np.log(10.0 * eps0)
        log_eps = np.log(eps0)
        log_eps_bar, h_bar = 0.0, 0.0
        late: list[float] = []
        total = self.n_warmup + n

        for i in range(total):
            self.step_size = float(np.exp(log_eps))
            m = self.mass if self.mass is not None else np.ones(d)
            p0 = rng.standard_normal(d) * np.sqrt(m)
            H0 = -lp + self._kinetic(p0)
            logu = -H0 + np.log(rng.random())

            xm = xp = x
            pm = pp = p0
            gm = gp = g
            lp_m = lp_p = lp
            nodes, ok, depth = 1, 1, 0
            alpha, n_alpha = 0.0, 0

            while ok == 1 and depth < self.max_depth:
                direction = 1 if rng.random() < 0.5 else -1
                if direction == 1:
                    tree = self._tree(xp, pp, gp, lp_p, logu, H0, 1, depth)
                    if tree.s == 1 and rng.random() < min(1.0, tree.n / nodes):
                        x = tree.xp
                    xp, pp, gp, lp_p = tree.xp, tree.pp, tree.gp, tree.lp_p
                else:
                    tree = self._tree(xm, pm, gm, lp_m, logu, H0, -1, depth)
                    if tree.s == 1 and rng.random() < min(1.0, tree.n / nodes):
                        x = tree.xm
                    xm, pm, gm, lp_m = tree.xm, tree.pm, tree.gm, tree.lp_m

                alpha += tree.alpha
                n_alpha += tree.n_alpha
                nodes += tree.n
                delta_x = xp - xm
                ok = tree.s * int(float(delta_x @ pm) >= 0.0 and float(delta_x @ pp) >= 0.0)
                depth += 1

            accept_stat = alpha / max(n_alpha, 1)
            self.depth_.append(depth)
            lp, g = self._lp(x), self._grad(x)

            if i < self.n_warmup:
                eta = 1.0 / (i + 1 + t0)
                h_bar = (1.0 - eta) * h_bar + eta * (delta - accept_stat)
                log_eps = mu - (i + 1) ** 0.5 / gamma * h_bar
                log_eps_bar = (i + 1) ** (-kappa) * log_eps + \
                    (1.0 - (i + 1) ** (-kappa)) * log_eps_bar
                if i == self.n_warmup - 1:
                    log_eps = log_eps_bar
            else:
                out[i - self.n_warmup] = x
                late.append(accept_stat)

        self.step_size_ = float(np.exp(log_eps))
        self.accept_rate_ = float(np.mean(late)) if late else 0.0
        return out

    def _tree(self, x, p, g, lp, logu, H0, direction, depth) -> _Node:
        """Grow one subtree; `logu` is the log of the slice height.

        Energies are kept in Hamiltonian form `H = −log π + K`, so the slice test
        is `log u <= −H` and the acceptance ratio is `exp(H0 − H1)`.
        """
        if depth == 0:
            x1, p1, g1, lp1 = self._leapfrog(x, p, g, direction * self.step_size)
            H1 = -lp1 + self._kinetic(p1)
            n = int(logu <= -H1)
            s = int(logu < -H1 + 1000.0)
            alpha = min(1.0, float(np.exp(H0 - H1)))
            return _Node(x1, p1, g1, x1, p1, g1, lp1, lp1, n, s, alpha, 1)

        a = self._tree(x, p, g, lp, logu, H0, direction, depth - 1)
        if direction == 1:
            b = self._tree(a.xp, a.pp, a.gp, a.lp_p, logu, H0, 1, depth - 1)
            xm, pm, gm, lp_m = a.xm, a.pm, a.gm, a.lp_m
            xp, pp, gp, lp_p = b.xp, b.pp, b.gp, b.lp_p
        else:
            b = self._tree(a.xm, a.pm, a.gm, a.lp_m, logu, H0, -1, depth - 1)
            xm, pm, gm, lp_m = b.xm, b.pm, b.gm, b.lp_m
            xp, pp, gp, lp_p = a.xp, a.pp, a.gp, a.lp_p

        delta = xp - xm
        no_uturn = int(float(delta @ pm) >= 0.0 and float(delta @ pp) >= 0.0)
        return _Node(
            xm, pm, gm, xp, pp, gp, lp_m, lp_p,
            a.n + b.n, a.s * b.s * no_uturn,
            a.alpha + b.alpha, a.n_alpha + b.n_alpha,
        )


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _normal_log_prob(mean, sd) -> LogProb:
    """Gaussian target; the closed-form reference for every kernel below."""
    mean = _vec(mean)
    sd = _vec(sd)

    def log_prob(x: np.ndarray) -> float:
        z = (x - mean) / sd
        return float(-0.5 * float(z @ z) - float(np.sum(np.log(sd)))
                     - 0.5 * x.size * np.log(2 * np.pi))

    return log_prob


def _cauchy_log_prob() -> LogProb:
    return lambda x: float(-np.log1p(x[0] ** 2))


def main() -> None:
    rng = np.random.default_rng(0)

    # ---- metropolis_hastings: the moments are the stationarity test ----- #
    target = _normal_log_prob([0.0, 0.0], [1.0, 2.0])
    rw = MCMC(target, variant="metropolis_hastings", scale=2.0)
    chain = rw.sample(120_000, [0.0, 0.0], rng=rng)
    mu, sd = chain.mean(0), chain.std(0)
    assert np.allclose(mu, 0.0, atol=0.05), mu
    assert np.allclose(sd, [1.0, 2.0], atol=0.07), sd
    assert 0.2 < rw.accept_rate_ < 0.6, rw.accept_rate_
    print(f"mcmc: metropolis_hastings — 120k draws give mean {np.round(mu, 3)} and "
          f"sd {np.round(sd, 3)} for the exact (0, 0) / (1, 2), acceptance "
          f"{rw.accept_rate_:.2f}")

    # ---- gibbs: exact conditionals, so nothing is ever rejected --------- #
    rho, mean = 0.8, np.array([1.0, -1.0])
    cov = np.array([[1.0, rho], [rho, 1.0]])
    prec = np.linalg.inv(cov)

    def cond0(x, r):
        """x0 | x1 ~ N(μ0 − P01/P00 (x1 − μ1), 1/P00)."""
        m = mean[0] - prec[0, 1] / prec[0, 0] * (x[1] - mean[1])
        return float(m + np.sqrt(1.0 / prec[0, 0]) * r.standard_normal())

    def cond1(x, r):
        m = mean[1] - prec[1, 0] / prec[1, 1] * (x[0] - mean[0])
        return float(m + np.sqrt(1.0 / prec[1, 1]) * r.standard_normal())

    gibbs = MCMC(None, variant="gibbs", conditionals=[cond0, cond1])
    gc = gibbs.sample(60_000, [0.0, 0.0], rng=rng)
    emp = np.cov(gc.T)
    assert np.allclose(gc.mean(0), mean, atol=0.03), gc.mean(0)
    assert np.allclose(emp, cov, atol=0.03), emp
    assert gibbs.accept_rate_ == 1.0
    print(f"mcmc: gibbs — exact conditionals recover mean {np.round(gc.mean(0), 2)} "
          f"and ρ = {emp[0, 1]:.3f} (target {rho}); acceptance 1.00 by construction")

    # the same atom, but the conditionals are sampled instead of derived
    joint = lambda x: float(-0.5 * (x - mean) @ prec @ (x - mean))  # noqa: E731
    sg = MCMC(None, variant="gibbs", width=1.2, log_conditionals=[joint, joint])
    sgc = sg.sample(20_000, [0.0, 0.0], rng=rng)
    assert np.allclose(np.cov(sgc.T), cov, atol=0.08), np.cov(sgc.T)
    print(f"mcmc: gibbs+slice — no algebra, conditionals sliced: ρ recovered as "
          f"{np.corrcoef(sgc.T)[0, 1]:.3f}")

    # ---- hmc: conserved energy is what buys the distance ----------------- #
    scales = np.array([1.0, 10.0])
    hmc_target = _normal_log_prob(np.zeros(2), scales)

    def hmc_grad(x: np.ndarray) -> np.ndarray:
        return -x / scales**2

    fd = np.array([(hmc_target(np.eye(1, 2, j)[0] * 1e-6)
                    - hmc_target(-np.eye(1, 2, j)[0] * 1e-6)) / 2e-6 for j in range(2)])
    assert np.allclose(fd, hmc_grad(np.zeros(2)), atol=1e-4)

    hmc = MCMC(hmc_target, variant="hmc", grad=hmc_grad, n_leapfrog=8,
               step_size=0.5, mass=1.0 / scales**2)
    hc = hmc.sample(10_000, np.zeros(2), rng=rng)
    assert np.allclose(hc.std(0), scales, rtol=0.06), hc.std(0)
    assert hmc.energy_error_ < 0.2, hmc.energy_error_

    mh = MCMC(hmc_target, variant="metropolis_hastings", scale=2.38 / np.sqrt(2))
    mc = mh.sample(10_000, np.zeros(2), rng=rng)
    # the bottleneck coordinate is the one that sets how long you must run, so
    # compare the *worst* coordinate's effective samples per evaluation
    ess_hmc = float(min(ess(hc[:, j]) for j in range(2)))
    ess_mh = float(min(ess(mc[:, j]) for j in range(2)))
    gain = (ess_hmc / (hmc.n_eval_ + hmc.n_grad_)) / (ess_mh / mh.n_eval_)
    assert np.allclose(mc.std(0), scales, rtol=0.2), mc.std(0)
    assert gain > 3.0, gain
    print(f"mcmc: hmc — mean |ΔH| {hmc.energy_error_:.3f}, sd "
          f"{np.round(hc.std(0), 2)} against {scales}, {gain:.1f}× the random walk's "
          f"effective samples per evaluation on the slow coordinate")

    # ---- nuts: trajectory length chosen by u-turns ----------------------- #
    nuts = MCMC(hmc_target, variant="nuts", grad=hmc_grad, mass=1.0 / scales**2,
                step_size=0.5, n_warmup=1500, max_depth=6)
    nc = nuts.sample(8_000, np.zeros(2), rng=rng)
    assert np.allclose(nc.std(0), scales, rtol=0.1), nc.std(0)
    assert np.allclose(nc.mean(0) / scales, 0.0, atol=0.06), nc.mean(0)
    assert 1.0 <= float(np.mean(nuts.depth_)) <= 6.0
    assert 0.5 < nuts.accept_rate_ < 1.0
    print(f"mcmc: nuts — dual averaging found step {nuts.step_size_:.3f}, mean depth "
          f"{np.mean(nuts.depth_):.1f}, acceptance {nuts.accept_rate_:.2f}, sd "
          f"{np.round(nc.std(0), 2)} against {scales}")

    # ---- slice_sampling: a target whose mean does not exist -------------- #
    cauchy = MCMC(_cauchy_log_prob(), variant="slice_sampling", width=2.0)
    cc = cauchy.sample(20_000, [0.0], rng=rng)[:, 0]
    q25, q50, q75 = np.quantile(cc, [0.25, 0.5, 0.75])
    assert abs(q50) < 0.05, q50
    assert abs(q25 + 1.0) < 0.06 and abs(q75 - 1.0) < 0.06, (q25, q75)
    assert cauchy.accept_rate_ == 1.0
    print(f"mcmc: slice_sampling — Cauchy quartiles {q25:.2f} / {q50:.2f} / {q75:.2f} "
          f"against the exact −1 / 0 / 1, no tuning and no rejections")

    # ---- adaptive_metropolis: it finds 2.38/√d on its own ---------------- #
    adapt = MCMC(_normal_log_prob([0.0], [1.0]), variant="adaptive_metropolis",
                 scale=0.05, n_warmup=4000, target_accept=0.44)
    ac = adapt.sample(30_000, [0.0], rng=rng)
    assert np.allclose(ac.std(), 1.0, atol=0.05)
    assert 1.2 < adapt.scale_ < 5.0, adapt.scale_
    assert 0.3 < adapt.accept_rate_ < 0.65, adapt.accept_rate_
    trace = adapt.trace_["proposal_scale"]
    assert len(trace) == 4000
    print(f"mcmc: adaptive_metropolis — scale 0.05 → {adapt.scale_:.2f} "
          f"(theory 2.38 for a unit Gaussian), acceptance {adapt.accept_rate_:.2f}, "
          f"frozen once warm-up ends")

    # ---- diagnostics: three numbers before any posterior summary -------- #
    # identical chains: B = 0, so R-hat is sqrt(1 − 1/n), i.e. 1 to three decimals
    base = rng.standard_normal(2000)
    assert abs(rhat(np.vstack([base, base])) - np.sqrt(1 - 1 / 2000)) < 1e-12
    iid = rng.standard_normal((4, 5000))
    assert 0.85 * 20_000 < ess(iid) < 1.15 * 20_000, ess(iid)
    phi = 0.9
    ar = np.empty((4, 5000))
    ar[:, 0] = rng.standard_normal(4)
    noise = rng.standard_normal((4, 5000)) * np.sqrt(1 - phi**2)
    for t in range(1, 5000):
        ar[:, t] = phi * ar[:, t - 1] + noise[:, t]
    tau_exact = (1 + phi) / (1 - phi)
    assert abs(ess(ar) / (20_000 / tau_exact) - 1.0) < 0.3, ess(ar)
    stuck = np.vstack([np.full(2000, -6.0) + 0.01 * rng.standard_normal(2000),
                       np.full(2000, 6.0) + 0.01 * rng.standard_normal(2000)])
    assert rhat(stuck) > 1.3, rhat(stuck)
    rep = diagnose(stuck)
    assert rep["rhat"] > 1.3 and np.isfinite(rep["ess"])
    print(f"mcmc: diagnostics — identical chains R-hat {rhat(np.vstack([base, base])):.3f} "
          f"(the 1/n factor keeps it just under 1), iid ESS "
          f"{ess(iid):.0f}/20000, AR(1) ρ=0.9 ESS {ess(ar):.0f} "
          f"(exact {20_000 / tau_exact:.0f}), split-mode R-hat {rhat(stuck):.2f}")

    print("mcmc demo passed")


if __name__ == "__main__":
    main()
