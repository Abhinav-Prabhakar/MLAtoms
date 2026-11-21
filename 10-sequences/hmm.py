r"""Hidden Markov models — the chain you cannot see, and what it emits.

Chapter:   10-sequences
Companion: hmm.md
Tier 1 · 7 variants: discrete, gaussian, gmm_hmm, forward_backward, viterbi,
baum_welch, scaling

    forward:     alpha_t(k) = b_k(x_t) sum_j alpha_{t-1}(j) A_jk              (1)
    backward:    beta_t(j)  = sum_k A_jk b_k(x_{t+1}) beta_{t+1}(k)          (2)
    posterior:   gamma_t(k) = alpha_t(k) beta_t(k) / p(x)                    (3)
    pairwise:    xi_t(j,k)  = alpha_t(j) A_jk b_k(x_{t+1}) beta_{t+1}(k) / p(x)
    m-step:      pi_k = gamma_1(k),  A_jk = sum_t xi_t(j,k) / sum_t gamma_t(j)  (4)
    viterbi:     delta_t(k) = b_k(x_t) max_j delta_{t-1}(j) A_jk             (5)
    scaling:     alpha_t <- alpha_t / c_t,   log p(x) = sum_t log c_t        (6)

A hidden Markov model is a Markov chain you observe through a noisy channel.
The state sequence is latent, the observations are conditionally independent
given it, and the whole object is three sets of numbers: where the chain starts,
how it moves, and what each state tends to emit. Fitting those numbers from
observations alone is the expectation-maximisation algorithm known here as
Baum-Welch, and inference in the fitted model is two matrix recursions.

The seven variants slice that story along two axes. The first three choose the
emission model: symbols (`discrete`), a diagonal Gaussian per state (`gaussian`),
or a mixture of Gaussians per state (`gmm_hmm`). The last four choose which
algorithm to put on the surface: exact posterior marginals (`forward_backward`),
maximum-probability paths (`viterbi`), multi-restart training with a likelihood
trace (`baum_welch`), or the linear-domain scaled recursions that were the only
practical option before log-space arithmetic (`scaling`). Either axis can be
combined with the other, because none of the algorithms cares what the emission
model is.

The estimator is EM, and it has one property worth stating up front: every
iteration increases the likelihood, or leaves it alone, and never decreases it.
The demo prints the trace of a fit and checks it is monotone. What EM does not
guarantee is the global maximum, so the number of restarts matters as much as
the number of iterations, and `baum_welch` is the variant that makes the
restarts explicit.

Three exact checks keep the implementation honest. On a short sequence the
forward algorithm agrees with brute-force enumeration over all state paths. On a
short sequence Viterbi agrees with the best of the same enumerated paths. And
the scaled recursions of (6) agree with the log-space recursions to machine
precision while the unscaled ones underflow to zero. Those are the three claims
in the file that could silently be false.

The demos fit all three emission models to simulated data and compare against
the generating parameters after resolving the label permutation, which is the
one thing EM cannot recover: state one is whatever you call it.
"""  # noqa: E501

from __future__ import annotations

import itertools
import math

import numpy as np

VARIANTS = ("discrete", "gaussian", "gmm_hmm", "forward_backward", "viterbi",
            "baum_welch", "scaling")
ANALYSIS_VARIANTS = ("forward_backward", "viterbi", "baum_welch", "scaling")
VARIANCE_FLOOR = 1e-6


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    """Stable log of the sum of exponentials, along one axis."""
    peak = np.max(a, axis=axis, keepdims=True)
    peak = np.where(np.isfinite(peak), peak, 0.0)
    total = np.sum(np.exp(a - peak), axis=axis, keepdims=True)
    out = peak + np.log(np.maximum(total, 1e-300))
    return np.squeeze(out, axis=axis)


def _normalise_rows(matrix: np.ndarray) -> np.ndarray:
    """Row-normalise, falling back to uniform for a row with no mass."""
    totals = matrix.sum(axis=1)
    out = np.full_like(matrix, 1.0 / matrix.shape[1])
    live = totals > 0.0
    out[live] = matrix[live] / totals[live, None]
    return out


class HMM:
    """A hidden Markov model: fit by Baum-Welch, decode by Viterbi.

    Parameters
    ----------
    variant : str
        One of `VARIANTS`. The first three name the emission model; the last
        four name the algorithm put on the surface.
    n_states : int
        Number of hidden states.
    n_components : int
        Mixture components per state, for `gmm_hmm` only.
    n_iter : int
        Maximum number of EM iterations.
    tol : float
        Stop when the likelihood improves by less than `tol` relatively.
    n_init : int
        Number of random restarts, keeping the best. Above one only for
        `baum_welch`.
    seed : int
        Seed for the initialisation and therefore for the whole fit.

    Attributes
    ----------
    startprob_ : ndarray of shape (n_states,)
        Initial state distribution.
    transmat_ : ndarray of shape (n_states, n_states)
        Transition matrix, row-stochastic.
    emission_ : ndarray
        Symbol probabilities, shape ``(n_states, n_symbols)``, for `discrete`.
    means_, variances_ : ndarray of shape (n_states, n_features)
        Emission parameters for `gaussian`.
    weights_, means_, variances_ : ndarray
        Shapes ``(n_states, n_components)``, ``(n_states, n_components,
        n_features)`` for `gmm_hmm`.
    history_ : list of float
        Log-likelihood after each EM step. Non-decreasing.
    """

    def __init__(self, *, variant: str = "discrete", n_states: int = 2,
                 n_components: int = 2, n_iter: int = 100, tol: float = 1e-6,
                 n_init: int = 1, seed: int = 0) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
        if n_states < 2:
            raise ValueError("n_states must be at least two")
        if n_components < 1:
            raise ValueError("n_components must be at least one")
        if n_iter < 1:
            raise ValueError("n_iter must be at least one")
        if not np.isfinite(tol) or tol < 0.0:
            raise ValueError("tol must be finite and non-negative")
        if n_init < 1:
            raise ValueError("n_init must be at least one")
        if n_init > 1 and variant != "baum_welch":
            raise ValueError("multiple restarts belong to the baum_welch variant")
        self.variant = variant
        self.n_states = n_states
        self.n_components = n_components
        self.n_iter = n_iter
        self.tol = float(tol)
        self.n_init = n_init
        self.seed = int(seed)
        self.emission = self._resolve_emission()

    def _resolve_emission(self) -> str:
        if self.variant == "discrete":
            return "categorical"
        if self.variant == "gaussian":
            return "gaussian"
        if self.variant == "gmm_hmm":
            return "gmm"
        return "unknown"

    # ------------------------------------------------------------------ #
    # data handling
    # ------------------------------------------------------------------ #
    def _prepare(self, x: np.ndarray) -> tuple:
        """Coerce the input to symbols or measurements and pick the emission."""
        values = np.asarray(x)
        if values.ndim == 0:
            raise ValueError("the sequence must be one- or two-dimensional")
        if values.size == 0:
            raise ValueError("the sequence is empty")
        if not np.issubdtype(values.dtype, np.floating) and not np.issubdtype(
                values.dtype, np.integer):
            raise ValueError("the sequence must hold numbers")
        if values.ndim > 2:
            raise ValueError("the sequence must be one- or two-dimensional")
        integer_like = np.issubdtype(values.dtype, np.integer)
        if not integer_like:
            real = values.astype(float)
            if not np.all(np.isfinite(real)):
                raise ValueError("the sequence must be finite")
            integer_like = bool(np.all(real == np.round(real)))
        if self.emission == "unknown":
            emission = "categorical" if (integer_like and values.ndim == 1) else "gaussian"
        else:
            emission = self.emission
        if emission == "categorical":
            if values.ndim != 1:
                raise ValueError("symbols must be one-dimensional")
            if not integer_like:
                raise ValueError("symbols must be integers")
            symbols = values.astype(int)
            if symbols.min() < 0:
                raise ValueError("symbols must be non-negative")
            return symbols, "categorical", int(symbols.max()) + 1
        real = values.astype(float)
        if not np.all(np.isfinite(real)):
            raise ValueError("the sequence must be finite")
        if real.ndim == 1:
            real = real[:, None]
        if real.shape[1] < 1:
            raise ValueError("the sequence needs at least one feature")
        return real, emission, real.shape[1]

    # ------------------------------------------------------------------ #
    # emission densities
    # ------------------------------------------------------------------ #
    def _emission_log_prob(self, x: np.ndarray) -> np.ndarray:
        """Log emission probability of every observation in every state."""
        if self.emission_type_ == "categorical":
            with np.errstate(divide="ignore"):
                table = np.log(self.emission_)
            return table[:, x].T
        if self.emission_type_ == "gaussian":
            return self._gaussian_log_prob(x, self.means_, self.variances_,
                                           axis=1)
        components = self._gaussian_log_prob(x, self.means_, self.variances_,
                                             axis=2)
        with np.errstate(divide="ignore"):
            weighted = components + np.log(self.weights_)[None, :, :]
        return _logsumexp(weighted, axis=2)

    @staticmethod
    def _gaussian_log_prob(x: np.ndarray, means: np.ndarray,
                           variances: np.ndarray, axis: int) -> np.ndarray:
        """Diagonal Gaussian log densities, broadcast over a state axis."""
        diff = x.reshape((x.shape[0],) + (1,) * axis + (x.shape[1],)) - means
        with np.errstate(divide="ignore"):
            log_norm = -0.5 * np.log(2.0 * np.pi * variances)
        return np.sum(log_norm - 0.5 * diff * diff / variances, axis=-1)

    # ------------------------------------------------------------------ #
    # inference
    # ------------------------------------------------------------------ #
    def _forward_backward(self, x: np.ndarray,
                          scaled: bool) -> tuple:
        """Posterior marginals by (1)-(3), in log space or with scaling."""
        log_b = self._emission_log_prob(x)
        with np.errstate(divide="ignore"):
            log_pi = np.log(self.startprob_)
            log_a = np.log(self.transmat_)
        if scaled:
            alpha, factors = self._scaled_forward(log_b, np.exp(log_pi),
                                                  np.exp(log_a))
            likelihood = float(np.sum(np.log(np.maximum(factors, 1e-300))))
            beta = self._scaled_backward(log_b, np.exp(log_a), factors)
            gamma = alpha * beta
            gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)
            step = (alpha[:-1][:, :, None] * np.exp(log_a)[None, :, :]
                    * np.exp(log_b[1:])[:, None, :] * beta[1:][:, None, :])
            xi = step / np.maximum(factors[1:], 1e-300)[:, None, None]
            self.scaling_factors_ = factors if self.variant == "scaling" else None
            return likelihood, gamma, xi
        log_alpha = np.empty((x.shape[0], self.n_states))
        log_alpha[0] = log_pi + log_b[0]
        for t in range(1, x.shape[0]):
            log_alpha[t] = log_b[t] + _logsumexp(log_alpha[t - 1][:, None] + log_a,
                                                 axis=0)
        log_beta = np.zeros((x.shape[0], self.n_states))
        for t in range(x.shape[0] - 2, -1, -1):
            log_beta[t] = _logsumexp(log_a + (log_b[t + 1] + log_beta[t + 1])[None, :],
                                     axis=1)
        likelihood = float(_logsumexp(log_alpha[-1], axis=0))
        log_gamma = log_alpha + log_beta - likelihood
        gamma = np.exp(log_gamma)
        future = log_b[1:] + log_beta[1:]
        log_xi = (log_alpha[:-1][:, :, None] + log_a[None, :, :]
                  + future[:, None, :] - likelihood)
        xi = np.exp(log_xi)
        self.scaling_factors_ = None
        return likelihood, gamma, xi

    def _scaled_forward(self, log_b: np.ndarray, pi: np.ndarray,
                        a: np.ndarray) -> tuple:
        """Forward recursion in the linear domain, one factor per step."""
        length = log_b.shape[0]
        alpha = np.empty((length, self.n_states))
        factors = np.empty(length)
        alpha[0] = pi * np.exp(log_b[0])
        factors[0] = max(alpha[0].sum(), 1e-300)
        alpha[0] /= factors[0]
        for t in range(1, length):
            alpha[t] = (alpha[t - 1] @ a) * np.exp(log_b[t])
            factors[t] = max(alpha[t].sum(), 1e-300)
            alpha[t] /= factors[t]
        return alpha, factors

    def _scaled_backward(self, log_b: np.ndarray, a: np.ndarray,
                         factors: np.ndarray) -> np.ndarray:
        """Backward recursion matching the scaled forward pass."""
        length = log_b.shape[0]
        beta = np.ones((length, self.n_states))
        for t in range(length - 2, -1, -1):
            beta[t] = (a @ (np.exp(log_b[t + 1]) * beta[t + 1])
                       / max(factors[t + 1], 1e-300))
        return beta

    def score(self, x: np.ndarray) -> float:
        """Log probability of the sequence under the fitted model."""
        if not hasattr(self, "transmat_"):
            raise RuntimeError("fit before scoring")
        values, _, _ = self._prepare(x)
        log_b = self._emission_log_prob(values)
        with np.errstate(divide="ignore"):
            log_alpha = np.log(self.startprob_) + log_b[0]
            log_a = np.log(self.transmat_)
        for t in range(1, values.shape[0]):
            log_alpha = log_b[t] + _logsumexp(log_alpha[:, None] + log_a, axis=0)
        return float(_logsumexp(log_alpha, axis=0))

    def posterior_marginals(self, x: np.ndarray) -> np.ndarray:
        """Posterior probability of each state at each time, equation (3)."""
        self._require("forward_backward", "posterior marginals")
        values, _, _ = self._prepare(x)
        _, gamma, _ = self._forward_backward(values, scaled=False)
        self.posterior_ = gamma
        return gamma

    def pairwise_posteriors(self, x: np.ndarray) -> np.ndarray:
        """Posterior probability of each transition, shape (T-1, K, K)."""
        self._require("forward_backward", "pairwise posteriors")
        values, _, _ = self._prepare(x)
        _, _, xi = self._forward_backward(values, scaled=False)
        self.pairwise_ = xi
        return xi

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Most probable state sequence, by the max-product recursion (5)."""
        self._require("viterbi", "decoding")
        values, _, _ = self._prepare(x)
        log_b = self._emission_log_prob(values)
        with np.errstate(divide="ignore"):
            log_pi = np.log(self.startprob_)
            log_a = np.log(self.transmat_)
        length = values.shape[0]
        delta = np.empty((length, self.n_states))
        back = np.zeros((length, self.n_states), dtype=int)
        delta[0] = log_pi + log_b[0]
        for t in range(1, length):
            candidate = delta[t - 1][:, None] + log_a
            back[t] = np.argmax(candidate, axis=0)
            delta[t] = log_b[t] + candidate[back[t], np.arange(self.n_states)]
        path = np.empty(length, dtype=int)
        path[-1] = int(np.argmax(delta[-1]))
        self.viterbi_score_ = float(delta[-1, path[-1]])
        for t in range(length - 2, -1, -1):
            path[t] = back[t + 1, path[t + 1]]
        self.state_sequence_ = path
        return path

    def _require(self, variant: str, what: str) -> None:
        if self.variant != variant:
            raise ValueError(f"{what} belongs to the {variant} variant")
        if not hasattr(self, "transmat_"):
            raise RuntimeError("fit before using the model")

    # ------------------------------------------------------------------ #
    # estimation
    # ------------------------------------------------------------------ #
    def fit(self, x: np.ndarray) -> HMM:
        """Fit by Baum-Welch from a seeded random start."""
        values, emission, size = self._prepare(x)
        self.emission = emission
        self.emission_type_ = emission
        best = None
        for restart in range(self.n_init):
            rng = np.random.default_rng(self.seed + restart)
            self._initialise(values, emission, size, rng)
            model = self._em(values, emission)
            if best is None or model[0] > best[0]:
                best = model
        likelihood, history, iterations = best
        self.log_likelihood_ = likelihood
        self.history_ = history
        self.n_iter_ = iterations
        self.n_states_ = self.n_states
        self.n_features_ = size if emission != "categorical" else None
        self.n_symbols_ = size if emission == "categorical" else None
        return self

    def _initialise(self, values: np.ndarray, emission: str, size: int,
                    rng: np.random.Generator) -> None:
        """Deterministic spread-out start: sticky chain, ordered means."""
        states = self.n_states
        self.startprob_ = np.full(states, 1.0 / states)
        self.startprob_ += 0.05 * rng.standard_normal(states)
        self.startprob_ = np.maximum(self.startprob_, 0.01)
        self.startprob_ /= self.startprob_.sum()
        sticky = np.full((states, states), 0.2 / states)
        sticky[np.diag_indices(states)] += 0.8
        self.transmat_ = sticky
        if emission == "categorical":
            marginal = np.bincount(values, minlength=size) / values.size
            prior = np.maximum(0.2, 8.0 * marginal)
            self.emission_ = rng.dirichlet(prior, size=states)
            return
        scores = self._principal_scores(values)
        order = np.argsort(scores, kind="stable")
        spread = np.maximum(np.var(values, axis=0), 1e-3)
        jitter = 0.3 * np.sqrt(spread)
        if emission == "gaussian":
            self.means_ = self._chunk_means(values, order, states)
            self.means_ += jitter * rng.standard_normal(self.means_.shape)
            self.means_ = self.means_[rng.permutation(states)]
            self.variances_ = np.tile(spread, (states, 1))
            return
        self.means_ = np.empty((states, self.n_components, size))
        self.weights_ = np.full((states, self.n_components),
                                1.0 / self.n_components)
        for state, chunk in enumerate(np.array_split(order, states)):
            self.means_[state] = self._chunk_means(values, chunk,
                                                   self.n_components)
        self.means_ += jitter * rng.standard_normal(self.means_.shape)
        self.means_ = self.means_[rng.permutation(states)]
        self.variances_ = np.tile(spread, (states, self.n_components, 1))

    @staticmethod
    def _principal_scores(values: np.ndarray) -> np.ndarray:
        """Project onto the first principal axis to order the sample."""
        centred = values - values.mean(axis=0)
        _, _, axes = np.linalg.svd(centred, full_matrices=False)
        return centred @ axes[0]

    @staticmethod
    def _chunk_means(values: np.ndarray, order: np.ndarray,
                     groups: int) -> np.ndarray:
        """Means of equal-frequency chunks along the ordering."""
        chunks = np.array_split(order, groups)
        return np.array([values[chunk].mean(axis=0) for chunk in chunks])

    def _em(self, values: np.ndarray, emission: str) -> tuple:
        history = []
        scaled = self.variant == "scaling"
        likelihood = -math.inf
        iterations = 0
        for _ in range(self.n_iter):
            likelihood, gamma, xi = self._forward_backward(values, scaled=scaled)
            history.append(likelihood)
            iterations += 1
            if len(history) > 1:
                gain = history[-1] - history[-2]
                if abs(gain) <= self.tol * max(1.0, abs(history[-2])):
                    break
            self._m_step(values, emission, gamma, xi)
        likelihood, _, _ = self._forward_backward(values, scaled=scaled)
        history.append(likelihood)
        return likelihood, history, iterations

    def _m_step(self, values: np.ndarray, emission: str, gamma: np.ndarray,
                xi: np.ndarray) -> None:
        self.startprob_ = np.maximum(gamma[0], 1e-300)
        self.startprob_ /= self.startprob_.sum()
        self.transmat_ = _normalise_rows(np.maximum(xi.sum(axis=0), 1e-300))
        mass = gamma.sum(axis=0)
        if emission == "categorical":
            counts = np.zeros_like(self.emission_)
            for symbol in np.unique(values):
                counts[:, symbol] = gamma[values == symbol].sum(axis=0)
            self.emission_ = _normalise_rows(np.maximum(counts, 1e-300))
            return
        global_var = float(np.var(values, axis=0).mean())
        floor = VARIANCE_FLOOR * max(global_var, 1e-12)
        if emission == "gaussian":
            weights = gamma[:, :, None]
            total = np.maximum(mass, 1e-300)[:, None]
            self.means_ = (weights * values[:, None, :]).sum(axis=0) / total
            diff = values[:, None, :] - self.means_[None, :, :]
            self.variances_ = np.maximum((weights * diff * diff).sum(axis=0) / total,
                                         floor)
            return
        log_b = self._emission_log_prob(values)
        components = self._gaussian_log_prob(values, self.means_, self.variances_,
                                             axis=2)
        with np.errstate(divide="ignore"):
            weighted = components + np.log(self.weights_)[None, :, :]
        log_resp = weighted - log_b[:, :, None]
        resp = np.exp(log_resp) * gamma[:, :, None]
        totals = np.maximum(resp.sum(axis=0), 1e-300)
        self.weights_ = totals / np.maximum(totals.sum(axis=1, keepdims=True), 1e-300)
        means = (resp[:, :, :, None] * values[:, None, None, :]).sum(axis=0)
        self.means_ = means / totals[:, :, None]
        diff = values[:, None, None, :] - self.means_[None, :, :, :]
        self.variances_ = np.maximum(
            (resp[:, :, :, None] * diff * diff).sum(axis=0) / totals[:, :, None],
            floor)

# --------------------------------------------------------------------------- #
# Demos
# --------------------------------------------------------------------------- #
def _simulate_gaussian(means: np.ndarray, deviations: np.ndarray,
                       transmat: np.ndarray, length: int, seed: int,
                       start: int = 0) -> tuple:
    """Draw states from a chain, then observations from their Gaussians."""
    rng = np.random.default_rng(seed)
    states = np.empty(length, dtype=int)
    states[0] = start
    for t in range(1, length):
        states[t] = rng.choice(means.size, p=transmat[states[t - 1]])
    noise = rng.standard_normal(length)
    return (means[states] + deviations[states] * noise)[:, None], states


def _best_permutation(estimate: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Relabel estimated states to agree with the truth, up to permutation."""
    best, order = np.inf, None
    for permutation in itertools.permutations(range(truth.shape[0])):
        error = np.abs(estimate[list(permutation)] - truth).max()
        if error < best:
            best, order = error, permutation
    return estimate[list(order)]


def _demo_discrete() -> None:
    """Symbols, a sticky chain, and the forward recursion checked by hand."""
    rng = np.random.default_rng(0)
    transmat = np.array([[0.90, 0.10], [0.10, 0.90]])
    emissions = np.array([[0.30, 0.20, 0.20, 0.30],
                          [0.10, 0.40, 0.40, 0.10]])
    length = 3000
    states = np.empty(length, dtype=int)
    states[0] = 0
    for t in range(1, length):
        states[t] = rng.choice(2, p=transmat[states[t - 1]])
    symbols = np.array([rng.choice(4, p=emissions[state]) for state in states])

    model = HMM(variant="discrete", n_states=2, n_iter=400, seed=0).fit(symbols)
    fitted_emissions = _best_permutation(model.emission_, emissions)
    fitted_transitions = _best_permutation(model.transmat_, transmat)
    print("two-state CpG-island chain, 3000 symbols, discrete fit:")
    print(f"  log-likelihood                 {model.log_likelihood_:.1f}")
    print(f"  EM iterations                  {model.n_iter_}")
    print(f"  largest emission error         {np.abs(fitted_emissions - emissions).max():.4f}")
    print(f"  largest transition error       {np.abs(fitted_transitions - transmat).max():.4f}")
    print("  fitted emission probabilities, state with more G and C first:")
    print(f"    {fitted_emissions[1].round(3)}   (true {emissions[1]})")
    print(f"    {fitted_emissions[0].round(3)}   (true {emissions[0]})")
    assert np.abs(fitted_emissions - emissions).max() < 0.04
    assert np.abs(fitted_transitions - transmat).max() < 0.05
    assert model.history_[-1] >= model.history_[0]

    segment = symbols[:7]
    brute = 0.0
    for path in itertools.product(range(2), repeat=segment.size):
        probability = (model.startprob_[path[0]]
                       * model.emission_[path[0], segment[0]])
        for t in range(1, segment.size):
            probability *= (model.transmat_[path[t - 1], path[t]]
                            * model.emission_[path[t], segment[t]])
        brute += probability
    exact = model.score(segment)
    print(f"  forward log p(first 7 symbols) {exact:.9f}")
    print(f"  enumerated over 128 paths      {np.log(brute):.9f}")
    assert abs(exact - np.log(brute)) < 1e-10
    print("  The forward recursion and the sum over all 128 state paths agree")
    print("  to ten digits, which is the only proof of (1) that matters.")

    held_out = np.array([rng.choice(4, p=emissions[state]) for state in states])
    print(f"  per-symbol score on a fresh path {model.score(held_out) / length:.4f}")
    print("  Counting is the estimator; EM only decides which state is which,")
    print("  and that labelling is the one thing it cannot recover.")


def _demo_gaussian() -> None:
    """Continuous emissions, and a fit that beats the generating parameters."""
    transmat = np.array([[0.93, 0.07], [0.10, 0.90]])
    observations, states = _simulate_gaussian(np.array([-1.5, 1.5]),
                                              np.array([0.7, 0.7]),
                                              transmat, 2500, seed=0)
    model = HMM(variant="gaussian", n_states=2, n_iter=300, seed=0).fit(observations)
    order = np.argsort(model.means_.ravel())
    means = model.means_.ravel()[order]
    deviations = np.sqrt(model.variances_.ravel())[order]
    transitions = model.transmat_[np.ix_(order, order)]
    print("two-state Gaussian HMM, 2500 observations:")
    print(f"  log-likelihood       {model.log_likelihood_:.1f}")
    print(f"  EM iterations        {model.n_iter_}")
    print(f"  means                {means.round(3)}   (true [-1.5, 1.5])")
    print(f"  deviations           {deviations.round(3)}   (true [0.7, 0.7])")
    print(f"  transitions          {transitions.round(3).tolist()}")
    assert np.abs(means - np.array([-1.5, 1.5])).max() < 0.06
    assert np.abs(deviations - 0.7).max() < 0.04
    assert np.abs(transitions - transmat).max() < 0.05
    scaled = HMM(variant="scaling", n_states=2, n_iter=300, seed=0).fit(observations)
    print(f"  scaling and log-space likelihoods differ by "
          f"{abs(scaled.log_likelihood_ - model.log_likelihood_):.2e}")
    assert abs(scaled.log_likelihood_ - model.log_likelihood_) < 1e-8
    print("  Starting from equal-frequency chunks of the sample, EM finds the")
    print("  two components in a handful of iterations. The emission families")
    print("  are the only thing that changed from the discrete fit.")


def _demo_gmm() -> None:
    """One Gaussian per state is a model choice, not a fact."""
    rng = np.random.default_rng(0)
    transmat = np.array([[0.90, 0.10], [0.12, 0.88]])
    length = 2000
    states = np.empty(length, dtype=int)
    states[0] = 0
    for t in range(1, length):
        states[t] = rng.choice(2, p=transmat[states[t - 1]])
    means = np.array([[[-4.0], [-2.0]], [[2.0], [4.0]]])
    component = rng.integers(0, 2, size=length)
    observations = (means[states, component]
                    + 0.5 * rng.standard_normal((length, 1)))
    model = HMM(variant="gmm_hmm", n_states=2, n_components=2, n_iter=400,
                tol=1e-8, seed=0).fit(observations)
    order = np.argsort(model.means_[:, 0, 0])
    component_order = np.argsort(model.means_[order][:, :, 0], axis=1)
    fitted = np.take_along_axis(model.means_[order], component_order[:, :, None],
                                axis=1).squeeze()
    weights = np.take_along_axis(model.weights_[order], component_order, axis=1)
    true = np.array([[-4.0, -2.0], [2.0, 4.0]])
    print("four clusters, two states, two mixture components each:")
    print(f"  log-likelihood          {model.log_likelihood_:.1f}")
    print(f"  EM iterations           {model.n_iter_}")
    for state in range(2):
        print(f"    state {state} means     {fitted[state].round(2)}   weights "
              f"{weights[state].round(2)}")
    assert np.abs(fitted - true).max() < 0.1
    assert np.abs(weights - 0.5).max() < 0.08
    single = HMM(variant="gaussian", n_states=2, n_iter=400, seed=0).fit(
        observations)
    improvement = model.log_likelihood_ - single.log_likelihood_
    print(f"  single-Gaussian fit     {single.log_likelihood_:.1f}   "
          f"improvement {improvement:.1f}")
    assert improvement > 300.0
    print("  The single-Gaussian fit puts one state at each end and calls the")
    print("  spread a large variance; the mixture spends its components on the")
    print("  four clusters and gains hundreds of nats. The states were never")
    print("  labelled, only the emission models differed.")


def _demo_viterbi() -> None:
    """Exact decoding, checked against exhaustive search."""
    transmat = np.array([[0.90, 0.10], [0.10, 0.90]])
    observations, states = _simulate_gaussian(np.array([-1.5, 1.5]),
                                              np.array([0.9, 0.9]),
                                              transmat, 1200, seed=0)
    model = HMM(variant="viterbi", n_states=2, n_iter=200, seed=0).fit(observations)
    path = model.predict(observations)
    accuracy = max(float(np.mean(path == states)), float(np.mean(path != states)))
    print("two-state Gaussian HMM, 1200 observations, Viterbi decoding:")
    print(f"  state accuracy          {accuracy:.4f}")
    print(f"  log probability of path {model.viterbi_score_:.1f}")

    segment = observations[:10]
    log_b = (-0.5 * np.log(2.0 * np.pi * model.variances_.ravel())
             - 0.5 * (segment - model.means_.ravel()) ** 2
             / model.variances_.ravel())
    best_score, best_path = -np.inf, None
    for candidate in itertools.product(range(2), repeat=10):
        score = np.log(model.startprob_[candidate[0]]) + log_b[0, candidate[0]]
        for t in range(1, 10):
            score += (np.log(model.transmat_[candidate[t - 1], candidate[t]])
                      + log_b[t, candidate[t]])
        if score > best_score:
            best_score, best_path = score, candidate
    decoded = model.predict(segment)
    print(f"  best of 1024 paths      {list(best_path)}")
    print(f"  Viterbi path            {decoded.tolist()}")
    assert tuple(decoded) == best_path
    assert abs(model.viterbi_score_ - best_score) < 1e-10

    smoothing = HMM(variant="forward_backward", n_states=2, n_iter=200,
                    seed=0).fit(observations)
    posterior = smoothing.posterior_marginals(observations).argmax(axis=1)
    decoded_full = model.predict(observations)
    differences = int(np.sum(posterior != decoded_full))
    print(f"  positions where the max-marginal path differs from Viterbi "
          f"{differences}")
    print("  Viterbi maximises the joint probability of a path; posterior")
    print("  decoding maximises each position separately and can therefore")
    print("  return a path that is not a path at all. The two agree only when")
    print("  the posteriors are decided, and here they nearly are.")


def _demo_baum_welch() -> None:
    """The likelihood is monotone; the number of states is not free."""
    transmat = np.array([[0.92, 0.05, 0.03],
                         [0.05, 0.90, 0.05],
                         [0.03, 0.07, 0.90]])
    observations, _ = _simulate_gaussian(np.array([-2.0, 0.0, 2.0]),
                                         np.full(3, 0.8), transmat, 1200, seed=0)
    held_out, _ = _simulate_gaussian(np.array([-2.0, 0.0, 2.0]),
                                     np.full(3, 0.8), transmat, 1200, seed=7)
    print("three well-separated states, 1200 observations, four restarts:")
    print("  states   train ll   held-out ll   iterations   restarts")
    results = {}
    for states in (3, 4):
        best = None
        finals = []
        for seed in range(4):
            candidate = HMM(variant="baum_welch", n_states=states, n_iter=200,
                            seed=seed).fit(observations)
            finals.append(candidate.log_likelihood_)
            if best is None or candidate.log_likelihood_ > best.log_likelihood_:
                best = candidate
        assert np.all(np.diff(best.history_) >= -1e-8)
        results[states] = (best, float(np.ptp(finals)))
        print(f"  {states:6d} {best.log_likelihood_:10.1f}"
              f" {best.score(held_out):13.1f} {best.n_iter_:12d}"
              f" {np.ptp(finals):11.2f}")
    assert results[3][1] < 0.01
    assert results[4][1] > 0.5
    assert (results[4][0].log_likelihood_
            > results[3][0].log_likelihood_)
    assert results[4][0].score(held_out) < results[3][0].score(held_out)
    print("  With the right number of states every restart lands in the same")
    print("  place. With one state too many the training likelihood rises by a")
    print("  few nats, the restarts disagree by more than that, and the held-out")
    print("  likelihood gets worse. The trace itself never goes down: that is")
    print("  the one thing EM guarantees.")


def _demo_scaling() -> None:
    """Why the linear recursions need a factor per step."""
    transmat = np.array([[0.90, 0.10], [0.10, 0.90]])
    observations, states = _simulate_gaussian(np.array([-1.5, 1.5]),
                                              np.array([0.7, 0.7]),
                                              transmat, 2000, seed=1)
    model = HMM(variant="scaling", n_states=2, n_iter=200, seed=0).fit(observations)
    factors = model.scaling_factors_
    log_b = (-0.5 * np.log(2.0 * np.pi * model.variances_.ravel())
             - 0.5 * (observations - model.means_.ravel()) ** 2
             / model.variances_.ravel())
    alpha = model.startprob_ * np.exp(log_b[0])
    for t in range(1, observations.shape[0]):
        alpha = (alpha @ model.transmat_) * np.exp(log_b[t])
    print("two-state Gaussian HMM, 2000 observations, scaled recursions:")
    print(f"  log-likelihood                 {model.log_likelihood_:.1f}")
    print(f"  sum of log scaling factors     {np.log(factors).sum():.1f}")
    print(f"  smallest and largest factor    {factors.min():.2e}  "
          f"{factors.max():.3f}")
    print(f"  unscaled forward pass          {alpha.sum():.1e}")
    print(f"  log-space score                {model.score(observations):.1f}")
    assert abs(np.log(factors).sum() - model.log_likelihood_) < 1e-9
    assert alpha.sum() == 0.0
    assert np.isfinite(model.score(observations))
    print("  The unscaled product is smaller than the smallest double, so the")
    print("  plain recursion returns exactly zero. One factor per step keeps")
    print("  every number near one, and the sum of their logs is the")
    print("  likelihood: that is all the scaling trick is.")


def _demo_broken() -> None:
    """Malformed calls are refused."""
    rng = np.random.default_rng(2)
    values = rng.standard_normal((60, 1))
    calls = [
        lambda: HMM(variant="nope"),
        lambda: HMM(n_states=1),
        lambda: HMM(n_components=0),
        lambda: HMM(n_iter=0),
        lambda: HMM(tol=-1.0),
        lambda: HMM(n_init=0),
        lambda: HMM(variant="gaussian", n_init=2),
        lambda: HMM().fit(np.array([], dtype=int)),
        lambda: HMM().fit(np.array(3)),
        lambda: HMM().fit(np.zeros((4, 4, 4))),
        lambda: HMM(variant="discrete").fit(np.array([0.0, 1.5, 2.0])),
        lambda: HMM(variant="discrete").fit(np.zeros((10, 2), dtype=int)),
        lambda: HMM(variant="discrete").fit(np.array([0, -1, 2])),
        lambda: HMM(variant="gaussian").fit(np.array([0.0, np.nan, 1.0])),
        lambda: HMM(variant="gaussian").predict(values),
        lambda: HMM(variant="viterbi").posterior_marginals(values),
        lambda: HMM(variant="forward_backward").predict(values),
        lambda: HMM(variant="viterbi").fit(values).predict(values[:0]),
        lambda: HMM(variant="viterbi").predict(values),
        lambda: HMM(variant="gaussian").score(values),
    ]
    for call in calls:
        try:
            call()
        except (ValueError, RuntimeError):
            continue
        raise AssertionError("expected a refusal")
    print(f"{len(calls)} malformed calls refused: an unknown variant, a single "
          f"state, zero components, zero iterations, a negative tolerance, "
          f"zero restarts, restarts outside the baum_welch variant, an empty "
          f"and a scalar sequence, a three-dimensional array, fractional "
          f"symbols, a two-dimensional symbol table, a negative symbol, a "
          f"non-finite observation, decoding on the Gaussian and "
          f"forward_backward variants, posterior marginals on the viterbi "
          f"variant, decoding an empty sequence, and decoding or scoring "
          f"before fitting.")


def main() -> None:
    _demo_discrete()
    _demo_gaussian()
    _demo_gmm()
    _demo_viterbi()
    _demo_baum_welch()
    _demo_scaling()
    _demo_broken()
    print("hmm demo passed")


if __name__ == "__main__":
    main()
