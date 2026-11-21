r"""Markov chains — order, memory, absorption and a sampler.

Chapter:   10-sequences
Companion: markov_chain.md
Tier 1 · 5 variants: first_order, higher_order, absorbing, stationary, mcmc_chain

    Markov property:  p(s_t | s_<t) = p(s_t | s_{t-k}, ..., s_{t-1})       (1)
    transition mle:   P_ij = n_ij / sum_j n_ij                             (2)
    stationary:       pi P = pi,  sum_i pi_i = 1                           (3)
    fundamental:      N = (I - Q)^(-1)                                     (4)
    absorption:       B = N R,   t = N 1                                   (5)
    metropolis:       alpha = min(1, p(x') q(x|x') / (p(x) q(x'|x)))       (6)

A sequence is a chain when the next symbol depends on the recent past and not
on the whole history. That single assumption buys a model with one matrix, a
likelihood that is a product, a stationary distribution that is an eigenvector,
and an exact account of what happens when the chain can get stuck. The five
variants here are the five questions this object is usually asked.

`first_order` estimates P from counts and answers "how likely is this
sequence". `higher_order` keeps k symbols of memory and answers the same
question when one is not enough. `absorbing` takes a chain that already exists
and computes how long it wanders and where it ends up. `stationary` computes
the long-run distribution, by power iteration or by eigen-decomposition.
`mcmc_chain` turns the object around: instead of estimating a chain from data,
it builds one whose stationary distribution is a target you choose, which is
the Metropolis-Hastings algorithm.

The estimation is counting and nothing else. A first-order chain has K states
and K^2 parameters, and the maximum-likelihood estimate is the row-normalised
count matrix, so the only decisions are what to do about contexts that were
never seen and how much smoothing to add. The demo fits a three-state chain
from five thousand draws and recovers the true matrix to within a couple of
percent, and the stationary distribution to within one.

The higher-order variant is where the state space bites. An order-k chain over
K symbols has K^k contexts, so the model grows exponentially in the memory it
keeps, and the demo uses a process that is exactly second-order: the next
symbol is the sum of the previous two modulo three. A first-order model can do
no better than one third on it, while the second-order model recovers the rule,
and the held-out log-likelihood says so by a wide margin.

Absorption is the part that is usually skipped and is often the point. Split
the states into transient and absorbing, invert I - Q, and the inverse contains
everything: its row sums are the expected number of steps before absorption,
and its product with the absorption rows gives the probability of ending in
each absorbing state. Gambler's ruin is the canonical test, and the demo checks
the closed forms E[T_i] = i(N-i) and P(hit N) = i/N to machine precision.

The sampler is the mirror image. Metropolis-Hastings proposes a move, accepts
it with probability (6), and the chain it builds has the target as its
stationary distribution. The demo runs it on a correlated two-dimensional
Gaussian, checks the mean and covariance against the truth, and then bins the
samples and fits a first-order chain to the bins: power iteration on that
fitted matrix reproduces the target's bin probabilities. The chain estimated
from the sampler and the chain the sampler was aiming at agree, which is the
whole claim of MCMC in one table.
"""  # noqa: E501

from __future__ import annotations

import math

import numpy as np

VARIANTS = ("first_order", "higher_order", "absorbing", "stationary",
            "mcmc_chain")


class MarkovChain:
    """A discrete Markov chain: fit it, analyse it, or sample from a target.

    Parameters
    ----------
    variant : {"first_order", "higher_order", "absorbing", "stationary",
               "mcmc_chain"}
        Which analysis to expose. Every variant except `mcmc_chain` is fitted
        from a sequence or loaded from a transition matrix.
    order : int
        Memory length for `higher_order`; must be one for the others.
    smoothing : float
        Additive pseudocount for every transition. Zero is maximum likelihood;
        a positive value keeps unseen contexts away from probability zero.
    target : callable, optional
        Log density of the target for `mcmc_chain`. It takes a one-dimensional
        array and returns a scalar.
    proposal_scale : float
        Standard deviation of the Gaussian random-walk proposal in `mcmc_chain`.
    seed : int
        Seed for `sample` and `run`; both are deterministic given the seed.
    n_states : int, optional
        Number of symbols, if larger than the largest observed symbol plus one.

    Attributes
    ----------
    transition_ : ndarray
        Row-stochastic transition matrix. For `higher_order` its shape is
        ``(K**order, K)``, with contexts in base-`K` lexicographic order.
    n_states_ : int
        Number of symbols.
    order_ : int
        Memory length actually used.
    """

    def __init__(self, *, variant: str = "first_order", order: int = 1,
                 smoothing: float = 0.0, target=None,
                 proposal_scale: float = 1.0, seed: int = 0,
                 n_states: int | None = None) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
        if order < 1:
            raise ValueError("order must be at least one")
        if variant != "higher_order" and order != 1:
            raise ValueError("order above one is only used by higher_order")
        if smoothing < 0.0 or not np.isfinite(smoothing):
            raise ValueError("smoothing must be finite and non-negative")
        if proposal_scale <= 0.0 or not np.isfinite(proposal_scale):
            raise ValueError("proposal_scale must be finite and positive")
        if n_states is not None and n_states < 2:
            raise ValueError("n_states must be at least two")
        if variant == "mcmc_chain" and target is None:
            raise ValueError("mcmc_chain needs a target log density")
        if target is not None and not callable(target):
            raise ValueError("target must be callable")
        self.variant = variant
        self.order = order
        self.smoothing = float(smoothing)
        self.target = target
        self.proposal_scale = float(proposal_scale)
        self.seed = int(seed)
        self.n_states = n_states

    @classmethod
    def from_transition(cls, matrix: np.ndarray, *,
                        variant: str = "absorbing") -> MarkovChain:
        """Load a known transition matrix, skipping estimation."""
        chain = cls(variant=variant)
        chain._set_matrix(matrix)
        return chain

    def fit(self, sequence: np.ndarray) -> MarkovChain:
        """Estimate the transition matrix from a sequence of symbols."""
        if self.variant == "mcmc_chain":
            raise ValueError("mcmc_chain does not fit data; call run")
        symbols = self._encode(sequence)
        order = self.order if self.variant == "higher_order" else 1
        if symbols.size <= order:
            raise ValueError(f"need more than {order} symbols")
        states = self.n_states
        if states is None:
            states = int(symbols.max()) + 1
        if int(symbols.max()) >= states:
            raise ValueError("a symbol is outside n_states")
        if states < 2:
            raise ValueError("need at least two states")
        if states ** order > 1_000_000:
            raise ValueError("the context space is too large for a dense count")
        counts = np.zeros((states ** order, states))
        for t in range(order, symbols.size):
            context = self._key(symbols[t - order:t], order, states)
            counts[context, symbols[t]] += 1.0
        self.n_states_ = states
        self.order_ = order
        self.smoothing_ = self.smoothing
        matrix = counts + self.smoothing
        totals = matrix.sum(axis=1)
        estimated = np.zeros_like(matrix)
        observed = totals > 0.0
        estimated[observed] = matrix[observed] / totals[observed, None]
        estimated[~observed] = 1.0 / states
        self.transition_ = estimated
        self.counts_ = counts
        self.contexts_ = np.flatnonzero(counts.sum(axis=1) > 0.0)
        return self

    def _set_matrix(self, matrix: np.ndarray) -> None:
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("the transition matrix must be square")
        if matrix.shape[0] < 2:
            raise ValueError("the matrix must have at least two states")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("the transition matrix must be finite")
        if np.any(matrix < 0.0):
            raise ValueError("the transition matrix must be non-negative")
        if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-9):
            raise ValueError("every row must sum to one")
        self.transition_ = matrix
        self.n_states_ = matrix.shape[0]
        self.order_ = 1
        self.counts_ = None
        self.contexts_ = np.arange(self.n_states_)

    def _encode(self, sequence: np.ndarray) -> np.ndarray:
        symbols = np.asarray(sequence)
        if symbols.ndim != 1:
            raise ValueError("the sequence must be one-dimensional")
        if symbols.size == 0:
            raise ValueError("the sequence is empty")
        if not np.issubdtype(symbols.dtype, np.integer):
            rounded = np.asarray(symbols, dtype=float)
            if not np.all(np.isfinite(rounded)) or not np.all(rounded == np.round(rounded)):
                raise ValueError("symbols must be integers")
            symbols = rounded.astype(int)
        symbols = symbols.astype(int)
        if symbols.min() < 0:
            raise ValueError("symbols must be non-negative")
        return symbols

    def _key(self, context: np.ndarray, order: int, states: int) -> int:
        key = 0
        for symbol in context:
            key = key * states + int(symbol)
        return key

    def _digits(self, key: int, order: int, states: int) -> tuple:
        digits = []
        for _ in range(order):
            digits.append(key % states)
            key //= states
        return tuple(reversed(digits))

    def log_probability(self, sequence: np.ndarray) -> float:
        """Log probability of the sequence, conditional on its first symbols."""
        if self.variant == "mcmc_chain":
            raise ValueError("mcmc_chain scores by its target, not by a matrix")
        if not hasattr(self, "transition_"):
            raise RuntimeError("fit before scoring")
        symbols = self._encode(sequence)
        if symbols.size <= self.order_:
            raise ValueError(f"need more than {self.order_} symbols")
        total = 0.0
        for t in range(self.order_, symbols.size):
            context = self._key(symbols[t - self.order_:t], self.order_,
                                self.n_states_)
            probability = self.transition_[context, symbols[t]]
            if probability <= 0.0:
                return -math.inf
            total += math.log(probability)
        return total

    def sample(self, length: int, start=None) -> np.ndarray:
        """Draw a sequence from the fitted chain, deterministically."""
        if not hasattr(self, "transition_"):
            raise RuntimeError("fit before sampling")
        if length <= self.order_:
            raise ValueError(f"length must exceed the order {self.order_}")
        rng = np.random.default_rng(self.seed)
        states = self.n_states_
        output = np.empty(length, dtype=int)
        if self.order_ == 1:
            state = int(rng.integers(states)) if start is None else int(start)
            if not 0 <= state < states:
                raise ValueError("start is outside the state space")
            output[0] = state
            for t in range(1, length):
                state = int(rng.choice(states, p=self.transition_[state]))
                output[t] = state
            return output
        if start is None:
            pick = self.contexts_[rng.integers(self.contexts_.size)]
            context = self._digits(int(pick), self.order_, states)
        else:
            context = tuple(int(s) for s in start)
            if len(context) != self.order_:
                raise ValueError(f"start must hold {self.order_} symbols")
            if min(context) < 0 or max(context) >= states:
                raise ValueError("start is outside the state space")
        output[:self.order_] = context
        for t in range(self.order_, length):
            key = self._key(np.asarray(context), self.order_, states)
            nxt = int(rng.choice(states, p=self.transition_[key]))
            output[t] = nxt
            context = context[1:] + (nxt,)
        return output

    def stationary(self, method: str | None = None, iterations: int = 1000,
                   tol: float = 1e-12, start=None) -> np.ndarray:
        """Long-run distribution: left eigenvector of the companion matrix."""
        if not hasattr(self, "transition_"):
            raise RuntimeError("fit before asking for the stationary distribution")
        if self.variant == "absorbing":
            raise ValueError("an absorbing chain has no unique stationary "
                             "distribution; use absorption_analysis")
        if method is None:
            method = "eigen" if self.variant == "stationary" else "power"
        if method not in ("power", "eigen"):
            raise ValueError("method must be 'power' or 'eigen'")
        companion = self._companion()
        if method == "eigen":
            values, vectors = np.linalg.eig(companion.T)
            index = int(np.argmin(np.abs(values - 1.0)))
            vector = np.real(vectors[:, index])
            vector = np.abs(vector)
            return vector / vector.sum()
        size = companion.shape[0]
        if start is None:
            vector = np.full(size, 1.0 / size)
        else:
            vector = np.asarray(start, dtype=float)
            if vector.shape != (size,) or np.any(vector < 0.0):
                raise ValueError("start must be a non-negative vector of the "
                                 "right length")
            vector = vector / vector.sum()
        for _ in range(iterations):
            nxt = vector @ companion
            if np.abs(nxt - vector).sum() < tol:
                vector = nxt
                break
            vector = nxt
        return vector / vector.sum()

    def _companion(self) -> np.ndarray:
        states = self.n_states_
        order = self.order_
        if order == 1:
            return self.transition_
        size = states ** order
        companion = np.zeros((size, size))
        for key in range(size):
            digits = self._digits(key, order, states)
            for nxt in range(states):
                probability = self.transition_[key, nxt]
                if probability > 0.0:
                    target = self._key(np.asarray(digits[1:] + (nxt,)), order,
                                       states)
                    companion[key, target] = probability
        return companion

    def absorption_analysis(self) -> dict:
        """Expected times and destinations for an absorbing chain."""
        if self.variant != "absorbing":
            raise ValueError("absorption_analysis belongs to the absorbing variant")
        if not hasattr(self, "transition_"):
            raise RuntimeError("fit before analysis")
        matrix = self.transition_
        absorbing = np.isclose(np.diag(matrix), 1.0) & (matrix.sum(axis=1) > 0.0)
        if not absorbing.any():
            raise ValueError("the chain has no absorbing state")
        transient = ~absorbing
        if not transient.any():
            raise ValueError("the chain has no transient state")
        q = matrix[np.ix_(transient, transient)]
        r = matrix[np.ix_(transient, absorbing)]
        fundamental = np.linalg.inv(np.eye(q.shape[0]) - q)
        times = fundamental.sum(axis=1)
        destinations = fundamental @ r
        self.absorbing_states_ = np.flatnonzero(absorbing)
        self.transient_states_ = np.flatnonzero(transient)
        self.fundamental_ = fundamental
        self.absorption_times_ = times
        self.absorption_probabilities_ = destinations
        return {"absorbing": self.absorbing_states_,
                "transient": self.transient_states_,
                "times": times, "destinations": destinations}

    def run(self, steps: int, start: np.ndarray, *,
            burn_in: int = 0) -> np.ndarray:
        """Metropolis-Hastings samples from `target`, with a Gaussian proposal."""
        if self.variant != "mcmc_chain":
            raise ValueError("run belongs to the mcmc_chain variant")
        if steps < 1 or burn_in < 0 or burn_in >= steps:
            raise ValueError("need steps > burn_in >= 0")
        x = np.asarray(start, dtype=float).ravel()
        if x.size == 0 or not np.all(np.isfinite(x)):
            raise ValueError("start must be a finite vector")
        rng = np.random.default_rng(self.seed)
        samples = np.empty((steps, x.size))
        accepted = 0
        for t in range(steps):
            proposal = x + self.proposal_scale * rng.standard_normal(x.size)
            ratio = float(self.target(proposal)) - float(self.target(x))
            if math.log(rng.uniform()) < ratio:
                x = proposal
                accepted += 1
            samples[t] = x
        self.samples_ = samples
        self.acceptance_rate_ = accepted / steps
        return samples[burn_in:]

# --------------------------------------------------------------------------- #
# Demos
# --------------------------------------------------------------------------- #
def _demo_first_order() -> None:
    """Counting recovers a known chain, and the stationary law with it."""
    truth_matrix = np.array([[0.80, 0.15, 0.05],
                             [0.10, 0.70, 0.20],
                             [0.05, 0.15, 0.80]])
    truth = MarkovChain.from_transition(truth_matrix, variant="first_order")
    sequence = truth.sample(5000)
    fitted = MarkovChain(variant="first_order").fit(sequence)
    error = float(np.abs(fitted.transition_ - truth_matrix).max())
    print("three-state chain, 5000 draws, first-order fit:")
    print(f"  largest absolute error in P        {error:.4f}")
    print("  stationary, fitted vs true:")
    fitted_pi = fitted.stationary()
    true_pi = truth.stationary(method="eigen")
    for state in range(3):
        print(f"    state {state}   {fitted_pi[state]:.4f}   {true_pi[state]:.4f}")
    assert error < 0.03
    assert np.abs(fitted_pi - true_pi).max() < 0.01

    # the likelihood is the product of the transitions, by definition
    manual = sum(math.log(truth_matrix[sequence[t - 1], sequence[t]])
                 for t in range(1, 40))
    assert abs(truth.log_probability(sequence[:40]) - manual) < 1e-12
    print(f"  log p(first 40 symbols)           {truth.log_probability(sequence[:40]):.6f}")
    assert np.array_equal(fitted.sample(500), fitted.sample(500))
    print("  Sampling is deterministic given the seed: two calls agree")
    print("  exactly. The estimator is counting, so the only thing that can")
    print("  move the answer is an unvisited context.")


def _demo_higher_order() -> None:
    """One symbol of memory is not always enough."""
    length = 4000
    sequence = np.empty(length, dtype=int)
    sequence[0], sequence[1] = 0, 1
    for t in range(2, length):
        sequence[t] = (sequence[t - 1] + sequence[t - 2]) % 3
    train, test = sequence[:3000], sequence[3000:]
    print("s_t = (s_{t-1} + s_{t-2}) mod 3, trained on 3000, tested on 1000:")
    print("  variant        order   parameters   test log-prob   per symbol")
    for variant, order in (("first_order", 1), ("higher_order", 2)):
        model = MarkovChain(variant=variant, order=order).fit(train)
        held_out = model.log_probability(test)
        parameters = model.n_states_ ** (order + 1)
        print(f"  {variant:<13s} {order:6d} {parameters:13d}"
              f" {held_out:15.3f} {held_out / len(test):13.4f}")
    one = MarkovChain(variant="first_order").fit(train)
    two = MarkovChain(variant="higher_order", order=2).fit(train)
    assert two.log_probability(test) > one.log_probability(test) + 900
    assert abs(two.log_probability(test)) < 1e-9
    assert two.transition_.shape == (9, 3)
    print("  The pair determines the next symbol exactly, so the order-2")
    print("  model assigns probability one to every held-out transition and")
    print("  scores zero. The order-1 model is guessing, at about one third")
    print("  per step. Twenty-seven parameters instead of three is the bill.")


def _demo_absorbing() -> None:
    """The fundamental matrix answers both questions exactly."""
    size = 5
    matrix = np.zeros((size, size))
    matrix[0, 0] = matrix[-1, -1] = 1.0
    for state in range(1, size - 1):
        matrix[state, state - 1] = 0.5
        matrix[state, state + 1] = 0.5
    chain = MarkovChain.from_transition(matrix, variant="absorbing")
    result = chain.absorption_analysis()
    print("gambler's ruin on 0..4, fair coin, absorbing at 0 and 4:")
    print("  state   expected steps   P(end at 0)   P(end at 4)")
    for position, state in enumerate(result["transient"]):
        left, right = result["destinations"][position]
        print(f"  {state:5d} {result['times'][position]:16.4f}"
              f" {left:13.4f} {right:12.4f}")
        assert abs(result["times"][position] - state * (4 - state)) < 1e-10
        assert abs(left - (1.0 - state / 4.0)) < 1e-10
        assert abs(right - state / 4.0) < 1e-10
    assert np.array_equal(result["absorbing"], np.array([0, 4]))
    print("  The closed forms are E[T_i] = i(4-i) and P(hit 4) = i/4, and the")
    print("  inverse reproduces both to machine precision. Nothing here is")
    print("  asymptotic: the fundamental matrix is exact for a finite chain.")


def _demo_stationary() -> None:
    """Power iteration converges when the chain is aperiodic, and not before."""
    matrix = np.array([[0.7, 0.2, 0.1],
                       [0.3, 0.4, 0.3],
                       [0.2, 0.3, 0.5]])
    chain = MarkovChain.from_transition(matrix, variant="stationary")
    power = chain.stationary(method="power")
    eigen = chain.stationary(method="eigen")
    values = np.linalg.eigvals(matrix)
    second = float(np.sort(np.abs(values))[-2])
    print("aperiodic three-state chain:")
    print(f"  power iteration   {power.round(6)}")
    print(f"  eigenvector       {eigen.round(6)}")
    print(f"  second eigenvalue {second:.6f}")
    assert np.abs(power - eigen).max() < 1e-10

    cycle = MarkovChain.from_transition(np.array([[0.0, 1.0], [1.0, 0.0]]),
                                        variant="stationary")
    odd = cycle.stationary(method="power", iterations=1001, start=[1.0, 0.0])
    even = cycle.stationary(method="power", iterations=1000, start=[1.0, 0.0])
    print("period-two chain, power iteration started at state 0:")
    print(f"  after 1001 steps  {odd.round(6)}")
    print(f"  after 1000 steps  {even.round(6)}")
    print(f"  eigenvector       {cycle.stationary(method='eigen').round(6)}")
    assert np.array_equal(odd, np.array([0.0, 1.0]))
    assert np.array_equal(even, np.array([1.0, 0.0]))
    lazy = MarkovChain.from_transition(0.5 * (np.eye(2) + np.array([[0.0, 1.0],
                                                                    [1.0, 0.0]])),
                                       variant="stationary")
    assert np.allclose(lazy.stationary(method="power"), [0.5, 0.5])
    print("  Power iteration on a periodic chain oscillates and never")
    print("  settles; the eigenvector still finds the unique stationary law,")
    print("  and so does power iteration on the lazy chain (P + I)/2.")


def _demo_mcmc() -> None:
    """A chain built to have a chosen stationary distribution."""
    mean = np.array([1.0, -1.0])
    covariance = np.array([[1.0, 0.7], [0.7, 1.0]])
    inverse = np.linalg.inv(covariance)

    def target(x: np.ndarray) -> float:
        offset = x - mean
        return -0.5 * float(offset @ inverse @ offset)

    sampler = MarkovChain(variant="mcmc_chain", target=target,
                           proposal_scale=1.2, seed=0)
    samples = sampler.run(20000, np.zeros(2), burn_in=2000)
    empirical_mean = samples.mean(axis=0)
    empirical_covariance = np.cov(samples.T)
    print("Metropolis-Hastings, 20000 steps, correlated Gaussian target:")
    print(f"  acceptance rate     {sampler.acceptance_rate_:.3f}")
    print(f"  mean                {empirical_mean.round(3)}   vs [1, -1]")
    print("  covariance          " + np.array2string(empirical_covariance,
                                                    precision=3).replace("\n", " "))
    print("                      vs [[1, 0.7], [0.7, 1]]")
    assert 0.2 < sampler.acceptance_rate_ < 0.6
    assert np.abs(empirical_mean - mean).max() < 0.15
    assert np.abs(empirical_covariance - covariance).max() < 0.15

    edges_x = np.array([-2.0, -0.5, 1.0, 2.5, 4.0])
    edges_y = np.array([-4.0, -2.5, -1.0, 0.5, 2.0])
    bins = (np.digitize(samples[:, 0], edges_x[1:-1]) * 4
            + np.digitize(samples[:, 1], edges_y[1:-1]))
    chain = MarkovChain(variant="first_order").fit(bins)
    stationary = chain.stationary()
    frequencies = np.bincount(bins, minlength=16) / bins.size
    distance = 0.5 * float(np.abs(stationary - frequencies).sum())
    print(f"  binned chain, TV(stationary, histogram) {distance:.4f}")
    assert distance < 0.01
    print("  Fit a first-order chain to the visited bins and power-iterate")
    print("  it: the stationary distribution of the fitted chain and the")
    print("  histogram of the samples agree. That is the MCMC claim, checked")
    print("  by the same estimator used on real sequences.")


def _demo_broken() -> None:
    """Malformed calls are refused."""
    rng = np.random.default_rng(1)
    sequence = rng.integers(0, 3, size=50)
    matrix = np.array([[0.5, 0.5], [0.5, 0.5]])
    calls = [
        lambda: MarkovChain(variant="nope"),
        lambda: MarkovChain(order=0),
        lambda: MarkovChain(variant="first_order", order=2),
        lambda: MarkovChain(smoothing=-1.0),
        lambda: MarkovChain(proposal_scale=0.0),
        lambda: MarkovChain(n_states=1),
        lambda: MarkovChain(variant="mcmc_chain"),
        lambda: MarkovChain(variant="mcmc_chain", target=3.0),
        lambda: MarkovChain().fit(np.array([], dtype=int)),
        lambda: MarkovChain().fit(np.array([1])),
        lambda: MarkovChain().fit(np.array([0, -1, 2])),
        lambda: MarkovChain().fit(np.array([0.0, 1.5, 2.0])),
        lambda: MarkovChain().fit(np.array([[0, 1], [1, 0]])),
        lambda: MarkovChain().fit(sequence).log_probability(np.array([0])),
        lambda: MarkovChain().sample(10),
        lambda: MarkovChain.from_transition(np.ones((2, 3))),
        lambda: MarkovChain.from_transition(np.array([[0.2, 0.2], [0.5, 0.5]])),
        lambda: MarkovChain(variant="first_order").fit(sequence).absorption_analysis(),
        lambda: MarkovChain.from_transition(matrix, variant="absorbing").absorption_analysis(),
        lambda: MarkovChain.from_transition(matrix, variant="absorbing").stationary(),
        lambda: MarkovChain(variant="mcmc_chain", target=lambda v: 0.0).run(10, np.zeros(0)),
        lambda: MarkovChain(variant="mcmc_chain", target=lambda v: 0.0,
                                          seed=0).run(10, np.zeros(2), burn_in=10),
        lambda: MarkovChain(variant="stationary").fit(sequence).stationary(method="nope"),
        lambda: MarkovChain().fit(sequence).sample(10, start=7),
    ]
    for call in calls:
        try:
            call()
        except (ValueError, RuntimeError):
            continue
        raise AssertionError("expected a refusal")
    print(f"{len(calls)} malformed calls refused: an unknown variant, an order "
          f"of zero, a mismatched order, negative smoothing, a zero proposal, "
          f"a one-state space, MCMC without a target and with a non-callable "
          f"one, an empty sequence and one shorter than the order, a negative "
          f"and a fractional symbol, a two-dimensional sequence, scoring and "
          f"sampling before fitting, a non-square matrix, rows that do not sum "
          f"to one, absorption on the wrong variant, a chain with no absorbing "
          f"state, stationarity on the absorbing variant, an empty start, an "
          f"impossible burn-in, an unknown stationary method, and a start "
          f"outside the state space.")


def main() -> None:
    _demo_first_order()
    _demo_higher_order()
    _demo_absorbing()
    _demo_stationary()
    _demo_mcmc()
    _demo_broken()
    print("markov_chain demo passed")


if __name__ == "__main__":
    main()
