# Markov chains — order, memory, absorption and a sampler

`markov_chain.py` · Sequences · Tier 1 · 5 variants

## Intuition

A sequence is a chain when the next symbol depends on the recent past and not on
the whole history. That is one assumption, and it is astonishing how much
follows: the model becomes one square matrix, the likelihood becomes a product
of matrix entries, the long-run behaviour becomes an eigenvector, and the
question "how long until this gets stuck" becomes a matrix inverse.

The five variants are the five questions this object is usually asked.

*How likely is this sequence?* Count the transitions, normalise the rows, and
multiply. That is `first_order`, and the estimator is nothing but counting — the
maximum-likelihood estimate has a closed form, so the only real decisions are
what to do with contexts that were never seen.

*What if one symbol of memory is not enough?* Keep $k$ of them. The state space
becomes $K^k$ contexts instead of $K$ states, and the same counting works, but
the cost grows exponentially in the memory. That is `higher_order`, and the demo
makes the bill explicit: an exactly second-order process is predicted perfectly
by an order-2 model and guessed at one third per symbol by an order-1 model.

*How long does it wander, and where does it end?* Split the states into
transient and absorbing, and invert $I-Q$. The inverse's row sums are expected
absorption times and its product with the absorption rows gives the
probabilities of each destination. That is `absorbing`, and gambler's ruin is
the canonical test: on $0..4$ the expected duration from $i$ is $i(4-i)$ and the
chance of ending at $4$ is $i/4$, both reproduced to machine precision.

*Where does it settle?* The stationary distribution solves $\pi P=\pi$, so it is
a left eigenvector. That is `stationary`, and the demo shows both the
convergence of power iteration and the case where it refuses to converge: on the
period-two chain $0\to1\to0$ power iteration alternates forever, while the
eigenvector finds $(\tfrac12,\tfrac12)$ and so does power iteration on the lazy
chain $(P+I)/2$.

*Can I build a chain with a distribution I choose?* Turn the object around.
Propose a move, accept it with probability (6), and the chain you get has your
target as its stationary law. That is `mcmc_chain`, and the demo checks it in
the least flattering way available: it runs Metropolis–Hastings, bins the
samples, fits a first-order chain to the bins, and power-iterates the fitted
matrix back to the histogram it came from. The agreement is $0.0003$ in total
variation.

```
      0.80       ┌─────────┐
     ┌──────────►│ state 0 │◄──┐
     │           └─────────┘   │ 0.05
     ▼ 0.15                 0.05│
  ┌─────────┐  0.20   ┌─────────┐
  │ state 1 │◄───────►│ state 2 │
  └─────────┘  0.15   └─────────┘
       ▲ 0.70              0.80
       └──────── state 1 ────────┘
```

Rows sum to one, so the matrix is a set of conditional distributions; the
diagonal is the sticky part and the off-diagonal is the drift. Five thousand
draws recover every entry to within $0.021$ and the stationary law to within
$0.004$.

## Variants

| variant | object | fit | main methods |
|:--|:--|:--|:--|
| `first_order` | $K\times K$ matrix | row-normalised counts | `log_probability`, `sample`, `stationary` |
| `higher_order` | $K^k\times K$ matrix | counts over $k$-symbol contexts | same, with $k$-symbol `start` |
| `absorbing` | transient block $Q$ | loaded or counted | `absorption_analysis` |
| `stationary` | left eigenvector | power iteration or `eig` | `stationary(method=...)` |
| `mcmc_chain` | proposal + target | Metropolis–Hastings | `run(steps, start, burn_in)` |

**`first_order`** is the default and the base case: one symbol of memory,
maximum-likelihood counting, and a likelihood that factorises. Everything else
in this file is a variation on it.

**`higher_order`** is the same estimator with the state redefined. A sequence’s
context is its last $k$ symbols, encoded in base $K$ so that a context is an
integer and the count table stays a matrix. It is the honest way to say "the
process has memory longer than one step" without inventing a hidden state.

**`absorbing`** is an analysis rather than a model: you supply a chain that has
absorbing states and it tells you the expected time to absorption and the
destination probabilities. It refuses chains with no absorbing state and chains
with no transient state, because both extreme cases make the block decomposition
trivial and misleading.

**`stationary`** is also an analysis: the same chain, asked about its long-run
behaviour. Two methods, because they fail differently — power iteration is
cheap and reveals periodicity by oscillating, while the eigen-decomposition
always returns the eigenvector but cannot tell you whether iteration would have
found it.

**`mcmc_chain`** inverts the direction of inference. Every other variant
estimates $P$ from data and derives $\pi$; this one starts from a target
$\pi\propto e^{-U}$ and constructs a $P$ that leaves it invariant. It is the
only variant whose "fit" is a simulation.

## Derivation

### The chain and its likelihood

An order-$k$ chain satisfies

$$
\Pr(s_t \mid s_{<t}) = \Pr(s_t \mid s_{t-k},\dots,s_{t-1}),
\qquad (1)
$$

for every $t>k$. Everything downstream is a consequence of that factorisation:
the joint probability of a sequence is the product of its conditional
probabilities, so the log-likelihood is a sum over transitions and separates
into independent rows.

Writing $n_{ij}$ for the number of times context $i$ was followed by symbol $j$,
the maximum-likelihood estimate is the row-normalised count,

$$
\hat P_{ij} = \frac{n_{ij}}{\sum_{j'} n_{ij'}},
\qquad (2)
$$

which is just the empirical conditional distribution. Additive smoothing
replaces $n_{ij}$ by $n_{ij}+\alpha$ before normalising; $\alpha=0$ is maximum
likelihood and $\alpha>0$ keeps an unseen transition from carrying probability
zero, at the cost of biasing every seen one.

The implementation is the loop in (2) and nothing else:

```python
        for t in range(order, symbols.size):
            context = self._key(symbols[t - order:t], order, states)
            counts[context, symbols[t]] += 1.0
```

`_key` is a base-$K$ positional encoding of the context, so a context is an
integer and `counts` is a dense matrix. Two things can go wrong: a context that
was never observed has a zero row, and a sequence shorter than the order has no
transitions at all. The first is handled by the fallback below; the second is
refused.

```python
        matrix = counts + self.smoothing
        totals = matrix.sum(axis=1)
        estimated = np.zeros_like(matrix)
        observed = totals > 0.0
        estimated[observed] = matrix[observed] / totals[observed, None]
        estimated[~observed] = 1.0 / states
```

The last line is a modelling decision that maximum likelihood cannot make:
an unobserved context has no information, so the fitted chain gives it the
uniform distribution rather than an arbitrary one. Without it the row would be
all zeros and the log-likelihood of any sequence passing through that context
would be $-\infty$.

### Stationarity

A distribution $\pi$ is stationary when one step leaves it unchanged,

$$
\pi P = \pi, \qquad \sum_i \pi_i = 1, \qquad \pi \ge 0,
\qquad (3)
$$

so $\pi$ is a left eigenvector of $P$ with eigenvalue one. Existence is
guaranteed for a finite chain — the all-ones vector is a right eigenvector, so
one is always an eigenvalue — but uniqueness needs irreducibility, and
convergence of the iterates needs aperiodicity.

Power iteration is (3) applied repeatedly from an arbitrary start: $\pi_{t+1} =
\pi_t P$. Its error shrinks like $|\lambda_2|^t$, where $\lambda_2$ is the
second-largest eigenvalue modulus, so the demo's aperiodic chain with
$|\lambda_2| = 0.473$ converges to twelve digits in about forty steps. A
periodic chain has $|\lambda_2|=1$ and the iteration never settles.

```python
        for _ in range(iterations):
            nxt = vector @ companion
            if np.abs(nxt - vector).sum() < tol:
                vector = nxt
                break
            vector = nxt
```

For an order-$k$ chain the object being iterated is not $P$ but the companion
matrix on $K^k$ contexts, whose entry from context $i$ to context $j$ is nonzero
only when $j$ is $i$ shifted by one symbol plus a new symbol. Building it is
$O(K^{k+1})$ and sparse in structure, but dense in storage — which is exactly
why high-order chains are rarely used beyond order three or four.

The eigen route is shorter and less informative:

```python
            values, vectors = np.linalg.eig(companion.T)
            index = int(np.argmin(np.abs(values - 1.0)))
            vector = np.real(vectors[:, index])
            vector = np.abs(vector)
            return vector / vector.sum()
```

The `np.abs` is not cosmetic. The eigenvector is real up to numerical error and
its overall sign is arbitrary, so a raw eigenvector can come back all-negative;
taking the modulus and renormalising makes the answer a distribution regardless.
The `argmin` picks the eigenvalue closest to one, which is the right choice
numerically — comparing to exactly $1.0$ can fail on a matrix whose eigenvalues
are computed as $0.9999999997$.

### Absorption

Suppose the states are ordered so that the $t$ transient ones come first. The
transition matrix is block triangular,

$$
P = \begin{pmatrix} Q & R \\ 0 & I \end{pmatrix},
$$

where $Q$ is transient-to-transient, $R$ is transient-to-absorbing, and the
absorbing block is the identity. The probability of being in transient state $j$
after $n$ steps, having started at $i$, is $(Q^n)_{ij}$, so the expected number
of visits to $j$ is $\sum_n (Q^n)_{ij}$, and the sum of the geometric series is
the fundamental matrix

$$
N = (I - Q)^{-1} = I + Q + Q^2 + \cdots
\qquad (4)
$$

Two consequences follow immediately:

$$
\mathbf{t} = N\mathbf{1}, \qquad B = N R,
\qquad (5)
$$

where $t_i$ is the expected number of steps before absorption starting from $i$,
and $B_{ik}$ is the probability of being absorbed in state $k$. Both are exact
for a finite chain — no asymptotics, no simulation.

```python
        q = matrix[np.ix_(transient, transient)]
        r = matrix[np.ix_(transient, absorbing)]
        fundamental = np.linalg.inv(np.eye(q.shape[0]) - q)
        times = fundamental.sum(axis=1)
        destinations = fundamental @ r
```

`np.ix_` is doing real work here: the transient and absorbing index sets are
interleaved in general, so slicing needs the cross product, not two independent
slices. On gambler's ruin the two rows come out as $(3, 4, 3)$ steps and

| start | expected steps | $P(\text{end at }0)$ | $P(\text{end at }4)$ |
|--:|--:|--:|--:|
| 1 | 3 | 0.75 | 0.25 |
| 2 | 4 | 0.50 | 0.50 |
| 3 | 3 | 0.25 | 0.75 |

which are $i(4-i)$ and $(1-i/4,\,i/4)$ exactly.

### Metropolis–Hastings

The sampler is the same equation read backwards. A proposal distribution
$q(x'\mid x)$ is easy to sample from; the target $p$ is only known up to a
constant. Propose $x'\sim q(\cdot\mid x)$ and accept it with probability

$$
\alpha = \min\!\left(1,\ \frac{p(x')\,q(x\mid x')}{p(x)\,q(x'\mid x)}\right).
\qquad (6)
$$

The ratio is what makes the unknown normalising constant cancel. The chain on
states has $p$ as its stationary distribution: detailed balance holds by
construction, $p(x)q(x'\mid x)\alpha = p(x')q(x\mid x')\alpha'$, and detailed
balance plus irreducibility implies (3). With a symmetric proposal the $q$ terms
cancel too and the rule is just $\min(1, p(x')/p(x))$, which is the random-walk
Metropolis algorithm.

```python
            proposal = x + self.proposal_scale * rng.standard_normal(x.size)
            ratio = float(self.target(proposal)) - float(self.target(x))
            if math.log(rng.uniform()) < ratio:
                x = proposal
                accepted += 1
```

The comparison is done in log space with the same decision rule as
`log(u) < log α`, because probabilities of interest routinely underflow. The
proposal scale is the entire tuning problem: too small and the chain crawls with
near-certain acceptance, too large and it bounces off the target with
near-certain rejection. The demo's correlated two-dimensional Gaussian accepts
$38\%$ of proposals at scale $1.2$, which is close to the rule of thumb that a
random-walk Metropolis in $d$ dimensions wants an acceptance rate near $0.234$
for large $d$ and considerably higher for small $d$.

## Code walkthrough

### Two ways to sample a chain

```python
            for t in range(1, length):
                state = int(rng.choice(states, p=self.transition_[state]))
                output[t] = state
```

Sampling is one `rng.choice` per step, and the state variable is the whole
memory of an order-1 chain. The order-$k$ branch keeps a tuple of the last $k$
symbols and slides it, which is the same loop with a wider state.

### The companion matrix

```python
                    target = self._key(np.asarray(digits[1:] + (nxt,)), order,
                                       states)
                    companion[key, target] = probability
```

Shifting the window is the only structure an order-$k$ chain has: from context
$(a,b)$ you can only reach $(b,c)$, and the probability of doing so is the
matrix entry for the new symbol. This is why the companion matrix has exactly
$K^{k+1}$ nonzeros out of $K^{2k}$ entries — at order ten over a three-symbol
alphabet that is $0.005\%$ dense, and dense storage is the wrong data structure
long before that point.

### Reading a chain out of a sampler

```python
    bins = (np.digitize(samples[:, 0], edges_x[1:-1]) * 4
            + np.digitize(samples[:, 1], edges_y[1:-1]))
    chain = MarkovChain(variant="first_order").fit(bins)
    stationary = chain.stationary()
    frequencies = np.bincount(bins, minlength=16) / bins.size
```

The demo's last check is the one worth remembering. Binning turns a continuous
sampler into a finite chain; fitting that chain by counting gives an estimate of
its transition matrix; power iteration gives that estimate's stationary
distribution. Comparing it to the histogram is a comparison of two things
derived from the same samples, so agreement is not evidence about the target —
but it *is* evidence that the two halves of this file, estimation and analysis,
are consistent. The total-variation distance is $0.0003$, which is what
consistency looks like.

## Complexity

Let $K$ be the number of symbols, $k$ the order, $n$ the sequence length, $T$
the number of sampler steps and $d$ the dimension of the target.

| operation | time | memory |
|:--|:--|:--|
| fit, order $k$ | $O(n)$ counting | $O(K^{k+1})$ |
| `log_probability` | $O(nk)$ | $O(1)$ |
| `sample` | $O(nK)$ expected | $O(n)$ |
| `stationary`, power | $O(I\,K^{2k})$ | $O(K^{2k})$ |
| `stationary`, eigen | $O(K^{3k})$ | $O(K^{2k})$ |
| `absorption_analysis` | $O(t^3)$ | $O(t^2)$ |
| `run` | $O(Td)$ | $O(Td)$ |

The estimation is linear and trivial. Everything expensive is in the state
space: the companion matrix is $K^k\times K^k$, so both stationary methods are
exponential in the memory length, and absorption is cubic in the number of
transient states because it inverts a matrix. The sampler is linear in the
number of steps, since each step costs one target evaluation.

## Failure modes

**Contexts you never saw.** With $K^k$ contexts and $n$ transitions, most
contexts are empty unless $n\gg K^k$. The default fills them with the uniform
distribution, which is defensible for a short sequence and dangerous for a long
one: a genuine deterministic transition that happened to be missing from the
training sample is assigned probability $1/K$. The `smoothing` parameter is the
knob, and the diagnostic is the fraction of contexts with zero counts.

**Order selection is a likelihood comparison with a price.** The demo's second
order gets $0$ on held-out data while the first order gets $-996$, but it also
has three times the parameters. That gap is enormous because the process is
exactly second-order; on real data the improvement is usually small enough that
the number of parameters matters, and the honest comparison is held-out
log-likelihood or BIC, not training likelihood, which always prefers more
memory.

**Periodicity stops power iteration, not stationarity.** The period-two chain
has a perfectly good stationary distribution, $(\tfrac12,\tfrac12)$, and power
iteration started at state $0$ visits only one state and oscillates forever
between the two basis vectors. This is a property of the iteration, not of the
chain, and the fix is structural: make the chain lazy, $P\to(P+I)/2$, which
costs nothing but keeps every state reachable at every step. The eigen route
sidesteps it entirely and therefore hides it, which is the argument for having
both.

**Reducibility makes the stationary distribution non-unique.** A chain with two
absorbing states and no path between them has infinitely many stationary
distributions — every mixture of the two. `eig` will return one of them, chosen
by numerical accident, and the answer is not wrong so much as undefined. That is
why `absorption_analysis` refuses a chain with no transient state and why
`stationary` refuses the `absorbing` variant outright rather than return a
number that would be mistaken for an answer.

**Proposal scale is the whole sampler.** Too small and the acceptance rate
approaches one while the chain explores a tiny neighbourhood; too large and the
acceptance rate approaches zero while the chain stands still. Both failures look
identical in a trace plot of one coordinate — a slowly mixing chain — and both
are fixed by looking at the acceptance rate and the autocorrelation, not at the
histogram. In several dimensions a single scalar scale is also the wrong
parameterisation: a correlated target needs a proposal shaped like the
covariance.

**Burn-in is a heuristic, and so is one chain.** The demo discards two thousand
steps from a target that starts at its mean, which is the easy case. A chain
started far out needs a burn-in long enough to reach the typical set, and no
diagnostic computed from a single chain can detect that it did not. The honest
practice is several chains from dispersed starts and a comparison of their
answers; the effective sample size, not the number of steps, is the budget.

**MCMC is exact in the limit and approximate at every finite step.** The samples
are correlated, so the Monte Carlo error shrinks like $\sqrt{\mathrm{ESS}}$, not
$\sqrt{T}$, and the demo's covariance entries land within $0.012$ of the truth
after twenty thousand steps. That is fine for a two-dimensional Gaussian; a
hundred-dimensional target with a badly chosen proposal can need orders of
magnitude more, and no amount of computing fixes a proposal that is structurally
wrong.

## Applications

**Language modelling and text generation.** The $n$-gram model of chapter 10 is
exactly the `higher_order` variant with $k \in \{1,2,3\}$ and a smoothing
scheme; the stationary distribution is a crude model of what the generator
drifts toward, and the kneser-ney atom in this chapter is the same counting
problem with a better backoff.

**PageRank and web graphs.** A random surfer is a Markov chain on pages, and the
PageRank vector is its stationary distribution; the damping factor is precisely
a lazy-chain correction that guarantees aperiodicity and irreducibility.
Everything in the `stationary` section applies, including the second-eigenvalue
convergence rate, which for the web is famously near one.

**Queues, inventory and reliability.** States are system conditions and
transitions are events; the stationary distribution is the long-run fraction of
time in each condition, and the transient absorption analysis is the answer to
"probability this system ever reaches total failure". Both are exact and both
are standard in operations research.

**Bayesian computation.** Metropolis–Hastings, Gibbs sampling and Hamiltonian
Monte Carlo are all chains built to have a posterior as their stationary
distribution. The acceptance ratio here is the simplest instance; the machinery
of diagnostics, effective sample size and convergence checks is the same in all
of them.

**Hidden Markov models.** The `hmm` atom in this chapter is a Markov chain on
latent states plus an emission model; forward–backward is the chain's filtering
recursion and Baum–Welch is the counting estimator of (2) applied to expected
counts instead of observed ones.

**Reinforcement learning and control.** A policy induces a Markov chain on
states, and its value function solves a linear system whose matrix is built from
the same transition matrix; the absorption analysis is the undiscounted special
case of policy evaluation for an episodic task.

## References

- Markov, A. A. (1906). Extension of the law of large numbers to dependent
  quantities. *Izvestiya Fiziko-Matematicheskogo Obshchestva pri Kazanskom
  Universitete* 15, 135–156. The original chain.
- Kemeny, J. G. & Snell, J. L. (1960). *Finite Markov Chains*. Van Nostrand.
  The fundamental matrix, equations (4) and (5), and absorption.
- Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H. & Teller,
  E. (1953). Equation of state calculations by fast computing machines.
  *Journal of Chemical Physics* 21(6), 1087–1092. The sampler.
- Hastings, W. K. (1970). Monte Carlo sampling methods using Markov chains and
  their applications. *Biometrika* 57(1), 97–109. The asymmetric acceptance
  ratio (6).
- Geyer, C. J. (1992). Practical Markov chain Monte Carlo. *Statistical Science*
  7(4), 473–483. Burn-in, autocorrelation and effective sample size.
- Roberts, G. O., Gelman, A. & Gilks, W. R. (1997). Weak convergence and optimal
  scaling of random walk Metropolis algorithms. *Annals of Applied Probability*
  7(1), 110–120. The $0.234$ acceptance rule.
- Page, L., Brin, S., Motwani, R. & Winograd, T. (1999). The PageRank citation
  ranking: bringing order to the web. Technical report, Stanford. The damping
  factor as a laziness correction.
- Levin, D. A., Peres, Y. & Wilmer, E. L. (2009). *Markov Chains and Mixing
  Times*. American Mathematical Society. The modern account of convergence and
  the second eigenvalue.
