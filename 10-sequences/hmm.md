# Hidden Markov models — the chain you cannot see

`hmm.py` · Sequences · Tier 1 · 7 variants

## Intuition

In a Markov chain you watch the state. Here you watch what the state emits. The
state sequence is latent, the observations are conditionally independent given
it, and every question you can ask splits into two halves: what the chain did,
and what each state sounds like.

That split is what makes the model useful. Weather is a two-state chain you
cannot observe directly; what you observe is whether people carry umbrellas.
Speech is a chain of phonemes you cannot observe directly; what you observe is a
waveform. A genome is a chain of CpG islands and background you cannot observe
directly; what you observe is a sequence of nucleotides. In each case the
interesting object is hidden and the observed object is noisy, and an HMM is the
smallest model that respects both facts.

Everything follows from three sets of numbers: where the chain starts, how it
moves, and what each state emits. Given those numbers, two matrix recursions
answer the inference questions — the forward pass for the likelihood and the
smoothed posteriors, the Viterbi pass for the single best path. Given only
observations, one algorithm estimates the numbers: Baum–Welch, which is
expectation–maximisation applied to the latent path.

```
          hidden:   ●───────●───────●───────●
                    │       │       │       │
                    ▼       ▼       ▼       ▼
          observed: x₁      x₂      x₃      x₄

          A: state → state       π: where it starts
          b: state → observation  the only part you can see
```

The seven variants slice the same object along two axes. Three of them name the
emission model: symbols (`discrete`), a diagonal Gaussian per state
(`gaussian`), or a mixture of Gaussians per state (`gmm_hmm`). Four of them name
the algorithm on the surface: exact posterior marginals (`forward_backward`),
maximum-probability paths (`viterbi`), multi-restart training with a likelihood
trace (`baum_welch`), or the scaled linear-domain recursions (`scaling`). The
axes compose, because none of the algorithms cares what the emission model is.

What the demos establish, beyond the usual parameter recovery, is that the three
claims in the file are true rather than plausible. On a seven-symbol sequence
the forward recursion agrees with enumeration over all $128$ state paths to ten
digits. On a ten-symbol sequence Viterbi agrees with the best of $1024$
enumerated paths exactly. And the scaled fit reaches the same likelihood as
the log-space fit to $8\times10^{-11}$, while the unscaled linear recursion
returns exactly zero.

The number worth remembering from the fits is $352$. That is how many nats a
two-component mixture per state gains over a single Gaussian per state on four
well-separated clusters. Choosing the emission model is not a detail; it is the
difference between modelling the data and modelling its average.

## Variants

| variant | emission | fit | surface |
|:--|:--|:--|:--|
| `discrete` | categorical symbols | Baum–Welch | score |
| `gaussian` | diagonal Gaussian | Baum–Welch | score |
| `gmm_hmm` | Gaussian mixture per state | Baum–Welch | score |
| `forward_backward` | inferred from the data | Baum–Welch | `posterior_marginals`, `pairwise_posteriors` |
| `viterbi` | inferred from the data | Baum–Welch | `predict`, `viterbi_score_` |
| `baum_welch` | inferred from the data | multi-restart EM | `history_`, `n_iter_` |
| `scaling` | inferred from the data | scaled linear EM | `scaling_factors_` |

**`discrete`** is the classical HMM of the speech and bioinformatics literature:
a finite alphabet, a table of emission probabilities per state, and counting as
the estimator. Its M-step is the simplest in the file: no responsibilities, no
variance floor, no moments.

**`gaussian`** replaces the table with one mean and one variance per state per
feature, so the M-step becomes weighted moments. It is the workhorse for
real-valued sequences and the model to try first when the state means are what
distinguish the states.

**`gmm_hmm`** gives every state a mixture instead of a single Gaussian, which is
the right model when a state is itself multimodal — a phoneme with several
allophones, a regime with two operating points. It costs a set of mixture
responsibilities per state and turns the M-step into a weighted soft-clustering.

**`forward_backward`** exposes the smoothing recursions: the posterior
probability of each state at each time, and of each transition. These are what
you plot, what you threshold, and what the other variants use internally.

**`viterbi`** exposes the max-product recursion and the score of the path it
returns. It answers "what is the single most likely explanation", which is a
different question from "what is each state's marginal" and can have a different
answer.

**`baum_welch`** is the training variant: it allows several random restarts,
keeps the best, and exposes the likelihood after every iteration. The trace is
the model's own convergence diagnostic and the restarts are the only defence
against local optima.

**`scaling`** runs the same EM in the linear domain with one normalising factor
per step, the technique that predates log-space arithmetic and is still how most
production implementations work. It is also the clearest demonstration of why
the factors are necessary.

## Derivation

### The model and the forward recursion

Let $z_{1:T}$ be the hidden states and $x_{1:T}$ the observations. The chain is
first-order and the emissions are conditionally independent given the state:

$$
p(z_{1:T}, x_{1:T}) = \pi_{z_1} b_{z_1}(x_1)\prod_{t=2}^{T} A_{z_{t-1}z_t}b_{z_t}(x_t).
$$

Summing over paths is exponential, but the sum factorises. Define the forward
variable $\alpha_t(k) = p(x_{1:t}, z_t = k)$; then

$$
\alpha_t(k) = b_k(x_t)\sum_{j} \alpha_{t-1}(j) A_{jk},
\qquad (1)
$$

with $\alpha_1(k)=\pi_k b_k(x_1)$ and $p(x_{1:T})=\sum_k \alpha_T(k)$. The
recursion has the shape of the chain: one $K\times K$ matrix product per
observation, $O(TK^2)$ in total, and the exponential disappears.

In log space the sum becomes a log-sum-exp, and the code is (1) with that
substitution:

```python
        log_alpha = np.empty((x.shape[0], self.n_states))
        log_alpha[0] = log_pi + log_b[0]
        for t in range(1, x.shape[0]):
            log_alpha[t] = log_b[t] + _logsumexp(log_alpha[t - 1][:, None] + log_a,
                                                 axis=0)
```

`_logsumexp` subtracts the row maximum before exponentiating. Without that, a
log-probability of $-2000$ underflows to zero and the sum is wrong; with it, the
only quantities that ever get exponentiated are at most one in magnitude.

The backward variable is the same idea run in reverse,
$\beta_t(j) = p(x_{t+1:T}\mid z_t=j)$:

$$
\beta_t(j) = \sum_k A_{jk} b_k(x_{t+1}) \beta_{t+1}(k),
\qquad (2)
$$

with $\beta_T(k)=1$. It does not depend on the past, which is why it can be
computed once and reused for every time step:

```python
        log_beta = np.zeros((x.shape[0], self.n_states))
        for t in range(x.shape[0] - 2, -1, -1):
            log_beta[t] = _logsumexp(log_a + (log_b[t + 1] + log_beta[t + 1])[None, :],
                                     axis=1)
```

### Posteriors

Multiplying the two messages and normalising gives the smoothed state
probability, and adding the transition in the middle gives the smoothed
transition probability:

$$
\gamma_t(k) = \frac{\alpha_t(k)\beta_t(k)}{p(x)},
\qquad
\xi_t(j,k) = \frac{\alpha_t(j)A_{jk}b_k(x_{t+1})\beta_{t+1}(k)}{p(x)}.
\qquad (3)
$$

Both are exact under the model: no sampling, no approximation, no iteration. The
same quantities computed from the scaled recursions are what every EM step
consumes.

The vectorised computation is worth showing because it removes a loop over the
sequence, which is where a naive implementation spends most of its time:

```python
        future = log_b[1:] + log_beta[1:]
        log_xi = (log_alpha[:-1][:, :, None] + log_a[None, :, :]
                  + future[:, None, :] - likelihood)
        xi = np.exp(log_xi)
```

The three factors broadcast against each other into a
$(T-1)\times K\times K$ array, one slice per transition. Memory is the price:
$O(TK^2)$ floats, which is fine for a few thousand observations and a handful of
states and is the reason long-sequence implementations use the
forward-only formulation of the M-step instead.

### The M-step

Baum–Welch maximises the expected complete-data log-likelihood. The expectation
is over the posterior of the hidden path, which (3) gives exactly, and the
maximisation separates because the complete-data likelihood is a sum of
independent terms. The result is counting with soft assignments:

$$
\pi_k = \gamma_1(k), \qquad
A_{jk} = \frac{\sum_t \xi_t(j,k)}{\sum_t \gamma_t(j)},
\qquad (4)
$$

and for symbols $b_k(v) = \sum_{t:x_t=v}\gamma_t(k) \big/ \sum_t \gamma_t(k)$.
The Gaussian and mixture updates are the same weighted moments with
$\gamma_t(k)$ as the weight; for the mixture, an extra responsibility
$r_t(k,m)$ splits each state's mass between its components.

Two implementation details make (4) survive real data. A row with no mass would
divide by zero, so it falls back to the uniform distribution — a state that died
is given no information rather than a NaN. And the emission estimate is a
weighted count, which the categorical branch computes directly:

```python
            counts = np.zeros_like(self.emission_)
            for symbol in np.unique(values):
                counts[:, symbol] = gamma[values == symbol].sum(axis=0)
            self.emission_ = _normalise_rows(np.maximum(counts, 1e-300))
```

`_normalise_rows` is the guard in one place:

```python
    totals = matrix.sum(axis=1)
    out = np.full_like(matrix, 1.0 / matrix.shape[1])
    live = totals > 0.0
    out[live] = matrix[live] / totals[live, None]
    return out
```

The small floor on the counts is not there to avoid the division —
`_normalise_rows` masks empty rows — but to keep every entry strictly positive,
so that `np.log` of the emission table never returns a negative infinity. A
symbol the model considers impossible then costs about $690$ nats, which is the
intended penalty rather than a special case.

The monotonicity of EM is the one guarantee worth stating. Each iteration
maximises a lower bound on the log-likelihood (Jensen's inequality gives the
bound), so the likelihood cannot decrease. The demo checks every fitted history for
monotonicity, and that is a property of the algorithm, not of the data.

### Viterbi

The most probable path is the maximum over paths of the same joint probability
that (1) summed. Swapping sum for max gives the max-product recursion

$$
\delta_t(k) = b_k(x_t)\max_j \delta_{t-1}(j)A_{jk},
\qquad (5)
$$

with $\delta_1(k)=\pi_k b_k(x_1)$ and the path recovered by backtracking through
the argmax at each step. It is dynamic programming over the same trellis as the
forward pass, with the same $O(TK^2)$ cost, and the log-domain version is
numerically identical to the linear one because there are no sums to underflow.

```python
        for t in range(1, length):
            candidate = delta[t - 1][:, None] + log_a
            back[t] = np.argmax(candidate, axis=0)
            delta[t] = log_b[t] + candidate[back[t], np.arange(self.n_states)]
```

The `candidate[back[t], np.arange(self.n_states)]` indexing is the vectorised
"take the maximum along each column"; writing it as a loop would be clearer and
twice as slow. The backtracking pass then walks `back` from the best final state
to the start.

The distinction from posterior decoding matters. Viterbi maximises
$p(z_{1:T}\mid x_{1:T})$ over whole paths. Posterior decoding picks
$\arg\max_k \gamma_t(k)$ at each $t$ independently, which maximises the expected
number of correctly labelled positions but can return a path the model considers
impossible. The demo's two decodings differ at three of twelve hundred
positions, which is typical when the posteriors are decisive and a warning when
they are not.

### Scaling

The linear-domain forward pass has the same recursion but no logs, and after a
few hundred observations it underflows: $p(x_{1:T})$ is a product of $T$
numbers smaller than one, and the smallest double is about $10^{-308}$. The fix
is to renormalise at every step and remember the factors:

$$
\hat\alpha_t = \frac{\alpha_t}{c_t}, \qquad c_t = \sum_k \alpha_t(k),
\qquad \log p(x_{1:T}) = \sum_t \log c_t.
\qquad (6)
$$

The normalised vector stays a probability distribution, so nothing underflows;
the product of the factors is the likelihood, and the sum of their logs is the
log-likelihood. It is exactly the same information as the log-space recursion,
carried in a different representation.

```python
        alpha[0] = pi * np.exp(log_b[0])
        factors[0] = max(alpha[0].sum(), 1e-300)
        alpha[0] /= factors[0]
        for t in range(1, length):
            alpha[t] = (alpha[t - 1] @ a) * np.exp(log_b[t])
            factors[t] = max(alpha[t].sum(), 1e-300)
            alpha[t] /= factors[t]
```

The backward pass must use the same factors, dividing by $c_{t+1}$ at each step,
or the product $\hat\alpha\hat\beta$ is not $\gamma$. That consistency condition
is the fiddly part of the classical algorithm and the reason log-space
implementations became popular.

### Emission models

For a diagonal Gaussian, the log density is a sum over features; the reshape is
what lets one expression serve both the per-state and the per-component case:

```python
        diff = x.reshape((x.shape[0],) + (1,) * axis + (x.shape[1],)) - means
```

With `axis=1` the result is $(T,K,d)$ and sums to a $(T,K)$ table; with `axis=2`
it is $(T,K,M,d)$ and sums to $(T,K,M)$, one column per mixture component. The
mixture then needs a log-sum-exp over components and a responsibility per
component, which is where the mixture M-step gets its extra factor.

The Gaussian M-step is weighted moments with $\gamma$ as the weight, and the
variance is floored at a tiny fraction of the global variance. The floor is a
guard against a component collapsing onto a single observation, where the
likelihood is infinite and EM would happily go there.

### Initialisation

EM is local, so the starting point decides which optimum is found. The
initialisation here is deliberately structured rather than random: sort the
sample along its first principal axis, cut it into equal-frequency chunks, use
the chunk means as the state means, add a small jitter, and start the transition
matrix sticky. The jitter and the seed are what make a restart a restart; the
structure is what keeps most restarts out of the degenerate optima.

## Code walkthrough

### The likelihood and the state marginals

`score` runs the recursion of (1) and nothing else, which makes it the cheapest
thing in the file and the natural place to check a fit. The `_require` gate is
not cosmetic: calling `predict` on the `forward_backward` variant, or
`posterior_marginals` on the `viterbi` variant, raises rather than silently
returning something that looks like an answer.

### The scaled pass

```python
            beta[t] = (a @ (np.exp(log_b[t + 1]) * beta[t + 1])
                       / max(factors[t + 1], 1e-300))
```

This is the backward half of (6). The division by `factors[t + 1]` is the
consistency condition: the forward vector was divided by that same number, so
the backward vector must be too, and then their product is a posterior instead
of a product of unrelated scalings.

### Mixture responsibilities

```python
        log_resp = weighted - log_b[:, :, None]
        resp = np.exp(log_resp) * gamma[:, :, None]
        totals = np.maximum(resp.sum(axis=0), 1e-300)
```

`weighted` holds $\log w_{km} + \log N(x_t;\mu_{km},\sigma^2_{km})$ and `log_b`
holds the log-sum-exp over $m$, so the difference is the log responsibility.
Multiplying by the state posterior gives the joint responsibility of state $k$
and component $m$ at time $t$, which is exactly the weight the mixture M-step
needs. The `1e-300` floor is there because a component can, in principle, be
responsible for nothing.

### Choosing the emission model

```python
        if self.emission == "unknown":
            emission = "categorical" if (integer_like and values.ndim == 1) else "gaussian"
```

The three emission variants fix their family in the constructor; the four
algorithm variants infer it. A one-dimensional array of integers is read as
symbols, anything else as measurements — which is a heuristic, and the reason
`discrete` exists as an explicit variant for the case where a float array really
does hold codes. The guards on length are the other half of `_prepare`: an empty
sequence has no first observation, so $\alpha_1$ is undefined, and a scalar has
no time axis at all.

## Complexity

Let $T$ be the sequence length, $K$ the number of states, $d$ the number of
features, $M$ the mixture components and $I$ the EM iterations.

| operation | time | memory |
|:--|:--|:--|
| one E-step, symbols or Gaussians | $O(TK^2)$ | $O(TK^2)$ |
| one E-step, mixture | $O(TK^2 + TKMd)$ | $O(TK^2)$ |
| one M-step, symbols | $O(TK)$ | $O(K^2)$ |
| one M-step, mixture | $O(TKMd)$ | $O(KMd)$ |
| fit | $O(I\,T K^2)$ | $O(TK^2)$ |
| `score` | $O(TK^2)$ | $O(TK)$ |
| `predict` | $O(TK^2)$ | $O(TK)$ |
| `posterior_marginals` | $O(TK^2)$ | $O(TK^2)$ |

The $O(TK^2)$ memory for the pairwise posteriors is the only avoidable term.
The standard alternative computes the M-step's transition counts in a single
forward pass over $\hat\alpha$, the transition matrix and $\hat\beta$, never
materialising $\xi$; that is what production implementations do, and it is
worth doing when $T$ is in the millions. Here the explicit array is what makes
the code readable.

## Failure modes

**The likelihood is multimodal and EM finds one mode.** With $K$ states there
are $K!$ labellings of the same model, all with identical likelihood, so the
fitted state numbering means nothing. That is why the demos relabel before
comparing. Less benign are genuinely different optima: fitting four states to
data from three, the demo's restarts disagree by $0.74$ nats while the
likelihood itself differs from the three-state fit by only $3$. The parameters
differ completely; the likelihood barely notices.

**More states is not more model.** The same demo fits three and four states to
three-state data. Training likelihood improves from $-1784.2$ to $-1781.2$,
while held-out likelihood degrades from $-1839.2$ to $-1845.4$. The extra state
is fitting noise. The number of states is a model-selection decision and needs
held-out data or a criterion, not the training likelihood.

**Underflow, if you skip the scaling.** The unscaled linear recursion in the
demo returns exactly $0.0$ after two thousand observations, while the
log-likelihood the model reports is $-2690.7$. The failure is silent: a zero
likelihood looks like a very bad model, not a broken implementation. If a
likelihood is exactly zero on a long sequence, this is the first thing to check.

**A dead state takes its row with it.** If no observation is ever assigned to a
state, its $\gamma$ column is zero, and the transition row becomes $0/0$. The
code falls back to uniform, which keeps the fit running but leaves a state that
means nothing. The symptom is a state with a uniform transition row and a
flat emission; the fix is fewer states or a better initialisation, not more
iterations.

**A Gaussian component can collapse onto one point.** The likelihood of a
Gaussian with $\sigma\to0$ at an observation goes to infinity, so the M-step
would happily set the variance to zero. The variance floor prevents the
singularity and keeps the model honest; without it, the fit can drive one
variance to zero within a few iterations and report a likelihood that is
infinite and meaningless. The same issue appears in
`gmm_hmm` and is the reason the floor is relative to the data's own variance.

**The emission model is a modelling assumption, not a detail.** On four
well-separated clusters, a single Gaussian per state loses $352$ nats to a
two-component mixture per state. In the other direction, a mixture with too many
components will split a unimodal state into two identical halves and call the
split structure. Compare emission families on held-out data, as the demo does.

**Viterbi and posterior decoding can disagree, and Viterbi can be worse by the
metric you care about.** Viterbi maximises the probability of the whole path;
posterior decoding maximises per-position accuracy. If the downstream task scores
positions, posterior decoding is the right decoder even though its path is less
probable. The three differing positions in twelve hundred are a reminder that
the two answers are answers to different questions.

**Conditional independence is doing a lot of work.** The model assumes the
observations are independent given the state. Correlated noise, overlapping
windows or a slowly varying measurement scale all violate that, and the fitted
states will encode the correlation rather than the phenomenon. The usual fixes
are richer emissions (full-covariance Gaussians, autoregressive HMMs) or a
different model class entirely.

## Applications

**Speech recognition.** The classical system was an HMM per phone, with Gaussian
mixture emissions over cepstral features, and the Viterbi and Baum–Welch
algorithms in this file are the ones that made it work. Modern recognisers
replaced the emission model with neural networks but kept the HMM decoding
layer for decades, and CTC — the `ctc` atom in this chapter — is the alignment
machinery that made the frame-level layer optional.

**Bioinformatics.** CpG islands, gene finding and copy-number segmentation are
all hidden-state problems over nucleotide or probe sequences; the `discrete`
demo is the toy version of a CpG-island model, and the forward–backward
posteriors are how the island boundaries are drawn.

**Regime switching in finance and economics.** Returns switch between
low-volatility and high-volatility regimes; a two-state Gaussian HMM with
time-varying variance is the standard starting point, and the smoothed
probability of the high-volatility state is a widely published risk indicator.

**Human activity and gesture recognition.** Sensor streams from phones and
wearables are segmented into activities by an HMM whose emissions are summary
features per window; the state sequence is the activity label and the transition
matrix encodes which activities follow which.

**Natural language processing.** Part-of-speech tagging was an HMM over word
emissions before it was a CRF, and the `crf` atom is its discriminative
successor. The Viterbi recursion is the same in the two models; only the
features change.

**Fault detection and predictive maintenance.** Machine condition is a hidden
state, sensor readings are the emissions, and the filtered state probability is
the alarm signal. The `change_point` atom is the alternative when the state is
persistent and the question is "when did it change" rather than "which state is
it in".

## References

- Baum, L. E. & Petrie, T. (1966). Statistical inference for probabilistic
  functions of finite state Markov chains. *Annals of Mathematical Statistics*
  37(6), 1554–1563. The estimation algorithm.
- Baum, L. E., Petrie, T., Soules, G. & Weiss, N. (1970). A maximization
  technique occurring in the statistical analysis of probabilistic functions of
  Markov chains. *Annals of Mathematical Statistics* 41(1), 164–171. The
  Baum–Welch form and its monotonicity.
- Viterbi, A. J. (1967). Error bounds for convolutional codes and an
  asymptotically optimum decoding algorithm. *IEEE Transactions on Information
  Theory* 13(2), 260–269. The max-product recursion.
- Forney, G. D. (1973). The Viterbi algorithm. *Proceedings of the IEEE* 61(3),
  268–278. The trellis picture and the backtracking.
- Dempster, A. P., Laird, N. M. & Rubin, D. B. (1977). Maximum likelihood from
  incomplete data via the EM algorithm. *Journal of the Royal Statistical
  Society B* 39(1), 1–38. Why the likelihood is monotone.
- Levinson, S. E., Rabiner, L. R. & Sondhi, M. M. (1983). An introduction to the
  application of the theory of probabilistic functions of a Markov process to
  automatic speech recognition. *Bell System Technical Journal* 62(4),
  1035–1074. Scaling and the practical recursions.
- Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected
  applications in speech recognition. *Proceedings of the IEEE* 77(2), 257–286.
  The canonical reference for everything in this file.
- Ephraim, Y. & Merhav, N. (2002). Hidden Markov processes. *IEEE Transactions
  on Information Theory* 48(6), 1518–1569. The modern statistical account.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, chapter 13.
  Springer. The EM derivation and the scaling discussion.
