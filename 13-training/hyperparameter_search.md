# Hyperparameter search — eight ways to spend a tuning budget

`hyperparameter_search.py` · Training · Tier 2 · 8 variants

## Intuition

Hyperparameter search is an optimisation problem where each evaluation is
expensive, noisy, and can often be stopped early. The eight variants here are
answers to two questions: how do you choose the next configuration, and how
much fidelity do you give it before deciding it is hopeless.

Grid and random search ignore previous results. TPE and GP-BO build a model of
the objective and sample where it looks good. Hyperband, ASHA, BOHB and the
median pruner exploit the training curve: a bad configuration is usually bad
after one epoch, so evaluate many configurations cheaply and promote only the
survivors.

The demo uses a synthetic loss surface over three knobs with a real fidelity
dimension, so a configuration's loss improves with epochs. At 60 full-fidelity
trials, random search's best is 0.805, TPE's is 0.256 and GP-BO's is 0.233.
With fidelity, ASHA reaches 0.291 using 1080 epochs and BOHB reaches 0.258
using 846, against 1620 epochs for the single-fidelity runs. The final table
also shows the honest counterexample: a 64-point grid reaches 0.294, and on
three knobs a grid is a perfectly good method. The lattice dies in ten
dimensions, not three.

## Variants

| variant | proposal | fidelity | keeps |
|:--|:--|:--|:--|
| `grid` | fixed lattice | full | all points |
| `random` | uniform iid | full | all points |
| `tpe` | KDE over good points | full | all points |
| `gp_bo` | GP posterior, expected improvement | full | all points |
| `hyperband` | random, several brackets | rungs | top $1/\eta$ per rung |
| `asha` | random, one ladder | rungs | top $1/\eta$ per rung |
| `bohb` | TPE inside Hyperband brackets | rungs | top $1/\eta$ per rung |
| `pruner` | random, one ladder | rungs | those above the rung median |

One class, one objective, one budget counter. Swapping `grid` for `bohb` is a
constructor argument, which is the point: the search strategy is a policy over
the same experiment.

## Derivation

### The problem

Let $x\in[0,1]^d$ be the normalised hyperparameters and $r$ a fidelity, such
as epochs or training-set size. The target is

$$
x^\star=\arg\min_{x\in[0,1]^d} f(x,R),
\qquad (1)
$$

where $R$ is the largest fidelity you can afford and $f(x,r)$ is the loss after
training at $r$. The return on a configuration is observed only at the
fidelities you actually run, and the demo's curve is deliberately
fidelity-dependent:

$$
f(x,r)=\underbrace{0.2+\sum_i c_i(x_i-x_i^\star)^2}_{\text{final loss}}
+\underbrace{a(x)\,e^{-r/6}}_{\text{transient}},
\qquad (2)
$$

so a configuration can look good at one epoch and bad at twenty-seven.

### Grid and random

A grid with $n$ values per axis costs

$$
N_{\text{grid}}=(n+1)^d
\qquad (3)
$$

evaluations, which is why it is the right choice for two knobs and the wrong
one for twenty. Random search draws $N$ points iid, so the probability that at
least one lands in a region of volume $v$ is

$$
P(\text{hit})=1-(1-v)^N\approx 1-e^{-Nv}.
\qquad (4)
$$

For $d=3$ and $N=27$ the demo's lattice places no point near the optimum and
scores 1.916; random scores 0.529 at the same 729 epochs. At $N=64$ the same
grid reaches 0.294. The lattice is not foolish, it is exponential.

### TPE

Tree-structured Parzen estimation splits the observations at a quantile
$\gamma$ and models the two sets separately:

$$
\ell(x)=p(x\mid y<y_\gamma),\qquad
g(x)=p(x\mid y\ge y_\gamma),\qquad
\text{propose}\ \propto\frac{\ell(x)}{g(x)}.
\qquad (5)
$$

Each density is a product of one-dimensional Gaussian kernels,

$$
\hat p(x)=\frac{1}{n}\sum_{i=1}^{n}\prod_{j=1}^{d}
\frac{1}{h_j}\varphi\!\left(\frac{x_j-x_{ij}}{h_j}\right),
\qquad
h_j=\max(\sigma_j,0.08)\,n^{-1/5},
\qquad (6)
$$

with a uniform floor mixed in so the ratio stays bounded. The demo samples a
proposal from $\ell$ and accepts it with probability
$\ell/(\ell+g)$, otherwise samples from $g$ — the accept/reject form of the
same ratio. That keeps some exploration while concentrating on the elite set.
At 60 trials it reaches 0.256 against random's 0.805.

### GP-BO

A Gaussian process puts a prior over smooth functions. With a Matern-5/2
kernel and Gaussian observation noise,

$$
\mu(x)=k(x)^{\top}(K+\sigma^2I)^{-1}y,\qquad
s^2(x)=k(x,x)-k(x)^{\top}(K+\sigma^2I)^{-1}k(x),
\qquad (7)
$$

and expected improvement over the best value $y^\star$ seen so far is

$$
\mathrm{EI}(x)=(y^\star-\mu(x))\,\Phi(z)+s(x)\,\varphi(z),\qquad
z=\frac{y^\star-\mu(x)}{s(x)}.
\qquad (8)
$$

The code standardises the targets, adds $10^{-6}$ to the diagonal, and uses a
Cholesky solve instead of an explicit inverse. It evaluates (8) on 512 random
candidates and takes the argmax, which is a cheap approximation to the
continuous acquisition optimisation. GP-BO scores 0.233 with 60 trials.

### Successive halving and Hyperband

Successive halving evaluates $n$ configurations at a rung, keeps the best
$1/\eta$, and multiplies the fidelity by $\eta$:

$$
n_{k+1}=\left\lfloor\frac{n_k}{\eta}\right\rfloor,\qquad
r_{k+1}=\eta\,r_k.
\qquad (9)
$$

Hyperband runs several such brackets, each starting at a different fidelity.
With $S=\lfloor\log_\eta R\rfloor$, bracket $s$ uses

$$
n_s=\left\lceil\frac{S+1}{s+1}\,\eta^s\right\rceil,\qquad
r_s=R\,\eta^{-s},
\qquad (10)
$$

so the cheap brackets have many configurations and the expensive ones have
few. Each bracket costs roughly $(S+1)R$ epochs, and the demo scales $n_s$ so
the total fits the budget. Hyperband scores 0.619 on seed 4: its widest bracket
spends most of the budget on one-epoch trials, and the one that reaches 27
epochs has only two configurations left.

### ASHA and median pruning

ASHA is Hyperband with a single ladder and asynchronous promotion: whenever a
rung has enough results, the top $1/\eta$ move up immediately. The demo runs it
synchronously because there is one worker, but the promotion schedule is the
same:

$$
\mathcal S_{k+1}=\mathrm{top}_{1/\eta}
\{f(x,r_k):x\in\mathcal S_k\}.
\qquad (11)
$$

The median pruner is simpler still: at each rung, stop every configuration
whose loss is worse than the median of that rung,

$$
\text{keep }x\iff f(x,r_k)\le \mathrm{median}\{f(x',r_k)\}.
\qquad (12)
$$

ASHA reaches 0.291 with 400 cheap evaluations and 1080 epochs; the pruner also
reaches 0.291, using 254 evaluations and 1104 epochs by keeping half rather
than a third at each rung. Both beat Hyperband here because one long ladder
spends its budget where the signal is.

### BOHB

BOHB replaces Hyperband's random proposals with TPE fitted to all completed
observations:

$$
\text{BOHB}=\text{Hyperband brackets}\quad+\quad
\text{TPE proposal}\ \ell/g,
\qquad (13)
$$

so new configurations at every rung are sampled where earlier good ones were,
not uniformly. It scores 0.258 with 846 epochs, the best of the multi-fidelity
methods on seed 4 and better than Hyperband's 0.619 with the same brackets.
The proposal model is the difference, since the bracket schedule is identical.

## Code walkthrough

The objective is a training curve plus a small deterministic perturbation, so
fidelity genuinely changes the ranking:

```python
def _curve(delta: np.ndarray, epochs: float) -> float:
    """A synthetic training curve: transient plus a quadratic bowl."""
    curvature = np.array([40.0, 15.0, 8.0])
    final = 0.2 + float(np.sum(curvature * delta ** 2)) + 1.5 * (
        delta[0] * delta[1])
    amplitude = 0.7 + 1.6 * float(delta[0] + 0.35)
    return final + amplitude * float(np.exp(-epochs / 6.0))
```

Every strategy goes through the same validated, cached evaluation, and the
budget is counted in epochs rather than calls:

```python
        key = (tuple(np.round(values, 12)), int(epochs))
        if key not in self._cache:
            delta = values - self.centre
            self._cache[key] = _curve(delta, epochs) + _noise(values, epochs)
        loss = self._cache[key]
        self.history.append((values.copy(), int(epochs), loss))
        self.epochs_used += int(epochs)
        return loss
```

The grid is the Cartesian product of one axis with itself, with the side chosen
so the point count is near `trials`:

```python
    def _grid(self) -> None:
        side = max(2, int(round(self.trials ** (1.0 / self.dim))))
        axis = np.linspace(0.0, 1.0, side)
        mesh = np.meshgrid(*([axis] * self.dim), indexing="ij")
        points = np.stack([grid.reshape(-1) for grid in mesh], axis=1)
        for point in points:
            self.evaluate(point, self.max_epochs)
```

TPE's density is a Gaussian mixture in log space, with a uniform floor so the
ratio is finite everywhere:

```python
    def _kde_logpdf(self, candidates: np.ndarray,
                    sample: np.ndarray) -> np.ndarray:
        if len(sample) < 2:
            return np.zeros(len(candidates))
        bandwidth = np.maximum(sample.std(axis=0), 0.05) * len(
            sample) ** (-0.2)
        gap = (candidates[:, None, :] - sample[None, :, :]) / bandwidth
        logp = -0.5 * gap ** 2 - np.log(bandwidth) - 0.5 * np.log(2.0 * pi)
        logp = _logsumexp(logp.reshape(len(candidates), -1)) - np.log(
            len(sample))
        return np.logaddexp(np.log(1.0 - self.gamma) + logp,
                            np.log(self.gamma))
```

The proposal samples from the elite set, then accepts against the bad density;
the rejection branch keeps the search from collapsing onto its first few
points:

```python
    def _tpe_point(self) -> np.ndarray:
        if len(self.history) < self.n_start:
            return self.rng.random(self.dim)
        losses = np.array([entry[2] for entry in self.history])
        cut = float(np.quantile(losses, self.gamma))
        points = np.array([entry[0] for entry in self.history])
        good = points[losses <= cut]
        bad = points[losses > cut]
        proposal = self._sample_kde(good)
        good_log = float(self._kde_logpdf(proposal[None], good)[0])
        bad_log = float(self._kde_logpdf(proposal[None], bad)[0])
        accept = float(np.exp(good_log - np.logaddexp(good_log, bad_log)))
        return proposal if self.rng.random() < accept else self._sample_kde(bad)
```

GP-BO is the posterior solve followed by expected improvement on a candidate
cloud:

```python
        kernel = _matern(points, points) + 1e-6 * np.eye(len(points))
        cholesky = np.linalg.cholesky(kernel)
        weights = np.linalg.solve(
            cholesky.T, np.linalg.solve(cholesky, target))
        candidates = self.rng.random((512, self.dim))
        cross = _matern(points, candidates)
        predicted = cross.T @ weights
        projected = np.linalg.solve(cholesky, cross)
        variance = np.maximum(1.0 - np.sum(projected ** 2, axis=0), 1e-9)
        deviation = np.sqrt(variance)
        best = float(target.min())
        z = (best - predicted) / deviation
        improvement = ((best - predicted) * _norm_cdf(z)
                       + deviation * np.exp(-0.5 * z ** 2) / sqrt(2.0 * pi))
        return candidates[int(np.argmax(improvement))]
```

Hyperband is four lines of bookkeeping per rung once the bracket sizes are
computed, and the proposal function is the only thing BOHB changes:

```python
        for start in range(levels, -1, -1):
            count = scale * int(np.ceil(
                (levels + 1) / (start + 1) * self.eta ** start))
            epochs = self.max_epochs / self.eta ** start
            points = [propose() for _ in range(count)]
            for _ in range(start + 1):
                results = [self.evaluate(point, int(round(epochs)))
                           for point in points]
                order = np.argsort(results, kind="stable")
                keep = max(1, count // self.eta)
                points = [points[index] for index in order[:keep]]
                count = keep
                epochs *= self.eta
```

The median pruner is the same ladder with a data-dependent stopping rule
instead of a fixed promotion quota:

```python
    def _pruner(self) -> None:
        count = max(4, self.budget // 8)
        points = [self.rng.random(self.dim) for _ in range(count)]
        for rung in (1, self.eta, self.eta ** 2, self.max_epochs):
            results = [self.evaluate(point, rung) for point in points]
            median = float(np.median(results))
            points = [point for point, loss in zip(points, results,
                                                   strict=True)
                      if loss <= median]
```

## Complexity

With $N$ trials, $d$ dimensions, $S$ brackets and a budget $B$ measured in
epochs:

- **Grid:** $(n+1)^d$ evaluations, exponential in $d$; the demo's 27-point
  grid is $3^3$ and its 64-point grid is $4^3$.
- **Random, TPE, GP-BO:** $N$ full-fidelity evaluations, so $NR$ epochs.
  TPE adds $O(N^2d)$ for the KDE comparisons and GP-BO adds $O(N^3)$ for the
  Cholesky solve at every proposal — fine below a few hundred trials, not
  beyond.
- **Hyperband, ASHA, BOHB, pruner:** $O(B)$ epochs total by construction. The
  promotion and pruning logic is $O(N\log N)$ per rung from the sort.
- **The demo:** the table runs 60 trials for each single-fidelity method and
  up to 400 evaluations for the multi-fidelity ones, plus 64 grid points.
  Every objective call is a few flops, so the whole run is well under a
  second; the wall-clock cost of real HPO is the training, not the scheduler.

## Failure modes

**Three dimensions flatter grids.** A $4^3$ grid reaches 0.294, close to GP-BO's
0.268. The argument against grids is $(n+1)^d$, not a bad result at $d=3$; at
$d=10$ the same density needs $4^{10}$ points.

**Random search has enormous variance.** At seed 6 it scores 1.521 and at seed
4 it scores 0.805 with the same 60 trials. Reporting one run of any method is
close to meaningless; the demo fixes seeds for reproducibility, not to suggest
that the ranking is stable.

**TPE is sensitive to its bandwidth and its split.** A floor that is too small
collapses the elite density onto the observed points and stops exploring; a
split that keeps too few points makes the density meaningless. The demo uses a
0.08 floor, a $1/5$ exponent, and $\gamma=0.25$, which are conventions rather
than laws.

**GP-BO is cubic in the number of observations.** Sixty trials is comfortable;
a few thousand is not, and the usual fixes are a sparse GP, a random-feature
approximation, or switching to TPE. The acquisition is also optimised on a
finite candidate cloud, so a narrow optimum between candidates can be missed.

**Hyperband can starve its expensive bracket.** With a small budget the widest
bracket is cheap and the final bracket keeps only one or two configurations;
the demo's 0.619 is worse than a single ASHA ladder at 0.291. Hyperband's
guarantees are asymptotic, and the constants matter at small scale.

**Early fidelity may not rank final fidelity.** The demo's transient term
depends on the configuration, so the ordering at one epoch is only correlated
with the ordering at 27. If low fidelity is anti-correlated with the target —
common when regularisation only pays off late — successive halving prunes the
eventual winner.

**ASHA in this file is synchronous.** True ASHA overlaps rungs across workers
and promotes as soon as results arrive, which changes both the budget and the
result. The demo implements the promotion schedule, not the distributed
scheduler.

**Median pruning assumes comparable noise.** The rule compares raw losses, so a
noisy rung can eliminate good configurations by chance. The demo's noise is
small and deterministic; with real training curves the median should be taken
over repeated measurements or replaced by a confidence bound.

**The budget is not the cost.** One epoch on 400 configurations is not the same
wall-clock cost as 27 epochs on 15, because per-trial overhead (data loading,
compilation, startup) is paid 400 times. The demo counts epochs because its
objective is free; a real scheduler must count both.

**The search can overfit the validation set.** Choosing the best of many
configurations on one split is a selection bias, and the winner's score is
optimistic. Nested cross-validation or a final holdout is the cure; neither is
in this file, because the landscape is synthetic.

**The unit cube is not the hyperparameter space.** The demo searches
$[0,1]^d$; real learning rates live on a log scale, and batch sizes on a
doubling grid. The transform belongs to the caller, and getting it wrong makes
the best point on the cube the wrong configuration in the model.

## Applications

Random search is the baseline every paper compares against, and for a handful
of knobs with a generous budget it is still hard to beat. TPE is the default in
Hyperopt and Optuna, and GP-BO is the default in Ax, BoTorch and Spearmint; both
are the standard choice when a single trial costs minutes rather than hours.

Multi-fidelity methods are how large organisations run sweeps: ASHA and
Hyperband underpin Optuna's pruners, Ray Tune's schedulers and Google Vizier,
and BOHB is the standard strong baseline in AutoML. The same ideas show up
outside HPO — early stopping in training loops, racing algorithms, and bandit
schedules are all the same bet that a cheap partial evaluation carries
information about the expensive final one.

In practice the search strategy is a small part of the result. The larger wins
come from a sensible search space (log scales, no redundant parameters), a
stable objective, and enough repetitions to see through noise. The demo is
deliberately built so that these choices are visible rather than hidden behind
a framework.

## References

- Bergstra, J., Bengio, Y. (2012). *Random search for hyper-parameter
  optimization.* JMLR. Random search as the baseline, (4).
- Bergstra, J., Bardenet, R., Bengio, Y., Kégl, B. (2011). *Algorithms for
  hyper-parameter optimization.* NeurIPS. TPE and the density ratio, (5)–(6).
- Snoek, J., Larochelle, H., Adams, R. P. (2012). *Practical Bayesian
  optimization of machine learning algorithms.* NeurIPS. GP-BO and expected
  improvement, (7)–(8).
- Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., Talwalkar, A. (2018).
  *Hyperband: a novel bandit-based approach to hyperparameter optimization.*
  JMLR. Brackets and (10).
- Li, L., Jamieson, K., Rostamizadeh, A., Gonina, E., Ben-Tzur, J., Hardt, M.,
  Recht, B., Talwalkar, A. (2020). *A system for massively parallel
  hyperparameter tuning.* MLSys. ASHA, (11).
- Falkner, S., Klein, A., Hutter, F. (2018). *BOHB: robust and efficient
  hyperparameter optimization at scale.* ICML. TPE inside Hyperband, (13).
- Golovin, D., Solnik, B., Moitra, S., Kochanski, G., Karro, J., Sculley, D.
  (2017). *Google Vizier: a service for black-box optimization.* KDD. Median
  stopping and the production scheduler.
- Akiba, T., Sano, S., Yanase, T., Ohta, T., Koyama, M. (2019). *Optuna: a
  next-generation hyperparameter optimization framework.* KDD. The pruner
  interface this atom imitates.
- Jamieson, K., Talwalkar, A. (2016). *Non-stochastic best arm identification
  and hyperparameter optimization.* AISTATS. Successive halving as bandit
  identification, (9).
