# Neural architecture search — six ways to choose the activations

`nas.py` · Training · Tier 2 · 6 variants

## Intuition

Architecture design is usually a human loop: try a network, read the
validation curve, change one thing, repeat. Neural architecture search (NAS)
automates the loop. The search space here is deliberately tiny so that every
method can be run to completion: two cells, each cell picks one of four
activations — identity, $\tanh$, ReLU or $\sin$ — for $4^2=16$ genomes. The
task is a periodic boundary, $\sin(6x)+0.3y>0$, so the genome that matters is
the one containing a periodic unit. Exhaustive search finds `sin/sin` at 0.894,
and the all-affine network scores 0.569: the signal is real, and the space is
small enough that "good search" has a definite meaning.

Six strategies attack the same 16-cell space. `darts` relaxes the discrete
choice into a softmax over operations and takes gradients through the
relaxation. `enas` keeps a controller that samples genomes and is trained by
REINFORCE from their rewards. `amoebanet` and `regularized_evolution` run
tournaments and mutate winners, with different replacement rules. `supernet`
trains one pool of weights by sampling paths, then reads every genome out of
the shared pool. `once_for_all` trains one wide network per second-cell choice
and slices narrower widths out of it for free.

The results are not a ranking of "best method". AmoebaNet, regularized
evolution and the supernet find `sin/sin` or `sin/tanh` at 0.894 and 0.879;
DARTS and once-for-all settle at 0.700, because the relaxed optimum and the
fixed trunk are not the discrete optimum; ENAS pays 9,900 gradient steps to
reach 0.757. That spread is the lesson: the search algorithm, the proxy it
optimises, and the budget it spends are three different things, and a NAS
paper that reports only the final accuracy is hiding two of them.

## Variants

| variant | mechanism | genome | score | nets | steps |
|:--|:--|:--|--:|--:|--:|
| `darts` | softmax relaxation, gradient on $\alpha$ | `sin/relu` | 0.700 | 3 | 1200 |
| `enas` | controller, REINFORCE | `relu/sin` | 0.757 | 33 | 9900 |
| `amoebanet` | tournament, aging population | `sin/sin` | 0.894 | 12 | 3600 |
| `regularized_evolution` | tournament, replace worst | `sin/tanh` | 0.879 | 24 | 7200 |
| `once_for_all` | one wide net, sliced widths | `tanh/sin` | 0.700 | 7 | 2100 |
| `supernet` | shared pool, sampled paths | `sin/sin` | 0.894 | 3 | 2100 |

Exhaustive search over all 16 genomes costs 48 networks and 14,400 steps for
the same 0.894. "Nets" counts every training, and each genome is trained three
times with fixed seeds so that the objective is deterministic; the seeds are
identical for every variant, so all six searches optimise the same benchmark.

## Derivation

### The objective

A genome $g=(g_1,g_2)$ selects one activation per cell. Training a genome
means minimising the usual softmax cross-entropy, and scoring it means
measuring test accuracy. Because a single initialisation is noisy, the
benchmark averages $R$ fixed seeds:

$$
g^\star=\arg\max_{g\in\mathcal G}\ \frac1R\sum_{r=1}^{R}
\mathrm{Acc}\big(\mathcal D_{\mathrm{test}};\theta(g,s_r)\big),
\qquad (1)
$$

with $|\mathcal G|=16$ and $R=3$. Exhaustive search evaluates all 16 terms of
(1): 48 trainings, 14,400 gradient steps, and the answer `sin/sin` at 0.894.
Every variant below is a way to avoid evaluating all 16.

### The network

Each cell is an affine map followed by the chosen activation:

$$
h_1=f_{g_1}(W_1x+b_1),\qquad
h_2=f_{g_2}(W_2h_1+b_2),\qquad
p=\mathrm{softmax}(W_3h_2+b_3),
\qquad (2)
$$

and the loss is $-\log p_y$. The two cells share a head but not their weights;
the search changes only which $f$ is applied where.

### ENAS: a controller and REINFORCE

ENAS parameterises a distribution over genomes with a controller
$\pi_\theta(g)=\prod_\ell \pi_{\theta_\ell}(g_\ell)$ and samples genomes from
it. The reward is the validation accuracy, and the controller is updated by

$$
\nabla_\theta J(\theta)=\mathbb E_{g\sim\pi_\theta}
\left[\big(R(g)-b\big)\nabla_\theta\log\pi_\theta(g)\right],
\qquad (3)
$$

where $b$ is a running baseline. The code uses the score-function form
directly: for the sampled cell, the gradient of $\log\pi$ is
$e_{g_\ell}-p_\ell$, and the update is

$$
\theta_\ell\leftarrow\theta_\ell+\eta\,\big(R-b\big)
\big(e_{g_\ell}-p_\ell\big).
\qquad (4)
$$

The baseline $b$ is an exponential moving average of rewards, which does not
change the expectation of (3) but removes most of its variance. ENAS draws 16
genomes, of which 11 are distinct, and the duplicates are free because
`evaluate` caches by genome.

### DARTS: relax the choice

DARTS replaces the discrete pick with a softmax over operations:

$$
\bar o_\ell(x)=\sum_{o\in\mathcal O}
\frac{\exp(\alpha_{\ell,o})}{\sum_{o'}\exp(\alpha_{\ell,o'})}\,o(x),
\qquad (5)
$$

so the network is differentiable in $\alpha$. With $w_{\ell,o}$ the softmax
weight, the derivative with respect to the logit is

$$
\frac{\partial L}{\partial\alpha_{\ell,o}}
=\sum_i \frac{\partial L}{\partial\bar o_\ell(x_i)}\,
w_{\ell,o}\big(o(x_i)-\bar o_\ell(x_i)\big),
\qquad (6)
$$

which is exactly the `d_alpha` accumulation in `_mix_grads`. The original
method also differentiates through a few steps of the inner training
(bilevel); the code takes the first-order shortcut,

$$
\nabla_\alpha L_{\mathrm{val}}
\approx \nabla_\alpha L_{\mathrm{val}}
\big(\theta-\xi\nabla_\theta L_{\mathrm{train}}\big),
\qquad (7)
$$

ignoring the Hessian term. The relaxed model trains for 300 steps, the
architecture logits take a descent step of 0.05 per iteration, and the winner
is the argmax of each row. The direction of that step is load-bearing: with
`alpha += ...` the search ascends the loss and returns a near-affine genome
instead. Here the descent direction lands on `sin/relu`.

### Weight sharing: the supernet

A supernet trains one parameter pool $\theta$ against a distribution over
paths:

$$
\min_\theta\ \mathbb E_{g\sim p(g)}\,
L\big(g;\theta\big),\qquad p\ \text{uniform on }\mathcal G .
\qquad (8)
$$

Each path uses its own $(W_1,W_2)$ blocks and the shared head, so sampling a
path updates the blocks it touches. The code samples in shuffled rounds of 16,
four rounds in total, so every genome is updated four times instead of the
sparse coverage of independent sampling. After training, every genome
is read out with the same $\theta$; the readout is the "shared estimate".

The estimate is biased. A shared pool is optimised for the *average* path, and
the head in particular sees a different hidden distribution at every step, so

$$
\mathrm{Acc}\big(g;\theta_{\mathrm{shared}}\big)\neq
\mathrm{Acc}\big(g;\theta(g)\big),
\qquad (9)
$$

and the ranking it induces need not agree with (1). Here the shared estimate
for the winner is 0.833 while the benchmark is 0.894, and in general the
correlation between the two rankings is well below one — which is why one-shot
NAS reports a *ranking*, then retrains the winner.

### Evolution and aging

The evolutionary variants keep a population of genomes, pick a parent by
$k$-way tournament, mutate one cell, and insert the child. The regularized
variant replaces the worst member; AmoebaNet replaces the *oldest*, a rule that
prevents a single lineage from taking over. Mutation changes one cell
with probability 0.4, the other with 0.4, and both with 0.2:

$$
g'=\begin{cases}(u,g_2)&u\sim U(\mathcal O),\ \text{prob }0.4\\
(g_1,u)&\text{prob }0.4\\
(u,v)&\text{prob }0.2\end{cases}
\qquad (10)
$$

The demo's AmoebaNet starts from the four distinct one-cell genomes, expands
them to `(c,c)`, and ages the population; it reaches `sin/sin` after 12
trainings. Regularized evolution starts from six random genomes and reaches
`sin/tanh` after 24; the extra cost is the price of the wider initial
population and of replacing the worst rather than the oldest.

### Once-for-all: elastic width

OFA trains one wide network and derives sub-networks by masking channels:

$$
h=\big(W\odot m_w\big)x,\qquad m_w\in\{0,1\}^{d},
\qquad (11)
$$

so a width-$w$ slice is a prefix of the wide weight matrix. The demo trains
one width-8 network per second-cell operation with the first cell fixed to
`tanh`, then evaluates widths 4, 6 and 8 by slicing. It finds `tanh/sin` at
0.700. The first cell is frozen, so the periodic unit can never appear where
the task needs it, and that ceiling is visible in the score: the method is
searching widths, not activations.

### What the budget buys

Counting gradient steps rather than wall-clock makes the comparison portable.
Each genome costs $R$ trainings of $T$ steps, so exhaustive search costs

$$
C_{\mathrm{exhaustive}}=|\mathcal G|\,R\,T=16\cdot 3\cdot 300=14{,}400,
\qquad (12)
$$

and a search that scores $N$ distinct genomes costs $N R T$ plus any
architecture-level overhead. AmoebaNet's $N=4$ gives 3,600 steps for the same
0.894; the supernet pays one 1,200-step shared pass and then three retrains for
2,100. ENAS's $N=11$ costs 9,900 and still misses the optimum — sample
efficiency is not automatically better than enumeration on a 16-cell space.

## Code walkthrough

The forward pass is the genome made concrete: two lookups into the parameter
dictionary, one activation each, and a shared head.

```python
def _forward(params: dict, x: np.ndarray, genome):
    pre_one = x @ params[(0, genome[0])]["W"] + params[(0, genome[0])]["b"]
    first = ACTS[genome[0]](pre_one)
    pre_two = first @ params[(1, genome[1])]["W"] + params[(1, genome[1])]["b"]
    second = ACTS[genome[1]](pre_two)
    logits = second @ params["head"]["W"] + params["head"]["b"]
    return logits, (x, pre_one, first, pre_two, second)
```

The benchmark is the mean over fixed seeds, and the cache is what makes
repeated sampling in ENAS and evolution cheap:

```python
            for seed in range(self.restarts):
                params = _init(np.random.default_rng(seed))
                _train(params, x, y, genome, self.steps, self.lr, self.decay)
                self.n_trained += 1
                self.n_steps += self.steps
                probs = _softmax(_forward(params, test_x, genome)[0])
                scores.append(float(np.mean(probs.argmax(axis=1) == test_y)))
            self.cache[genome] = float(np.mean(scores))
```

DARTS builds the mixture (5) by evaluating all four operations on the same
pre-activation and combining the results with the softmax weights:

```python
    def _mix_forward(self, params, x, alpha):
        weight1 = _softmax(alpha[0])
        weight2 = _softmax(alpha[1])
        pre1 = [x @ params[(0, op)]["W"] + params[(0, op)]["b"] for op in OPS]
        act1 = [ACTS[op](value) for op, value in zip(OPS, pre1, strict=True)]
        hidden = sum(weight1[i] * act1[i] for i in range(len(OPS)))
```

and then takes the descent step on the logits — the sign that decides whether
the search works:

```python
            grads, d_alpha = self._mix_grads(params, cache, delta)
            _apply(params, grads, first, second, step, self.lr, self.decay)
            alpha -= 0.05 * d_alpha
```

ENAS's controller update is (4) with a score-function gradient per cell and a
moving-average baseline:

```python
            baseline = 0.7 * baseline + 0.3 * score
            for layer in range(2):
                grad = -probs[layer].copy()
                grad[OPS.index(genome[layer])] += 1.0
                theta[layer] += 0.5 * (score - baseline) * grad
```

The two evolutionary variants share the tournament and the mutation, and
differ only in what leaves the population. AmoebaNet ages:

```python
            score = self.evaluate((child, child), x, y, test_x, test_y)
            population.pop(0)
            scores.pop(0)
            population.append(child)
            scores.append(score)
```

while regularized evolution replaces its worst member. The supernet trains in
shuffled rounds so that every genome is visited before any is revisited:

```python
        for step in range(1, rounds * self.steps + 1):
            slot = (step - 1) % len(GRID)
            if slot == 0:
                order = [int(index) for index in self.rng.permutation(
                    len(GRID))]
            genome = GRID[order[slot]]
```

OFA derives a narrower subnet by slicing the prefix of each weight matrix and
running the same forward pass:

```python
                sliced = {
                    (0, "tanh"): {
                        "W": params[(0, "tanh")]["W"][:, :width],
                        "b": params[(0, "tanh")]["b"][:width]},
                    (1, op): {
                        "W": params[(1, op)]["W"][:width, :width],
                        "b": params[(1, op)]["b"][:width]},
                    "head": {"W": params["head"]["W"][:width],
                             "b": params["head"]["b"]}}
```

## Complexity

Let $|\mathcal G|$ be the number of genomes, $T$ the steps per training and
$R$ the restarts.

- **Exhaustive:** $|\mathcal G|RT$ gradient steps. Here $16\cdot3\cdot300
  =14{,}400$, and it is the reference the other five are measured against.
- **DARTS:** one relaxed training plus one retrain, $2T$ steps and one
  architecture level. The mixture evaluates all four operations per cell per
  step, so each step is four times the arithmetic of a discrete forward pass;
  the step count does not show that multiplier.
- **ENAS:** $N R T$ for $N$ distinct sampled genomes, plus a controller whose
  cost is negligible at this scale.
- **Evolution:** $N R T$ with $N$ the number of distinct genomes ever scored;
  the tournament and mutation are $O(1)$.
- **Supernet:** one shared pass of four rounds of $|\mathcal G|$ steps, plus
  $R T$ for the retrain. The shared pass is cheap per step but sees only one
  path at a time.
- **OFA:** one training per first-cell choice, then a forward pass per slice;
  the slice evaluation is free compared with training.

The exhaustive probe alone is 14,400 steps; the six searches add about 26,000
more. Every training is a two-layer network on 240 points, so the constant is
small; the point of the table is the ratio between $N R T$ and
$|\mathcal G|RT$, not absolute time.

## Failure modes

**The relaxed optimum is not the discrete optimum.** DARTS trains the mixture
in (5) and then takes an argmax, but the mixture's best weights need not point
at the best discrete genome. Here it picks `sin/relu` at 0.700 while
`sin/sin` scores 0.894. The gap is the price of the relaxation, and it grows
when the operations have very different scales — four activations with
different output ranges are close to a worst case.

**Weight sharing biases the ranking.** The supernet's shared estimate is
0.833 for a genome that retrains to 0.894, and every one-shot method inherits
some version of (9). Reporting the shared number as the architecture's
accuracy overstates the method by a wide margin. Always retrain the winner
before quoting a score.

**The controller needs more samples than the space has cells.** ENAS spends
9,900 steps and 33 trainings to reach 0.757, more than AmoebaNet's 3,600 for
0.894. REINFORCE on a 16-armed bandit is sample-hungry; on a space this small,
enumeration is cheaper than learning a policy. ENAS pays off when the space is
combinatorial and exhaustive search is impossible.

**A frozen cell caps the score.** OFA fixes the first cell to `tanh` and
searches only the second, so it can never build the periodic unit where the
task needs it; its 0.700 is a ceiling, not a search failure. Elastic-width
methods assume the macro architecture is already right.

**Seed noise can beat the search signal.** Before the restarts in (1), the
same genome scored anywhere from 0.55 to 0.93 depending on initialisation, and
a search could "win" by drawing a lucky seed. Averaging three seeds closes most
of that gap; it does not eliminate it, and a search that reports one run is
reporting one draw.

**The benchmark is not the deployment task.** Every variant here optimises
test accuracy on the same distribution it is scored on. In practice the search
runs on a proxy — a smaller dataset, a shorter schedule, a validation split —
and the proxy's ranking can differ from the full task's. The shared-estimate
bias and the proxy bias compound.

**Budget accounting hides the constant.** Counting trainings treats a DARTS
mixture step and a discrete step as equal, but the mixture evaluates four
operations. Counting steps ignores that a supernet step touches one path's
blocks. There is no single fair unit; state the unit and the constant.

**Architecture search can overfit the validation set.** With 16 genomes and a
3-seed average the risk is small, but a search that evaluates thousands of
genomes against one validation split will select noise. The usual fix is a
held-out test set that the search never sees, which is what the demo's
`test_x`/`test_y` would have to be if the space were larger.

**Search spaces encode assumptions.** Four activations and two cells cannot
express a residual connection, a convolution or a normalisation layer. The
winner here is "the activation that matches the data", which is a statement
about this space, not about NAS in general.

**Mutation is not local in a two-gene genome.** Changing one of two cells is a
large jump: there is no "nearby" architecture to hill-climb. Evolution works
here because the space is tiny; on real cell-based spaces the mutation
operators are designed to keep most of the graph intact.

**One shared pool can co-adapt.** The head sees a different hidden
distribution on every supernet step, so it learns a compromise. That is the
mechanism behind the 0.833 estimate; techniques such as path dropout and
progressive shrinking exist to reduce it.

## Applications

NAS entered practice through image-classification backbones, where the
searched cell is repeated at several resolutions and the search cost is
amortised over a family of models. AmoebaNet and its descendants produced
backbones that were competitive with hand-designed ones, and DARTS made the
search cheap enough to run on a single machine — at the cost of the
relaxation gap above.

Weight-sharing methods are now the default because they decouple search cost
from the number of architectures: train one supernet, evaluate thousands of
subnets, retrain a handful. Once-for-all turned that into a deployment story,
where one trained network serves a range of latency budgets by slicing width,
depth and resolution, and hardware-aware variants add latency to the objective
so the search returns models that fit a device.

Evolutionary search survives where the space is discrete and the objective is
noisy or non-differentiable: hardware-aware NAS, compiler autotuning, feature
selection, and neural-network pruning, where each candidate is a mask rather
than a differentiable weight. The controller approach is less common now but
remains the cleanest example of treating architecture choice as a policy.

Outside deep learning, the same patterns recur: Bayesian optimisation over
hyperparameters (chapter 13's `hyperparameter_search`), bandit algorithms for
allocating a training budget, and evolutionary search over pipelines. The
vocabulary changes; the tension between a cheap proxy and the true objective
does not.

## References

- Zoph, B., Le, Q. V. (2017). *Neural architecture search with reinforcement
  learning.* ICLR. The controller and REINFORCE formulation, (3)–(4).
- Pham, H., Guan, M. Y., Zoph, B., Le, Q. V., Dean, J. (2018). *Efficient
  neural architecture search via parameter sharing.* ICML. ENAS.
- Liu, H., Simonyan, K., Yang, Y. (2019). *DARTS: differentiable architecture
  search.* ICLR. The relaxation (5)–(7).
- Bender, G., Kindermans, P.-J., Zoph, B., Vasudevan, V., Le, Q. V. (2018).
  *Understanding and simplifying one-shot architecture search.* ICML. The
  shared-pool objective (8) and the ranking bias (9).
- Real, E., Aggarwal, A., Huang, Y., Le, Q. V. (2019). *Regularized evolution
  for image classifier architecture search.* AAAI. Tournament, aging and
  (10).
- Real, E., Aggarwal, A., Huang, Y., Le, Q. V. (2019). *Aging evolution for
  image classifier architecture search.* GECCO. The AmoebaNet replacement
  rule.
- Cai, H., Gan, C., Wang, T., Zhang, Z., Han, S. (2020). *Once-for-all: train
  one network and specialize it for efficient deployment.* ICLR. Elastic
  width and (11).
- Elsken, T., Metzen, J. H., Hutter, F. (2019). *Neural architecture search:
  a survey.* JMLR. The taxonomy and the proxy-versus-objective framing.
- White, C., Neiswanger, W., Nolen, S., Savani, Y. (2021). *A study on
  encodings in neural architecture search.* AutoML. Why the search space,
  not only the optimiser, determines the result.
