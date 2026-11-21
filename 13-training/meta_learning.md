# Meta-learning — seven ways to learn from a handful of examples

`meta_learning.py` · Training · Tier 2 · 7 variants

## Intuition

A classifier that has never seen a task cannot learn it from five examples.
A meta-learner has seen hundreds of tasks drawn from the same family, so five
examples are enough to identify *which* member of the family it is facing. The
difference is the distribution, not the architecture.

The family here is concentric rings: three classes, each a ring of random
radius, with the radii resampled for every task. A ring's mean is the origin,
so nearest-mean on the raw inputs is nearly useless — it scores 0.358, and a
random encoder scores 0.454. A representation that encodes the radius can tell
the rings apart, and learning that representation across tasks is the
meta-learning problem.

Seven algorithms attack it. `maml` adapts a copy of the parameters on the
support set and differentiates through the adaptation to score the query set.
`first_order_maml` drops the second-order term. `reptile` moves the
initialisation toward the adapted parameters without ever touching the query
set. `meta_sgd` learns a per-parameter inner learning rate. The three metric
methods skip adaptation entirely: `prototypical` classifies by distance to the
class means in embedding space, `matching` by a soft nearest-neighbour over
the support set, and `relation` by a learned similarity module.

After 400 meta-iterations the prototype head leads at 0.850, matching reaches
0.838, the relation module 0.848, MAML 0.795, first-order MAML 0.788, meta-SGD
0.755 and Reptile 0.684. The ordering is not a verdict on the algorithms: it
says that this task family is metric, so a fixed nearest-prototype rule over a
good embedding is exactly the right inductive bias, and adaptation on five
points adds variance without adding information.

## Variants

| variant | adapts at test time | what moves | score |
|:--|:--|:--|--:|
| `maml` | yes, full second order | initialisation | 0.795 |
| `first_order_maml` | yes, first order | initialisation | 0.788 |
| `reptile` | yes, first order | initialisation | 0.684 |
| `meta_sgd` | yes, learned steps | initialisation and $\alpha$ | 0.755 |
| `prototypical` | no | encoder | 0.850 |
| `matching` | no | encoder | 0.838 |
| `relation` | no | encoder and relation module | 0.848 |

The five learned variants share one encoder — a 2→64→16 ReLU network — and the
three MAML-family variants add a linear head on top. The metric variants use
the encoder alone. Every variant sees the same 400 meta-iterations of four
tasks, each with three classes, five support points and fifteen query points.

## Derivation

### The episodic objective

A task $\mathcal T=(\mathcal D_{\mathrm{sup}},\mathcal D_{\mathrm{qry}})$ is
drawn from a family $p(\mathcal T)$. Meta-training minimises the query loss
after adapting on the support set:

$$
\min_\theta\ \mathbb E_{\mathcal T\sim p(\mathcal T)}
\left[L_{\mathrm{qry}}\big(\theta'(\theta,\mathcal D_{\mathrm{sup}})\big)
\right],
\qquad (1)
$$

and meta-testing repeats the same procedure on a task the learner has never
seen. The number that matters is the expected query accuracy after adaptation,
which is what `evaluate` measures over 200 fresh tasks.

### MAML

The inner loop takes one or more gradient steps on the support loss:

$$
\theta'=\theta-\alpha\nabla_\theta L_{\mathrm{sup}}(\theta).
\qquad (2)
$$

Differentiating (1) through (2) gives the meta-gradient

$$
\frac{\partial L_{\mathrm{qry}}(\theta')}{\partial\theta}
=\left(I-\alpha\nabla^2_\theta L_{\mathrm{sup}}(\theta)\right)
\nabla_{\theta'}L_{\mathrm{qry}}(\theta'),
\qquad (3)
$$

so the outer update needs a Hessian-vector product at every inner step. The
code computes it by finite difference,

$$
Hv\approx\frac{\nabla L_{\mathrm{sup}}(\theta+\epsilon v)
-\nabla L_{\mathrm{sup}}(\theta)}{\epsilon},
\qquad (4)
$$

which costs two support gradients and is exact to $O(\epsilon)$ without a
second-order autodiff. The reverse pass then multiplies the factors in (3):

$$
v\leftarrow v-\alpha\,H_t\,v,\qquad t=T-1,\dots,0 .
\qquad (5)
$$

First-order MAML drops the Hessian and uses
$\nabla_{\theta'}L_{\mathrm{qry}}$ directly. On this task the difference is
small — 0.795 against 0.788 — because the inner loop takes three small steps
and the curvature term contributes little. On tasks where the inner loop moves
far, the gap is larger, and (4) is what closes it.

### Reptile

Reptile never evaluates the query loss during meta-training. It adapts on the
support set and pulls the initialisation toward the adapted parameters:

$$
\theta\leftarrow\theta+\beta\big(\theta'-\theta\big).
\qquad (6)
$$

With one inner step, (6) is $\beta\alpha\nabla L_{\mathrm{sup}}$, an SGD step
on the support loss; with several steps it approximates the gradient of a
multi-step objective without the Hessian. The demo uses $\beta=0.5$ and gets
0.684, the weakest of the seven — the price of ignoring the query set when the
support set is only 15 points.

### Meta-SGD

Meta-SGD replaces the scalar inner rate with a vector $\alpha$ of the same
shape as $\theta$:

$$
\theta'=\theta-\alpha\odot\nabla_\theta L_{\mathrm{sup}}(\theta),
\qquad (7)
$$

and learns $\alpha$ alongside the initialisation. The gradient with respect to
$\alpha$ follows from (7) and the chain rule:

$$
\frac{\partial L_{\mathrm{qry}}}{\partial\alpha}
=-g_t\odot v_{t+1},
\qquad (8)
$$

where $g_t$ is the support gradient at step $t$ and $v_{t+1}$ is the
backpropagated query gradient at the next iterate. The code accumulates (8)
while running the same reverse recursion as MAML, and clips $\alpha$ to
$(10^{-3},1)$ to keep the inner loop stable. Meta-SGD reaches 0.755: better
than Reptile, worse than a plain MAML with a well-chosen scalar rate, because
a second copy of every parameter is a lot to learn from 400 tasks.

### Prototypical networks

The encoder maps each point to $\mathbb R^{16}$. A class prototype is the mean
of its support embeddings, and a query is classified by negative squared
distance:

$$
p(y=c\mid q)=\mathrm{softmax}_c\left(-\|\phi(q)-\mu_c\|^2\right),\qquad
\mu_c=\frac1K\sum_{k}\phi(s_{ck}).
\qquad (9)
$$

There is no inner loop: the support set *is* the classifier. The demo reaches
0.850, the best score, because the task family is metric — the rings are
separable in a space where radius is a coordinate, and the mean is the right
summary of a ring.

### Matching networks

Matching replaces the mean with a soft nearest-neighbour over the support
points of each class:

$$
\ell_{qc}=\log\sum_{k=1}^{K}
\exp\left(-\frac{\|\phi(q)-\phi(s_{ck})\|^2}{\tau}\right),
\qquad (10)
$$

which is a log-sum-exp of negative distances. As $\tau\to0$ it becomes
1-nearest-neighbour; as $\tau\to\infty$ it becomes a class mean. The demo uses
$\tau=0.5$ and reaches 0.838, slightly below prototypical, because five points
per class make the attention noisy and the mean is the lower-variance summary.
The original matching network also conditions on the whole support set with an
LSTM; (10) is the attention-only core.

### Relation networks

The relation module learns the similarity instead of fixing it:

$$
r(q,c)=f_\psi\big([\phi(q),\mu_c,|\phi(q)-\mu_c|]\big),
\qquad (11)
$$

with a two-layer MLP $f_\psi$ over the concatenation of the query, the
prototype and their absolute difference. The score replaces the distance in
(9), and the encoder and the relation module are trained end-to-end on the
episodic loss. It reaches 0.848, essentially tying prototypical: on a task
where Euclidean distance is already the right metric, learning a metric buys
nothing, and the extra parameters are pure variance.

### Shots, and what the support set buys

The support set is the only supervision at test time, so its size sets the
ceiling. Prototypical with the same encoder and 250 meta-iterations scores
0.813 at one shot, 0.818 at three and 0.844 at five. The gain is small because
the rings are well separated and one point already pins down a radius. On
tasks where classes overlap more, the curve is steeper, and the difference
between a good and a bad meta-learner is largest at one shot.

### Cost

Let $T$ be the inner steps and $G$ the cost of one support gradient. MAML's
per-task cost is

$$
C_{\mathrm{MAML}}=(T+1)G+2TG=(3T+1)G,
\qquad (12)
$$

while prototypical costs one forward pass over support and query and no
adaptation at all. At $T=3$ the second-order pass is about ten support
gradients per task; the demo still runs in seconds because the network is
tiny, but on a real backbone that factor decides whether the method is
affordable. This is why first-order MAML and Reptile exist: they trade a small
accuracy gap for a large constant.

## Code walkthrough

The task generator is the whole inductive bias: rings of random radius, each
one class, so the mean of every class is the origin.

```python
    angles = rng.uniform(0.0, 2.0 * np.pi, (n_way, n_shot + n_query))
    points = radii[:, None, None] * np.stack(
        [np.cos(angles), np.sin(angles)], axis=-1)
    points = points + rng.normal(0.0, 0.10, points.shape)
    return points[:, :n_shot], points[:, n_shot:]
```

The encoder is two ReLU layers, and the head is linear; the same forward pass
serves the MAML family and, through `_metric_logits`, the metric family.

```python
def _encode(params: dict, x: np.ndarray):
    z1 = x @ params["W1"] + params["b1"]
    a1 = np.maximum(z1, 0.0)
    z2 = a1 @ params["W2"] + params["b2"]
    a2 = np.maximum(z2, 0.0)
    return a2, (x, z1, a1, z2)
```

The inner loop saves every iterate, because the reverse pass of (5) needs the
Hessian at each one:

```python
    def _adapt(self, params: dict, sup: np.ndarray, steps: int, **kwargs):
        trail = [params]
        gradients = []
        for _ in range(steps):
            grads, _ = self._support_grads(trail[-1], sup)
            gradients.append(grads)
            scaling = kwargs.get("alpha")
            if scaling is None:
                step = {key: self.inner_lr * grads[key] for key in grads}
            else:
                step = {key: scaling[key] * grads[key] for key in grads}
            trail.append({key: trail[-1][key] - step[key] for key in trail[-1]})
        return trail[-1], trail, gradients
```

The Hessian-vector product of (4) is two support gradients, no second-order
machinery:

```python
    def _hessian_vector(self, params: dict, sup: np.ndarray,
                        vector: dict) -> dict:
        base, _ = self._support_grads(params, sup)
        eps = 1e-4
        shifted = {key: params[key] + eps * vector[key] for key in params}
        moved, _ = self._support_grads(shifted, sup)
        return {key: (moved[key] - base[key]) / eps for key in base}
```

and the reverse pass applies (5) from the last inner step to the first:

```python
            vector = query
            for index in range(self.inner_steps - 1, -1, -1):
                product = self._hessian_vector(trail[index], sup, vector)
                vector = {key: vector[key] - self.inner_lr * product[key]
                          for key in vector}
            return vector, None
```

Reptile is the same inner loop with a different outer signal — the difference
between the adapted and original parameters, with no query loss in sight:

```python
        if self.variant == "reptile":
            adapted, _, _ = self._adapt(self.params, sup, self.inner_steps)
            return ({key: adapted[key] - self.params[key]
                     for key in self.params}, None)
```

Meta-SGD accumulates the learning-rate gradient (8) during the same reverse
recursion, then steps $\alpha$ with its own optimiser:

```python
            for index in range(self.inner_steps - 1, -1, -1):
                d_alpha = {key: d_alpha[key] - gradients[index][key] * vector[key]
                           for key in d_alpha}
                product = self._hessian_vector(trail[index], sup, vector)
                vector = {key: vector[key] - self.alpha[key] * product[key]
                          for key in vector}
            return vector, d_alpha
```

The prototypical head is (9) in three lines, and its gradient is the chain rule
through the mean:

```python
    if variant == "prototypical":
        centre = s.mean(axis=1)
        logits = -((q[:, :, None, :] - centre[None, None, :, :]) ** 2).sum(-1)
        return logits, (q, s, centre, s_cache, q_cache)
```

Matching is the same idea with a log-sum-exp over the support points:

```python
    if variant == "matching":
        pair = q[:, :, None, None, :] - s[None, None, :, :, :]
        similarity = -(pair ** 2).sum(-1) / TAU
        shift = similarity.max(axis=-1, keepdims=True)
        weight = np.exp(similarity - shift)
        total = weight.sum(axis=-1, keepdims=True)
        logits = (shift + np.log(total))[..., 0]
        return logits, (q, s, pair, weight / total, s_cache, q_cache)
```

The relation module concatenates the query, the prototype and their gap, then
scores the triple with a two-layer MLP; its gradients come from the same
cross-entropy, backpropagated through the encoder.

## Complexity

With $N$ meta-iterations, $B$ tasks per iteration, $K$ shots, $Q$ queries,
$T$ inner steps and an encoder of $P$ parameters:

- **Prototypical and matching:** one encoder pass over $(K+Q)\,N_{\mathrm{way}}$
  points per task, $O(NB(K+Q)P)$ for the whole run. No adaptation.
- **Relation:** the same plus $N_{\mathrm{way}}^2$ MLP evaluations per query;
  the module is small, so the encoder still dominates.
- **MAML:** $(3T+1)$ support or query gradients per task, from (12). The
  demo's $T=3$ means about ten gradient evaluations per task; the finite
  difference is two of them per inner step.
- **First-order MAML:** $T+1$ gradients per task, roughly a third of the
  second-order cost.
- **Reptile:** $T$ gradients per task, the cheapest of the adaptive family,
  because the query set is never evaluated during meta-training.
- **Meta-SGD:** MAML's cost plus one extra parameter tensor of size $P$; the
  inner loop is elementwise, so the arithmetic is nearly unchanged.

The demo's 400 iterations × 4 tasks × 7 variants is a few seconds, dominated
by the second-order variants. The constant that matters in practice is (12):
the same search on a ResNet backbone is where MAML's Hessian becomes the
reason people use first-order methods.

## Failure modes

**The support set can be too small for adaptation.** MAML with three inner
steps and 15 support points overfits the support loss: the adapted model fits
the five points per class and the query accuracy falls. The demo's 0.795
versus prototypical's 0.850 is that effect. Fewer steps, a smaller inner rate
or a head-only inner loop are the usual fixes.

**Meta-overfitting.** With 400 tasks the encoder can memorise the family
rather than learn a transferable representation. The tell is a large gap
between two independent evaluations — here the demo's two 200-task runs agree
to within 0.02, which is why its table is trustworthy. On a small task family,
hold out task *parameters*, not just samples.

**Second-order is not automatically better.** First-order MAML scores 0.788
against 0.795, a difference smaller than the task-to-task noise. The Hessian
term matters when the inner loop moves far; with three small steps it is a
rounding error at ten times the cost. Measure the gap before paying for it.

**Reptile ignores the query set.** Its outer signal is the support-adapted
parameters, so it optimises for support-set performance, not query-set
performance. When support and query are drawn from the same task that is fine;
when the task has a distribution shift between the two splits, it is not.

**Metric methods assume the right metric.** Prototypical wins here because
squared distance in a learned embedding is the correct inductive bias for
rings. On a task where class membership depends on a relation between points
rather than their position, a fixed metric fails and the relation module's
learned similarity is the only one of the three that can recover.

**The prototype is a mean, and means hide structure.** A class with two modes
has a prototype between them, in a region with no data. The same failure
appears in the raw baseline: every ring's mean is the origin. Clustering the
support before averaging, or using attention as in (10), is the usual remedy.

**A learned inner rate can collapse or explode.** Meta-SGD's $\alpha$ is
clipped because the unclipped version drives some coordinates to zero and
others past one within a few hundred iterations. The clip is part of the
algorithm, not a detail.

**Evaluation leaks through the task sampler.** If meta-training and meta-test
tasks share the same random radii drawn once, the test set is not fresh. The
demo resamples the radii for every task; the `evaluate` loop draws new tasks
from the same generator, which is the correct protocol but a subtle one to get
right in a real pipeline.

**The number of classes is fixed in the code.** All heads have `N_WAY=3`
outputs, so a four-way task would need a different network. Metric methods
generalise to a new $N_{\mathrm{way}}$ at test time; MAML's linear head does
not without resizing. That asymmetry is easy to miss when comparing scores.

**Few-shot accuracy is a noisy statistic.** With 15 queries per class and 200
tasks the standard error is around 0.01, so a 0.005 gap between two methods is
not a result. Report the interval, not the third decimal.

## Applications

Few-shot image classification is the benchmark that drove the field: Omniglot
and miniImageNet episodes with five classes and one or five shots. Prototypical
and matching networks are still strong baselines there, and MAML-style
adaptation is used when the task shift is large enough that a fixed metric
fails.

Meta-learning also appears wherever a model must be personalised from a few
examples: user-specific keyboards and speech models, per-patient adaptation in
medical imaging, per-robot calibration. In those settings the "task" is a
user or a device, the support set is their history, and the meta-learner
provides the initialisation that a few gradient steps can specialise.

Reptile and first-order MAML are the practical choices for large backbones
because they avoid the Hessian. Metric methods are popular in retrieval and
open-set recognition, where the number of classes is not known in advance and
a classifier head cannot be fixed. Relation modules appear in Siamese
architectures for verification, where the learned similarity is the product.

Outside supervised learning, the same episodic objective underlies meta-RL,
where the inner loop is a policy update, and gradient-based hyperparameter
search, where the inner loop is training and the outer loop scores validation
loss — the connection to chapter 13's `hyperparameter_search`.

## References

- Thrun, S., Pratt, L. (1998). *Learning to Learn.* Springer. The framing of
  meta-learning as learning across a task distribution.
- Finn, C., Abbeel, P., Levine, S. (2017). *Model-agnostic meta-learning for
  fast adaptation of deep networks.* ICML. Equations (2)–(5).
- Nichol, A., Achiam, J., Schulman, J. (2018). *On first-order meta-learning
  algorithms.* arXiv. First-order MAML and Reptile, (6).
- Li, Z., Zhou, F., Chen, F., Li, H. (2017). *Meta-SGD: learning to learn
  quickly for few-shot learning.* arXiv. Equations (7)–(8).
- Snell, J., Swersky, K., Zemel, R. (2017). *Prototypical networks for
  few-shot learning.* NeurIPS. Equation (9).
- Vinyals, O., Blundell, C., Lillicrap, T., Wierstra, D. (2016). *Matching
  networks for one-shot learning.* NeurIPS. Attention over the support set,
  (10).
- Sung, F., Yang, Y., Zhang, L., Xiang, T., Torr, P. H. S., Hospedales, T. M.
  (2018). *Learning to compare: relation network for few-shot learning.*
  CVPR. Equation (11).
- Hospedales, T., Antoniou, A., Micaelli, P., Storkey, A. (2022).
  *Meta-learning in neural networks: a survey.* IEEE TPAMI. The taxonomy and
  the evaluation pitfalls.
