# Pruning — eight ways to delete weights and keep the accuracy

`pruning.py` · Training · Tier 2 · 8 variants

## Intuition

A trained network contains far more weights than the task needs, so most of
them can be set to zero with little loss. That part is easy. The two decisions
that actually matter are *which* weights to delete and *when* to delete them,
and the eight variants here are different answers to those two questions.

The implementation makes the deletion explicit with a binary mask per matrix.
The mask multiplies the weight in the forward pass, so a pruned weight
contributes nothing, and it multiplies the gradient in the backward pass, so a
pruned weight never moves again. Every variant then reduces to a rule for
filling those masks: the weight's magnitude, its accumulated movement, the
norm of the hidden unit it belongs to, the singular values of the matrix, or a
schedule that grows the masks over training.

The demo trains a 643-parameter MLP on a three-class problem, then compresses
it. At 80% sparsity seven of the eight variants keep at least 0.979 accuracy,
which says something about how redundant the model is, not about how clever the
criteria are. The separation appears at 98%: a single prune collapses to 0.333,
iterative pruning reaches 0.662, rewinding to the initial weights reaches 0.967,
and a gradual schedule reaches 0.996. When the budget is genuinely tight, *when*
matters more than *which*.

## Variants

| variant | score | unit | decision |
|:--|:--|:--|:--|
| `magnitude` | $\lvert w\rvert$ | weight | iterative, per layer |
| `one_shot` | $\lvert w\rvert$ | weight | once, after training |
| `unstructured` | $\lvert w\rvert$ | weight | once, global threshold |
| `movement` | $\sum_t\lvert w^{(t)}g^{(t)}\rvert$ | weight | once, accumulated |
| `structured` | $\lVert W_{:,j}\rVert_2\lVert W_{j,:}\rVert_2$ | hidden unit | once |
| `gradual` | $\lvert w\rvert$ | weight | cubic schedule during training |
| `lottery_ticket` | $\lvert w\rvert$ | weight | iterative, rewind to init |
| `svd_prune` | singular values | rank | low-rank factorisation |

The schedule column is the interesting one once the budget gets tight. The
score column decides *which* weights die; the schedule decides whether the
network gets a chance to route around the damage.

## Derivation

### The masked network

Let $M_l$ be a binary mask the shape of $W_l$. The forward pass and the
gradient are both masked:

$$
h=\tanh\!\left((W_1\odot M_1)\,x+b_1\right),\qquad
z=(W_2\odot M_2)\,h+b_2,
\qquad (1)
$$

$$
\frac{\partial L}{\partial W_l}=M_l\odot G_l,\qquad
W_l\leftarrow W_l-\eta\,M_l\odot G_l.
\qquad (2)
$$

Equation (2) is why a mask is not merely a view: an entry with $M_{ij}=0$ has
zero gradient, so it stays exactly where it was and can be restored later
without a saved copy. The pruned values are dead weight in memory but they are
never read by the forward pass, which is why `n_stored` counts the mask rather
than the array.

For a matrix with $k$ survivors this costs

$$
\text{stored}=k+\underbrace{\dim b_1+\dim b_2}_{\text{never pruned}},
\qquad (3)
$$

so the demo's per-layer 80% model keeps 121 of 608 weights plus 35 biases and
reports 156 numbers against 643.

### What to remove

The simplest score is the weight itself. Magnitude pruning keeps the $k$
largest entries,

$$
M_{ij}=\mathbf{1}\!\left[\,|W_{ij}|\ge \tau_k\,\right],
\qquad (4)
$$

where $\tau_k$ is the $k$-th largest magnitude. This is a first-order
approximation to the loss change from deleting a weight. Expanding around the
trained point, the second-order model of LeCun's optimal brain damage is

$$
\delta L_i\approx \tfrac12\,W_i^2\,H_{ii},
\qquad (5)
$$

so magnitude is exact only when the diagonal Hessian is constant across
weights. It ignores the fact that a weight can be large and still redundant
because another unit already computes the same feature.

Movement pruning replaces the static score with the first-order term
accumulated along the training trajectory:

$$
s_{ij}=\sum_{t}\left|W^{(t)}_{ij}\,g^{(t)}_{ij}\right|,
\qquad (6)
$$

which is the total first-order loss change attributable to that weight, and it
is what the demo accumulates in the last 100 steps before masking. On the demo
task it flips 104 of 512 first-layer decisions relative to a pure magnitude
rule at the same sparsity, and scores 0.967 instead of 1.000: a different mask
is not automatically a better one.

### Global against per-layer budgets

A per-layer rule spends the sparsity evenly, $k_l=\lfloor(1-s)|W_l|\rfloor$,
while a global rule spends it where the large weights are:

$$
\sum_l k_l=(1-s)\sum_l|W_l|.
\qquad (7)
$$

Both target the same total, but the global threshold is free to empty a layer
whose weights happen to be small. In the demo it zeroes 82% of `W1` and 70% of
`W2`, and still scores 0.988; the layer-wise rule lands exactly on 80/80 and
scores 1.000. Neither is universally right: global is better when layers have
very different scales, per-layer when every layer is load-bearing.

### Structured pruning

Structured pruning removes whole hidden units, which is the only kind that
makes the matrix physically smaller. The score for unit $j$ multiplies its
incoming and outgoing norms:

$$
s_j=\lVert W_{1,:,j}\rVert_2\;\lVert W_{2,j,:}\rVert_2,
\qquad (8)
$$

and the same indicator zeroes column $j$ of $W_1$ and row $j$ of $W_2$. In the
demo six of 32 units survive at 80% sparsity, accuracy stays at 1.000, and the
stored count is 149 — the smallest of the mask-based variants, because the
surviving rows and columns are genuinely gone rather than multiplied by zero.

### Schedules: gradual, iterative, rewinding

A gradual schedule raises the target sparsity during one training run:

$$
s_t=s\left(1-\left(1-\tfrac{t}{T}\right)^3\right),
\qquad (9)
$$

so the model is dense when it is still learning the task and sparse only once
the easy structure has been found. Iterative pruning does the same in discrete
rounds and retrains between them:

$$
s_r=1-(1-s)^{r/R},
\qquad (10)
$$

with $R$ rounds. The lottery-ticket variant adds one extra step: before each
round it rewinds the weights to their initial values,

$$
W^{(r)}\leftarrow W^{(0)}\odot M^{(r-1)},
\qquad (11)
$$

so the mask and the weights are selected together instead of the mask being
fitted to whatever weights the previous round happened to reach. On the demo,
at 98% sparsity, one prune gives 0.333, iterative pruning 0.662, rewinding
0.967, and the gradual schedule 0.996.

### Low-rank pruning

The last variant does not mask at all. It replaces each matrix by its truncated
singular value decomposition,

$$
W\approx U_r\,\mathrm{diag}(\sigma_{1:r})\,V_r^{\top},
\qquad (12)
$$

which stores $r(d+k)$ numbers instead of $dk$. The demo allocates rank greedily
across both matrices, always buying the singular value with the largest energy
per stored number:

$$
\text{gain}_i=\frac{\sigma_i^2}{d+k},
\qquad (13)
$$

until the entry budget or $1-\text{tolerance}$ of the total energy is reached.
At 50% it keeps ranks 4 and 3 and scores 1.000; at 80% the budget buys rank 1
for `W1` and rank 2 for `W2`, and accuracy falls to 0.667. A rank-one output
layer cannot separate three classes, and no amount of clever allocation fixes
that: the constraint is the rank, not the criterion.

## Code walkthrough

The mask enters in exactly two places, and both are one expression. Forward:

```python
def _forward(parameters: dict, masks: dict, x: np.ndarray):
    pre = x @ (parameters["W1"] * masks["W1"]) + parameters["b1"]
    hidden = np.tanh(pre)
    logits = hidden @ (parameters["W2"] * masks["W2"]) + parameters["b2"]
    return logits, {"x": x, "pre": pre, "hidden": hidden}
```

Backward multiplies each gradient by the same mask, including the hidden-layer
gradient that flows through the masked second matrix:

```python
    dhidden = (dlogits @ (parameters["W2"] * masks["W2"]).T) * (
        1.0 - cache["hidden"] ** 2)
    grads = {
        "W1": x.T @ dhidden * masks["W1"],
        "b1": dhidden.sum(axis=0),
        "W2": cache["hidden"].T @ dlogits * masks["W2"],
        "b2": dlogits.sum(axis=0),
    }
```

Choosing the survivors is a stable top-$k$ over the flattened score, so ties do
not depend on the platform's sort order:

```python
def _keep_top(scores: np.ndarray, sparsity: float, shape) -> np.ndarray:
    flat = np.asarray(scores, dtype=float).reshape(-1)
    keep = max(1, int(round((1.0 - sparsity) * flat.size)))
    keep = min(keep, flat.size)
    order = np.argsort(-flat, kind="stable")[:keep]
    mask = np.zeros(flat.size)
    mask[order] = 1.0
    return mask.reshape(shape)
```

The global variant concatenates both layers before taking the top-$k$, then
slices the result back apart; every other mask variant applies the same helper
per layer:

```python
        if self.variant == "unstructured":
            flat = np.concatenate([scores[name].reshape(-1)
                                   for name in LEARNED])
            keep = _keep_top(flat, target, flat.shape)
            offset = 0
            for name in LEARNED:
                size = scores[name].size
                mask = keep[offset:offset + size].reshape(scores[name].shape)
                self.masks[name] = self.masks[name] * mask
                offset += size
```

Movement accumulates the product of weight and gradient at every step, which
is the discrete form of (6), before the same `_prune_masks` call consumes it:

```python
    def _accumulate(self, x, y, steps: int, lr: float) -> None:
        for _ in range(steps):
            _, grads, _ = _loss_and_grads(self.parameters, self.masks, x, y)
            for name in LEARNED:
                self.scores[name] += np.abs(
                    self.parameters[name] * grads[name])
            self.history.append(self._step(x, y, lr))
```

The gradual schedule is (9) evaluated once per step, with a prune every
`interval` steps and a final prune at the end:

```python
    def _fit_gradual(self, x, y, steps: int, lr: float,
                     interval: int) -> None:
        for step in range(steps):
            fraction = (step + 1) / steps
            target = self.sparsity * (1.0 - (1.0 - fraction) ** 3)
            if step % interval == 0 or step == steps - 1:
                self._prune_masks(target)
            self.history.append(self._step(x, y, lr))
```

Iterative pruning is the same loop over rounds, and the `rewind` flag is the
only difference between `magnitude` and `lottery_ticket`:

```python
    def _fit_iterative(self, x, y, steps: int, lr: float,
                       rewind: bool) -> None:
        round_steps = max(1, steps // self.rounds)
        for round_index in range(self.rounds):
            if rewind:
                for name in LEARNED:
                    self.parameters[name] = self.init[name] * self.masks[name]
                for key in ("b1", "b2"):
                    self.parameters[key] = self.init[key].copy()
            self._train(x, y, round_steps, lr)
```

Structured pruning scores units rather than entries, then broadcasts one
indicator into a column mask and a row mask:

```python
    def _prune_structured(self) -> None:
        norm_in = np.linalg.norm(self.parameters["W1"], axis=0)
        norm_out = np.linalg.norm(self.parameters["W2"], axis=1)
        score = norm_in * norm_out
        keep = max(1, int(round((1.0 - self.sparsity) * self.hidden)))
        order = np.argsort(-score, kind="stable")[:keep]
        keep_mask = np.zeros(self.hidden)
        keep_mask[order] = 1.0
        self.masks["W1"] = keep_mask[None, :] * np.ones_like(self.masks["W1"])
        self.masks["W2"] = keep_mask[:, None] * np.ones_like(self.masks["W2"])
```

The low-rank variant is the only one that changes the arrays rather than the
mask, and its loop is (13) written as a greedy auction over ranks:

```python
        while recovered < (1.0 - self.tolerance) * total:
            best = None
            for name in LEARNED:
                singular = factor[name][1]
                if ranks[name] >= singular.size:
                    continue
                cost = (self.parameters[name].shape[0]
                        + self.parameters[name].shape[1])
                if used + cost > budget:
                    continue
                gain = float(singular[ranks[name]] ** 2) / cost
                if best is None or gain > best[0]:
                    best = (gain, name, cost)
            if best is None:
                break
            _, name, cost = best
            recovered += float(factor[name][1][ranks[name]] ** 2)
            ranks[name] += 1
            used += cost
```

## Complexity

For a weight matrix of shape $d\times k$ and a batch of $B$ examples:

- **Masked forward and backward:** still $O(Bdk)$; multiplying by a mask does
  not skip work. A sparse mask saves memory, not time, unless the storage and
  kernel are also sparse.
- **Choosing a mask:** $O(dk\log(dk))$ for the sort, done once per prune event.
  Gradual pruning repeats it every `interval` steps, which is why the interval
  is a real cost knob.
- **Structured pruning:** reduces the hidden width from $h$ to $h'$, so the
  matmuls shrink to $O(Bdh')+O(Bh'c)$ — the only variant here that is faster
  at inference on ordinary hardware.
- **SVD pruning:** one $O(\min(d,k)\,dk)$ factorisation per matrix, then
  $O(r(d+k))$ storage and $O(Br(d+k))$ per forward pass.
- **The demo:** eight variants over 500 steps on 240 examples, plus four
  schedules and a finite-difference-free training loop. The whole run is under a
  second, and the assertions are on stored counts and held-out accuracy rather
  than on gradients, because pruning never changes the gradients, only gates
  them.

## Failure modes

**A single prune at 98% destroys the model.** The demo reports 0.333 for
`one_shot` against 1.000 for the dense model. The mask was chosen from weights
that had never operated under sparsity, and the next two variants show what a
schedule buys: 0.662 for four rounds of prune-and-retrain, 0.967 for rewinding.

**Iterative pruning is not the same as a good mask.** `magnitude` reaches 0.662
at 98%, better than one shot and far worse than rewinding or gradual. Four
rounds of prune-and-retrain help, but the later rounds are fitting weights that
already carry the damage of the early rounds.

**The lottery ticket's win is over reinitialisation, not over magnitude.** At
80% every method scores 1.000; at 98% rewinding reaches 0.967. The control in
the original paper is a random reinitialisation at the same mask, which the
demo does not include, so the honest reading is that rewinding helps this task,
not that it is universally the best criterion.

**Global thresholds can empty a layer.** The demo's global rule gives `W2` 70%
sparsity while `W1` gets 82%; on a network with one narrow layer, the same rule
can delete all of it. The per-layer rule exists for that reason.

**Structured pruning can delete a class.** Unit scores are aggregated over all
outputs, so a unit that only serves one class can look unimportant. The demo is
lucky — six units carry three classes — but the failure is the reason structured
pruning is usually followed by fine-tuning.

**Sparsity is not speed.** A mask of 80% zeros still runs a dense matmul at
full cost. Only `structured` and `svd_prune` reduce compute, and only the
structured one reduces it on hardware that cannot exploit sparsity.

**The mask is applied to gradients, not to the optimiser state.** If the
optimiser keeps momentum for a pruned weight, restoring the mask later resumes
with stale momentum. This implementation is plain SGD and sidesteps the issue;
Adam-based pruning stacks have to zero the state explicitly.

**Pruning before training is a different algorithm.** `one_shot` here prunes
after training. Pruning at initialisation and then training the sparse mask is
the other one-shot recipe, and it behaves differently because the surviving
weights never see the dense solution.

**Low rank is a hard constraint, not a budget.** At 80% sparsity `svd_prune`
keeps rank 1 in `W1` and rank 2 in `W2` and scores 0.667, because three classes
need a rank-two output. Allocating the same number of stored values as a mask
does not buy the same function class.

**The reported sparsity is over prunable weights only.** Biases are never
pruned here, so total sparsity is lower than the per-matrix number. The demo
prints both `n_stored` and `n_dense` for this reason.

## Applications

Magnitude and movement pruning are the default compression step for large
language models, usually combined with quantisation: prune to 50–60%, then
quantise what remains. Movement pruning's accumulated score is the basis of
several production recipes because it can be computed during fine-tuning
without a separate pass. Structured pruning is what ships to phones and
accelerators, because removing a channel is the only kind of sparsity that a
dense kernel can exploit.

The lottery-ticket line of work changed how people think about over-parameterised
training: the mask found on a dense run is often trainable from the original
initialisation, which suggests the dense network's role is partly to find a
good sparse subnetwork. Low-rank pruning is the bridge to the previous atom:
truncated SVD is the same decomposition that LoRA learns, applied to a frozen
weight instead of an update. In practice the strongest compression pipelines
stack all three — prune for zeros, factorise for rank, quantise for bits — and
distil the result into a smaller dense model.

## References

- LeCun, Y., Denker, J. S., Solla, S. A. (1990). *Optimal brain damage.*
  NeurIPS. The diagonal-Hessian score of (5).
- Hassibi, B., Stork, D. G. (1993). *Second order derivatives for network
  pruning: optimal brain surgeon.* NeurIPS. The full-Hessian correction that
  magnitude ignores.
- Han, S., Pool, J., Tran, J., Dally, W. J. (2015). *Learning both weights and
  connections for efficient neural networks.* NeurIPS. Iterative magnitude
  pruning, (10).
- Li, H., Kadav, A., Durdanovic, I., Samet, H., Graf, H. P. (2017). *Pruning
  filters for efficient ConvNets.* ICLR. The structured score of (8).
- Zhu, M., Gupta, S. (2017). *To prune, or not to prune: exploring the efficacy
  of pruning for model compression.* arXiv:1710.01878. The cubic schedule, (9).
- Frankle, J., Carbin, M. (2019). *The lottery ticket hypothesis: finding sparse,
  trainable neural networks.* ICLR. Rewinding, (11).
- Sanh, V., Wolf, T., Rush, A. M. (2020). *Movement pruning: adaptive sparsity
  by fine-tuning.* NeurIPS. The accumulated score of (6).
- Denton, E., Zaremba, W., Bruna, J., LeCun, Y., Fergus, R. (2014). *Exploiting
  linear structure within convolutional networks for efficient evaluation.*
  NeurIPS. Low-rank factorisation of trained weights, (12).
- Frankle, J., Dziugaite, G. K., Roy, D. M., Carbin, M. (2020). *Linear mode
  connectivity and the lottery ticket hypothesis.* ICML. Why rewinding beats
  reinitialisation.
- Sun, M., Liu, Z., Bair, A., Kolter, J. Z. (2023). *A simple and effective
  pruning approach for large language models.* arXiv:2306.11695. Magnitude with
  activations, the modern descendant of (4).
