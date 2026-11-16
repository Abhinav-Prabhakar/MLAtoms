# SGD

`sgd.py` · Optimization · Tier 1 · 8 variants

## Intuition

Gradient descent is Euler's method on the gradient-flow ODE `θ̇ = −∇f(θ)`, with the
learning rate as the step size. Everything else in this file is a correction for
the fact that Euler's method is a bad ODE solver:

- **Momentum** adds inertia, so the iterate keeps moving through flat regions and
  averages out gradient noise.
- **Nesterov** evaluates the gradient where the momentum is *about to take you*,
  which is what lets it brake before overshooting.
- **Dampening** controls how fast the velocity builds.
- **Weight decay** shrinks parameters that the loss does not care about.
- **Clipping** puts a ceiling on the step when a single batch produces a huge
  gradient.
- **Restarts** empty the velocity when it has become wrong.

## Variants

| Variant | Update | Gains | Costs |
|:--|:--|:--|:--|
| `vanilla` | `θ ← θ − ηg` | nothing to tune, exact convergence theory | slow on ill-conditioned problems |
| `momentum` | `v ← μv + g`, `θ ← θ − ηv` | `√κ` instead of `κ` iterations | one extra buffer per parameter |
| `nesterov` | gradient at `θ − ημv` | brakes earlier; better on convex problems | same memory, one more multiply |
| `dampening` | `v ← μv + (1−d)g` | controls initial velocity, stabilises warmup | another hyper-parameter |
| `weight_decay_l2` | `g ← g + λθ` | shrinks weights through the same path | decay is scaled by `1/√v` in adaptive methods |
| `decoupled_wd` | `θ ← θ − ηλθ − η·update` | exact shrinkage, the AdamW fix | none |
| `clipping` | `g ← g·min(1, c/‖g‖)` | survives a bad batch or an RNN | hides real gradient explosions |
| `nesterov_restart` | Nesterov, `v ← 0` on loss increase | recovers from overshoot | needs a loss value each step |

## Derivation

**Momentum is an EMA.** Unrolling `v_t = μv_{t−1} + g_t`:

$$
v_t = \sum_{k=0}^{t} \mu^{t-k} g_k
$$

so the update is a gradient average with effective horizon `1/(1−μ)` (10 steps at
`μ=0.9`, 100 at `μ=0.99`). On a quadratic with condition number `κ`, heavy-ball
momentum with the optimal `μ = ((√κ−1)/(√κ+1))²` reaches `ε` accuracy in
`O(√κ log(1/ε))` steps instead of `O(κ log(1/ε))` — the single biggest reason
momentum is on by default.

**Nesterov is a lookahead.** Classical momentum uses `g_t`, Nesterov uses
`∇f(θ_t − ημv_t)`:

$$
v_{t+1} = \mu v_t + \eta\,\nabla f(\theta_t - \mu v_t),\qquad \theta_{t+1} = \theta_t - v_{t+1}
$$

which is a first-order approximation of the *implicit* update (`∇f` at the next
iterate). For a quadratic it makes the overshoot correction exact, and it is the
form that attains the `O(1/t²)` accelerated rate.

The implementation uses the algebraically equivalent "gradient + momentum" form
from PyTorch (`g_eff = g + μv`, then `θ ← θ − η·g_eff`), which is why the demo's
first step equals `(1−dampening)·g`.

**L2 vs decoupled decay.** Adding `λθ` to the gradient is *not* the same as
subtracting `ηλθ` from the parameter, unless the update is a plain multiple of the
gradient. With momentum the decay term is also accumulated in `v`, so the effective
decay is `ηλ/(1−μ)` — 10× the nominal coefficient at `μ = 0.9`. With an adaptive
denominator the discrepancy is worse (see `adam.md`). Decoupling is the fix that
made AdamW work.

**Clipping as a trust region.** `g ← g·min(1, c/‖g‖)` bounds the update norm by
`ηc`. It is a cheap, first-order alternative to a line search: it does not change
the direction, only the length, so it cannot fix a wrong direction — only a wrong
scale.

**Restarts.** Momentum is only optimal for a fixed curvature. When the loss
increases, the velocity is pointing uphill; zeroing it reverts to a fresh descent
step. This is the simplest instance of the restart schedules that make accelerated
methods work on non-convex objectives.

## Code walkthrough

The class stores `params`, one velocity buffer per parameter, and a variant name;
every variant is the same loop with a few flags switched on in `__init__`:

```python
        self.momentum = 0.0 if variant == "vanilla" else momentum
        ...
        self.nesterov = nesterov or variant in ("nesterov", "nesterov_restart")
```

That is why `vanilla` ignores a `momentum=0.9` argument silently: the variant, not
the flag, decides. It also means `SGD(params, variant="nesterov")` works without
passing `nesterov=True`, which is the ergonomic point of having variants at all.

`_clip` computes one norm over the concatenation of every parameter's gradient:

```python
        norm = np.sqrt(sum(float((g * g).sum()) for g in grads))
        if norm <= self.clip_norm:
            return grads
        scale = self.clip_norm / (norm + 1e-12)
        return [g * scale for g in grads]
```

Global-norm clipping, not per-tensor: the same `scale` multiplies *every* gradient,
so the direction of the stacked vector is preserved. Per-tensor clipping changes the
relative sizes of the layers' updates and is a different algorithm. The `+ 1e-12`
only guards the division when `norm` is exactly zero, which cannot reach this branch
because `norm <= clip_norm` would have returned.

`step` applies the restart *before* touching the velocity:

```python
        if self.variant == "nesterov_restart" and loss is not None:
            if loss > self.best_loss:
                restart = True
                self.buf = [np.zeros_like(b) for b in self.buf]
            self.best_loss = min(self.best_loss, loss)
```

`best_loss` starts at `inf`, so the first step never restarts. The comparison is
strict (`>`), so a flat loss does not trigger a restart every step. Note that the
buffer is replaced with fresh zeros rather than multiplied by zero: aliasing bugs in
optimiser state are usually about *who else holds a reference* to the array, and a
rebind is the safe form.

The per-parameter body is five lines and the order matters:

```python
            if self.weight_decay and self.variant != "decoupled_wd":
                g = g + self.weight_decay * p

            if self.momentum:
                v *= self.momentum
                v += (1.0 - self.dampening) * g
                g = v.copy() if (not self.nesterov or restart) else g + self.momentum * v
            p -= self.lr * g

            if self.weight_decay and self.variant == "decoupled_wd":
                p -= self.lr * self.weight_decay * p
```

The decay is added **before** the momentum update, so it enters `v` and accumulates
over steps — the doc's `ηλ/(1−μ)` effect. If the `g = g + wd*p` line were moved below
the momentum block, `weight_decay_l2` would silently become decoupled and the demo's
`l2 < dw` assertion would flip. Decoupled decay is applied after the parameter step,
directly to `p`, so it is a true multiplicative shrinkage `p ← p(1 − ηλ)`.

`v *= self.momentum` mutates the buffer in place, which is why `self.buf` can be
created with `np.zeros_like` and never reallocated. `(1−dampening)` multiplies the
*new* gradient only, not the accumulated velocity — that is PyTorch's convention and
the reason the demo's first step is exactly `(1−d)·g`.

The Nesterov line is the one that differs between the three published forms:

```python
                g = v.copy() if (not self.nesterov or restart) else g + self.momentum * v
```

The `else` branch is the "gradient + momentum" form: `g_eff = g + μv`, with `v`
already updated to include `g`. Substituting gives `g_eff = (1+μ)g + μ²v_old`, which
is algebraically the same first step as `∇f(θ − ημv_old)` to `O(η²)`. The `if` branch
covers both non-Nesterov momentum (use `v` directly) *and* the restart step. On a
restart, `v` has just been set to `(1−d)g`; using `g + μv` there would add `μ(1−d)g`
on top of `g` and take a step `(1+μ)` times too large at exactly the moment the
optimiser is trying to recover. That `restart` term inside the conditional is the
non-obvious part of the file and the demo (`restarted < 0.1 * plain`) is what pins it.

## Complexity

`O(P)` time and `O(P)` extra memory per step for every variant (`P` = parameter
count), because the velocity buffer is the same size as the parameters. Vanilla is
`O(P)` time and `O(1)` extra. Nothing here is asymptotically expensive; the cost of
momentum is memory bandwidth, which is why optimiser state often decides whether a
model fits on a device.

## Failure modes

1. **Learning rate above `2/L`.** For a quadratic with largest curvature `L`, SGD
   diverges for `η > 2/L`. The symptom is a loss that decreases for a few steps and
   then explodes; the fix is a warmup or `η ≤ 2/L`.
2. **Momentum + weight decay.** As derived above, the decay is multiplied by
   `1/(1−μ)`. A recipe tuned with `μ=0` will over-regularise at `μ=0.9`.
3. **Nesterov formula confusion.** There are three common forms in the literature
   (lookahead gradient, gradient+momentum, and the two-step "extrapolated point"
   form) that differ by `O(η²μ)` terms. Copying half of one and half of another
   produces a scheme that is neither. Pick one and check its first step.
4. **Clipping as a band-aid.** If you need `clip_norm=1` on a transformer, the
   gradients are exploding for a reason (init, LR, or a missing normalisation);
   clipping makes training survive, not correct.
5. **Restart on noise.** With a noisy minibatch loss, restarts fire constantly and
   the effective momentum collapses. Compare smoothed losses, or restart only when
   the loss exceeds the best by a margin.

## Applications

- **CNNs / small models:** vanilla SGD + momentum + `weight_decay_l2`, cosine or
  step schedule. Often generalises better than Adam at the same budget.
- **RNNs / any recurrent stack:** global-norm clipping is not optional.
- **Fine-tuning:** decoupled weight decay, small `lr`, no momentum warm restart.
- **Batch-size scaling:** linear LR scaling + warmup; momentum stays fixed.

## References

- Polyak, *Some methods in the solution of optimization problems*, 1964 (heavy ball).
- Nesterov, *A method of solving a convex programming problem with convergence rate O(1/k²)*, 1983.
- Sutskever et al., *On the importance of initialization and momentum in deep learning*, ICML 2013.
- Loshchilov & Hutter, *Decoupled Weight Decay Regularization*, 2019. arXiv:1711.05101
- Pascanu, Mikolov & Bengio, *On the difficulty of training recurrent neural networks* (clipping), 2013.
