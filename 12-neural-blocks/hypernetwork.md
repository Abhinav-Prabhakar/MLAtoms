# Hypernetworks — weights that are functions of a conditioning vector

`hypernetwork.py` · Neural blocks · Tier 2 · 4 variants

## Intuition

A hypernetwork is a network whose output is another network's weights. Instead
of learning one weight matrix, it learns a function from a conditioning vector
$c$ to a weight matrix $W(c)$. At inference, a new $c$ produces a new target
network without any gradient steps.

The four variants differ in how much of the target they generate.
`hypernet` emits a full $W$ and $b$ for one linear layer — the most expressive
and the most expensive. `weight_generation` emits all four matrices of a
two-layer target, which is weight generation for a small deep network.
`lora_hypernet` emits a low-rank update to a frozen base, so the generated part
is $B(c)A(c)^{\top}$ with $r\ll\min(d_{\text{in}},d_{\text{out}})$.
`conditioned_weights` emits only a per-channel scale and shift, which is the
cheapest form: one frozen matrix and a family of affine maps.

The demo's central check is that a batch of conditions behaves exactly like a
loop of single examples — each row gets its own weights, and no matrix is
shared across the batch. It also checks the low-rank structure of the LoRA
update, that the frozen base really stays frozen, and every gradient against
central differences.

## Variants

| variant | generated | target | frozen base |
|:--|:--|:--|:--|
| `hypernet` | $W,b$ | one linear layer | none |
| `weight_generation` | $W_1,b_1,W_2,b_2$ | two-layer MLP | none |
| `lora_hypernet` | $A,B,b$ | $W_0+BA^{\top}$ | $W_0$ |
| `conditioned_weights` | $\gamma,\beta$ | $xW_0^{\top}$ scaled and shifted | $W_0$ |

At $d_{\text{in}}=4$, $d_{\text{out}}=3$, $d_c=2$, hidden 6, target width 5 and
rank 2, the totals are 123, 319, 149 and 72 parameters. The trainable counts
are 123, 319, 137 and 60 — the frozen base is the difference in the last two.

## Derivation

### The definition

$$
\theta=h(c),\qquad y=f\bigl(x;\theta\bigr).
\qquad (1)
$$

The hypernetwork $h$ is trained end to end: the loss on $y$ backpropagates
through the target's forward pass into $\theta$ and then into $h$'s parameters.
That is the only structural difference from an ordinary model — the target's
"weights" are activations of $h$, so they receive gradients like any other
intermediate array.

### Full weight generation

For one linear target with $W(c)\in\mathbb R^{d_{\text{out}}\times d_{\text{in}}}$,

$$
W(c)=\mathrm{reshape}\bigl(h(c)_{1:\,d_{\text{in}}d_{\text{out}}}\bigr),\qquad
b(c)=h(c)_{d_{\text{in}}d_{\text{out}}+1:\,},
\qquad (2)
$$

$$
y=xW(c)^{\top}+b(c).
\qquad (3)
$$

The cost is $d_{\text{in}}d_{\text{out}}+d_{\text{out}}$ emitted numbers per
example. At the demo's sizes that is 15 numbers from a 6-wide head, and the
whole model is 123 parameters. Two conditions produce two different matrices:
the demo shifts $c$ and confirms that every entry of the $3\times4$ matrix
moves.

### Generating a deep target

For a two-layer target with hidden width $w$,

$$
u=\mathrm{gelu}\bigl(xW_1(c)^{\top}+b_1(c)\bigr),\qquad
y=uW_2(c)^{\top}+b_2(c),
\qquad (4)
$$

with all four objects slices of the same emitted vector. The emitted size grows
as $d_{\text{in}}w+w+wd_{\text{out}}+d_{\text{out}}$, which is why this variant
is the most expensive at 319 parameters. The demo keeps the forward and
backward explicit rather than hiding the target behind a class, because the
point is that the target's parameters are just slices of $h(c)$.

### LoRA as a generated low-rank update

$$
W(c)=W_0+B(c)A(c)^{\top},\qquad
A(c)\in\mathbb R^{d_{\text{in}}\times r},\quad
B(c)\in\mathbb R^{d_{\text{out}}\times r},
\qquad (5)
$$

$$
y=xW_0^{\top}+x\,A(c)\,B(c)^{\top}+b(c),
\qquad (6)
$$

so the update to the weight matrix has rank at most $r$. $W_0$ is a parameter
that is deliberately excluded from `self.gradients`; the only trainable path
into the base is the generated correction. At $r=2$ on a $3\times4$ matrix, the
demo computes the update $BA^{\top}$ and checks that its singular values beyond
the second are zero, then takes a step and confirms that $W_0$ is bit-identical
afterwards.

### Conditioned weights are FiLM

$$
y=\bigl(xW_0^{\top}\bigr)\odot\bigl(1+\gamma(c)\bigr)+\beta(c).
\qquad (7)
$$

Only $2d_{\text{out}}$ numbers are emitted, and the target's mixing of input
channels is still the frozen $W_0$. This is feature-wise linear modulation: the
condition can rescale and offset each output channel but cannot change which
inputs are combined. That is a real restriction — it is the cheapest variant
here and the least expressive.

### Gradients through generated weights

For (3), the weight gradient is an outer product per example:

$$
\frac{\partial L}{\partial W(c)_b}
=\frac{\partial L}{\partial y_b}\,x_b^{\top},\qquad
\frac{\partial L}{\partial b(c)_b}=\frac{\partial L}{\partial y_b},
\qquad (8)
$$

and for (6), with $z_b=x_bA(c)_b$,

$$
\frac{\partial L}{\partial A(c)_b}=x_b^{\top}
\frac{\partial L}{\partial z_b},\qquad
\frac{\partial L}{\partial B(c)_b}
=\frac{\partial L}{\partial y_b}\,z_b^{\top}.
\qquad (9)
$$

Both are batched outer products, which is why the backward pass is a pair of
`einsum` calls. The gradients then flow through the head and the trunk like any
MLP. The demo's worst errors are between $3.4\times10^{-10}$ and
$1.1\times10^{-9}$ across the four variants, with the input and the condition
included in the check.

### Parameter counts

$$
P=(\underbrace{d_cH+H}_{\text{trunk}})
+\bigl(\underbrace{H E+E}_{\text{head}}\bigr)
+\underbrace{\text{base}}_{\text{if frozen}},
\qquad (10)
$$

where $E$ is the number of emitted values: $d_{\text{in}}d_{\text{out}}+
d_{\text{out}}$, or $(d_{\text{in}}w+w)+(wd_{\text{out}}+d_{\text{out}})$, or
$d_{\text{in}}r+d_{\text{out}}r+d_{\text{out}}$, or $2d_{\text{out}}$. At the
demo's defaults the head costs 105, 301, 119 or 42 parameters, and the trunk
costs 18 in every case.

### Complexity

For batch $B$, input width $d_{\text{in}}$, output width $d_{\text{out}}$,
condition width $d_c$ and trunk width $H$:

- **Trunk and head:** $O(Bd_cH+BHE)$ per forward pass; the head dominates
  whenever the emitted vector is large.
- **Full generation:** $O(Bd_{\text{in}}d_{\text{out}})$ to apply the generated
  matrix, and the same again in the backward pass. Memory is the issue: a
  separate $d_{\text{out}}\times d_{\text{in}}$ matrix per example.
- **Deep target:** $O(B(d_{\text{in}}w+wd_{\text{out}}))$, the cost of the
  generated network itself.
- **LoRA:** $O(Br(d_{\text{in}}+d_{\text{out}}))$ instead of
  $O(Bd_{\text{in}}d_{\text{out}})$; linear in $r$.
- **FiLM:** $O(Bd_{\text{out}})$ for the modulation plus the frozen matmul.
- **The demo:** widths of at most 8 and batches of 3, so every product is
  small; the finite-difference loops and the 400-step fit dominate, and the
  file runs in well under a second.

## Code walkthrough

The trunk is an ordinary MLP over the condition. Everything else is a reshape
of its output:

```python
    def _trunk(self, c: np.ndarray):
        p = self.parameters
        pre = c @ p["trunk.W"] + p["trunk.b"]
        hidden = _gelu(pre)
        return hidden, {"c": c, "pre": pre, "hidden": hidden}
```

The full-generation forward slices the emitted vector into a matrix and a bias,
then applies them with a batched contraction. The `boi` index order is
`(batch, out, in)`, so the same einsum would work if every row had a different
matrix:

```python
        if self.variant == "hypernet":
            split = self.in_dim * self.out_dim
            weight = raw[:, :split].reshape(batch, self.out_dim, self.in_dim)
            bias = raw[:, split:]
            out = np.einsum("bi,boi->bo", x, weight) + bias
            cache.update(weight=weight, bias=bias)
            return out, cache
```

Weight generation for a deep target is two einsums with the generated matrices
in the middle. The intermediate activation is cached because the backward pass
needs it for both generated layers:

```python
            pre_hidden = np.einsum("bi,bwi->bw", x, first) + first_bias
            hidden = _gelu(pre_hidden)
            out = np.einsum("bw,bow->bo", hidden, second) + second_bias
```

LoRA keeps the base product separate and adds the low-rank path through the
bottleneck. `reduced` is $x_bA(c)_b$, the only quantity that both weight
gradients need:

```python
            base = x @ p["base.W"].T
            reduced = np.einsum("bi,bir->br", x, down)
            out = base + np.einsum("br,bor->bo", reduced, up) + bias
            cache.update(down=down, up=up, bias=bias, base=base,
                         reduced=reduced)
```

Conditioned weights are five lines, and the forward pass stores the base so
the backward pass can compute the scale gradient without recomputing it:

```python
        gamma = raw[:, :self.out_dim]
        beta = raw[:, self.out_dim:]
        base = x @ p["base.W"].T
        out = base * (1.0 + gamma) + beta
        cache.update(gamma=gamma, beta=beta, base=base)
```

Every variant ends at the same head and trunk. The emitted gradient is
accumulated in `d_raw`, and the rest is an ordinary MLP backward pass:

```python
        g["head.W"] += cache["trunk"]["hidden"].T @ d_raw
        g["head.b"] += d_raw.sum(axis=0)
        d_hidden = d_raw @ p["head.W"].T
        trunk = cache["trunk"]
        d_pre = d_hidden * _gelu_grad(trunk["pre"])
        g["trunk.W"] += trunk["c"].T @ d_pre
        g["trunk.b"] += d_pre.sum(axis=0)
        d_c = d_pre @ p["trunk.W"].T
        return d_x, d_c
```

The Gauss–Newton structure of (8) is visible directly: the weight gradient is
the outer product of the upstream gradient and the input, accumulated over the
batch axis.

## Failure modes

**Conditioning collapse.** Nothing forces $h(c)$ to use $c$. If the trunk's
weights shrink or the head becomes insensitive, the hypernetwork becomes a
constant generator and the target loses its adaptation. Checking that two
conditions give different weights, as the demo does, is a cheap smoke test; a
real training run needs a diversity term or a task distribution that cannot be
solved by one function.

**Per-example weights break the usual batching intuition.** A generated
$W(c_b)$ exists only for row $b$; there is no shared matrix to inspect or prune.
The demo's loop test is the invariant: batched output must equal the loop over
examples. A bug that broadcasts one condition across the batch produces a
plausible loss and wrong semantics.

**Memory grows with the emitted size.** At $B$ examples and a $d\times d$
target, the forward pass holds $B d^2$ weight entries plus their gradients.
Full weight generation is practical for small targets only; LoRA and FiLM exist
because of this wall.

**LoRA's rank is a hard ceiling.** The generated update cannot exceed rank $r$,
so the family of functions the hypernetwork can produce is limited even if the
hypernetwork itself is large. The demo checks the rank directly rather than
trusting the algebra.

**A frozen base must be frozen in the optimiser, not just the backward pass.**
Excluding `base.W` from `self.gradients` keeps the analytic gradient correct,
but an external optimiser that iterates all parameters and applies weight decay
will still move it. Freezing is a property of the update rule.

**FiLM cannot change the input mixing.** Equation (7) rescales and shifts output
channels only. A task that needs a different combination of inputs cannot be
represented, no matter how large the hypernetwork is. That is the expressivity
price of emitting $2d_{\text{out}}$ numbers instead of $d_{\text{in}}
d_{\text{out}}$.

**Generated weights need their own initialisation.** The head's output is
reshaped into a matrix, so the scale of the emitted entries depends on the head
width and on the target's fan-in. The demo's $1/\sqrt H$ head initialisation is
not tuned for the target's fan-in; a real hypernetwork initialises the head so
that the generated matrix has the target's usual variance.

**Condition distribution shift.** The hypernetwork only knows the $c$ values it
was trained on. A new condition far outside that range produces a weight
matrix, but not necessarily a sensible one — the function $h$ extrapolates
however its architecture happens to extrapolate.

**The target's nonlinearity is fixed by the variant.** `weight_generation`
always inserts a GELU between its two generated layers; changing the target's
activation changes the model, not a hyperparameter. That is the same trade-off
as hard-coding an architecture anywhere else.

**No guarantee that the generated network is well-conditioned.** A generated
$W(c)$ can be near-singular for some $c$, and the loss surface in $c$ can be
sharp. Weight normalisation or spectral constraints on the emitted matrix are
the usual mitigations; none is in this minimal implementation.

## Applications

Hypernetworks are used for continual learning, where a task embedding generates
the weights of a task-specific model and the shared hypernetwork carries
knowledge across tasks. They appear in one-shot learning as feed-forward weight
predictors, in neural architecture search as generators of candidate networks,
and in implicit neural representations where a latent code modulates a small
MLP.

The two cheap variants are the ones most often seen in practice. LoRA and its
hypernetwork-generated relatives adapt large language models with a low-rank
update per task or per example. FiLM conditioning is standard in visual
reasoning, style transfer and diffusion models, where a small modulation vector
steers a frozen backbone. Full weight generation remains a research tool for
small targets, but it is the cleanest way to see what "a network that outputs a
network" actually computes.

## References

- Ha, D., Dai, A., Le, Q. V. (2017). *HyperNetworks.* ICLR. The original
  formulation, equations (1)–(3).
- Bertinetto, L., Henriques, J. F., Valmadre, J., Torr, P. H. S., Vedaldi, A.
  (2016). *Learning feed-forward one-shot learners.* NeurIPS. Weight prediction
  for a target classifier.
- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L.,
  Chen, W. (2022). *LoRA: low-rank adaptation of large language models.* ICLR.
  Equations (5) and (6).
- Perez, E., Strub, F., de Vries, H., Dumoulin, V., Courville, A. (2018).
  *FiLM: visual reasoning with a general conditioning layer.* AAAI. Equation
  (7).
- von Oswald, J., Henning, C., Grewe, B. F., Sacramento, J. (2020). *Continual
  learning with hypernetworks.* ICANN. Task-conditioned weight generation for
  sequential tasks.
