# Backpropagation — six ways to differentiate a program

`backprop.py` · Training · Tier 1 · 6 variants

## Intuition

Every training step needs the derivative of one scalar loss with respect to
every parameter. Backpropagation is not a formula for neural networks; it is a
way of running the chain rule over a computation graph so that all those
derivatives cost about as much as one extra forward pass.

The six variants are six answers to "which derivative, and which direction?".
The `scalar_engine` builds a graph of individual numbers, so each operation
shows up as one node with one local derivative. `reverse_mode` does the same
thing over arrays, which is what a deep-learning framework does. `forward_mode`
carries a derivative forward alongside every value, which is cheap when there
are few inputs and many outputs. `jacobian` uses repeated reverse sweeps to
fill a matrix of derivatives. `hessian_vector` differentiates the gradient
itself. `gradient_check` compares the analytic answer against two numerical
ones.

The demo differentiates the same two-layer tanh network four ways and demands
that they agree. Reverse mode and central differences match to $4.8\times
10^{-10}$, forward mode reproduces the reverse-mode gradient to $4.4\times
10^{-16}$, and the complex step lands at machine precision.

## Variants

| variant | object | one call returns | cost |
|:--|:--|:--|:--|
| `scalar_engine` | `Scalar` graph | value and one derivative per input | one sweep |
| `reverse_mode` | `Tensor` graph | full gradient of a scalar | one sweep |
| `forward_mode` | `Dual` numbers | full gradient | one pass per parameter |
| `jacobian` | `Tensor` graph | $m\times n$ matrix | one sweep per output |
| `hessian_vector` | gradients | $Hv$ for a direction $v$ | two gradient calls |
| `gradient_check` | any callable | numeric vs complex error | $3P$ evaluations |

All six share one `mlp_loss`, written with operators only, so the same source
runs on `Scalar`, `Tensor`, `Dual` and plain arrays. That is the point of the
file: autodiff is a property of the operators, not of the model.

## Derivation

### The chain rule on a graph

For a scalar loss $L$ computed by a directed acyclic graph, define the adjoint
of a node $v$ as $\bar v=\partial L/\partial v$. Reverse mode seeds
$\bar L=1$ and visits nodes in reverse topological order:

$$
\bar p \mathrel{+}= \bar v\,\frac{\partial v}{\partial p}
\quad\text{for every edge } p\to v.
\qquad (1)
$$

The sum is essential: if $p$ feeds two consumers, both contributions must be
added. A node that is used $k$ times receives $k$ terms, which is why the
scalar engine stores edges and accumulates rather than assigns.

### The scalar engine

Each operation allocates a node holding its value, its parents, and one
closure per parent that maps the incoming adjoint to that parent's share:

$$
v=ab,\qquad \frac{\partial v}{\partial a}=b,\qquad
\frac{\partial v}{\partial b}=a.
\qquad (2)
$$

```python
    def __mul__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        return Scalar(self.value * other.value, (self, other),
                      (lambda g: g * other.value,
                       lambda g: g * self.value))
```

A node does not know the graph above it, only the local derivative below it.
The sweep in (1) supplies the rest.

### Reverse mode is one forward and one backward pass

Let $C$ be the cost of the forward pass and $P$ the number of parameters.
Reverse mode costs

$$
O(C)_{\text{forward}} + O(C)_{\text{backward}} = O(C),
\qquad (3)
$$

independent of $P$. Forward mode costs $O(PC)$ because each parameter needs
its own seeded pass. For a network with $P=10^6$ and a scalar loss, reverse
mode is the only option. For a function with one input and $10^6$ outputs, the
inequality reverses.

The price of reverse mode is memory: every intermediate needed by a local
derivative must survive until the backward pass. That is the tape, and it is
why checkpointing exists as a separate atom.

### Broadcasting has to be undone

A forward add of shapes $(3,4)$ and $(4,)$ produces $(3,4)$. The adjoint
arrives with shape $(3,4)$, but the second parent only owns $4$ numbers, so the
extra axes are summed away:

$$
\frac{\partial L}{\partial b_j}=\sum_{i}\bar y_{ij}.
\qquad (4)
$$

```python
def _unbroadcast(gradient: np.ndarray, shape: tuple) -> np.ndarray:
    gradient = np.asarray(gradient, dtype=float)
    while gradient.ndim > len(shape):
        gradient = gradient.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and gradient.shape[axis] != 1:
            gradient = gradient.sum(axis=axis, keepdims=True)
    return gradient
```

Getting this wrong is the most common autodiff bug: the gradient has the right
values but the wrong shape, and NumPy silently broadcasts it into something
plausible.

### Forward mode with dual numbers

Adjoin a nilpotent $\epsilon$ with $\epsilon^2=0$. Then

$$
f(a+b\epsilon)=f(a)+f'(a)\,b\,\epsilon,
\qquad (5)
$$

so evaluating $f$ once on a dual number returns both the value and the
directional derivative. Every rule follows mechanically:

$$
(a+b\epsilon)(c+d\epsilon)=ac+(ad+bc)\epsilon,\qquad
\tanh(a+b\epsilon)=\tanh(a)+(1-\tanh^2 a)\,b\,\epsilon.
\qquad (6)
$$

One pass gives $Jv$ for a single direction $v$; $n$ passes over the basis
directions give the full gradient. The demo's single dual pass reproduces
$g^{\top}v$ to $4.4\times10^{-16}$.

### Jacobians

For $f:\mathbb R^n\to\mathbb R^m$ the Jacobian is $J_{ij}=\partial y_i/
\partial x_j$. Reverse mode produces one row per sweep, so the full matrix
costs $m$ backward passes:

$$
J=\begin{bmatrix}\nabla y_1^{\top}\\ \vdots\\ \nabla y_m^{\top}\end{bmatrix}.
\qquad (7)
$$

The demo differentiates $y=\tanh(Ax+b)$ with $A$ of shape $3\times4$. The
closed form is

$$
J=\mathrm{diag}\bigl(1-\tanh^2(Ax+b)\bigr)A,
\qquad (8)
$$

and the three reverse sweeps reproduce it exactly — the difference is $0.0$,
not merely small, because both sides perform the same floating-point products.

### Hessian-vector products

The Hessian is the Jacobian of the gradient. A product $Hv$ therefore costs one
derivative of a gradient, and the implementation here takes the cheapest route
available with only reverse mode:

$$
Hv\approx\frac{\nabla f(\theta+\epsilon v)-\nabla f(\theta-\epsilon v)}
{2\epsilon}.
\qquad (9)
$$

That is two gradient calls and a truncation error of $O(\epsilon^2)$ times the
third derivative. For the demo's quadratic $f=\tfrac12\theta^{\top}A\theta$ the
third derivative is zero, so the product is exact up to roundoff
($9.8\times10^{-12}$). A production HVP instead does forward-over-reverse and
avoids the finite difference entirely; the demo prints the number so the
approximation is visible rather than hidden.

### Why two numerical checks

Central differences truncate at $O(\epsilon^2)$ but subtract two nearly equal
numbers, so the roundoff term grows like machine precision over $\epsilon$:

$$
\text{error}\;\approx\;\frac{\epsilon^2}{6}\bigl|f'''\bigr|
+\frac{2\,\text{eps}_{\text{mach}}\,\bigl|f\bigr|}{\epsilon}.
\qquad (10)
$$

The best $\epsilon$ sits near $10^{-5}$, and the error floor is around
$10^{-10}$ — exactly the $4.8\times10^{-10}$ the demo reports. The complex step
has no subtraction:

$$
\mathrm{Im}\,f(x+ih)/h=f'(x)-\frac{h^2}{6}f'''(x)+O(h^4),
\qquad (11)
$$

so $h=10^{-20}$ gives a relative error at the machine floor
($4.4\times10^{-16}$ here). It requires $f$ to be analytic and implemented
without `abs`, `max`, or comparisons.

## Code walkthrough

The whole reverse pass is a reverse topological sweep with accumulation. Note
that the seed is assigned after the zeroing, and that a node visited twice is
visited once:

```python
def _backward(node, seed: float = 1.0) -> None:
    for item in _topological(node):
        item.grad = np.zeros_like(item.value, dtype=float)
    node.grad = np.asarray(seed, dtype=float) * np.ones_like(node.value)
    for item in reversed(_topological(node)):
        for parent, local in zip(item.parents, item.local, strict=True):
            parent.grad = parent.grad + local(item.grad)
```

`Tensor` needs `__array_priority__` so that `x @ W` with a NumPy `x` defers to
`W.__rmatmul__` instead of trying to convert the graph node into an array. The
matrix-multiply rule is the two adjoints of $y=xW$:

```python
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        if self.value.ndim == 1 and other.value.ndim == 1:
            return Tensor(float(self.value @ other.value), (self, other),
                          (lambda g: g * other.value,
                           lambda g: g * self.value))
        return Tensor(self.value @ other.value, (self, other),
                      (lambda g: g @ other.value.T,
                       lambda g: self.value.T @ g))
```

The reverse driver builds a node per parameter, calls the function, and
requires a scalar. Returning a vector would be a silent bug, so it is checked:

```python
def _reverse_gradient(fn, params: list[np.ndarray]) -> list[np.ndarray]:
    nodes = [Tensor(p) for p in params]
    out = fn(nodes)
    if not isinstance(out, Tensor) or out.value.ndim != 0:
        raise ValueError("the function must return a scalar")
    out.backward()
    return [node.grad.copy() for node in nodes]
```

Forward mode is the same loop shape, but the seed is a unit tangent instead of
an adjoint. Its cost is visible in the nesting: one pass per parameter entry:

```python
    for position, parameter in enumerate(params):
        for index in np.ndindex(parameter.shape):
            seeded = [Dual(p, np.zeros_like(p)) for p in params]
            seeded[position].tangent[index] = 1.0
            out = fn(seeded)
            if not isinstance(out, Dual) or out.value.ndim != 0:
                raise ValueError("the function must return a scalar")
            gradients[position][index] = out.tangent
```

The Jacobian reuses one graph and re-seeds the output. Each `backward` call
resets every adjoint first, so the rows do not contaminate each other:

```python
        flat = out.value.reshape(-1)
        rows = []
        for row in range(flat.size):
            seed = np.zeros_like(out.value)
            seed.reshape(-1)[row] = 1.0
            out.backward(seed)
            rows.append(column.grad.reshape(-1).copy())
        return np.stack(rows)
```

The Hessian-vector product is two gradient evaluations around the point, which
is why it needs no second-order machinery:

```python
        plus = [p + self.eps * v for p, v in zip(params, vector,
                                                  strict=True)]
        minus = [p - self.eps * v for p, v in zip(params, vector,
                                                   strict=True)]
        ahead = _reverse_gradient(fn, plus)
        behind = _reverse_gradient(fn, minus)
        return [(a - b) / (2.0 * self.eps)
                for a, b in zip(ahead, behind, strict=True)]
```

The two numerical checks share one loop. The complex probe perturbs by an
imaginary $10^{-20}$ and divides the imaginary part; there is no subtraction,
which is exactly why it is more accurate than the central difference beside it:

```python
                probe = [p.astype(complex) for p in params]
                probe[position][index] += 1e-20j
                complex_step[position][index] = np.imag(fn(probe)) / 1e-20
```

Because `mlp_loss` is written with `@`, `+`, `-`, `*` and `.mean()`, every
engine above drives it unchanged:

```python
    first, first_bias, second, second_bias = params
    hidden = _tanh(x @ first + first_bias)
    prediction = hidden @ second + second_bias
    difference = prediction - y
    return (difference * difference).mean()
```

## Complexity

Let $C$ be the cost of one forward evaluation, $P$ the number of parameters,
$m$ the number of outputs and $N$ the number of graph nodes.

- **Scalar engine:** $O(N)$ time and $O(N)$ memory; one Python object per
  operation, so it is a teaching tool rather than a runtime.
- **Reverse mode:** one forward and one backward sweep, roughly $2$–$3C$, with
  $O(\text{tape})$ memory. The cost is independent of $P$.
- **Forward mode:** $P$ dual passes, $O(PC)$; memory $O(1)$ beyond the values.
- **Jacobian:** $m$ reverse sweeps, $O(mC)$, or $n$ forward sweeps for a wider
  than tall map. Choose the smaller dimension.
- **Hessian-vector:** two gradient calls, $O(C)$ each, plus the vector
  arithmetic; accuracy $O(\epsilon^2)$.
- **Gradient check:** $2P$ forward evaluations plus $P$ complex ones, so
  $O(PC)$ with a large constant; used on small probes, never in a loop.
- **The demo:** 17 parameters, one hidden layer of width 4, batches of 3. The
  17-parameter loop in the check is 51 evaluations; the 300-step training run
  is 300 forward-backward pairs. Everything is milliseconds.

## Failure modes

**Aliasing breaks the tape.** An in-place update of an array that a node still
references changes the value the backward pass will read. Reverse mode requires
that every tensor saved for the backward pass is treated as immutable.

**`__array_priority__` missing.** Without it, `ndarray @ Tensor` calls the
ndarray's own `matmul`, which tries to convert the node to a 0-d object array
and fails with a shape error far from the real cause. Any wrapper object that
sits on the left of a NumPy operator needs the priority attribute and a
reflected method.

**Broadcast gradients are not summed.** A parameter of shape $(4,)$ used
against a batch of shape $(3,4)$ accumulates three rows. Skipping
`_unbroadcast` gives a shape error at best and a silently transposed gradient
at worst.

**Accumulate, do not assign.** Two consumers of one node must both add into its
adjoint. Assigning the last contribution is the classic bug: the loss still
decreases, just more slowly, and the gradient check is what catches it.

**Reverse mode's memory grows with depth.** The tape holds every activation
whose local derivative needs it. A 100-layer network with large activations
can need more memory for the backward pass than for the parameters.

**Forward mode's cost grows with the parameter count.** For a network it is
$P$ passes, which is why frameworks rarely use it for training. It is the right
tool for a tall Jacobian or a single directional derivative.

**Finite differences have a floor.** The demo's $4.8\times10^{-10}$ is not a
bug; it is the sum of the two error terms in (10). Tuning $\epsilon$ below
$10^{-6}$ makes the answer worse, not better, because cancellation takes over.

**The complex step assumes analyticity.** `abs`, `max`, `relu` and any branch
on a comparison destroy it. The check would still run and return a number, so a
wrong result looks like a right one.

**Gradient checks at kinks are meaningless.** A ReLU has no derivative at zero
and different one-sided derivatives next to it. Central differences there
report a large error for a correct implementation.

**The Hessian-vector product is not exact.** Equation (9) approximates $Hv$ and
is only exact for quadratics. A curvature-aware optimiser that assumes an exact
product inherits the truncation error.

**Node identity, not value identity, defines the graph.** `_topological` keys
on `id()`, so nodes must stay alive for the duration of the sweep, and two
equal arrays are still two nodes.

## Applications

Backpropagation is the engine of every gradient-trained model. Reverse mode is
what PyTorch, TensorFlow, JAX and tinygrad implement; the array graph here is a
condensed version of the same design. Forward mode appears in
sensitivity analysis, in computing directional derivatives for adversarial
robustness, and in differentiating tall maps where the output dimension is
small.

Jacobians and Hessian-vector products are the workhorses of second-order
methods: natural gradient, K-FAC, Hessian-free optimisation, influence
functions and sharpness-aware training all call an HVP. Gradient checking is
the standard unit test for a new layer, and the complex step is the sharpest
version of it when the layer is analytic. The scalar engine is the standard
pedagogical starting point because a graph with one edge per operation makes
the chain rule visible with nothing else in the way.

## References

- Baydin, A. G., Pearlmutter, B. A., Radul, A. A., Siskind, J. M. (2018).
  *Automatic differentiation in machine learning: a survey.* JMLR 18(153).
  The taxonomy of forward and reverse mode.
- Griewank, A., Walther, A. (2008). *Evaluating Derivatives: Principles and
  Techniques of Algorithmic Differentiation.* SIAM. The graph formulation and
  the cost model in (3).
- Pearlmutter, B. A. (1994). *Fast exact multiplication by the Hessian.*
  Neural Computation 6(1). The R-operator behind exact Hessian-vector
  products.
- Maclaurin, D., Duvenaud, D., Adams, R. P. (2015). *Autograd: effortless
  gradients in NumPy.* ICML AutoML workshop. The array-level tape design.
- Martins, J. R. R. A., Sturdza, P., Alonso, J. J. (2003). *The complex-step
  derivative approximation.* ACM TOMS 29(3). Equation (11).
- Karpathy, A. (2022). *micrograd.* A scalar autodiff engine in about a hundred
  lines, the model for the `scalar_engine` variant.
