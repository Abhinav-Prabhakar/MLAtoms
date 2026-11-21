"""Backpropagation — six ways to differentiate a program.

Chapter:   13-training
Companion: backprop.md
Tier 1 · 6 variants: scalar_engine, reverse_mode, forward_mode, jacobian,
hessian_vector, gradient_check
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("scalar_engine", "reverse_mode", "forward_mode", "jacobian",
            "hessian_vector", "gradient_check")


# --------------------------------------------------------------------------- #
# Generic graph machinery
# --------------------------------------------------------------------------- #
def _topological(node) -> list:
    order, seen = [], set()

    def visit(current):
        if id(current) in seen:
            return
        seen.add(id(current))
        for parent in current.parents:
            visit(parent)
        order.append(current)

    visit(node)
    return order


def _backward(node, seed: float = 1.0) -> None:
    for item in _topological(node):
        item.grad = np.zeros_like(item.value, dtype=float)
    node.grad = np.asarray(seed, dtype=float) * np.ones_like(node.value)
    for item in reversed(_topological(node)):
        for parent, local in zip(item.parents, item.local, strict=True):
            parent.grad = parent.grad + local(item.grad)


def _unbroadcast(gradient: np.ndarray, shape: tuple) -> np.ndarray:
    gradient = np.asarray(gradient, dtype=float)
    while gradient.ndim > len(shape):
        gradient = gradient.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and gradient.shape[axis] != 1:
            gradient = gradient.sum(axis=axis, keepdims=True)
    return gradient


def _tanh(value):
    method = getattr(value, "tanh", None)
    return method() if method is not None else np.tanh(value)


def mlp_loss(params, x: np.ndarray, y: np.ndarray):
    """A two-layer tanh network with a mean-squared error.

    Written with operators only, so the same function runs on `Scalar`,
    `Tensor`, `Dual` and plain NumPy arrays.
    """
    first, first_bias, second, second_bias = params
    hidden = _tanh(x @ first + first_bias)
    prediction = hidden @ second + second_bias
    difference = prediction - y
    return (difference * difference).mean()


# --------------------------------------------------------------------------- #
# Variant 1: a scalar graph
# --------------------------------------------------------------------------- #
class Scalar:
    """A node in a scalar expression graph, micrograd style."""

    def __init__(self, value: float, parents=(), local=()):
        self.value = float(value)
        self.grad = 0.0
        self.parents = parents
        self.local = local

    def __add__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        return Scalar(self.value + other.value, (self, other),
                      (lambda g: g, lambda g: g))

    __radd__ = __add__

    def __mul__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        return Scalar(self.value * other.value, (self, other),
                      (lambda g: g * other.value,
                       lambda g: g * self.value))

    __rmul__ = __mul__

    def __neg__(self):
        return Scalar(-self.value, (self,), (lambda g: -g,))

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __pow__(self, power):
        return Scalar(self.value ** power, (self,),
                      (lambda g: g * power * self.value ** (power - 1),))

    def tanh(self):
        out = np.tanh(self.value)
        return Scalar(out, (self,), (lambda g: g * (1.0 - out ** 2),))

    def exp(self):
        out = np.exp(self.value)
        return Scalar(out, (self,), (lambda g: g * out,))

    def backward(self, seed: float = 1.0) -> None:
        _backward(self, seed)


# --------------------------------------------------------------------------- #
# Variant 2: reverse mode over arrays
# --------------------------------------------------------------------------- #
class Tensor:
    """A node in an array expression graph with a reverse pass."""

    __array_priority__ = 1000.0

    def __init__(self, value, parents=(), local=()):
        self.value = np.asarray(value, dtype=float)
        self.grad = np.zeros_like(self.value)
        self.parents = parents
        self.local = local

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor(self.value + other.value, (self, other),
                      (lambda g: _unbroadcast(g, self.value.shape),
                       lambda g: _unbroadcast(g, other.value.shape)))

    __radd__ = __add__

    def __neg__(self):
        return Tensor(-self.value, (self,), (lambda g: -g,))

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor(self.value * other.value, (self, other),
                      (lambda g: _unbroadcast(g * other.value,
                                              self.value.shape),
                       lambda g: _unbroadcast(g * self.value,
                                              other.value.shape)))

    __rmul__ = __mul__

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        if self.value.ndim == 1 and other.value.ndim == 1:
            return Tensor(float(self.value @ other.value), (self, other),
                          (lambda g: g * other.value,
                           lambda g: g * self.value))
        return Tensor(self.value @ other.value, (self, other),
                      (lambda g: g @ other.value.T,
                       lambda g: self.value.T @ g))

    def __rmatmul__(self, other):
        return Tensor(other) @ self

    def tanh(self):
        out = np.tanh(self.value)
        return Tensor(out, (self,), (lambda g: g * (1.0 - out ** 2),))

    def reshape(self, *shape):
        return Tensor(self.value.reshape(shape), (self,),
                      (lambda g: g.reshape(self.value.shape),))

    def mean(self):
        size = self.value.size
        return Tensor(self.value.mean(), (self,),
                      (lambda g: np.full_like(self.value, g / size),))

    def sum(self):
        return Tensor(self.value.sum(), (self,),
                      (lambda g: np.full_like(self.value, g),))

    def backward(self, seed: float = 1.0) -> None:
        _backward(self, seed)


# --------------------------------------------------------------------------- #
# Variant 3: forward mode with dual numbers
# --------------------------------------------------------------------------- #
class Dual:
    """A value and its derivative along one direction."""

    __array_priority__ = 1000.0

    def __init__(self, value, tangent):
        self.value = np.asarray(value, dtype=float)
        tangent = np.asarray(tangent, dtype=float)
        if tangent.shape != self.value.shape:
            tangent = np.broadcast_to(tangent, self.value.shape)
        self.tangent = tangent

    def __add__(self, other):
        other = other if isinstance(other, Dual) else Dual(other, 0.0)
        return Dual(self.value + other.value, self.tangent + other.tangent)

    __radd__ = __add__

    def __neg__(self):
        return Dual(-self.value, -self.tangent)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        other = other if isinstance(other, Dual) else Dual(other, 0.0)
        return Dual(self.value * other.value,
                    self.tangent * other.value + self.value * other.tangent)

    __rmul__ = __mul__

    def __matmul__(self, other):
        other = other if isinstance(other, Dual) else Dual(other, 0.0)
        return Dual(self.value @ other.value,
                    self.tangent @ other.value
                    + self.value @ other.tangent)

    def __rmatmul__(self, other):
        return Dual(other, 0.0) @ self

    def tanh(self):
        out = np.tanh(self.value)
        return Dual(out, (1.0 - out ** 2) * self.tangent)

    def mean(self):
        return Dual(self.value.mean(), self.tangent.mean())


# --------------------------------------------------------------------------- #
# The techniques
# --------------------------------------------------------------------------- #
def _reverse_gradient(fn, params: list[np.ndarray]) -> list[np.ndarray]:
    nodes = [Tensor(p) for p in params]
    out = fn(nodes)
    if not isinstance(out, Tensor) or out.value.ndim != 0:
        raise ValueError("the function must return a scalar")
    out.backward()
    return [node.grad.copy() for node in nodes]


def _forward_gradient(fn, params: list[np.ndarray]) -> list[np.ndarray]:
    gradients = [np.zeros_like(p) for p in params]
    for position, parameter in enumerate(params):
        for index in np.ndindex(parameter.shape):
            seeded = [Dual(p, np.zeros_like(p)) for p in params]
            seeded[position].tangent[index] = 1.0
            out = fn(seeded)
            if not isinstance(out, Dual) or out.value.ndim != 0:
                raise ValueError("the function must return a scalar")
            gradients[position][index] = out.tangent
    return gradients


class Backprop:
    """Entry points for the six techniques, gated by the variant."""

    def __init__(self, variant: str = "reverse_mode", eps: float = 1e-6):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if not np.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be positive and finite")
        self.variant = variant
        self.eps = eps

    def _require(self, *names: str) -> None:
        if self.variant not in names:
            raise ValueError(f"{self.variant!r} does not provide this method")

    def scalar(self, fn, args: tuple[float, ...]):
        self._require("scalar_engine")
        nodes = [Scalar(a) for a in args]
        out = fn(*nodes)
        if not isinstance(out, Scalar):
            raise ValueError("the expression must return a Scalar")
        out.backward()
        return out.value, [node.grad for node in nodes]

    def gradient(self, fn, params: list[np.ndarray]) -> list[np.ndarray]:
        self._require("reverse_mode", "forward_mode")
        params = [np.asarray(p, dtype=float) for p in params]
        if not params or any(p.size == 0 for p in params):
            raise ValueError("params must be a non-empty list of arrays")
        if not all(np.all(np.isfinite(p)) for p in params):
            raise ValueError("params must be finite")
        if self.variant == "forward_mode":
            return _forward_gradient(fn, params)
        return _reverse_gradient(fn, params)

    def jacobian(self, fn, x: np.ndarray) -> np.ndarray:
        self._require("jacobian")
        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or x.size == 0:
            raise ValueError("x must be a non-empty vector")
        column = Tensor(x.reshape(-1, 1))
        out = fn(column)
        if not isinstance(out, Tensor):
            raise ValueError("the function must return a Tensor")
        flat = out.value.reshape(-1)
        rows = []
        for row in range(flat.size):
            seed = np.zeros_like(out.value)
            seed.reshape(-1)[row] = 1.0
            out.backward(seed)
            rows.append(column.grad.reshape(-1).copy())
        return np.stack(rows)

    def hessian_vector(self, fn, params: list[np.ndarray],
                       vector: list[np.ndarray]) -> list[np.ndarray]:
        self._require("hessian_vector")
        if len(params) != len(vector):
            raise ValueError("params and vector must have the same length")
        plus = [p + self.eps * v for p, v in zip(params, vector,
                                                  strict=True)]
        minus = [p - self.eps * v for p, v in zip(params, vector,
                                                   strict=True)]
        ahead = _reverse_gradient(fn, plus)
        behind = _reverse_gradient(fn, minus)
        return [(a - b) / (2.0 * self.eps)
                for a, b in zip(ahead, behind, strict=True)]

    def check(self, fn, params: list[np.ndarray]):
        self._require("gradient_check")
        params = [np.asarray(p, dtype=float) for p in params]
        analytic = _reverse_gradient(fn, params)
        numeric = [np.zeros_like(p) for p in params]
        complex_step = [np.zeros_like(p) for p in params]
        for position, parameter in enumerate(params):
            for index in np.ndindex(parameter.shape):
                forward = [p.copy() for p in params]
                backward = [p.copy() for p in params]
                forward[position][index] += self.eps
                backward[position][index] -= self.eps
                numeric[position][index] = (
                    fn(forward) - fn(backward)) / (2.0 * self.eps)
                probe = [p.astype(complex) for p in params]
                probe[position][index] += 1e-20j
                complex_step[position][index] = np.imag(fn(probe)) / 1e-20
        numeric_error = max(float(np.abs(a - b).max())
                            for a, b in zip(analytic, numeric, strict=True))
        complex_error = max(float(np.abs(a - b).max())
                            for a, b in zip(analytic, complex_step,
                                            strict=True))
        return numeric_error, complex_error


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def demo() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(3, 2))
    teacher = [rng.normal(size=(2, 4)), rng.normal(size=(4,)),
               rng.normal(size=(4, 1)), rng.normal(size=(1,))]
    y = x @ teacher[0] @ np.tanh(teacher[2]) + teacher[3]
    params = [rng.normal(size=(2, 4)) * 0.5, np.zeros(4),
              rng.normal(size=(4, 1)) * 0.5, np.zeros(1)]

    print("backprop. the scalar engine matches a hand derivative:")
    engine = Backprop("scalar_engine")
    value, gradient = engine.scalar(
        lambda a, b: (a * b + a ** 2).tanh(), (0.7, -1.3))
    expected = np.tanh(0.7 * -1.3 + 0.7 ** 2)
    slope = 1.0 - expected ** 2
    assert np.isclose(value, expected)
    assert np.isclose(gradient[0], slope * (-1.3 + 2 * 0.7))
    assert np.isclose(gradient[1], slope * 0.7)
    print(f"  f = tanh(ab + a^2) at (0.7, -1.3): value {value:.6f},")
    print(f"  df/da {gradient[0]:.6f}, df/db {gradient[1]:.6f}")

    print("backprop. reverse mode differentiates the whole network:")
    reverse = Backprop("reverse_mode")
    gradient = reverse.gradient(lambda p: mlp_loss(p, x, y), params)
    numeric = []
    for position, parameter in enumerate(params):
        block = np.zeros_like(parameter)
        for index in np.ndindex(parameter.shape):
            ahead = [p.copy() for p in params]
            behind = [p.copy() for p in params]
            ahead[position][index] += 1e-6
            behind[position][index] -= 1e-6
            block[index] = (mlp_loss(ahead, x, y)
                            - mlp_loss(behind, x, y)) / 2e-6
        numeric.append(block)
    worst = max(float(np.abs(a - b).max())
                for a, b in zip(gradient, numeric, strict=True))
    assert worst < 1e-6
    print(f"  {sum(p.size for p in params)} parameters, one backward pass,")
    print(f"  worst error against central differences {worst:.1e}")

    print("backprop. forward mode matches reverse mode:")
    direction = [rng.normal(size=p.shape) for p in params]
    forward = Backprop("forward_mode")
    forward_gradient = forward.gradient(
        lambda p: mlp_loss(p, x, y), params)
    error = max(float(np.abs(a - b).max())
                for a, b in zip(forward_gradient, gradient, strict=True))
    assert error < 1e-9
    directional = sum(float(np.sum(g * d))
                      for g, d in zip(gradient, direction, strict=True))
    seeded = [Dual(p, d) for p, d in zip(params, direction, strict=True)]
    assert np.isclose(mlp_loss(seeded, x, y).tangent, directional)
    print(f"  full gradient error {error:.1e}; one dual pass gives the")
    print(f"  directional derivative {directional:.6f}")

    print("backprop. the Jacobian is checked against its closed form:")
    matrix = rng.normal(size=(3, 4))
    bias = rng.normal(size=(3, 1))
    jacobian = Backprop("jacobian")
    entries = rng.normal(size=4)
    jac = jacobian.jacobian(
        lambda v: _tanh(matrix @ v + bias), entries)
    out = np.tanh(matrix @ entries.reshape(-1, 1) + bias).reshape(-1)
    expected = np.diag(1.0 - out ** 2) @ matrix
    assert np.allclose(jac, expected)
    assert jac.shape == (3, 4)
    print("  a 3x4 Jacobian from three reverse sweeps matches")
    print(f"  diag(1 - tanh^2) A to {np.abs(jac - expected).max():.1e}")

    print("backprop. a Hessian-vector product costs two gradients:")
    square = rng.normal(size=(4, 4))
    square = square + square.T
    point = rng.normal(size=4)
    vector = rng.normal(size=4)
    hessian = Backprop("hessian_vector", eps=1e-5)
    product = hessian.hessian_vector(
        lambda p: 0.5 * (p[0] @ square @ p[0]), [point], [vector])[0]
    assert np.allclose(product, square @ vector, atol=1e-7)
    other = rng.normal(size=4)
    left = float(vector @ (square @ other))
    right = float(other @ (square @ vector))
    assert np.isclose(left, right)
    print("  for f = p^T A p / 2 the product is A v exactly, to")
    print(f"  {np.abs(product - square @ vector).max():.1e}; symmetry holds")

    print("backprop. central differences and the complex step:")
    checker = Backprop("gradient_check", eps=1e-6)
    numeric_error, complex_error = checker.check(
        lambda p: mlp_loss(p, x, y), params)
    assert numeric_error < 1e-6
    assert complex_error < 1e-11
    print(f"  central differences err {numeric_error:.1e}, complex step")
    print(f"  err {complex_error:.1e}: the complex step is not subtracted")

    print("backprop. training by the gradients we computed:")
    trained = [p.copy() for p in params]
    start = float(mlp_loss(trained, x, y))
    for _ in range(300):
        gradient = _reverse_gradient(
            lambda p: mlp_loss(p, x, y), trained)
        trained = [p - 0.2 * g
                   for p, g in zip(trained, gradient, strict=True)]
    end = float(mlp_loss(trained, x, y))
    assert end < 0.01 * start
    print(f"  300 steps take the loss from {start:.4f} to {end:.6f}")

    print("backprop. malformed calls are refused:")
    guard = Backprop("reverse_mode")
    bad = [
        lambda: Backprop("autograd"),
        lambda: Backprop("reverse_mode", eps=0.0),
        lambda: Backprop("reverse_mode", eps=-1.0),
        lambda: Backprop("forward_mode", eps=np.inf),
        lambda: Backprop("jacobian").gradient(
            lambda p: mlp_loss(p, x, y), params),
        lambda: Backprop("reverse_mode").jacobian(
            lambda v: v, np.zeros(2)),
        lambda: Backprop("reverse_mode").hessian_vector(
            lambda p: mlp_loss(p, x, y), params, params),
        lambda: guard.check(lambda p: mlp_loss(p, x, y), params),
        lambda: guard.scalar(lambda a: a, (1.0,)),
        lambda: guard.gradient(lambda p: 1.0, params),
        lambda: guard.gradient(lambda p: p[0] * 2.0, params),
        lambda: Backprop("forward_mode").gradient(
            lambda p: p[0], params),
        lambda: guard.gradient(lambda p: mlp_loss(p, x, y), []),
        lambda: guard.gradient(
            lambda p: mlp_loss(p, x, y), [np.full((2, 2), np.nan)]),
        lambda: Backprop("scalar_engine").scalar(
            lambda a, b: 2.0, (1.0, 2.0)),
        lambda: Backprop("jacobian").jacobian(
            lambda v: v, np.zeros((2, 2))),
        lambda: Backprop("jacobian").jacobian(
            lambda v: v, np.zeros(0)),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except ValueError:
            refused += 1
    assert refused == len(bad)
    print(f"  {refused} of {len(bad)}: an unknown variant, a bad or")
    print("  infinite step, a method used on the wrong variant, a vector-")
    print("  output or constant expression, empty or non-finite")
    print("  parameters, a mismatched Hessian direction, and a Jacobian")
    print("  input that is not a non-empty vector")
    print("backprop demo passed")

if __name__ == "__main__":
    demo()
