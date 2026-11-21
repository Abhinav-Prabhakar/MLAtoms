"""SGD — 8 variants of the update rule that everything else is compared to.

Chapter:   01-optimization
Companion: sgd.md
Tier 1 · 8 variants: vanilla, momentum, nesterov, dampening, weight_decay_l2,
decoupled_wd, clipping, nesterov_restart

The class holds a list of parameter arrays and the matching state, so one object
optimises a whole model. `step(grads, loss=...)` is the only entry point; `loss`
is used by `nesterov_restart` and ignored by the rest.

Every variant is the same three lines with different bookkeeping:

    v <- mu*v + (1-dampening)*grad
    grad_eff <- grad + mu*v        (only for nesterov)
    p <- p - lr*(grad_eff + wd*p)  (coupled weight decay)
"""

from __future__ import annotations

import numpy as np

VARIANTS = frozenset(
    ["vanilla", "momentum", "nesterov", "dampening", "weight_decay_l2",
     "decoupled_wd", "clipping", "nesterov_restart"]
)


class SGD:
    """Stochastic gradient descent with the usual knobs.

    params : list of float arrays, modified in place.
    variant selects the flavour; the flags (momentum, dampening, weight_decay,
    clip_norm) are shared, and each variant turns on the ones it needs.
    """

    def __init__(
        self,
        params: list[np.ndarray],
        lr: float = 0.01,
        variant: str = "vanilla",
        momentum: float = 0.9,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        clip_norm: float | None = None,
        nesterov: bool = False,
    ) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        self.params = params
        self.lr = lr
        self.variant = variant
        self.momentum = 0.0 if variant == "vanilla" else momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.clip_norm = clip_norm
        self.nesterov = nesterov or variant in ("nesterov", "nesterov_restart")
        if variant == "nesterov_restart":
            self.momentum = momentum
        self.buf = [np.zeros_like(p) for p in params]
        self.best_loss = np.inf
        self.t = 0

    # -- helpers ----------------------------------------------------------- #
    def _clip(self, grads: list[np.ndarray]) -> list[np.ndarray]:
        if self.clip_norm is None:
            return grads
        norm = np.sqrt(sum(float((g * g).sum()) for g in grads))
        if norm <= self.clip_norm:
            return grads
        scale = self.clip_norm / (norm + 1e-12)
        return [g * scale for g in grads]

    # -- the update -------------------------------------------------------- #
    def step(self, grads: list[np.ndarray], loss: float | None = None) -> None:
        self.t += 1
        grads = self._clip([np.asarray(g, dtype=float) for g in grads])

        restart = False
        if self.variant == "nesterov_restart" and loss is not None:
            if loss > self.best_loss:
                restart = True
                self.buf = [np.zeros_like(b) for b in self.buf]
            self.best_loss = min(self.best_loss, loss)

        for p, g, v in zip(self.params, grads, self.buf, strict=True):
            # Coupled L2 (classic) vs decoupled (AdamW-style) weight decay.
            if self.weight_decay and self.variant != "decoupled_wd":
                g = g + self.weight_decay * p

            if self.momentum:
                v *= self.momentum
                v += (1.0 - self.dampening) * g
                # Nesterov evaluates the gradient one momentum step ahead.
                g = v.copy() if (not self.nesterov or restart) else g + self.momentum * v
            p -= self.lr * g

            if self.weight_decay and self.variant == "decoupled_wd":
                p -= self.lr * self.weight_decay * p


def quadratic(A: np.ndarray, b: np.ndarray):
    """f(x) = 1/2 x^T A x - b^T x with analytic gradient, for the demo."""
    return (
        lambda x: 0.5 * float(x @ A @ x) - float(b @ x),
        lambda x: A @ x - b,
    )


def main() -> None:
    # 1. Vanilla SGD converges on a convex quadratic, at the expected rate.
    A = np.diag([1.0, 10.0])
    b = np.array([1.0, -2.0])
    f, grad = quadratic(A, b)
    x_star = np.linalg.solve(A, b)

    x = np.array([5.0, 5.0])
    opt = SGD([x], lr=0.05, variant="vanilla")
    for _ in range(300):
        opt.step([grad(x)])
    assert np.linalg.norm(x - x_star) < 1e-3, x
    print(f"vanilla: ||x - x*|| = {np.linalg.norm(x - x_star):.2e}")

    # 2. Momentum gets there in fewer steps than vanilla.
    def steps_to_tol(variant: str, tol: float = 1e-6, iters: int = 500) -> int:
        x = np.array([5.0, 5.0])
        o = SGD([x], lr=0.02, variant=variant, momentum=0.9)
        for i in range(iters):
            o.step([grad(x)])
            if np.linalg.norm(x - x_star) < tol:
                return i + 1
        return iters

    n_van, n_mom, n_nes = steps_to_tol("vanilla"), steps_to_tol("momentum"), steps_to_tol("nesterov")
    assert n_mom < n_van and n_nes <= n_mom + 5, (n_van, n_mom, n_nes)
    print(f"steps to 1e-6: vanilla {n_van}, momentum {n_mom}, nesterov {n_nes}")

    # 3. Dampening slows the momentum build-up: after one step,
    #    buf = (1-dampening)*g, so the first step is exactly lr*(1-dampening)*g.
    p0 = np.array([1.0])
    o = SGD([p0], lr=1.0, variant="dampening", momentum=0.9, dampening=0.5)
    o.step([np.array([2.0])])
    assert np.allclose(o.buf[0], 1.0) and np.allclose(p0, [0.0])
    print("dampening: first step = (1-dampening)*g")

    # 4. Coupled vs decoupled decay differ once momentum is in play: the coupled
    #    version pushes the decay term through the momentum buffer, so it decays
    #    *harder* than the same coefficient applied decoupled.
    l2 = np.array([1.0])
    dw = np.array([1.0])
    o1 = SGD([l2], lr=0.1, variant="weight_decay_l2", weight_decay=0.1, momentum=0.9)
    o2 = SGD([dw], lr=0.1, variant="decoupled_wd", weight_decay=0.1, momentum=0.9)
    for _ in range(10):
        o1.step([np.array([0.01])])
        o2.step([np.array([0.01])])
    assert not np.allclose(l2, dw), (l2, dw)
    assert l2[0] < dw[0], (l2, dw)
    print(f"weight decay after 10 steps: coupled {l2[0]:.4f}, decoupled {dw[0]:.4f}")

    # 5. Gradient clipping bounds the global norm, and only when it exceeds it.
    p = np.array([0.0, 0.0])
    o = SGD([p], lr=1.0, variant="clipping", clip_norm=1.0)
    o.step([np.array([3.0, 4.0])])
    assert np.allclose(p, [-0.6, -0.8]), p          # 5.0 -> 1.0
    small = np.array([0.0])
    o = SGD([small], lr=1.0, variant="clipping", clip_norm=1.0)
    o.step([np.array([0.1])])
    assert np.allclose(small, [-0.1])
    print("clipping: norm 5 -> 1, small gradients untouched")

    # 6. Restarts fix the oscillation that high momentum causes when curvature
    #    changes. Same lr, same momentum; only the restart rule differs.
    A2 = np.diag([1.0, 400.0])       # badly conditioned on purpose
    f2, grad2 = quadratic(A2, np.zeros(2))

    def run(with_restart: bool, iters: int = 200) -> float:
        x = np.array([1.0, 1.0])
        o = SGD([x], lr=0.002, variant="nesterov_restart" if with_restart else "nesterov",
                momentum=0.95)
        for _ in range(iters):
            o.step([grad2(x)], loss=f2(x))
        return f2(x)

    plain, restarted = run(False), run(True)
    assert np.isfinite(plain) and restarted < 0.1 * plain, (plain, restarted)
    print(f"restart: loss {restarted:.3e} vs plain nesterov {plain:.3e}")
    print("sgd demo passed")


if __name__ == "__main__":
    main()
