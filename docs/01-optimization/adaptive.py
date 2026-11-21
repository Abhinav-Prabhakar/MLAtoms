"""Adaptive methods — 7 alternatives to Adam's second moment.

Chapter:   01-optimization
Companion: adaptive.md
Tier 1 · 7 variants: adagrad, adadelta, rmsprop, rmsprop_centered, yogi, signsgd,
sm3

Each one answers "what should the denominator be?" differently: the total squared
gradient (Adagrad), an EMA of it (RMSProp), an EMA of the *variance* (centred
RMSProp), a monotone approximation of it (Yogi), nothing at all (SignSGD), or a
factorised version that costs `O(sum d_i)` instead of `O(prod d_i)` memory (SM3).
"""

from __future__ import annotations

import numpy as np

VARIANTS = frozenset(
    ["adagrad", "adadelta", "rmsprop", "rmsprop_centered", "yogi", "signsgd", "sm3"]
)


class Adaptive:
    """One class, seven denominators. `params` are updated in place."""

    def __init__(
        self,
        params: list[np.ndarray],
        lr: float = 0.01,
        variant: str = "rmsprop",
        beta: float = 0.9,
        beta2: float = 0.999,
        rho: float = 0.95,
        eps: float = 1e-8,
        momentum: float = 0.0,
    ) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        self.params = params
        self.lr = lr
        self.variant = variant
        self.beta = beta
        self.beta2 = beta2
        self.rho = rho
        self.eps = eps
        self.momentum = momentum
        self.t = 0

        # Adagrad / RMSProp / Yogi / SM3
        self.acc = [np.zeros_like(p) for p in params]
        self.acc2 = [np.zeros_like(p) for p in params]      # centred RMSProp mean
        self.delta = [np.zeros_like(p) for p in params]      # Adadelta update RMS
        self.vel = [np.zeros_like(p) for p in params]        # optional momentum
        # SM3: one accumulator per mode, e.g. (rows,) and (cols,) for a matrix.
        self.sm3 = [[np.zeros(s) for s in p.shape] for p in params]

    def _sm3_denominator(self, i: int, g: np.ndarray) -> np.ndarray:
        """v_hat = min over modes of the mode accumulator, broadcast back."""
        accs = self.sm3[i]
        for k, acc in enumerate(accs):
            axes = tuple(ax for ax in range(g.ndim) if ax != k)
            acc += (g * g).sum(axis=axes)
        v = None
        for k, acc in enumerate(accs):
            shape = [1] * g.ndim
            shape[k] = g.shape[k]
            shaped = acc.reshape(shape)
            v = shaped if v is None else np.minimum(v, shaped)
        return v

    def step(self, grads: list[np.ndarray]) -> None:
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads, strict=True)):
            g = np.asarray(g, dtype=float)

            if self.variant == "signsgd":
                d = np.sign(g)
            elif self.variant == "adagrad":
                self.acc[i] += g * g
                d = g / (np.sqrt(self.acc[i]) + self.eps)
            elif self.variant == "adadelta":
                self.acc[i] = self.rho * self.acc[i] + (1 - self.rho) * g * g
                rms_d = np.sqrt(self.delta[i] + self.eps)
                rms_g = np.sqrt(self.acc[i] + self.eps)
                d = (rms_d / rms_g) * g
                self.delta[i] = self.rho * self.delta[i] + (1 - self.rho) * d * d
            elif self.variant == "rmsprop":
                self.acc[i] = self.beta * self.acc[i] + (1 - self.beta) * g * g
                d = g / (np.sqrt(self.acc[i]) + self.eps)
            elif self.variant == "rmsprop_centered":
                self.acc[i] = self.beta * self.acc[i] + (1 - self.beta) * g * g
                self.acc2[i] = self.beta * self.acc2[i] + (1 - self.beta) * g
                var = np.maximum(self.acc[i] - self.acc2[i] ** 2, 0.0)
                d = g / (np.sqrt(var) + self.eps)
            elif self.variant == "yogi":
                self.acc[i] += (1 - self.beta2) * np.sign(g * g - self.acc[i]) * g * g
                self.acc2[i] = self.beta * self.acc2[i] + (1 - self.beta) * g
                mhat = self.acc2[i] / (1 - self.beta ** self.t)
                vhat = self.acc[i] / (1 - self.beta2 ** self.t)
                d = mhat / (np.sqrt(vhat) + self.eps)
            elif self.variant == "sm3":
                vhat = self._sm3_denominator(i, g)
                d = g / (np.sqrt(vhat) + self.eps)
            else:
                raise ValueError(self.variant)

            if self.momentum:
                self.vel[i] = self.momentum * self.vel[i] + d
                d = self.vel[i]
            p -= self.lr * d


def main() -> None:
    rng = np.random.default_rng(0)
    A = np.diag([1.0, 15.0])
    b = np.array([1.0, -1.0])
    xs = np.linalg.solve(A, b)
    grad = lambda x: A @ x - b  # noqa: E731

    # 1. Adagrad's effective rate is lr/sqrt(t) under a constant gradient, so the
    #    distance travelled grows like 2*sqrt(T) rather than linearly.
    p = np.array([0.0])
    opt = Adaptive([p], lr=1.0, variant="adagrad")
    T = 100
    for _ in range(T):
        opt.step([np.ones(1)])
    # sum_{t<=T} t^-1/2 -> 2*sqrt(T) + zeta(1/2), and zeta(1/2) = -1.4604
    approx = 2.0 * np.sqrt(T) - 1.4603545
    assert abs(-p[0] - approx) < 0.1, p
    print(f"adagrad: after {T} unit gradients, |p| = {-p[0]:.2f} "
          f"(2*sqrt(T) + zeta(1/2) = {approx:.2f})")

    # 2. Adadelta needs no learning rate at all: the numerator carries its own
    #    scale. It still converges on the same quadratic.
    x = np.array([4.0, 4.0])
    opt = Adaptive([x], lr=1.0, variant="adadelta", rho=0.9)
    for _ in range(20_000):
        opt.step([grad(x)])
    # The step size is dimensionless, so there is nothing to tune: it converges
    # to the optimum exactly, just slowly (the numerator starts at sqrt(eps)).
    assert np.linalg.norm(x - xs) < 1e-6, x
    print(f"adadelta (lr=1, untuned): ||x - x*|| = {np.linalg.norm(x - xs):.1e}")

    # 3. Centred RMSProp subtracts the mean, so on a *constant* gradient the
    #    denominator collapses to eps and the step explodes. That is the
    #    trade-off: lower variance in the estimate, no floor on the step.
    p1, p2 = np.array([0.0]), np.array([0.0])
    plain = Adaptive([p1], lr=1.0, variant="rmsprop", beta=0.9)
    centred = Adaptive([p2], lr=1.0, variant="rmsprop_centered", beta=0.9)
    for _ in range(100):
        plain.step([np.ones(1)])
        centred.step([np.ones(1)])
    assert abs(p1[0]) < 200 and abs(p2[0]) > 3 * abs(p1[0])
    print(f"constant gradient: rmsprop {p1[0]:.2f}, centred {p2[0]:.2f}")

    # 4. Yogi's accumulator is additive in |g|^2, so after a spike followed by
    #    quiet steps it stays near the spike level, while Adam's decays
    #    geometrically and the effective LR explodes.
    beta2 = 0.9
    v_yogi, v_adam = 0.0, 0.0
    seq = [1.0] + [0.01] * 200
    for gt in seq:
        v_yogi += (1 - beta2) * np.sign(gt * gt - v_yogi) * gt * gt
        v_adam = beta2 * v_adam + (1 - beta2) * gt * gt
    assert v_yogi > 100 * v_adam, (v_yogi, v_adam)
    print(f"after a spike: yogi v = {v_yogi:.4f}, adam v = {v_adam:.1e} "
          f"(adam's LR grows by {np.sqrt(v_yogi / v_adam):.0f}x)")

    # 5. SignSGD moves exactly lr per coordinate and therefore converges to a
    #    ball of radius ~lr/L, not to the optimum.
    x = np.array([3.0, 3.0])
    opt = Adaptive([x], lr=0.01, variant="signsgd")
    for _ in range(500):
        opt.step([grad(x)])
    assert 0 < np.linalg.norm(x - xs) < 0.1, x
    print(f"signsgd: settles {np.linalg.norm(x - xs):.3f} from the optimum (lr = 0.01)")

    # 6. SM3 uses O(sum d_i) memory instead of O(prod d_i). For a 2x3 matrix that
    #    is 5 numbers instead of 6 — the win grows with dimension.
    p = rng.normal(size=(2, 3))
    opt = Adaptive([p], lr=0.1, variant="sm3")
    opt.step([rng.normal(size=(2, 3))])
    stored = sum(a.size for a in opt.sm3[0])
    assert stored == 5 and stored < 6
    print(f"sm3: {stored} accumulator entries for a {p.shape} tensor")

    # 7. Every variant reduces the loss on the shared quadratic.
    for variant, lr in [("adagrad", 0.3), ("adadelta", 1.0), ("rmsprop", 0.01),
                        ("rmsprop_centered", 0.01), ("yogi", 0.01), ("signsgd", 0.005),
                        ("sm3", 0.05)]:
        x = np.array([4.0, 4.0])
        opt = Adaptive([x], lr=lr, variant=variant)
        f0 = 0.5 * x @ A @ x - b @ x
        for _ in range(3000):
            opt.step([grad(x)])
        f1 = 0.5 * x @ A @ x - b @ x
        assert f1 < f0, (variant, f0, f1)
    print("all 7 variants reduce the quadratic objective")
    print("adaptive demo passed")


if __name__ == "__main__":
    main()
