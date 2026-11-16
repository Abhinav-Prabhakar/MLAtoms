"""Adam — 8 adaptive-moment variants.

Chapter:   01-optimization
Companion: adam.md
Tier 1 · 8 variants: adam, adamw, amsgrad, nadam, adamax, radam, adan,
eps_inside_sqrt

All variants keep two exponential averages of the gradient and its square, then
scale the step by their ratio. What separates them is *which* statistic is used,
*where* epsilon sits, and whether weight decay is coupled to the gradient or
applied directly to the parameters.
"""

from __future__ import annotations

import numpy as np

VARIANTS = frozenset(
    ["adam", "adamw", "amsgrad", "nadam", "adamax", "radam", "adan", "eps_inside_sqrt"]
)


class Adam:
    """Adam family optimiser over a list of parameter arrays.

    betas = (b1, b2) are the decay rates for the first and second moments.
    adan uses (b1, b2, b3) with the Adan defaults if you pass a 3-tuple.
    """

    def __init__(
        self,
        params: list[np.ndarray],
        lr: float = 1e-2,
        variant: str = "adam",
        betas: tuple[float, ...] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        self.params = params
        self.lr = lr
        self.variant = variant
        self.eps = eps
        self.wd = weight_decay
        self.b1 = betas[0]
        self.b2 = betas[1]
        self.b3 = betas[2] if len(betas) > 2 else 0.99
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.vmax = [np.zeros_like(p) for p in params]     # amsgrad
        self.g_prev = [np.zeros_like(p) for p in params]    # adan
        self.u = [np.zeros_like(p) for p in params]         # adamax
        self.n = [np.zeros_like(p) for p in params]         # adan
        self.t = 0

    # -- per-variant denominator ------------------------------------------- #
    def _denom(self, v, vmax, eps):
        if self.variant == "eps_inside_sqrt":
            return np.sqrt(v + eps)
        if self.variant == "amsgrad":
            return np.sqrt(np.maximum(vmax, v)) + eps
        return np.sqrt(v) + eps

    def step(self, grads: list[np.ndarray]) -> None:
        self.t += 1
        t = self.t
        grads = [np.asarray(g, dtype=float) for g in grads]

        for i, (p, g) in enumerate(zip(self.params, grads, strict=True)):
            if self.variant == "adamw" and self.wd:
                p -= self.lr * self.wd * p  # decoupled: no gradient term

            if self.variant == "adan":
                self._adan_step(p, g, i, t)
                continue

            if self.wd and self.variant != "adamw":
                g = g + self.wd * p

            m, v = self.m[i], self.v[i]
            m *= self.b1
            m += (1.0 - self.b1) * g
            if self.variant == "adamax":
                self.u[i] = np.maximum(self.b2 * self.u[i], np.abs(g))
                mhat = m / (1.0 - self.b1**t)
                p -= (self.lr / (1.0 - self.b1**t)) * m / (self.u[i] + self.eps)
                continue

            v *= self.b2
            v += (1.0 - self.b2) * g * g
            mhat = m / (1.0 - self.b1**t)

            if self.variant == "radam":
                rho_inf = 2.0 / (1.0 - self.b2) - 1.0
                rho_t = rho_inf - 2.0 * t * self.b2**t / (1.0 - self.b2**t)
                if rho_t > 4.0:
                    r = np.sqrt(
                        ((rho_t - 4.0) * (rho_t - 2.0) * rho_inf)
                        / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t)
                    )
                    l_t = np.sqrt(1.0 - self.b2**t) / (np.sqrt(v) + self.eps)
                    p -= self.lr * r * l_t * m / (1.0 - self.b1**t)
                else:
                    p -= self.lr * mhat
                continue

            if self.variant == "nadam":
                # Nesterov looks one momentum step ahead: b1*mhat + (1-b1)*g/(1-b1^t).
                g_eff = self.b1 * mhat + (1.0 - self.b1) * g / (1.0 - self.b1**t)
                vhat = v / (1.0 - self.b2**t)
                p -= self.lr * g_eff / (np.sqrt(vhat) + self.eps)
                continue

            vhat = v / (1.0 - self.b2**t)
            if self.variant == "amsgrad":
                self.vmax[i] = np.maximum(self.vmax[i], vhat)
                denom = np.sqrt(self.vmax[i]) + self.eps
            elif self.variant == "eps_inside_sqrt":
                denom = np.sqrt(vhat + self.eps)
            else:
                denom = np.sqrt(vhat) + self.eps
            p -= self.lr * mhat / denom

    def _adan_step(self, p, g, i, t) -> None:
        """Adan (Xie et al., 2022): Nesterov-style, with the *gradient difference*
        as the second statistic. m over g, v over (g-g_prev), n over (g-g_prev)^2."""
        self.m[i] = (1.0 - self.b1) * self.m[i] + self.b1 * g
        d = g - self.g_prev[i]
        self.v[i] = (1.0 - self.b2) * self.v[i] + self.b2 * d
        self.n[i] = (1.0 - self.b3) * self.n[i] + self.b3 * d * d
        self.g_prev[i] = g

        mhat = self.m[i] / (1.0 - self.b1**t)
        vhat = self.v[i] / (1.0 - self.b2**t)
        nhat = self.n[i] / (1.0 - self.b3**t)
        p -= self.lr * (mhat + self.b2 * vhat) / (np.sqrt(nhat) + self.eps)


def main() -> None:
    rng = np.random.default_rng(0)
    A = np.diag([1.0, 20.0])
    b = np.array([1.0, -1.0])
    x_star = np.linalg.solve(A, b)
    grad = lambda x: A @ x - b  # noqa: E731

    # 1. Every variant converges on the same convex quadratic.
    for variant in sorted(VARIANTS):
        lr = 0.05 if variant in ("adam", "adamw", "amsgrad", "nadam", "eps_inside_sqrt") else 0.02
        if variant == "adan":
            # Adan's difference statistic vanishes at the optimum, so its
            # stationary error is proportional to lr: use a smaller step.
            lr = 0.005
        x = np.array([4.0, 4.0])
        opt = Adam([x], lr=lr, variant=variant)
        for _ in range(1500):
            opt.step([grad(x)])
        err = float(np.linalg.norm(x - x_star))
        assert np.isfinite(err) and err < 1e-2, (variant, err)
    print("all 8 variants converge on a 2-D quadratic")

    # 2. Bias correction makes the first Adam step exactly lr in magnitude:
    #    mhat/sqrt(vhat) = sign(g), so |dp| = lr on every coordinate.
    x = np.array([0.0, 0.0])
    opt = Adam([x], lr=0.1, variant="adam")
    opt.step([np.array([3.0, -0.5])])
    assert np.allclose(np.abs(x), 0.1, atol=1e-7), x
    print(f"adam first step: |dp| = {np.abs(x)} with lr 0.1")

    # 3. Without bias correction the first step would be ~lr/10, which is the
    #    whole reason bias correction exists.
    m = 0.9 * 0.0 + 0.1 * 3.0
    v = 0.999 * 0.0 + 0.001 * 9.0
    naive = 0.1 * m / np.sqrt(v)          # no 1-b^t: 3.16x too large here
    corrected = 0.1 * (m / 0.1) / np.sqrt(v / 0.001)
    assert abs(naive - 0.1 * 0.1 / np.sqrt(0.001)) < 1e-3 and abs(corrected - 0.1) < 1e-9
    print(f"bias correction: {naive:.4f} uncorrected vs {corrected:.4f} corrected")

    # 4. AMSGrad's denominator is monotone: vmax never decreases.
    x = np.array([1.0])
    opt = Adam([x], lr=0.01, variant="amsgrad")
    prev = -np.inf
    for _ in range(20):
        opt.step([np.array([rng.normal()])])
        cur = float(opt.vmax[0][0])
        assert cur >= prev
        prev = cur
    print("amsgrad: vmax non-decreasing")

    # 5. Adamax's denominator is a running max of |g|, not a second moment.
    x = np.array([0.0])
    opt = Adam([x], lr=0.1, variant="adamax", betas=(0.9, 0.999))
    opt.step([np.array([2.0])])
    opt.step([np.array([-5.0])])
    assert np.allclose(opt.u[0], [5.0]), opt.u[0]
    assert np.allclose(opt.m[0], [0.9 * 0.2 + 0.1 * -5.0])
    print("adamax: u = running max |g|")

    # 6. RAdam is conservative at the start, then matches Adam. At t=1 the
    #    rectification factor is 0 (rho_t < 4), so the step is the SGD-momentum
    #    step, not the full adaptive step.
    xa, xr = np.array([1.0]), np.array([1.0])
    a = Adam([xa], lr=0.01, variant="adam")
    r = Adam([xr], lr=0.01, variant="radam")
    a.step([np.array([1.0])])
    r.step([np.array([1.0])])
    assert np.allclose(xa, [0.99]) and np.allclose(xr, [0.99])
    print("radam: first step = momentum step (no adaptive scaling)")

    # 7. AdamW differs from Adam under weight decay: decoupled decay multiplies
    #    the parameter by (1 - lr*wd) exactly, while the coupled version has its
    #    decay pushed through the adaptive denominator.
    xa, xw = np.array([1.0]), np.array([1.0])
    opt_a = Adam([xa], lr=0.1, variant="adam", weight_decay=0.5)
    opt_w = Adam([xw], lr=0.1, variant="adamw", weight_decay=0.5)
    for _ in range(5):
        opt_a.step([np.array([0.0])])
        opt_w.step([np.array([0.0])])
    assert not np.allclose(xa, xw) and xa[0] < xw[0], (xa, xw)
    print(f"decay after 5 zero-grad steps: adam {xa[0]:.4f}, adamw {xw[0]:.4f}")

    # 8. Adan tracks gradient differences, so a constant gradient has a vanishing
    #    second statistic and the step is dominated by m/sqrt(eps) — the reason
    #    the epsilon must not be too small in float32.
    x = np.array([1.0])
    opt = Adam([x], lr=1e-3, variant="adan")
    opt.step([np.array([1.0])])
    n1 = float(opt.n[0][0])
    for _ in range(3):
        opt.step([np.array([1.0])])          # d = 0 from here on
    assert np.allclose(opt.n[0][0], n1 * (1.0 - opt.b3) ** 3, rtol=1e-9)
    print(f"adan: constant gradient -> difference statistic decays {n1:.2e} -> {opt.n[0][0]:.2e}")
    print("adam demo passed")


if __name__ == "__main__":
    main()
