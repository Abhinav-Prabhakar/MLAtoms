"""Meta-learning — seven ways to learn from a handful of examples.

Chapter:   13-training
Companion: meta_learning.md
Tier 2 · 7 variants: maml, first_order_maml, reptile, prototypical, matching,
relation, meta_sgd
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("maml", "first_order_maml", "reptile", "prototypical", "matching",
            "relation", "meta_sgd")
METRIC = ("prototypical", "matching", "relation")
N_WAY = 3
N_SHOT = 5
N_QUERY = 15
EMBED = 16
HIDDEN = 64
SCALE = 8.0
TAU = 0.5


# --------------------------------------------------------------------------- #
# The task family
# --------------------------------------------------------------------------- #
def make_task(rng: np.random.Generator, n_way: int = N_WAY,
              n_shot: int = N_SHOT, n_query: int = N_QUERY):
    """Concentric rings of random radius, each ring one class.

    A ring's mean is the origin, so nearest-mean on the raw inputs cannot tell
    the classes apart. A representation that encodes the radius can.
    """
    radii = np.sort(rng.uniform(0.5, 2.5, n_way))
    angles = rng.uniform(0.0, 2.0 * np.pi, (n_way, n_shot + n_query))
    points = radii[:, None, None] * np.stack(
        [np.cos(angles), np.sin(angles)], axis=-1)
    points = points + rng.normal(0.0, 0.10, points.shape)
    return points[:, :n_shot], points[:, n_shot:]


def _labels(count: int, n_way: int = N_WAY) -> np.ndarray:
    return np.repeat(np.arange(n_way), count)


# --------------------------------------------------------------------------- #
# A two-layer encoder and a linear head
# --------------------------------------------------------------------------- #
def _init(rng: np.random.Generator) -> dict:
    return {
        "W1": rng.normal(0.0, np.sqrt(2.0 / 2.0), (2, HIDDEN)),
        "b1": np.zeros(HIDDEN),
        "W2": rng.normal(0.0, np.sqrt(2.0 / HIDDEN), (HIDDEN, EMBED)),
        "b2": np.zeros(EMBED),
        "Wh": rng.normal(0.0, np.sqrt(2.0 / EMBED), (EMBED, N_WAY)),
        "bh": np.zeros(N_WAY),
        "R1": rng.normal(0.0, np.sqrt(2.0 / (3 * EMBED)),
                         (3 * EMBED, HIDDEN)),
        "r1": np.zeros(HIDDEN),
        "R2": rng.normal(0.0, np.sqrt(2.0 / HIDDEN), (HIDDEN, 1)),
        "r2": np.zeros(1),
    }


def _encode(params: dict, x: np.ndarray):
    z1 = x @ params["W1"] + params["b1"]
    a1 = np.maximum(z1, 0.0)
    z2 = a1 @ params["W2"] + params["b2"]
    a2 = np.maximum(z2, 0.0)
    return a2, (x, z1, a1, z2)


def _encode_back(params: dict, d_out: np.ndarray, cache) -> dict:
    x, z1, a1, z2 = cache
    d2 = d_out * (z2 > 0.0)
    d1 = (d2 @ params["W2"].T) * (z1 > 0.0)
    return {"W1": x.T @ d1, "b1": d1.sum(0),
            "W2": a1.T @ d2, "b2": d2.sum(0)}


def _full_forward(params: dict, x: np.ndarray):
    emb, cache = _encode(params, x)
    logits = emb @ params["Wh"] + params["bh"]
    return logits, (cache, emb)


def _full_back(params: dict, d_logits: np.ndarray, cache) -> dict:
    cache, emb = cache
    grads = {"Wh": emb.T @ d_logits, "bh": d_logits.sum(0)}
    grads.update(_encode_back(params, d_logits @ params["Wh"].T, cache))
    for key in ("R1", "r1", "R2", "r2"):
        grads[key] = np.zeros_like(params[key])
    return grads


def _cross_entropy(logits: np.ndarray, labels: np.ndarray):
    shifted = logits - logits.max(axis=1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    loss = -float(np.mean(logp[np.arange(len(labels)), labels]))
    grad = np.exp(logp)
    grad[np.arange(len(labels)), labels] -= 1.0
    return loss, grad / len(labels)


# --------------------------------------------------------------------------- #
# The metric heads
# --------------------------------------------------------------------------- #
def _metric_logits(params: dict, sup: np.ndarray, qry: np.ndarray,
                   variant: str, tau: float = TAU):
    """Return (logits, cache) where logits is (n_way, n_query, n_way)."""
    s, s_cache = _encode(params, sup.reshape(-1, 2))
    q, q_cache = _encode(params, qry.reshape(-1, 2))
    s = s.reshape(N_WAY, sup.shape[1], -1)
    q = q.reshape(N_WAY, qry.shape[1], -1)
    if variant == "prototypical":
        centre = s.mean(axis=1)
        logits = -((q[:, :, None, :] - centre[None, None, :, :]) ** 2).sum(-1)
        return logits, (q, s, centre, s_cache, q_cache)
    if variant == "matching":
        pair = q[:, :, None, None, :] - s[None, None, :, :, :]
        similarity = -(pair ** 2).sum(-1) / tau
        shift = similarity.max(axis=-1, keepdims=True)
        weight = np.exp(similarity - shift)
        total = weight.sum(axis=-1, keepdims=True)
        logits = (shift + np.log(total))[..., 0]
        return logits, (q, s, pair, weight / total, s_cache, q_cache)
    centre = s.mean(axis=1)
    pair = np.concatenate([
        np.broadcast_to(q[:, :, None, :], (N_WAY, qry.shape[1], N_WAY,
                                           q.shape[-1])),
        np.broadcast_to(centre[None, None, :, :], (N_WAY, qry.shape[1],
                                                   N_WAY, centre.shape[-1])),
        np.abs(q[:, :, None, :] - centre[None, None, :, :])], axis=-1)
    flat = pair.reshape(-1, pair.shape[-1])
    hidden = np.maximum(flat @ params["R1"] + params["r1"], 0.0)
    score = (hidden @ params["R2"] + params["r2"]).reshape(
        N_WAY, qry.shape[1], N_WAY)
    return score, (q, s, centre, pair, hidden, s_cache, q_cache)


def _metric_back(params: dict, d_logits: np.ndarray, cache,
                 variant: str, tau: float = TAU) -> dict:
    q, s = cache[0], cache[1]
    grads = {key: np.zeros_like(value) for key, value in params.items()}
    if variant == "prototypical":
        centre, s_cache, q_cache = cache[2], cache[3], cache[4]
        count = s.shape[1]
        d_q = -2.0 * (d_logits.sum(axis=2)[:, :, None] * q
                      - np.einsum("ijc,cd->ijd", d_logits, centre))
        d_centre = 2.0 * (np.einsum("iqj,iqd->jd", d_logits, q)
                          - d_logits.sum(axis=(0, 1))[:, None] * centre)
        d_s = np.repeat(d_centre[:, None, :] / count, count, axis=1)
    elif variant == "matching":
        pair, attention, s_cache, q_cache = cache[2], cache[3], cache[4], \
            cache[5]
        d_sim = attention * d_logits[:, :, :, None]
        d_q = (2.0 / tau) * (
            np.einsum("iqck,ckd->iqd", d_sim, s)
            - d_sim.sum(axis=(2, 3))[:, :, None] * q)
        d_s = (2.0 / tau) * (
            np.einsum("iqck,iqd->ckd", d_sim, q)
            - d_sim.sum(axis=(0, 1))[:, :, None] * s)
        del pair
    else:
        centre, pair, hidden, s_cache, q_cache = cache[2], cache[3], cache[4], \
            cache[5], cache[6]
        d_score = d_logits.reshape(-1)
        grads["R2"] = hidden.T @ d_score[:, None]
        grads["r2"] = d_score[:, None].sum(axis=0)
        d_hidden = d_score[:, None] @ params["R2"].T
        d_pre = d_hidden * (hidden > 0.0)
        grads["R1"] = pair.reshape(-1, pair.shape[-1]).T @ d_pre
        grads["r1"] = d_pre.sum(axis=0)
        d_pair = (d_pre @ params["R1"].T).reshape(pair.shape)
        width = q.shape[-1]
        gap = q[:, :, None, :] - centre[None, None, :, :]
        sign = np.sign(gap)
        d_q = (d_pair[..., :width] + sign * d_pair[..., 2 * width:]).sum(
            axis=2)
        d_centre = (d_pair[..., width:2 * width]
                    - sign * d_pair[..., 2 * width:]).sum(axis=(0, 1))
        d_s = np.repeat(d_centre[:, None, :] / s.shape[1], s.shape[1], axis=1)
    encoder = _encode_back(params, d_s.reshape(-1, s.shape[-1]), s_cache)
    for key, value in encoder.items():
        grads[key] += value
    encoder = _encode_back(params, d_q.reshape(-1, q.shape[-1]), q_cache)
    for key, value in encoder.items():
        grads[key] += value
    return grads


# --------------------------------------------------------------------------- #
# The learner
# --------------------------------------------------------------------------- #
class MetaLearner:
    """A shared encoder, one meta-learning rule, and a few-shot task family.

    `fit()` runs the meta-training loop and `evaluate()` reports mean query
    accuracy on freshly sampled tasks. The MAML family adapts a copy of the
    parameters on the support set at test time; the metric family classifies by
    prototype, soft neighbour or a learned relation, with no inner loop.
    """

    def __init__(self, variant: str = "maml", *, steps: int = 400,
                 meta_lr: float = 3e-3, inner_lr: float = 0.03,
                 inner_steps: int = 3, batch: int = 4, tasks: int = 200,
                 tau: float = TAU, seed: int = 0):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if steps < 1 or tasks < 1 or batch < 1:
            raise ValueError("steps, tasks and batch must be positive")
        if inner_steps < 1:
            raise ValueError("inner_steps must be positive")
        if not np.isfinite(meta_lr) or meta_lr <= 0.0:
            raise ValueError("meta_lr must be positive and finite")
        if not np.isfinite(inner_lr) or inner_lr <= 0.0:
            raise ValueError("inner_lr must be positive and finite")
        if not np.isfinite(tau) or tau <= 0.0:
            raise ValueError("tau must be positive and finite")
        self.variant = variant
        self.steps = steps
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.batch = batch
        self.tasks = tasks
        self.tau = tau
        self.rng = np.random.default_rng(seed)
        self.params = _init(np.random.default_rng(seed + 1))
        self.alpha = {key: np.full_like(value, inner_lr)
                      for key, value in self.params.items()}
        self.history: list[float] = []

    # -- inner loop ----------------------------------------------------------- #
    def _support_grads(self, params: dict, sup: np.ndarray):
        logits, cache = _full_forward(params, sup.reshape(-1, 2))
        loss, grad = _cross_entropy(logits, _labels(sup.shape[1]))
        return _full_back(params, grad, cache), loss

    def _query_grads(self, params: dict, qry: np.ndarray):
        logits, cache = _full_forward(params, qry.reshape(-1, 2))
        loss, grad = _cross_entropy(logits, _labels(qry.shape[1]))
        return _full_back(params, grad, cache), loss

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

    def _hessian_vector(self, params: dict, sup: np.ndarray,
                        vector: dict) -> dict:
        base, _ = self._support_grads(params, sup)
        eps = 1e-4
        shifted = {key: params[key] + eps * vector[key] for key in params}
        moved, _ = self._support_grads(shifted, sup)
        return {key: (moved[key] - base[key]) / eps for key in base}

    # -- one meta-gradient ---------------------------------------------------- #
    def gradient(self, sup: np.ndarray, qry: np.ndarray):
        """Return (meta-gradient, alpha-gradient) for one task.

        The second value is only non-`None` for `meta_sgd`, which learns a
        per-parameter inner learning rate alongside the initialisation.
        """
        if self.variant in ("maml", "first_order_maml"):
            adapted, trail, _ = self._adapt(self.params, sup, self.inner_steps)
            query, _ = self._query_grads(adapted, qry)
            if self.variant == "first_order_maml":
                return query, None
            vector = query
            for index in range(self.inner_steps - 1, -1, -1):
                product = self._hessian_vector(trail[index], sup, vector)
                vector = {key: vector[key] - self.inner_lr * product[key]
                          for key in vector}
            return vector, None
        if self.variant == "reptile":
            adapted, _, _ = self._adapt(self.params, sup, self.inner_steps)
            return ({key: adapted[key] - self.params[key]
                     for key in self.params}, None)
        if self.variant == "meta_sgd":
            adapted, trail, gradients = self._adapt(
                self.params, sup, self.inner_steps, alpha=self.alpha)
            query, _ = self._query_grads(adapted, qry)
            vector = query
            d_alpha = {key: np.zeros_like(value)
                       for key, value in self.alpha.items()}
            for index in range(self.inner_steps - 1, -1, -1):
                d_alpha = {key: d_alpha[key] - gradients[index][key] * vector[key]
                           for key in d_alpha}
                product = self._hessian_vector(trail[index], sup, vector)
                vector = {key: vector[key] - self.alpha[key] * product[key]
                          for key in vector}
            return vector, d_alpha
        logits, cache = _metric_logits(self.params, sup, qry, self.variant,
                                       self.tau)
        labels = _labels(qry.shape[1])
        flat = logits.reshape(-1, N_WAY)
        _, grad = _cross_entropy(flat, labels)
        return _metric_back(self.params, grad.reshape(logits.shape), cache,
                            self.variant, self.tau), None

    # -- outer loop ----------------------------------------------------------- #
    def _adam(self, state: list, grads: dict, step: int, lr: float) -> None:
        first, second = state
        for key, grad in grads.items():
            first[key] = 0.9 * first[key] + 0.1 * grad
            second[key] = 0.999 * second[key] + 0.001 * grad ** 2
            self.params[key] -= lr * (first[key] / (1.0 - 0.9 ** step)) / (
                np.sqrt(second[key] / (1.0 - 0.999 ** step)) + 1e-8)

    def fit(self) -> MetaLearner:
        state = ({key: np.zeros_like(value) for key, value in self.params.items()},
                 {key: np.zeros_like(value) for key, value in self.params.items()})
        alpha_state = ({key: np.zeros_like(value) for key, value in self.alpha.items()},
                       {key: np.zeros_like(value) for key, value in self.alpha.items()})
        for step in range(1, self.steps + 1):
            total = {key: np.zeros_like(value)
                     for key, value in self.params.items()}
            alpha_total = {key: np.zeros_like(value)
                           for key, value in self.alpha.items()}
            for _ in range(self.batch):
                sup, qry = make_task(self.rng)
                grad, alpha_grad = self.gradient(sup, qry)
                for key in total:
                    total[key] += grad[key] / self.batch
                if alpha_grad is not None:
                    for key in alpha_total:
                        alpha_total[key] += alpha_grad[key] / self.batch
            if self.variant == "reptile":
                blended = {key: 0.5 * total[key] + self.params[key]
                           for key in self.params}
                self.params = blended
            else:
                self._adam(state, total, step, self.meta_lr)
                if self.variant == "meta_sgd":
                    first, second = alpha_state
                    for key, grad in alpha_total.items():
                        first[key] = 0.9 * first[key] + 0.1 * grad
                        second[key] = 0.999 * second[key] + 0.001 * grad ** 2
                        self.alpha[key] -= 1e-3 * (
                            first[key] / (1.0 - 0.9 ** step)) / (
                            np.sqrt(second[key] / (1.0 - 0.999 ** step))
                            + 1e-8)
                        self.alpha[key] = np.clip(self.alpha[key], 1e-3, 1.0)
            self.history.append(float(np.mean(
                [np.abs(g).mean() for g in total.values()])))
        return self

    # -- inference ------------------------------------------------------------ #
    def _adapted(self, sup: np.ndarray) -> dict:
        if self.variant in METRIC:
            return self.params
        adapted, _, _ = self._adapt(self.params, sup, self.inner_steps)
        return adapted

    def predict(self, sup: np.ndarray, qry: np.ndarray) -> np.ndarray:
        support = np.asarray(sup, dtype=float)
        query = np.asarray(qry, dtype=float)
        if (support.ndim != 3 or support.shape[0] != N_WAY
                or support.shape[2] != 2):
            raise ValueError("support must be (n_way, n_shot, 2)")
        if (query.ndim != 3 or query.shape[0] != N_WAY
                or query.shape[2] != 2):
            raise ValueError("query must be (n_way, n_query, 2)")
        if not np.all(np.isfinite(support)) or not np.all(np.isfinite(query)):
            raise ValueError("the inputs must be finite")
        sup, qry = support, query
        if self.variant in METRIC:
            logits, _ = _metric_logits(self.params, sup, qry, self.variant,
                                       self.tau)
        else:
            logits, _ = _full_forward(self._adapted(sup), qry.reshape(-1, 2))
            logits = logits.reshape(N_WAY, qry.shape[1], N_WAY)
        return logits.argmax(axis=-1)

    def evaluate(self, tasks: int | None = None,
                 shots: int = N_SHOT) -> float:
        total = 0.0
        count = self.tasks if tasks is None else tasks
        for _ in range(count):
            sup, qry = make_task(self.rng, n_shot=shots)
            prediction = self.predict(sup, qry)
            total += float(np.mean(prediction == np.arange(N_WAY)[:, None]))
        return total / count


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def raw_prototypes(tasks: int = 200, seed: int = 77) -> float:
    rng = np.random.default_rng(seed)
    total = 0.0
    for _ in range(tasks):
        sup, qry = make_task(rng)
        centre = sup.mean(axis=1)
        distance = ((qry[:, :, None, :] - centre[None, None, :, :]) ** 2).sum(-1)
        total += float(np.mean(distance.argmin(axis=2) == _labels(1)[:, None]))
    return total / tasks


def random_encoder(tasks: int = 200, seed: int = 78) -> float:
    params = _init(np.random.default_rng(seed))
    rng = np.random.default_rng(seed + 1)
    total = 0.0
    for _ in range(tasks):
        sup, qry = make_task(rng)
        logits, _ = _metric_logits(params, sup, qry, "prototypical")
        total += float(np.mean(logits.argmax(axis=2) == _labels(1)[:, None]))
    return total / tasks


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def demo() -> None:
    print("meta. three classes on concentric rings, five shots each:")
    print("  each ring's mean is the origin, so raw prototypes see nothing")
    raw = raw_prototypes()
    random = random_encoder()
    print(f"  raw-input prototypes score {raw:.3f}")
    print(f"  a random encoder with prototypes scores {random:.3f}")

    print("meta. seven learners, four hundred meta-iterations each:")
    table = {}
    learners = {}
    for variant in VARIANTS:
        learner = MetaLearner(variant, seed=2).fit()
        learners[variant] = learner
        table[variant] = learner.evaluate()
        print(f"    {variant:<18s} {table[variant]:.3f} accuracy")

    print("meta. the metric family needs no inner loop at test time:")
    for variant in METRIC:
        print(f"    {variant:<18s} {learners[variant].evaluate():.3f}")

    print("meta. the MAML family adapts on the support set:")
    for variant in ("maml", "first_order_maml", "meta_sgd"):
        print(f"    {variant:<18s} {learners[variant].evaluate():.3f}"
              f"  inner lr {learners[variant].inner_lr}")

    print("meta. shots matter more than the algorithm:")
    shallow = MetaLearner("prototypical", seed=2, steps=250).fit()
    for shots in (1, 3, 5):
        print(f"    {shots}-shot prototypical  {shallow.evaluate(shots=shots):.3f}")

    print("meta. the two ends of the cost scale:")
    print("  maml differentiates through the inner loop, so each task costs")
    print("  inner_steps support gradients plus inner_steps Hessian-vector")
    print("  products; prototypical costs one forward pass and no adaptation")

    best = max(table, key=table.get)
    print(f"  {best} wins with {table[best]:.3f} against {raw:.3f} raw")
    assert all(score > raw + 0.25 for score in table.values())
    assert table["prototypical"] > 0.8
    assert table["matching"] > 0.75
    assert table["relation"] > 0.75
    assert table["maml"] > 0.78
    assert table["first_order_maml"] > 0.77
    assert table["reptile"] > 0.65
    assert table["meta_sgd"] > 0.75
    assert raw < 0.5 and random < 0.6

    print("meta. malformed calls are refused:")
    sup, qry = make_task(np.random.default_rng(0))
    bad = [
        lambda: MetaLearner("maml_v2"),
        lambda: MetaLearner("maml", steps=0),
        lambda: MetaLearner("maml", batch=0),
        lambda: MetaLearner("maml", tasks=0),
        lambda: MetaLearner("maml", inner_steps=0),
        lambda: MetaLearner("maml", meta_lr=0.0),
        lambda: MetaLearner("maml", inner_lr=-1.0),
        lambda: MetaLearner("matching", tau=0.0),
        lambda: MetaLearner("maml", seed=1).fit().predict(
            sup[:, :, :1], qry),
        lambda: MetaLearner("maml", seed=1).fit().predict(
            sup[:-1], qry),
        lambda: MetaLearner("maml", seed=1).fit().predict(
            np.full_like(sup, np.nan), qry),
        lambda: MetaLearner("prototypical", seed=1).fit().predict(
            sup, qry[:, :, :1]),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except (ValueError, IndexError):
            refused += 1
    assert refused == len(bad)
    print(f"  {refused} of {len(bad)}: an unknown variant, a zero step,")
    print("  batch, task or inner-step count, a non-positive meta or inner")
    print("  learning rate, a non-positive temperature, and inputs whose")
    print("  shape or finiteness is wrong")
    print("meta demo passed")


if __name__ == "__main__":
    demo()
