"""Continual learning — eight ways to keep old tasks alive.

Chapter:   13-training
Companion: continual_learning.md
Tier 3 · 8 variants: ewc, si, mas, replay, gem, lwf, packnet, lora_merge
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("ewc", "si", "mas", "replay", "gem", "lwf", "packnet",
            "lora_merge")

N_TASKS = 5
PER_TASK = 2
N_CLASSES = N_TASKS * PER_TASK
WIDTH = 32
RADIUS = 2.0
NOISE = 0.45
SAMPLES = 200
STEPS = 600
LR = 2e-2
PRETRAIN_STEPS = 2500
PRETRAIN_LR = 1e-2
EWC_LAMBDA = 20.0
SI_LAMBDA = 2.0
MAS_LAMBDA = 0.05
LWF_LAMBDA = 4.0
GEM_LR = 0.02
MEMORY = 20
PACK_BUDGET = 4
LORA_RANK = 2
LORA_LR = 1e-4

_BASE_CACHE: dict[int, dict] = {}


# --------------------------------------------------------------------------- #
# The data: two classes per ring, one task per adjacent pair
# --------------------------------------------------------------------------- #
def make_sequence(seed: int = 50) -> list[tuple]:
    """Five tasks of two classes each, sampled from separate ring sectors."""
    rng = np.random.default_rng(seed)
    angles = 2.0 * np.pi * np.arange(N_CLASSES) / N_CLASSES
    angles = angles + rng.uniform(-0.15, 0.15, N_CLASSES)
    centres = RADIUS * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    x = np.vstack([centres[c] + rng.normal(0.0, NOISE, (SAMPLES, 2))
                   for c in range(N_CLASSES)])
    y = np.concatenate([np.full(SAMPLES, c) for c in range(N_CLASSES)])
    tasks = []
    for t in range(N_TASKS):
        classes = np.arange(t * PER_TASK, (t + 1) * PER_TASK)
        keep = np.isin(y, classes)
        xt, yt = x[keep], y[keep]
        order = rng.permutation(len(xt))
        xt, yt = xt[order], yt[order]
        cut = int(0.7 * len(xt))
        tasks.append((xt[:cut], yt[:cut], xt[cut:], yt[cut:]))
    return tasks


def make_pretraining(seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """A disjoint ring family that teaches the trunk generic circle features."""
    rng = np.random.default_rng(seed)
    angles = 2.0 * np.pi * np.arange(N_CLASSES) / N_CLASSES
    angles = angles + rng.uniform(-0.3, 0.3, N_CLASSES)
    centres = RADIUS * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    x = np.vstack([centres[c] + rng.normal(0.0, NOISE, (SAMPLES, 2))
                   for c in range(N_CLASSES)])
    y = np.concatenate([np.full(SAMPLES, c) for c in range(N_CLASSES)])
    return x, y


# --------------------------------------------------------------------------- #
# A two-layer trunk and a linear head, with optional masks and adapters
# --------------------------------------------------------------------------- #
def _init(seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0.0, np.sqrt(2.0 / 2.0), (2, WIDTH)),
        "b1": np.zeros(WIDTH),
        "W2": rng.normal(0.0, np.sqrt(2.0 / WIDTH), (WIDTH, WIDTH)),
        "b2": np.zeros(WIDTH),
        "W3": rng.normal(0.0, np.sqrt(2.0 / WIDTH), (WIDTH, N_CLASSES)),
        "b3": np.zeros(N_CLASSES),
    }


def _forward(params: dict, x: np.ndarray, mask1=None, mask2=None,
             ad1=None, ad2=None):
    w1 = params["W1"] if ad1 is None else params["W1"] + ad1[0] @ ad1[1]
    w2 = params["W2"] if ad2 is None else params["W2"] + ad2[0] @ ad2[1]
    z1 = x @ w1 + params["b1"]
    a1 = np.maximum(z1, 0.0)
    if mask1 is not None:
        a1 = a1 * mask1
    z2 = a1 @ w2 + params["b2"]
    a2 = np.maximum(z2, 0.0)
    if mask2 is not None:
        a2 = a2 * mask2
    return a2 @ params["W3"] + params["b3"], (x, z1, a1, z2, a2, w1, w2)


def _backward(params: dict, x: np.ndarray, y: np.ndarray, rows=None,
              mask1=None, mask2=None, ad1=None, ad2=None,
              dlogits=None) -> dict:
    logits, cache = _forward(params, x, mask1, mask2, ad1, ad2)
    x_, z1, a1, z2, a2, w1, w2 = cache
    if dlogits is None:
        shifted = logits - logits.max(axis=1, keepdims=True)
        prob = np.exp(shifted)
        prob /= prob.sum(axis=1, keepdims=True)
        d = prob.copy()
        d[np.arange(len(y)), y] -= 1.0
        d /= len(y)
        if rows is not None:
            allowed = np.zeros(N_CLASSES)
            allowed[np.asarray(rows)] = 1.0
            d *= allowed
    else:
        d = dlogits
    grads = {"W3": a2.T @ d, "b3": d.sum(0)}
    dz2 = (d @ params["W3"].T) * (z2 > 0.0)
    if mask2 is not None:
        dz2 = dz2 * mask2
    grads["W2"] = a1.T @ dz2
    grads["b2"] = dz2.sum(0)
    dz1 = (dz2 @ w2.T) * (z1 > 0.0)
    if mask1 is not None:
        dz1 = dz1 * mask1
    grads["W1"] = x_.T @ dz1
    grads["b1"] = dz1.sum(0)
    if ad1 is not None:
        grads["A1"] = grads["W1"] @ ad1[1].T
        grads["B1"] = ad1[0].T @ grads["W1"]
    if ad2 is not None:
        grads["A2"] = grads["W2"] @ ad2[1].T
        grads["B2"] = ad2[0].T @ grads["W2"]
    return grads


def _zeros(params: dict) -> dict:
    return {key: np.zeros_like(value) for key, value in params.items()}


def _new_state(params: dict) -> tuple[dict, dict]:
    return _zeros(params), _zeros(params)


def _adam(cur: dict, grads: dict, state: tuple, step: int, lr: float) -> None:
    first, second = state
    for key, grad in grads.items():
        first[key] = 0.9 * first[key] + 0.1 * grad
        second[key] = 0.999 * second[key] + 0.001 * grad ** 2
        cur[key] -= lr * (first[key] / (1.0 - 0.9 ** step)) / (
            np.sqrt(second[key] / (1.0 - 0.999 ** step)) + 1e-8)


def _train(params: dict, x: np.ndarray, y: np.ndarray, rows, steps: int = STEPS,
           lr: float = LR, penalty=None, extra=None, gradmask=None,
           mask1=None, mask2=None) -> None:
    state = _new_state(params)
    for step in range(1, steps + 1):
        grads = _backward(params, x, y, rows, mask1, mask2)
        if penalty is not None:
            for key, value in penalty(params).items():
                grads[key] += value
        if extra is not None:
            for key, value in extra(params, x, y).items():
                grads[key] += value
        if gradmask is not None:
            grads = {key: grads[key] * gradmask[key] for key in grads}
        _adam(params, grads, state, step, lr)


# --------------------------------------------------------------------------- #
# The pretrained trunk, shared by every variant
# --------------------------------------------------------------------------- #
def _pretrain(seed: int) -> dict:
    params = _init(seed)
    x, y = make_pretraining()
    _train(params, x, y, np.arange(N_CLASSES),
           steps=PRETRAIN_STEPS, lr=PRETRAIN_LR)
    rng = np.random.default_rng(seed + 98)
    params["W3"] = rng.normal(0.0, np.sqrt(2.0 / WIDTH),
                              (WIDTH, N_CLASSES))
    params["b3"] = np.zeros(N_CLASSES)
    return params


def _base(seed: int = 1) -> dict:
    if seed not in _BASE_CACHE:
        _BASE_CACHE[seed] = _pretrain(seed)
    return {key: value.copy() for key, value in _BASE_CACHE[seed].items()}


# --------------------------------------------------------------------------- #
# The continual learner
# --------------------------------------------------------------------------- #
class Continual:
    """A pretrained trunk, a task sequence, and one forgetting defence.

    `fit()` walks the five tasks and records the per-task inference state.
    `evaluate()` returns the mean accuracy over the tasks; `scores` holds the
    five per-task numbers. Variants that isolate parameters (packnet, LoRA)
    also record a mask or an adapter for each task.
    """

    def __init__(self, variant: str = "ewc", *, seed: int = 1,
                 steps: int = STEPS, lr: float = LR):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if steps < 1:
            raise ValueError("steps must be positive")
        if not np.isfinite(lr) or lr <= 0.0:
            raise ValueError("lr must be positive and finite")
        self.variant = variant
        self.seed = seed
        self.steps = steps
        self.lr = lr
        self.data = make_sequence()
        self.params = _base(seed)
        self.scores: list[float] = []
        self.inference: list[dict] | None = None

    # -- dispatch ------------------------------------------------------------- #
    def fit(self) -> Continual:
        if self.variant == "ewc":
            self._fit_ewc()
        elif self.variant == "si":
            self._fit_si()
        elif self.variant == "mas":
            self._fit_mas()
        elif self.variant == "replay":
            self._fit_replay()
        elif self.variant == "gem":
            self._fit_gem()
        elif self.variant == "lwf":
            self._fit_lwf()
        elif self.variant == "packnet":
            self._fit_packnet()
        else:
            self._fit_lora()
        return self

    # -- evaluation ----------------------------------------------------------- #
    def predict(self, x: np.ndarray, task: int) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError("x must be (n, 2)")
        if not 0 <= task < N_TASKS:
            raise ValueError(f"task must be in [0, {N_TASKS})")
        if not np.all(np.isfinite(x)):
            raise ValueError("x must be finite")
        if self.inference is None:
            logits = _forward(self.params, x)[0]
        else:
            state = self.inference[task]
            logits = _forward(self.params, x, state["mask1"], state["mask2"],
                              state["ad1"], state["ad2"])[0]
        keep = np.arange(task * PER_TASK, (task + 1) * PER_TASK)
        return keep[logits[:, keep].argmax(axis=1)]

    def evaluate(self) -> float:
        total = []
        for t in range(N_TASKS):
            x, _, xt, yt = self.data[t]
            total.append(float(np.mean(self.predict(xt, t) == yt)))
        self.scores = total
        return float(np.mean(total))

    # -- EWC ------------------------------------------------------------------ #
    def _fit_ewc(self) -> None:
        fisher = _zeros(self.params)
        for t in range(N_TASKS):
            rows = np.arange(t * PER_TASK, (t + 1) * PER_TASK)
            anchor = {key: value.copy() for key, value in self.params.items()}
            penalty = None
            if t:
                def penalty(p, anchor=anchor, fisher=fisher):
                    return {key: EWC_LAMBDA * fisher[key] * (p[key] - anchor[key])
                            for key in fisher}
            _train(self.params, self.data[t][0], self.data[t][1], rows,
                   self.steps, self.lr, penalty=penalty)
            count = len(self.data[t][0])
            task_fisher = _zeros(self.params)
            for i in range(count):
                grads = _backward(self.params, self.data[t][0][i:i + 1],
                                  self.data[t][1][i:i + 1], rows)
                for key in task_fisher:
                    task_fisher[key] += grads[key] ** 2 / count
            for key in fisher:
                fisher[key] += task_fisher[key]

    # -- SI ------------------------------------------------------------------- #
    def _fit_si(self) -> None:
        importance = _zeros(self.params)
        for t in range(N_TASKS):
            rows = np.arange(t * PER_TASK, (t + 1) * PER_TASK)
            anchor = {key: value.copy() for key, value in self.params.items()}

            def penalty(p, anchor=anchor, importance=importance):
                return {key: 2.0 * SI_LAMBDA * importance[key]
                        * (p[key] - anchor[key]) for key in importance}

            state = _new_state(self.params)
            previous = {key: value.copy() for key, value in self.params.items()}
            path = _zeros(self.params)
            for step in range(1, self.steps + 1):
                grads = _backward(self.params, self.data[t][0], self.data[t][1],
                                  rows)
                if t:
                    for key, value in penalty(self.params).items():
                        grads[key] += value
                _adam(self.params, grads, state, step, self.lr)
                for key in grads:
                    path[key] += -grads[key] * (self.params[key] - previous[key])
                    previous[key] = self.params[key].copy()
            for key in importance:
                importance[key] = np.maximum(importance[key] + path[key], 0.0)

    # -- MAS ------------------------------------------------------------------ #
    def _fit_mas(self) -> None:
        importance = _zeros(self.params)
        for t in range(N_TASKS):
            rows = np.arange(t * PER_TASK, (t + 1) * PER_TASK)
            anchor = {key: value.copy() for key, value in self.params.items()}

            def penalty(p, anchor=anchor, importance=importance):
                return {key: 2.0 * MAS_LAMBDA * importance[key]
                        * (p[key] - anchor[key]) for key in importance}

            _train(self.params, self.data[t][0], self.data[t][1], rows,
                   self.steps, self.lr, penalty=penalty if t else None)
            logits = _forward(self.params, self.data[t][0])[0]
            grads = _backward(self.params, self.data[t][0], self.data[t][1],
                              rows, dlogits=logits / len(self.data[t][0]))
            for key in importance:
                weight = np.abs(grads[key])
                importance[key] += weight / (weight.mean() + 1e-12)

    # -- replay --------------------------------------------------------------- #
    def _fit_replay(self) -> None:
        examples: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        for t in range(N_TASKS):
            x, y = self.data[t][0], self.data[t][1]
            rows = np.arange(t * PER_TASK, (t + 1) * PER_TASK)
            if examples:
                x = np.vstack([x] + examples)
                y = np.concatenate([y] + labels)
                rows = np.unique(y)
            _train(self.params, x, y, rows, self.steps, self.lr)
            rng = np.random.default_rng(t)
            chosen = rng.choice(len(self.data[t][0]),
                                min(MEMORY, len(self.data[t][0])),
                                replace=False)
            examples.append(self.data[t][0][chosen])
            labels.append(self.data[t][1][chosen])

    # -- GEM ------------------------------------------------------------------ #
    def _fit_gem(self) -> None:
        memory: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for t in range(N_TASKS):
            rows = np.arange(t * PER_TASK, (t + 1) * PER_TASK)
            for _ in range(self.steps):
                grads = _backward(self.params, self.data[t][0],
                                  self.data[t][1], rows)
                if memory:
                    grads = _project(grads, memory, self.params)
                for key in grads:
                    self.params[key] -= GEM_LR * grads[key]
            rng = np.random.default_rng(t)
            chosen = rng.choice(len(self.data[t][0]),
                                min(MEMORY, len(self.data[t][0])),
                                replace=False)
            memory.append((self.data[t][0][chosen], self.data[t][1][chosen],
                           rows))
        self.inference = None

    # -- LwF ------------------------------------------------------------------ #
    def _fit_lwf(self) -> None:
        for t in range(N_TASKS):
            rows = np.arange(t * PER_TASK, (t + 1) * PER_TASK)
            old = {key: value.copy() for key, value in self.params.items()}

            def extra(p, x, y, old=old):
                temperature = 2.0
                old_logits = _forward(old, x)[0]
                target = np.exp((old_logits - old_logits.max(axis=1,
                                                            keepdims=True))
                                / temperature)
                target /= target.sum(axis=1, keepdims=True)
                logits, cache = _forward(p, x)
                current = np.exp((logits - logits.max(axis=1, keepdims=True))
                                 / temperature)
                current /= current.sum(axis=1, keepdims=True)
                d = LWF_LAMBDA * temperature * (current - target)
                x_, z1, a1, z2, a2, w1, w2 = cache
                grads = {"W3": a2.T @ d / len(x), "b3": d.sum(0) / len(x)}
                dz2 = (d @ p["W3"].T) * (z2 > 0.0)
                grads["W2"] = a1.T @ dz2 / len(x)
                grads["b2"] = dz2.sum(0) / len(x)
                dz1 = (dz2 @ w2.T) * (z1 > 0.0)
                grads["W1"] = x_.T @ dz1 / len(x)
                grads["b1"] = dz1.sum(0) / len(x)
                return grads

            _train(self.params, self.data[t][0], self.data[t][1], rows,
                   self.steps, self.lr, extra=extra if t else None)

    # -- packnet -------------------------------------------------------------- #
    def _fit_packnet(self) -> None:
        free1 = np.ones(WIDTH, dtype=bool)
        free2 = np.ones(WIDTH, dtype=bool)
        masks = []
        for t in range(N_TASKS):
            rows = np.arange(t * PER_TASK, (t + 1) * PER_TASK)
            mask = _mask_for(self.params, free1, free2, rows)
            _train(self.params, self.data[t][0], self.data[t][1], rows,
                   self.steps, self.lr, gradmask=mask,
                   mask1=free1.astype(float), mask2=free2.astype(float))
            score1 = np.linalg.norm(self.params["W2"][free1][:, free2], axis=1)
            score2 = np.linalg.norm(self.params["W3"][free2][:, rows], axis=1)
            index1 = np.where(free1)[0]
            index2 = np.where(free2)[0]
            take1 = index1[np.argsort(-score1)[:PACK_BUDGET]]
            take2 = index2[np.argsort(-score2)[:PACK_BUDGET]]
            keep1 = np.zeros(WIDTH, dtype=bool)
            keep2 = np.zeros(WIDTH, dtype=bool)
            keep1[take1] = True
            keep2[take2] = True
            mask = _mask_for(self.params, keep1, keep2, rows)
            _train(self.params, self.data[t][0], self.data[t][1], rows,
                   steps=200, lr=self.lr, gradmask=mask,
                   mask1=keep1.astype(float), mask2=keep2.astype(float))
            free1 &= ~keep1
            free2 &= ~keep2
            masks.append((keep1.astype(float), keep2.astype(float)))
        self.inference = [{"mask1": masks[t][0], "mask2": masks[t][1],
                           "ad1": None, "ad2": None}
                          for t in range(N_TASKS)]

    # -- LoRA merge ----------------------------------------------------------- #
    def _fit_lora(self) -> None:
        adapters = []
        saved = self.params
        for t in range(N_TASKS):
            rows = np.arange(t * PER_TASK, (t + 1) * PER_TASK)
            head_mask = _mask_for(saved, np.zeros(WIDTH, dtype=bool),
                                  np.zeros(WIDTH, dtype=bool), rows)
            state = _new_state(saved)
            for step in range(1, 301):
                grads = _backward(saved, self.data[t][0], self.data[t][1],
                                  rows)
                grads = {key: grads[key] * head_mask[key] for key in grads}
                _adam(saved, grads, state, step, self.lr)
            rng = np.random.default_rng(t)
            ad1 = (rng.normal(0.0, 0.01, (2, LORA_RANK)),
                   np.zeros((LORA_RANK, WIDTH)))
            ad2 = (rng.normal(0.0, 0.01, (WIDTH, LORA_RANK)),
                   np.zeros((LORA_RANK, WIDTH)))
            adapter_state = _new_state({"A1": ad1[0], "B1": ad1[1],
                                        "A2": ad2[0], "B2": ad2[1]})
            for step in range(1, 401):
                grads = _backward(saved, self.data[t][0], self.data[t][1],
                                  rows, ad1=ad1, ad2=ad2)
                cur = {"A1": ad1[0], "B1": ad1[1], "A2": ad2[0], "B2": ad2[1]}
                _adam(cur, {key: grads[key] for key in cur},
                      adapter_state, step, LORA_LR)
                ad1 = (cur["A1"], cur["B1"])
                ad2 = (cur["A2"], cur["B2"])
                head = {key: grads[key] * head_mask[key]
                        for key in ("W3", "b3")}
                _adam(saved, head, state, step + 300, self.lr)
            adapters.append((ad1, ad2))
        self.params = saved
        self.inference = [{"mask1": None, "mask2": None,
                           "ad1": adapters[t][0], "ad2": adapters[t][1]}
                          for t in range(N_TASKS)]


# --------------------------------------------------------------------------- #
# GEM's constrained projection
# --------------------------------------------------------------------------- #
def _mask_for(params: dict, free1: np.ndarray, free2: np.ndarray,
              rows: np.ndarray) -> dict:
    mask = _zeros(params)
    mask["W1"][:, free1] = 1.0
    mask["b1"][free1] = 1.0
    mask["W2"][free1, :] = 1.0
    mask["W2"][:, free2] = 1.0
    mask["b2"][free2] = 1.0
    mask["W3"][:, rows] = 1.0
    mask["b3"][rows] = 1.0
    return mask


def _flatten(grads: dict) -> np.ndarray:
    return np.concatenate([grads[key].ravel() for key in sorted(grads)])


def _project(grads: dict, memory: list, params: dict) -> dict:
    keys = sorted(grads)
    current = _flatten(grads)
    constraints = np.stack([_flatten(_backward(params, x, y, rows))
                            for x, y, rows in memory])
    gram = constraints @ constraints.T + 1e-9 * np.eye(len(memory))
    linear = constraints @ current
    dual = np.zeros(len(memory))
    rate = 1.0 / (np.linalg.norm(gram, 2) + 1e-9)
    for _ in range(200):
        dual = np.maximum(0.0, dual - rate * (gram @ dual + linear))
    projected = current + constraints.T @ dual
    out, offset = {}, 0
    for key in keys:
        size = grads[key].size
        out[key] = projected[offset:offset + size].reshape(grads[key].shape)
        offset += size
    return out


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def joint_accuracy(seed: int = 1) -> float:
    """One model trained on all five tasks at once."""
    params = _base(seed)
    data = make_sequence()
    x = np.vstack([task[0] for task in data])
    y = np.concatenate([task[1] for task in data])
    _train(params, x, y, None, steps=1500, lr=PRETRAIN_LR)
    total = []
    for t in range(N_TASKS):
        xt, yt = data[t][2], data[t][3]
        keep = np.arange(t * PER_TASK, (t + 1) * PER_TASK)
        logits = _forward(params, xt)[0]
        total.append(float(np.mean(keep[logits[:, keep].argmax(axis=1)] == yt)))
    return float(np.mean(total))


def naive_accuracy(seed: int = 1) -> float:
    """Sequential fine-tuning with no defence at all."""
    params = _base(seed)
    data = make_sequence()
    for t in range(N_TASKS):
        rows = np.arange(t * PER_TASK, (t + 1) * PER_TASK)
        _train(params, data[t][0], data[t][1], rows)
    total = []
    for t in range(N_TASKS):
        keep = np.arange(t * PER_TASK, (t + 1) * PER_TASK)
        logits = _forward(params, data[t][2])[0]
        total.append(float(np.mean(keep[logits[:, keep].argmax(axis=1)]
                                   == data[t][3])))
    return float(np.mean(total))


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def demo() -> None:
    print("continual. five tasks, two ring classes each, ten classes total:")
    print("  a pretrained trunk, five fresh heads, and one shared model")
    joint = joint_accuracy()
    naive = naive_accuracy()
    print(f"  joint training on all five tasks  {joint:.3f}")
    print(f"  naive sequential fine-tuning      {naive:.3f}")

    print("continual. eight defences, five tasks each:")
    results = {}
    for variant in VARIANTS:
        learner = Continual(variant).fit()
        results[variant] = learner.evaluate()
        per_task = " ".join(f"{score:.2f}" for score in learner.scores)
        print(f"    {variant:<11s} {results[variant]:.3f}   [{per_task}]")

    print("continual. what each defence preserves:")
    print("  ewc, si and mas anchor important weights; replay and gem")
    print("  rehearse stored examples; lwf distils the old logits; packnet")
    print("  and lora_merge isolate the parameters each task may touch")

    print("continual. malformed calls are refused:")
    bad = [
        lambda: Continual("ewc_v2"),
        lambda: Continual("ewc", steps=0),
        lambda: Continual("ewc", lr=0.0),
        lambda: Continual("ewc").fit().predict(np.zeros((4, 3)), 0),
        lambda: Continual("ewc").fit().predict(np.zeros((4, 2)), 5),
        lambda: Continual("ewc").fit().predict(
            np.full((4, 2), np.nan), 0),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except (ValueError, IndexError):
            refused += 1
    assert refused == len(bad)

    best = max(results, key=results.get)
    print(f"  {refused} of {len(bad)} malformed calls refused")
    print(f"  {best} leads with {results[best]:.3f} against "
          f"{naive:.3f} naive")
    assert joint > 0.9
    assert naive < 0.8
    assert results["replay"] > 0.85
    assert results["packnet"] > 0.85
    assert results["lora_merge"] > 0.85
    assert results["gem"] > 0.75
    assert results["si"] > 0.70
    assert results["mas"] > 0.65
    assert results["ewc"] > 0.60
    assert results["lwf"] < naive
    print("continual demo passed")


if __name__ == "__main__":
    demo()
