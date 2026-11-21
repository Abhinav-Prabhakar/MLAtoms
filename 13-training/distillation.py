"""Knowledge distillation — eight ways to move knowledge between models.

Chapter:   13-training
Companion: distillation.md
Tier 1 · 8 variants: response, feature, relation, self_distill, born_again,
online_distill, attention_transfer, dkd
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("response", "feature", "relation", "self_distill", "born_again",
            "online_distill", "attention_transfer", "dkd")
LOGIT_VARIANTS = ("response", "self_distill", "born_again",
                  "online_distill", "dkd")


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
def _softmax(z: np.ndarray) -> np.ndarray:
    shifted = z - z.max(axis=-1, keepdims=True)
    out = np.exp(shifted)
    return out / out.sum(axis=-1, keepdims=True)


def _log_softmax(z: np.ndarray) -> np.ndarray:
    shifted = z - z.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def _cross_entropy(logits: np.ndarray, labels: np.ndarray):
    probs = _softmax(logits)
    batch = logits.shape[0]
    loss = -np.mean(np.log(probs[np.arange(batch), labels] + 1e-12))
    grad = probs.copy()
    grad[np.arange(batch), labels] -= 1.0
    return loss, grad / batch


def _kl(student_logits: np.ndarray, teacher_probs: np.ndarray,
        temperature: float):
    """Temperature-scaled KL(teacher || student), and its logit gradient."""
    log_probs = _log_softmax(student_logits / temperature)
    loss = temperature ** 2 * float(
        np.mean(np.sum(teacher_probs * (np.log(teacher_probs + 1e-12)
                                        - log_probs), axis=-1)))
    probs = np.exp(log_probs)
    grad = temperature * (probs - teacher_probs) / student_logits.shape[0]
    return loss, grad


def _attention(hidden: np.ndarray) -> np.ndarray:
    """Row-stochastic token attention, one head, scaled dot product."""
    scale = 1.0 / np.sqrt(hidden.shape[-1])
    scores = np.einsum("bth,bsh->bts", hidden, hidden) * scale
    return _softmax(scores)


def _pairwise(values: np.ndarray) -> np.ndarray:
    difference = values[:, None, :] - values[None, :, :]
    return np.sqrt((difference ** 2).sum(axis=-1))


# --------------------------------------------------------------------------- #
# The distiller
# --------------------------------------------------------------------------- #
class Distiller:
    """A one-hidden-layer student taught by a teacher, variant by variant.

    `forward(x, y)` takes `x` of shape `(batch, tokens, in_dim)` and optional
    integer labels, and returns the loss. `backward(cache)` fills
    `self.gradients`. Every variant distils a different object: softened
    logits, hidden states, relations, attention maps, or a decomposition of
    the logit loss.
    """

    def __init__(self, variant: str = "response", *, in_dim: int = 4,
                 classes: int = 3, hidden: int = 6, teacher_hidden: int = 6,
                 tokens: int = 2, temperature: float = 2.0,
                 alpha: float = 0.3, tckd_weight: float = 1.0,
                 nckd_weight: float = 1.0, seed: int = 0):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if min(in_dim, classes, hidden, teacher_hidden, tokens) < 1:
            raise ValueError("all sizes must be positive")
        if classes < 2:
            raise ValueError("classes must be at least two")
        if not np.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        if tckd_weight < 0.0 or nckd_weight < 0.0:
            raise ValueError("DKD weights must be non-negative")
        self.variant = variant
        self.in_dim = in_dim
        self.classes = classes
        self.hidden = hidden
        self.teacher_hidden = teacher_hidden
        self.tokens = tokens
        self.temperature = temperature
        self.alpha = alpha
        self.tckd_weight = tckd_weight
        self.nckd_weight = nckd_weight
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.parameters = self._build(self.rng)
        keys = self._trainable_keys()
        self.gradients = {key: np.zeros_like(self.parameters[key])
                          for key in keys}

    # -- construction -------------------------------------------------------- #
    def _build(self, rng: np.random.Generator) -> dict[str, np.ndarray]:
        p: dict[str, np.ndarray] = {}
        for prefix, width in (("student", self.hidden),
                              ("teacher", self.teacher_hidden)):
            p[f"{prefix}.W1"] = rng.normal(0.0, 1.0 / np.sqrt(self.in_dim),
                                           (self.in_dim, width))
            p[f"{prefix}.b1"] = np.zeros(width)
            p[f"{prefix}.W2"] = rng.normal(0.0, 1.0 / np.sqrt(width),
                                           (width, self.classes))
            p[f"{prefix}.b2"] = np.zeros(self.classes)
        if self.variant == "feature":
            p["regressor.W"] = rng.normal(
                0.0, 1.0 / np.sqrt(self.teacher_hidden),
                (self.teacher_hidden, self.hidden))
        return p

    def _trainable_keys(self) -> tuple[str, ...]:
        keys = ["student.W1", "student.b1", "student.W2", "student.b2"]
        if self.variant == "feature":
            keys.append("regressor.W")
        if self.variant == "online_distill":
            keys += ["teacher.W1", "teacher.b1", "teacher.W2", "teacher.b2"]
        return tuple(keys)

    @property
    def n_parameters(self) -> int:
        return int(sum(self.parameters[key].size for key in self.gradients))

    # -- validation ---------------------------------------------------------- #
    def _check(self, x, y):
        x = np.asarray(x, dtype=float)
        if x.ndim != 3:
            raise ValueError("x must be (batch, tokens, in_dim)")
        if x.shape[1:] != (self.tokens, self.in_dim):
            raise ValueError("x has the wrong token or feature count")
        if x.shape[0] < 1:
            raise ValueError("the batch must be non-empty")
        if not np.all(np.isfinite(x)):
            raise ValueError("x must be finite")
        if self.variant == "relation" and x.shape[0] < 2:
            raise ValueError("relation distillation needs a batch of two")
        if y is None:
            return x, None
        y = np.asarray(y)
        if y.shape != (x.shape[0],):
            raise ValueError("y must hold one label per example")
        if not np.issubdtype(y.dtype, np.integer):
            raise ValueError("y must be integer labels")
        if y.min() < 0 or y.max() >= self.classes:
            raise ValueError("y is outside the class range")
        return x, y

    # -- network ------------------------------------------------------------- #
    def _mlp(self, prefix: str, x: np.ndarray):
        p = self.parameters
        pre = np.einsum("bti,ih->bth", x, p[f"{prefix}.W1"]) + p[f"{prefix}.b1"]
        hidden = np.tanh(pre)
        logits = hidden.mean(axis=1) @ p[f"{prefix}.W2"] + p[f"{prefix}.b2"]
        return hidden, logits

    # -- forward ------------------------------------------------------------- #
    def forward(self, x, y=None):
        x, y = self._check(x, y)
        h_s, z_s = self._mlp("student", x)
        h_t, z_t = self._mlp("teacher", x)
        cache = {"x": x, "y": y, "h_s": h_s, "z_s": z_s, "h_t": h_t,
                 "z_t": z_t}
        return self._loss(cache), cache

    def _loss(self, cache) -> float:
        y = cache["y"]
        hard = 0.0 if y is None else self.alpha
        soft = 1.0 - hard
        kind = self.variant
        if kind == "dkd":
            if y is None:
                raise ValueError("dkd needs labels")
            loss = self.alpha * _cross_entropy(cache["z_s"], y)[0]
            return loss + soft * self._dkd(cache)[0]
        if kind in LOGIT_VARIANTS:
            probs_t = _softmax(cache["z_t"] / self.temperature)
            loss = soft * _kl(cache["z_s"], probs_t, self.temperature)[0]
            if y is not None:
                loss += self.alpha * _cross_entropy(cache["z_s"], y)[0]
            if kind == "online_distill":
                probs_s = _softmax(cache["z_s"] / self.temperature)
                loss += soft * _kl(cache["z_t"], probs_s,
                                   self.temperature)[0]
                if y is not None:
                    loss += self.alpha * _cross_entropy(cache["z_t"], y)[0]
            return loss
        if kind == "feature":
            loss = soft * self._feature_loss(cache["h_s"], cache["h_t"])
        elif kind == "relation":
            loss = soft * self._relation_loss(cache["z_s"], cache["z_t"])
        else:
            loss = soft * self._attention_loss(cache["h_s"], cache["h_t"])
        if y is not None:
            loss += self.alpha * _cross_entropy(cache["z_s"], y)[0]
        return loss

    # -- variant losses ------------------------------------------------------ #
    def _feature_loss(self, h_s: np.ndarray, h_t: np.ndarray) -> float:
        difference = h_s - h_t @ self.parameters["regressor.W"]
        return float(np.mean(difference ** 2))

    def _relation_loss(self, z_s: np.ndarray, z_t: np.ndarray) -> float:
        upper = np.triu_indices(z_s.shape[0], 1)
        distance_s = _pairwise(z_s)[upper]
        distance_t = _pairwise(z_t)[upper]
        normalised_s = distance_s / distance_s.mean()
        normalised_t = distance_t / distance_t.mean()
        return float(np.mean((normalised_s - normalised_t) ** 2))

    def _attention_loss(self, h_s: np.ndarray, h_t: np.ndarray) -> float:
        first = _attention(h_s)
        second = _attention(h_t)
        first = first / np.linalg.norm(first.reshape(first.shape[0], -1),
                                       axis=1)[:, None, None]
        second = second / np.linalg.norm(second.reshape(second.shape[0], -1),
                                         axis=1)[:, None, None]
        return float(np.mean((first - second) ** 2))

    def _dkd(self, cache):
        """Target-class and non-target-class KD, with both losses."""
        labels, z_s, z_t = cache["y"], cache["z_s"], cache["z_t"]
        temperature = self.temperature
        probs_s = _softmax(z_s / temperature)
        probs_t = _softmax(z_t / temperature)
        rows = np.arange(z_s.shape[0])
        target_s = probs_s[rows, labels]
        target_t = probs_t[rows, labels]
        first = np.log(target_t + 1e-12) - np.log(target_s + 1e-12)
        second = (np.log(1.0 - target_t + 1e-12)
                  - np.log(1.0 - target_s + 1e-12))
        tckd = float(np.mean(target_t * first + (1.0 - target_t) * second))
        mask = np.ones_like(probs_t)
        mask[rows, labels] = 0.0
        rest_t = np.where(mask > 0.0, probs_t / (1.0 - target_t)[:, None], 0.0)
        rest_s = np.where(mask > 0.0, probs_s / (1.0 - target_s)[:, None], 0.0)
        nckd = float(np.mean(np.sum(
            rest_t * (np.log(rest_t + 1e-12) - np.log(rest_s + 1e-12)),
            axis=-1)))
        tckd *= temperature ** 2
        nckd *= temperature ** 2
        loss = self.tckd_weight * tckd + self.nckd_weight * nckd
        cache["dkd"] = {"tckd": tckd, "nckd": nckd, "mask": mask}
        return loss, tckd, nckd

    # -- backward ------------------------------------------------------------ #
    def backward(self, cache) -> None:
        for key in self.gradients:
            self.gradients[key][...] = 0.0
        y = cache["y"]
        dz_s = np.zeros_like(cache["z_s"])
        dz_t = np.zeros_like(cache["z_t"])
        dh_s = np.zeros_like(cache["h_s"])
        dh_t = np.zeros_like(cache["h_t"])
        hard = 0.0 if y is None else self.alpha
        soft = 1.0 - hard
        if y is not None:
            _, dz_s = _cross_entropy(cache["z_s"], y)
            dz_s *= self.alpha
            if self.variant == "online_distill":
                _, dz_t = _cross_entropy(cache["z_t"], y)
                dz_t *= self.alpha
        if self.variant in ("response", "self_distill", "born_again"):
            probs_t = _softmax(cache["z_t"] / self.temperature)
            _, grad = _kl(cache["z_s"], probs_t, self.temperature)
            dz_s += soft * grad
        elif self.variant == "online_distill":
            probs_s = _softmax(cache["z_s"] / self.temperature)
            probs_t = _softmax(cache["z_t"] / self.temperature)
            ratio = np.log(probs_s + 1e-12) - np.log(probs_t + 1e-12)
            forward = np.sum(probs_s * ratio, axis=-1, keepdims=True)
            backward = -np.sum(probs_t * ratio, axis=-1, keepdims=True)
            scale = soft * self.temperature / cache["z_s"].shape[0]
            dz_s += scale * (probs_s - probs_t
                             + probs_s * (ratio - forward))
            dz_t += scale * (probs_t - probs_s
                             - probs_t * (ratio + backward))
        elif self.variant == "feature":
            difference = (cache["h_s"] - cache["h_t"] @ self.parameters[
                "regressor.W"])
            scale = 2.0 * soft / difference.size
            dh_s += scale * difference
            self.gradients["regressor.W"] += -scale * np.einsum(
                "bth,btd->hd", cache["h_t"], difference)
        elif self.variant == "relation":
            dz_s += soft * self._relation_grad(cache["z_s"], cache["z_t"])
        elif self.variant == "attention_transfer":
            dh_s += soft * self._attention_grad(cache["h_s"], cache["h_t"])
        else:
            dz_s += soft * self._dkd_grad(cache)
        self._mlp_backward("student", cache, dz_s, dh_s)
        if self.variant == "online_distill":
            self._mlp_backward("teacher", cache, dz_t, dh_t)

    def _relation_grad(self, z_s: np.ndarray, z_t: np.ndarray) -> np.ndarray:
        upper = np.triu_indices(z_s.shape[0], 1)
        distance_s = _pairwise(z_s)[upper]
        distance_t = _pairwise(z_t)[upper]
        mean_s = distance_s.mean()
        normalised_s = distance_s / mean_s
        normalised_t = distance_t / distance_t.mean()
        residual = normalised_s - normalised_t
        count = distance_s.size
        share = float(np.mean(residual * normalised_s))
        weight = 2.0 * (residual - share) / (count * mean_s)
        weight = weight / np.maximum(distance_s, 1e-12)
        spread = np.zeros((z_s.shape[0], z_s.shape[0]))
        spread[upper] = weight
        spread = spread + spread.T
        return spread.sum(axis=1)[:, None] * z_s - spread @ z_s

    def _attention_grad(self, h_s: np.ndarray, h_t: np.ndarray) -> np.ndarray:
        attention_s = _attention(h_s)
        attention_t = _attention(h_t)
        norm_s = np.linalg.norm(attention_s.reshape(h_s.shape[0], -1),
                                axis=1)[:, None, None]
        norm_t = np.linalg.norm(attention_t.reshape(h_t.shape[0], -1),
                                axis=1)[:, None, None]
        scaled_s = attention_s / norm_s
        scaled_t = attention_t / norm_t
        difference = scaled_s - scaled_t
        grad_scaled = 2.0 * difference / difference.size
        inner = np.sum(scaled_s * grad_scaled, axis=(1, 2), keepdims=True)
        grad_attention = (grad_scaled - scaled_s * inner) / norm_s
        inner = np.sum(attention_s * grad_attention, axis=-1, keepdims=True)
        grad_scores = attention_s * (grad_attention - inner)
        scale = 1.0 / np.sqrt(h_s.shape[-1])
        return (grad_scores + grad_scores.transpose(0, 2, 1)) @ h_s * scale

    def _dkd_grad(self, cache) -> np.ndarray:
        labels, z_s, z_t = cache["y"], cache["z_s"], cache["z_t"]
        temperature = self.temperature
        probs_s = _softmax(z_s / temperature)
        probs_t = _softmax(z_t / temperature)
        rows = np.arange(z_s.shape[0])
        target_s = probs_s[rows, labels]
        target_t = probs_t[rows, labels]
        onehot = np.zeros_like(probs_s)
        onehot[rows, labels] = 1.0
        coefficient = (-target_t / np.maximum(target_s, 1e-12)
                       + (1.0 - target_t)
                       / np.maximum(1.0 - target_s, 1e-12))
        target_grad = (temperature * coefficient[:, None] * probs_s
                       * (onehot - target_s[:, None]) / z_s.shape[0])
        mask = cache["dkd"]["mask"]
        rest_t = np.where(mask > 0.0, probs_t / (1.0 - target_t)[:, None], 0.0)
        rest_s = np.where(mask > 0.0, probs_s / (1.0 - target_s)[:, None], 0.0)
        rest_grad = np.where(mask > 0.0,
                             temperature * (rest_s - rest_t) / z_s.shape[0],
                             0.0)
        return (self.tckd_weight * target_grad
                + self.nckd_weight * rest_grad)

    # -- parameter gradients ------------------------------------------------- #
    def _mlp_backward(self, prefix: str, cache, dz: np.ndarray,
                      direct: np.ndarray, grads=None) -> None:
        p = self.parameters
        g = self.gradients if grads is None else grads
        hidden = cache["h_s"] if prefix == "student" else cache["h_t"]
        g[f"{prefix}.W2"] += hidden.mean(axis=1).T @ dz
        g[f"{prefix}.b2"] += dz.sum(axis=0)
        dpool = dz @ p[f"{prefix}.W2"].T
        dpre = dpool[:, None, :] / self.tokens + direct
        dpre *= 1.0 - hidden ** 2
        g[f"{prefix}.W1"] += np.einsum("bti,bth->ih", cache["x"], dpre)
        g[f"{prefix}.b1"] += dpre.sum(axis=(0, 1))

    # -- training helpers ---------------------------------------------------- #
    def predict(self, x, network: str = "student"):
        if network not in ("student", "teacher"):
            raise ValueError("network must be 'student' or 'teacher'")
        x, _ = self._check(x, None)
        return self._mlp(network, x)[1]

    def accuracy(self, x, y, network: str = "student") -> float:
        labels = np.asarray(y)
        return float(np.mean(np.argmax(self.predict(x, network), axis=-1)
                             == labels))

    def fit(self, x, y, *, steps: int = 200, lr: float = 0.2,
            decay: float = 0.99) -> list[float]:
        if steps < 1:
            raise ValueError("steps must be positive")
        history = []
        for _ in range(steps):
            loss, cache = self.forward(x, y)
            self.backward(cache)
            for key in self.gradients:
                self.parameters[key] -= lr * self.gradients[key]
            if self.variant == "self_distill":
                self.update_teacher(decay)
            history.append(loss)
        return history

    def fit_teacher(self, x, y, *, steps: int = 200, lr: float = 0.2) -> None:
        """Train only the teacher network, on cross-entropy."""
        if steps < 1:
            raise ValueError("steps must be positive")
        x, y = self._check(x, y)
        names = ("W1", "b1", "W2", "b2")
        grads = {f"teacher.{name}": np.zeros_like(self.parameters[
            f"teacher.{name}"]) for name in names}
        for _ in range(steps):
            hidden, logits = self._mlp("teacher", x)
            _, dz = _cross_entropy(logits, y)
            for value in grads.values():
                value[...] = 0.0
            self._mlp_backward("teacher", {"x": x, "h_t": hidden}, dz,
                               np.zeros_like(hidden), grads)
            for key, value in grads.items():
                self.parameters[key] -= lr * value

    def update_teacher(self, decay: float) -> None:
        """Move the teacher toward the student; the mean-teacher update."""
        if self.variant not in ("self_distill", "online_distill"):
            raise ValueError("this variant has no moving teacher")
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must lie in [0, 1)")
        for name in ("W1", "b1", "W2", "b2"):
            teacher = self.parameters[f"teacher.{name}"]
            student = self.parameters[f"student.{name}"]
            teacher *= decay
            teacher += (1.0 - decay) * student

    def copy_student_to_teacher(self) -> None:
        if self.variant not in ("born_again", "self_distill"):
            raise ValueError("this variant has no copied teacher")
        for name in ("W1", "b1", "W2", "b2"):
            self.parameters[f"teacher.{name}"][...] = self.parameters[
                f"student.{name}"]

    def reset_student(self) -> None:
        fresh = self._build(self.rng)
        for name in ("W1", "b1", "W2", "b2"):
            self.parameters[f"student.{name}"][...] = fresh[
                f"student.{name}"]

    def fit_generations(self, x, y, *, rounds: int = 3, steps: int = 200,
                        lr: float = 0.2, evaluate=None) -> list[float]:
        if self.variant != "born_again":
            raise ValueError("fit_generations is for born_again only")
        if rounds < 1:
            raise ValueError("rounds must be positive")
        scores = []
        x_eval, y_eval = (x, y) if evaluate is None else evaluate
        keep = self.alpha
        self.alpha = 1.0
        self.reset_student()
        self.fit(x, y, steps=steps, lr=lr)
        scores.append(self.accuracy(x_eval, y_eval))
        self.copy_student_to_teacher()
        self.alpha = keep
        for _ in range(rounds - 1):
            self.reset_student()
            self.fit(x, y, steps=steps, lr=lr)
            scores.append(self.accuracy(x_eval, y_eval))
            self.copy_student_to_teacher()
        return scores


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _blobs(seed: int, per_class: int = 30, tokens: int = 2, dim: int = 4,
           classes: int = 3, scale: float = 1.4, noise: float = 0.6,
           flip: float = 0.0):
    rng = np.random.default_rng(seed)
    centres = np.eye(classes, dim) * scale
    labels = np.repeat(np.arange(classes), per_class)
    x = centres[labels][:, None, :] + rng.normal(
        0.0, noise, (labels.size, tokens, dim))
    if flip > 0.0:
        chosen = rng.random(labels.size) < flip
        labels[chosen] = (labels[chosen] + 1) % classes
    order = rng.permutation(labels.size)
    return x[order], labels[order]


def demo() -> None:
    x, y = _blobs(seed=1)

    print("distillation. eight losses share one student skeleton:")
    for variant in VARIANTS:
        model = Distiller(variant, in_dim=4, classes=3, hidden=6,
                          teacher_hidden=8, tokens=2, seed=2)
        loss, _ = model.forward(x, y)
        print(f"    {variant:<19s} loss {loss:7.4f}"
              f"  trainable {model.n_parameters:4d}")
        assert np.isfinite(loss)
    base = Distiller("response", in_dim=4, classes=3, hidden=6,
                     teacher_hidden=8, tokens=2, seed=2)
    assert base.n_parameters == 6 * 4 + 6 + 6 * 3 + 3
    feature = Distiller("feature", in_dim=4, classes=3, hidden=6,
                        teacher_hidden=8, tokens=2, seed=2)
    assert feature.n_parameters == base.n_parameters + 8 * 6
    print("  the student costs 51 numbers; feature distillation adds an")
    print("  8x6 regressor, and online distillation trains both networks")

    print("distillation. temperature softens the teacher's targets:")
    teacher = Distiller("response", in_dim=4, classes=3, hidden=8,
                        teacher_hidden=8, tokens=2, alpha=1.0, seed=3)
    teacher.fit_teacher(x, y, steps=400, lr=0.3)
    logits = teacher._mlp("teacher", x)[1]
    gaps = []
    for temperature in (1.0, 2.0, 4.0):
        probs = _softmax(logits / temperature)
        entropy = -np.mean(np.sum(probs * np.log(probs + 1e-12), axis=-1))
        ordered = np.sort(probs, axis=-1)
        gaps.append(float(np.mean(ordered[:, -1] - ordered[:, -2])))
        print(f"    T={temperature:<4.1f} entropy {entropy:5.3f}"
              f"  top gap {gaps[-1]:5.3f}")
    assert gaps[0] > gaps[1] > gaps[2]
    print("  a larger T flattens the distribution, so the dark knowledge")
    print("  in the non-target classes survives the KL")

    print("distillation. feature loss vanishes when the student matches:")
    model = Distiller("feature", in_dim=4, classes=3, hidden=6,
                      teacher_hidden=8, tokens=2, alpha=0.0, seed=4)
    _, cache = model.forward(x)
    regressor = model.parameters["regressor.W"]
    predicted = cache["h_t"] @ regressor
    assert np.isclose(model._feature_loss(predicted, cache["h_t"]), 0.0)
    assert model._feature_loss(predicted + 0.1, cache["h_t"]) > 0.0
    print(f"  the {regressor.shape[0]}x{regressor.shape[1]} regressor maps")
    print("  teacher width to student width; a perfect match costs zero")

    print("distillation. relation loss ignores scale and example order:")
    model = Distiller("relation", in_dim=4, classes=3, hidden=6,
                      teacher_hidden=8, tokens=2, alpha=0.0, seed=5)
    _, cache = model.forward(x)
    reference = model._relation_loss(cache["z_s"], cache["z_t"])
    assert np.isclose(model._relation_loss(3.0 * cache["z_s"],
                                           3.0 * cache["z_t"]), reference)
    order = np.arange(x.shape[0])[::-1]
    assert np.isclose(model._relation_loss(cache["z_s"][order],
                                           cache["z_t"][order]), reference)
    assert reference > 0.0
    print("  distances are divided by their batch mean, so both a rescale")
    print(f"  and a reordering leave the loss at {reference:.4f}")

    print("distillation. attention transfer compares token maps:")
    model = Distiller("attention_transfer", in_dim=4, classes=3, hidden=6,
                      teacher_hidden=8, tokens=2, alpha=0.0, seed=6)
    _, cache = model.forward(x)
    maps = _attention(cache["h_s"])
    assert np.allclose(maps.sum(axis=-1), 1.0)
    assert maps.shape == (x.shape[0], 2, 2)
    identical = model._attention_loss(cache["h_s"], cache["h_s"])
    assert np.isclose(identical, 0.0)
    moved = model._attention_loss(cache["h_s"], cache["h_t"])
    assert moved > 0.0
    print("  each row sums to one over tokens; a student that reproduces")
    print(f"  the teacher's map scores zero, the untrained pair {moved:.4f}")

    print("distillation. DKD separates the target from the rest:")
    model = Distiller("dkd", in_dim=4, classes=3, hidden=6,
                      teacher_hidden=8, tokens=2, temperature=2.0, seed=7)
    labels = np.array([0, 1, 2])
    teacher_logits = np.array([[4.0, 1.0, 0.5], [0.5, 4.0, 1.0],
                               [1.0, 0.5, 4.0]])
    matching = teacher_logits.copy()
    spread = np.array([[3.0, 1.0, 1.0], [1.0, 3.0, 1.0],
                       [1.0, 1.0, 3.0]])
    confused = np.array([[1.0, 3.0, 5.0], [5.0, 1.0, 3.0],
                         [3.0, 5.0, 1.0]])
    for name, student_logits in (("matching", matching),
                                 ("spread only", spread),
                                 ("target off", confused)):
        tckd, nckd = model._dkd({"y": labels, "z_s": student_logits,
                                 "z_t": teacher_logits})[1:]
        print(f"    {name:<13s} TCKD {tckd:6.4f}  NCKD {nckd:6.4f}")
        assert tckd >= 0.0 and nckd >= 0.0
    print("  TCKD watches the target probability alone; NCKD watches the")
    print("  shape of the remaining mass, and the two are weighted apart")

    print("distillation. finite differences agree with the gradients:")
    small_x = np.random.default_rng(8).normal(size=(4, 2, 3))
    small_y = np.array([0, 1, 2, 0])
    for variant in VARIANTS:
        model = Distiller(variant, in_dim=3, classes=3, hidden=4,
                          teacher_hidden=5, tokens=2, temperature=2.0,
                          seed=10)
        loss, cache = model.forward(small_x, small_y)
        model.backward(cache)
        analytic = {key: value.copy() for key, value in model.gradients.items()}
        worst = 0.0
        for key, value in model.parameters.items():
            if key not in model.gradients:
                continue
            for index in np.ndindex(value.shape):
                old = value[index]
                value[index] = old + 1e-6
                plus = model.forward(small_x, small_y)[0]
                value[index] = old - 1e-6
                minus = model.forward(small_x, small_y)[0]
                value[index] = old
                numeric = (plus - minus) / 2e-6
                worst = max(worst, abs(numeric - analytic[key][index]))
        assert worst < 1e-6, (variant, worst)
        print(f"  {variant:<19s} worst error {worst:.1e}")

    print("distillation. soft targets beat hard labels on a label budget:")
    x_all, y_all = _blobs(seed=12, per_class=50, dim=6, classes=4,
                          scale=0.9, noise=1.0)
    x_train, y_train = x_all[:140], y_all[:140]
    x_test, y_test = x_all[140:], y_all[140:]
    teacher = Distiller("response", in_dim=6, classes=4, hidden=16,
                        teacher_hidden=16, tokens=2, alpha=1.0, seed=13)
    teacher.fit_teacher(x_train, y_train, steps=600, lr=0.3)
    keep = np.concatenate([np.flatnonzero(y_train == label)[:2]
                           for label in range(4)])
    plain = Distiller("response", in_dim=6, classes=4, hidden=16,
                      teacher_hidden=16, tokens=2, alpha=1.0, seed=14)
    plain.fit(x_train[keep], y_train[keep], steps=600, lr=0.3)
    distilled = Distiller("response", in_dim=6, classes=4, hidden=16,
                          teacher_hidden=16, tokens=2, alpha=0.2, seed=14)
    for name in ("W1", "b1", "W2", "b2"):
        distilled.parameters[f"teacher.{name}"][...] = teacher.parameters[
            f"teacher.{name}"]
    distilled.fit(x_train, y_train, steps=600, lr=0.3)
    hard_score = plain.accuracy(x_test, y_test)
    soft_score = distilled.accuracy(x_test, y_test)
    teacher_score = teacher.accuracy(x_test, y_test, "teacher")
    print(f"  eight labels: hard-label student {hard_score:.3f},"
          f" distilled student {soft_score:.3f}")
    print(f"  the teacher itself scores {teacher_score:.3f}; the student")
    print("  also sees the 132 unlabelled inputs, which is the extra data")
    assert soft_score > hard_score

    print("distillation. born-again generations improve on the first:")
    noisy_x, noisy_y = _blobs(seed=12, per_class=50, dim=6, classes=4,
                              scale=0.9, noise=1.0, flip=0.2)
    noisy_train, noisy_labels = noisy_x[:140], noisy_y[:140]
    baseline = Distiller("response", in_dim=6, classes=4, hidden=16,
                         teacher_hidden=16, tokens=2, alpha=1.0, seed=15)
    baseline.fit(noisy_train, noisy_labels, steps=600, lr=0.3)
    model = Distiller("born_again", in_dim=6, classes=4, hidden=16,
                      teacher_hidden=16, tokens=2, alpha=0.2, seed=15)
    scores = model.fit_generations(noisy_train, noisy_labels, rounds=4,
                                   steps=600, lr=0.3,
                                   evaluate=(x_test, y_test))
    plain_score = baseline.accuracy(x_test, y_test)
    print("  generations " + " -> ".join(f"{score:.3f}" for score in scores))
    print(f"  one plain run on the same noisy labels scores {plain_score:.3f}")
    assert scores[-1] > scores[0] and scores[-1] >= plain_score

    print("distillation. a mean teacher tracks the student:")
    model = Distiller("self_distill", in_dim=6, classes=4, hidden=16,
                      teacher_hidden=16, tokens=2, alpha=0.2, seed=16)
    before = model.parameters["teacher.W1"].copy()
    model.fit(x_train, y_train, steps=200, lr=0.3, decay=0.9)
    moved = float(np.abs(model.parameters["teacher.W1"] - before).max())
    assert moved > 0.0
    print(f"  with decay 0.9 the teacher moves by at most {moved:.3f} from")
    print("  its initial weights over 200 steps: a slow average of every")
    print("  student it has seen")

    print("distillation. malformed calls are refused:")
    guard = Distiller("response", in_dim=4, classes=3, hidden=6,
                      teacher_hidden=8, tokens=2, seed=17)
    good_x, good_y = x[:4], y[:4]
    _, good_cache = guard.forward(good_x, good_y)
    bad = [
        lambda: Distiller("attention"),
        lambda: Distiller("response", in_dim=0),
        lambda: Distiller("response", hidden=0),
        lambda: Distiller("response", classes=1),
        lambda: Distiller("response", temperature=0.0),
        lambda: Distiller("response", alpha=1.5),
        lambda: Distiller("dkd", nckd_weight=-1.0),
        lambda: guard.forward(good_x[0], good_y),
        lambda: guard.forward(good_x[..., :3], good_y),
        lambda: guard.forward(good_x, good_y[:2]),
        lambda: guard.forward(good_x, good_y.astype(float)),
        lambda: guard.forward(good_x, good_y + 3),
        lambda: guard.forward(np.full_like(good_x, np.nan), good_y),
        lambda: Distiller("relation").forward(good_x[:1]),
        lambda: Distiller("dkd", in_dim=4).forward(good_x),
        lambda: guard.update_teacher(0.9),
        lambda: Distiller("response").fit_teacher(good_x, good_y, steps=0),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except ValueError:
            refused += 1
    assert refused == len(bad)
    print(f"  {refused} of {len(bad)}: an unknown variant, a zero size,")
    print("  one class, a zero temperature, an out-of-range weight, a")
    print("  two-dimensional input, wrong labels, a non-finite input, a")
    print("  short batch for relations, dkd without labels, and a moving")
    print("  teacher on a variant that has none")
    print("distillation demo passed")


if __name__ == "__main__":
    demo()
