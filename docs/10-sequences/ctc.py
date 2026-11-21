r"""Connectionist temporal classification — summing over alignments.

Chapter:   10-sequences
Companion: ctc.md
Tier 2 · 4 variants: ctc_loss, greedy_decode, beam_decode, prefix_beam

    p(l | x) = sum over every frame alignment pi of x that collapses to l    (1)
    L(x, l)  = -log p(l | x)                                                (2)

A sequence model that emits one distribution over symbols per frame produces a
label for every frame, but the training data gives labels only for the sequence
as a whole. CTC closes the gap by defining the probability of a label sequence as
the sum over every frame-level path that collapses to it: merge repeated symbols,
then delete the blank. The sum is over C^T paths, so it is never enumerated; on
the extended lattice it factors, and one forward pass gives the exact likelihood.

The four variants are one loss and three decoders. `ctc_loss` is the
forward-backward pass in the extended lattice, with the gradient with respect to
the frame logits. `greedy_decode` takes the most likely symbol at each frame and
collapses it. `beam_decode` searches over extended paths, keeping the best `beam`
of them at each frame. `prefix_beam` searches over collapsed prefixes instead,
tracking the probability of ending in a blank and of ending in a label, which is
the same search with the alignments of one prefix merged as it goes.

Everything is numpy and everything is exact where it claims to be. The loss and
its gradient are checked against enumeration of every path on a small lattice,
and against central differences through a softmax. The decoders are checked
against an exact marginal decoder that enumerates every output sequence. The
numbers to carry away: the loss matches the enumeration of all 81 frame paths to
4e-16, the symbol marginals to 2.2e-16 and the gradient to 8.4e-10, while the
best single alignment carries only 7% of the target's probability. On 24 random
problems greedy decoding agrees with the exact marginal decoder 9 times, a
width-eight beam over extended paths 13 times, and a width-eight prefix beam 23
times, which reaches 24 of 24 at width 32.
"""  # noqa: E501

from __future__ import annotations

import itertools

import numpy as np

VARIANTS = ("ctc_loss", "greedy_decode", "beam_decode", "prefix_beam")
NEG = -np.inf


def _lse(values: np.ndarray) -> float:
    """Log-sum-exp of a one-dimensional array, safe when everything is -inf."""
    peak = float(values.max())
    if not np.isfinite(peak):
        return NEG
    return peak + float(np.log(np.exp(values - peak).sum()))


def _lse_rows(values: np.ndarray) -> np.ndarray:
    """Log-sum-exp along axis 0 of a stack of equally shaped rows."""
    peak = values.max(axis=0)
    safe = np.where(np.isfinite(peak), peak, 0.0)
    with np.errstate(divide="ignore"):
        total = safe + np.log(np.exp(values - safe).sum(axis=0))
    return np.where(peak > NEG, total, NEG)


class CTC:
    """CTC loss, posterior and the three decoders.

    Parameters
    ----------
    variant : str
        One of `VARIANTS`.
    blank : int
        Channel reserved for the blank symbol.
    beam : int
        Number of beams kept by the two search decoders.

    Attributes
    ----------
    n_states_ : int
        Length of the extended label sequence for the last call.
    """

    def __init__(self, *, variant: str = "ctc_loss", blank: int = 0,
                 beam: int = 8) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
        if not isinstance(blank, (int, np.integer)) or blank < 0:
            raise ValueError("blank must be a non-negative integer")
        if not isinstance(beam, (int, np.integer)) or beam < 1:
            raise ValueError("beam must be a positive integer")
        self.variant = variant
        self.blank = int(blank)
        self.beam = int(beam)

    # ------------------------------------------------------------------ #
    # input handling
    # ------------------------------------------------------------------ #
    @staticmethod
    def _probs(probs: np.ndarray) -> np.ndarray:
        array = np.asarray(probs, dtype=float)
        if array.ndim == 2:
            array = array[None]
        if array.ndim != 3:
            raise ValueError("probs must be (frames, channels) or a batch of those")
        if min(array.shape) < 1:
            raise ValueError("frames and channels must be positive")
        if not np.all(np.isfinite(array)):
            raise ValueError("probs must be finite")
        if np.any(array < 0.0):
            raise ValueError("probs must be non-negative")
        if not np.allclose(array.sum(axis=-1), 1.0, atol=1e-6):
            raise ValueError("probs must sum to one along the channel axis")
        return array

    def _channels(self, probs: np.ndarray) -> int:
        channels = probs.shape[-1]
        if self.blank >= channels:
            raise ValueError("blank is outside the channel range")
        return channels

    def _targets(self, targets, batch: int, frames: int, channels: int,
                 lengths=None) -> list:
        if isinstance(targets, np.ndarray) and targets.ndim == 1:
            targets = [targets]
        if isinstance(targets, np.ndarray) and targets.ndim == 2:
            rows = [targets[index] for index in range(targets.shape[0])]
        elif isinstance(targets, (list, tuple)):
            rows = list(targets)
        else:
            raise ValueError("targets must be a sequence of label arrays")
        if lengths is None:
            lengths = [len(row) for row in rows]
        lengths = [int(length) for length in lengths]
        if len(rows) != batch or len(lengths) != batch:
            raise ValueError("one target per sequence is required")
        cleaned = []
        for row, length in zip(rows, lengths, strict=True):
            labels = np.asarray(row)
            if labels.ndim != 1:
                raise ValueError("each target must be one-dimensional")
            if length < 0 or length > labels.size:
                raise ValueError("target length is outside the array")
            labels = labels[:length]
            if not np.issubdtype(labels.dtype, np.integer):
                numeric = labels.astype(float)
                if not np.all(numeric == np.round(numeric)):
                    raise ValueError("labels must be integers")
                labels = numeric.astype(int)
            labels = labels.astype(int)
            if labels.size and (labels.min() < 0 or labels.max() >= channels):
                raise ValueError("a label is outside the channel range")
            if labels.size and np.any(labels == self.blank):
                raise ValueError("a label may not be the blank symbol")
            if labels.size > frames:
                raise ValueError("the target is longer than the frame sequence")
            cleaned.append(labels)
        return cleaned

    # ------------------------------------------------------------------ #
    # the extended lattice
    # ------------------------------------------------------------------ #
    def _extended(self, target: np.ndarray) -> np.ndarray:
        """Interleave blanks: blank, l1, blank, l2, ..., lS, blank."""
        extended = np.full(2 * target.size + 1, self.blank, dtype=int)
        extended[1::2] = target
        return extended

    def _skip(self, extended: np.ndarray) -> np.ndarray:
        """May state s be reached from s - 2, skipping the symbol between?"""
        allowed = np.zeros(extended.size, dtype=bool)
        index = np.arange(extended.size)
        later = index >= 2
        allowed[later] = ((extended[later] != self.blank)
                          & (extended[later] != extended[index[later] - 2]))
        return allowed

    # ------------------------------------------------------------------ #
    # forward-backward
    # ------------------------------------------------------------------ #
    def _alpha(self, log_probs: np.ndarray, extended: np.ndarray,
               skip: np.ndarray) -> tuple:
        """Prefix probabilities and the log likelihood of the target."""
        frames, width = log_probs.shape[0], extended.size
        alpha = np.full((frames, width), NEG)
        alpha[0, 0] = log_probs[0, extended[0]]
        if width > 1:
            alpha[0, 1] = log_probs[0, extended[1]]
        for t in range(1, frames):
            previous = alpha[t - 1]
            stay = previous
            step = np.concatenate(([NEG], previous[:-1]))
            jump = np.concatenate(([NEG, NEG], previous[:-2])) if width > 2 else (
                np.full(width, NEG))
            jump = np.where(skip, jump, NEG)
            alpha[t] = log_probs[t, extended] + _lse_rows(np.stack([stay, step, jump]))
        finals = alpha[-1, -1:] if width == 1 else alpha[-1, -2:]
        return alpha, _lse(finals)

    def _beta(self, log_probs: np.ndarray, extended: np.ndarray,
              skip: np.ndarray) -> np.ndarray:
        """Suffix probabilities, excluding the emission at the current frame."""
        frames, width = log_probs.shape[0], extended.size
        beta = np.full((frames, width), NEG)
        beta[-1, -1] = 0.0
        if width > 1:
            beta[-1, -2] = 0.0
        for t in range(frames - 2, -1, -1):
            following = beta[t + 1] + log_probs[t + 1, extended]
            stay = following
            step = np.concatenate((following[1:], [NEG]))
            jump = np.concatenate((following[2:], [NEG, NEG])) if width > 2 else (
                np.full(width, NEG))
            jumps = np.zeros(width, dtype=bool)
            if width > 2:
                jumps[:-2] = skip[2:]
            jump = np.where(jumps, jump, NEG)
            beta[t] = _lse_rows(np.stack([stay, step, jump]))
        return beta

    def _check_target(self, target: np.ndarray, frames: int) -> None:
        if target.size > frames:
            raise ValueError("the target is longer than the frame sequence")

    # ------------------------------------------------------------------ #
    # loss, posterior, gradient
    # ------------------------------------------------------------------ #
    def loss(self, probs, targets, lengths=None) -> float:
        """Mean negative log probability of the targets."""
        return self.loss_and_gradient(probs, targets, lengths)[0]

    def posterior(self, probs, target) -> np.ndarray:
        """Posterior over the extended states, shape ``(frames, 2S + 1)``."""
        batch = self._probs(probs)
        if batch.shape[0] != 1:
            raise ValueError("posterior takes one sequence at a time")
        target = self._targets(target, 1, batch.shape[1], batch.shape[2])[0]
        log_probs = np.log(batch[0])
        extended = self._extended(target)
        skip = self._skip(extended)
        alpha, log_p = self._alpha(log_probs, extended, skip)
        if not np.isfinite(log_p):
            raise ValueError("the target cannot be aligned with the frames")
        beta = self._beta(log_probs, extended, skip)
        self.n_states_ = extended.size
        return np.exp(alpha + beta - log_p)

    def loss_and_gradient(self, probs, targets, lengths=None) -> tuple:
        """Mean negative log likelihood and its gradient with respect to logits."""
        batch = self._probs(probs)
        sequences, frames, channels = batch.shape
        self._channels(batch)
        cleaned = self._targets(targets, sequences, frames, channels, lengths)
        log_probs = np.log(batch)
        gradient = np.zeros_like(batch)
        losses = []
        for index, target in enumerate(cleaned):
            extended = self._extended(target)
            skip = self._skip(extended)
            alpha, log_p = self._alpha(log_probs[index], extended, skip)
            if not np.isfinite(log_p):
                raise ValueError("a target cannot be aligned with the frames")
            beta = self._beta(log_probs[index], extended, skip)
            posterior = np.exp(alpha + beta - log_p)
            mass = np.zeros((frames, channels))
            for state in range(extended.size):
                mass[:, extended[state]] += posterior[:, state]
            gradient[index] = batch[index] - mass
            losses.append(-log_p)
        self.n_states_ = 2 * max((target.size for target in cleaned), default=0) + 1
        return float(np.mean(losses)), gradient / sequences

    # ------------------------------------------------------------------ #
    # decoding
    # ------------------------------------------------------------------ #
    def _collapse(self, path) -> np.ndarray:
        """Merge repeats, then delete the blank."""
        output = []
        previous = -1
        for symbol in path:
            if symbol != previous and symbol != self.blank:
                output.append(int(symbol))
            previous = int(symbol)
        return np.array(output, dtype=int)

    def greedy_decode(self, probs) -> list:
        """The most likely symbol per frame, collapsed."""
        batch = self._probs(probs)
        self._channels(batch)
        return [self._collapse(path) for path in batch.argmax(axis=-1)]

    def beam_decode(self, probs) -> list:
        """Beam search over extended frame paths, then collapse and merge."""
        batch = self._probs(probs)
        channels = self._channels(batch)
        log_probs = np.log(batch)
        outputs = []
        for sequence in log_probs:
            beams = {(): 0.0}
            for frame in sequence:
                candidates: dict = {}
                for path, score in beams.items():
                    for symbol in range(channels):
                        extension = path + (symbol,)
                        value = score + float(frame[symbol])
                        old = candidates.get(extension)
                        candidates[extension] = (
                            value if old is None else float(np.logaddexp(old, value)))
                ranked = sorted(candidates.items(), key=lambda item: -item[1])
                beams = dict(ranked[:self.beam])
            merged: dict = {}
            for path, score in beams.items():
                collapsed = tuple(self._collapse(path).tolist())
                old = merged.get(collapsed)
                merged[collapsed] = (
                    score if old is None else float(np.logaddexp(old, score)))
            best = max(merged.items(), key=lambda item: item[1])[0]
            outputs.append(np.array(best, dtype=int))
        return outputs

    def prefix_beam(self, probs) -> list:
        """Prefix beam search with separate blank and non-blank probabilities."""
        batch = self._probs(probs)
        channels = self._channels(batch)
        log_probs = np.log(batch)
        outputs = []
        for sequence in log_probs:
            beams = {(): (0.0, NEG)}
            for frame in sequence:
                updated: dict = {}
                for prefix, (p_blank, p_nonblank) in beams.items():
                    total = float(np.logaddexp(p_blank, p_nonblank))
                    last = prefix[-1] if prefix else -1
                    for symbol in range(channels):
                        score = float(frame[symbol])
                        if symbol == self.blank:
                            blank, nonblank = updated.get(prefix, (NEG, NEG))
                            updated[prefix] = (float(np.logaddexp(blank, total + score)),
                                              nonblank)
                        elif prefix and symbol == last:
                            blank, nonblank = updated.get(prefix, (NEG, NEG))
                            merged = float(np.logaddexp(nonblank, p_nonblank + score))
                            updated[prefix] = (blank, merged)
                            longer = prefix + (symbol,)
                            blank, nonblank = updated.get(longer, (NEG, NEG))
                            reopened = float(np.logaddexp(nonblank, p_blank + score))
                            updated[longer] = (blank, reopened)
                        else:
                            longer = prefix + (symbol,)
                            blank, nonblank = updated.get(longer, (NEG, NEG))
                            updated[longer] = (
                                blank, float(np.logaddexp(nonblank, total + score)))
                ranked = sorted(updated.items(),
                                key=lambda item: -float(np.logaddexp(*item[1])))
                beams = dict(ranked[:self.beam])
            best = max(beams.items(), key=lambda item: float(np.logaddexp(*item[1])))[0]
            outputs.append(np.array(best, dtype=int))
        return outputs

    def decode(self, probs) -> list:
        """Decode with the search that the variant names."""
        if self.variant in ("ctc_loss", "greedy_decode"):
            return self.greedy_decode(probs)
        if self.variant == "beam_decode":
            return self.beam_decode(probs)
        return self.prefix_beam(probs)

    # ------------------------------------------------------------------ #
    # alignments
    # ------------------------------------------------------------------ #
    def best_alignment(self, probs, target) -> np.ndarray:
        """The single most likely frame path that collapses to the target."""
        batch = self._probs(probs)
        if batch.shape[0] != 1:
            raise ValueError("best_alignment takes one sequence at a time")
        target = self._targets(target, 1, batch.shape[1], batch.shape[2])[0]
        log_probs = np.log(batch[0])
        extended = self._extended(target)
        skip = self._skip(extended)
        frames, width = log_probs.shape[0], extended.size
        delta = np.full((frames, width), NEG)
        back = np.zeros((frames, width), dtype=int)
        delta[0, 0] = log_probs[0, extended[0]]
        if width > 1:
            delta[0, 1] = log_probs[0, extended[1]]
        for t in range(1, frames):
            previous = delta[t - 1]
            stay = previous
            step = np.concatenate(([NEG], previous[:-1]))
            jump = np.concatenate(([NEG, NEG], previous[:-2])) if width > 2 else (
                np.full(width, NEG))
            jump = np.where(skip, jump, NEG)
            candidates = np.stack([stay, step, jump])
            back[t] = candidates.argmax(axis=0)
            delta[t] = candidates.max(axis=0) + log_probs[t, extended]
        finals = delta[-1, -1:] if width == 1 else delta[-1, -2:]
        state = width - 1 - int(finals.argmax())
        path = np.empty(frames, dtype=int)
        for t in range(frames - 1, -1, -1):
            path[t] = extended[state]
            if t:
                state -= int(back[t, state])
        return path

    def count_alignments(self, frames: int, target) -> int:
        """How many frame paths collapse to the target, exactly."""
        if not isinstance(frames, (int, np.integer)) or frames < 1:
            raise ValueError("frames must be a positive integer")
        target = np.asarray(target)
        if target.ndim != 1:
            raise ValueError("the target must be one-dimensional")
        if target.size > frames:
            return 0
        extended = self._extended(target.astype(int))
        skip = self._skip(extended)
        counts = [0] * extended.size
        counts[0] = 1
        if extended.size > 1:
            counts[1] = 1
        for _ in range(1, frames):
            following = [0] * extended.size
            for state in range(extended.size):
                total = counts[state]
                if state >= 1:
                    total += counts[state - 1]
                if state >= 2 and skip[state]:
                    total += counts[state - 2]
                following[state] = total
            counts = following
        return counts[-1] + (counts[-2] if extended.size > 1 else 0)


# ---------------------------------------------------------------------- #
# demo
# ---------------------------------------------------------------------- #
def _random_probs(seed: int, frames: int, channels: int, scale: float = 1.0) -> np.ndarray:
    """A batch of one sequence, softly peaked by the scale of its logits."""
    rng = np.random.default_rng(seed)
    logits = scale * rng.standard_normal((frames, channels))
    logits = logits - logits.max(axis=-1, keepdims=True)
    weights = np.exp(logits)
    return (weights / weights.sum(axis=-1, keepdims=True))[None]


def _all_outputs(channels: int, frames: int, blank: int) -> list:
    """Every label sequence short enough to be alignable, for exact decoding."""
    symbols = [symbol for symbol in range(channels) if symbol != blank]
    outputs = [np.zeros(0, dtype=int)]
    for length in range(1, frames + 1):
        for combination in itertools.product(symbols, repeat=length):
            outputs.append(np.array(combination, dtype=int))
    return outputs


def _exact_decode(model: CTC, probs: np.ndarray, outputs: list) -> tuple:
    """The output with the largest marginal probability, by enumeration."""
    best, best_log = None, NEG
    for output in outputs:
        try:
            log_p = -model.loss(probs, [output])
        except ValueError:
            continue
        if log_p > best_log:
            best, best_log = output, log_p
    return best, best_log


def _demo_lattice() -> None:
    """The loss, the posterior and the gradient, against brute force."""
    model = CTC()
    probs = _random_probs(0, 4, 3)
    target = np.array([1, 2])
    extension = model._extended(target)
    total = 0.0
    symbol_mass = np.zeros((4, 3))
    aligned = 0
    for path in itertools.product(range(3), repeat=4):
        weight = float(np.prod([probs[0, t, symbol] for t, symbol in enumerate(path)]))
        if np.array_equal(model._collapse(path), target):
            total += weight
            aligned += 1
            for t, symbol in enumerate(path):
                symbol_mass[t, symbol] += weight
    enumerated = -np.log(total)
    modelled = model.loss(probs, [target])
    counted = model.count_alignments(4, target)
    gamma = model.posterior(probs, target)
    per_symbol = np.zeros((4, 3))
    for state in range(extension.size):
        per_symbol[:, extension[state]] += gamma[:, state]
    best = model.best_alignment(probs, target)
    best_share = float(np.prod([probs[0, t, symbol] for t, symbol in enumerate(best)])) / total
    print("four frames, three channels, the target 1 2:")
    print(f"  {aligned} of {3 ** 4} frame paths collapse to it; the lattice count is {counted}")
    print(f"  loss {modelled:.6f} against the enumerated {enumerated:.6f}, "
          f"difference {abs(modelled - enumerated):.0e}")
    print(f"  posterior rows sum to {gamma.sum(axis=1).min():.1f} and match the "
          f"enumerated symbol marginals to {np.abs(per_symbol - symbol_mass / total).max():.1e}")
    print(f"  the best single alignment [{' '.join(map(str, best))}] carries "
          f"{best_share:.1%} of the target's probability")
    print("  so the label sequence is not any one alignment; it is the sum")

    logits = np.random.default_rng(1).standard_normal((2, 4, 3))
    targets = [np.array([1, 2]), np.array([2, 1, 1])]

    def objective(z: np.ndarray) -> float:
        shifted = np.exp(z - z.max(axis=-1, keepdims=True))
        return CTC().loss(shifted / shifted.sum(axis=-1, keepdims=True), targets)

    shifted = np.exp(logits - logits.max(axis=-1, keepdims=True))
    loss, gradient = CTC().loss_and_gradient(
        shifted / shifted.sum(axis=-1, keepdims=True), targets)
    numeric = np.zeros_like(gradient)
    for index in np.ndindex(logits.shape):
        plus, minus = logits.copy(), logits.copy()
        plus[index] += 1e-6
        minus[index] -= 1e-6
        numeric[index] = (objective(plus) - objective(minus)) / 2e-6
    error = np.abs(numeric - gradient).max() / np.abs(gradient).max()
    print(f"  the gradient of the mean loss {loss:.6f} matches central differences "
          f"to {error:.1e}")
    print(f"  every frame column sums to {np.abs(gradient.sum(axis=-1)).max():.1e}, "
          "as a softmax gradient must")
    assert abs(modelled - enumerated) < 1e-12
    assert counted == aligned
    assert np.abs(per_symbol - symbol_mass / total).max() < 1e-12
    assert error < 1e-8


def _demo_repeats() -> None:
    """The collapse rule, and what a repeated label costs in probability."""
    model = CTC()
    print("collapse merges repeats and deletes blanks:")
    for path in ([1, 1, 1], [1, 0, 1], [0, 1, 0], [0, 0, 0], [1, 2, 2, 0, 2]):
        collapsed = model._collapse(path)
        print(f"  {' '.join(map(str, path)):<12} -> "
              f"{' '.join(map(str, collapsed)) if collapsed.size else '(empty)'}")
    probs = np.full((1, 5, 2), 0.5)
    print("a two-symbol alphabet, uniform at every frame, five frames:")
    for labels in ([1], [1, 1], [1, 1, 1]):
        target = np.array(labels)
        count = model.count_alignments(5, target)
        probability = np.exp(-model.loss(probs, [target]))
        print(f"  alignments of {' '.join(map(str, labels)):<6} {count:>3}   "
              f"probability {probability:.4f} = {count}/32")
    print("  A path collapses to one run of the label per group of ones, so each")
    print("  repetition needs its own blank separator; three runs need the exact")
    print("  alternation 1 0 1 0 1 and there is only one way to write it here.")
    assert model.count_alignments(5, np.array([1])) == 15
    assert model.count_alignments(5, np.array([1, 1])) == 15
    assert model.count_alignments(5, np.array([1, 1, 1])) == 1


def _demo_decoders() -> None:
    """Four decoders against the exact marginal decoder on random problems."""
    frames, channels, problems = 6, 3, 24
    outputs = _all_outputs(channels, frames, 0)
    names = ("greedy", "beam, 8", "prefix, 8", "beam, 32", "prefix, 32")
    hits = dict.fromkeys(names, 0)
    example = None
    for seed in range(problems):
        probs = _random_probs(seed, frames, channels)
        truth, _ = _exact_decode(CTC(), probs, outputs)
        predictions = {
            "greedy": CTC(variant="greedy_decode").decode(probs)[0],
            "beam, 8": CTC(variant="beam_decode", beam=8).decode(probs)[0],
            "prefix, 8": CTC(variant="prefix_beam", beam=8).decode(probs)[0],
            "beam, 32": CTC(variant="beam_decode", beam=32).decode(probs)[0],
            "prefix, 32": CTC(variant="prefix_beam", beam=32).decode(probs)[0],
        }
        for name, answered in predictions.items():
            hits[name] += int(np.array_equal(answered, truth))
        if example is None and not np.array_equal(predictions["greedy"], truth):
            example = (seed, truth, predictions)
    print(f"{problems} random problems, {frames} frames, {channels} channels, "
          f"{len(outputs)} candidate outputs:")
    for name in names:
        print(f"  {name:<11} agrees with the exact marginal decode {hits[name]:>2}"
              f" of {problems}")
    seed, truth, predictions = example
    print(f"  problem {seed}: the exact answer is "
          f"[{' '.join(map(str, truth))}], greedy says "
          f"[{' '.join(map(str, predictions['greedy']))}], prefix beam says "
          f"[{' '.join(map(str, predictions['prefix, 8']))}]")
    print("  Greedy takes the best frame path, which is not the best label sequence;")
    print("  a prefix beam merges the alignments of a prefix before pruning, so the")
    print("  same width covers far more of the probability mass.")
    assert hits["prefix, 32"] == problems
    assert hits["beam, 32"] < hits["prefix, 32"]
    assert hits["prefix, 8"] > hits["greedy"]


def _demo_width() -> None:
    """How much of the mass a beam keeps as its width grows."""
    probs = _random_probs(13, 7, 3)
    outputs = _all_outputs(3, 7, 0)
    truth, truth_log = _exact_decode(CTC(), probs, outputs)
    print(f"one problem, the exact answer is [{' '.join(map(str, truth))}] "
          f"at log probability {truth_log:.4f}:")
    print("  width   beam outcome            kept    prefix outcome"
          "           kept")
    for width in (1, 2, 4, 8, 16, 32):
        beam = CTC(variant="beam_decode", beam=width).decode(probs)[0]
        prefix = CTC(variant="prefix_beam", beam=width).decode(probs)[0]
        beam_log = -CTC().loss(probs, [beam])
        prefix_log = -CTC().loss(probs, [prefix])
        beam_text = " ".join(map(str, beam)) or "empty"
        prefix_text = " ".join(map(str, prefix)) or "empty"
        print(f"  {width:>5}   {beam_text:<22}{np.exp(beam_log - truth_log):>7.3f}   "
              f"{prefix_text:<22}{np.exp(prefix_log - truth_log):>7.3f}")
    assert np.array_equal(CTC(variant="prefix_beam", beam=32).decode(probs)[0], truth)


def _demo_guards() -> None:
    """Malformed configurations, inputs and targets, all refused."""
    probs = _random_probs(5, 4, 3)
    attempts = [
        lambda: CTC(variant="nope"),
        lambda: CTC(blank=-1),
        lambda: CTC(beam=0),
        lambda: CTC(blank=3).loss(probs, [np.array([1])]),
        lambda: CTC().loss(probs[0, :, 0], [np.array([1])]),
        lambda: CTC().loss(probs * 2.0, [np.array([1])]),
        lambda: CTC().loss(probs - 1.0, [np.array([1])]),
        lambda: CTC().loss(np.full((1, 4, 3), np.nan), [np.array([1])]),
        lambda: CTC().loss(np.zeros((1, 0, 3)), [np.zeros(0, dtype=int)]),
        lambda: CTC().loss(probs, [np.array([1, 2, 1, 2, 1])]),
        lambda: CTC().loss(probs, [np.array([3])]),
        lambda: CTC().loss(probs, [np.array([-1])]),
        lambda: CTC().loss(probs, [np.array([0])]),
        lambda: CTC().loss(probs, [np.array([1.5])]),
        lambda: CTC().loss(probs, [np.array([[1, 2]])]),
        lambda: CTC().loss(np.concatenate([probs, probs]), [np.array([1])]),
        lambda: CTC().loss(probs, [np.array([1])], lengths=[2]),
        lambda: CTC().posterior(np.concatenate([probs, probs]), np.array([1])),
        lambda: CTC().count_alignments(0, np.array([1])),
        lambda: CTC().loss(probs, [np.array([1, 1, 1])]),
    ]
    refused = 0
    for attempt in attempts:
        try:
            attempt()
        except (ValueError, RuntimeError):
            refused += 1
    print(f"{refused} malformed calls refused: an unknown variant, a negative blank,")
    print("  a zero beam, a blank outside the channels, a one-dimensional input, a")
    print("  distribution that does not sum to one, a negative or non-finite entry,")
    print("  an empty frame axis, a target longer than the frames, a label outside")
    print("  the channels, at the blank, negative or fractional, a two-dimensional")
    print("  target, a batch mismatch, a length past the array, a batch passed to the")
    print("  one-sequence methods, zero frames, and a target that needs a")
    print("  separator blank the frames cannot provide.")
    assert refused == len(attempts)


def main() -> None:
    _demo_lattice()
    _demo_repeats()
    _demo_decoders()
    _demo_width()
    _demo_guards()
    print("ctc demo passed")


if __name__ == "__main__":
    main()
