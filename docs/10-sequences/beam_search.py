r"""Beam search — choosing a sequence from a next-token distribution.

Chapter:   10-sequences
Companion: beam_search.md
Tier 2 · 7 variants: greedy, beam, top_k, top_p, temperature,
                    repetition_penalty, contrastive

    score(y) = sum_t log p(y_t | y_(1:t-1))                            (1)
    beam     = the B highest-scoring prefixes at every step            (2)

A language model gives a distribution over the next token given everything
before it. Turning that into a sentence is a search problem: the number of
sequences of length T is V^T, and the model only ever scores one step at a time.
This atom implements the standard answers.

Greedy decoding takes the most likely token at each step. Beam search keeps the
B most likely prefixes and extends all of them, which finds a better sequence at
the cost of B times the work. Sampling replaces the argmax with a draw, and the
three sampling variants shape the distribution first: temperature rescales the
logits, top-k keeps the k best tokens, and top-p keeps the smallest set whose
mass reaches p. Repetition penalty divides the scores of tokens already in the
context. Contrastive decoding subtracts the log-probabilities of a weaker
amateur model, which suppresses tokens that are likely for generic reasons rather
than because of the context.

Everything is numpy and deterministic. The model is a callable that maps a token
prefix to logits, so the atom is independent of any particular language model and
the demos can use exact tables. The numbers to carry away: on a two-step table
where greedy takes the locally better first token it ends at probability 0.3000
while a beam of two finds the 0.3600 sequence, a beam wide enough to keep every
prefix matches brute force over all 27 sequences exactly, temperature T makes the
distribution exactly softmax(z/T), top-p at 0.7 keeps the two tokens holding 0.75
of the mass, a repetition penalty of two sends a one-token loop on a four-token
pass before it locks, and contrastive decoding flips a choice the expert alone
would make.
"""  # noqa: E501

from __future__ import annotations

import numpy as np

VARIANTS = ("greedy", "beam", "top_k", "top_p", "temperature",
            "repetition_penalty", "contrastive")


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    """Log probabilities, with -inf entries allowed and masked rows refused."""
    peak = float(logits.max())
    if not np.isfinite(peak):
        raise ValueError("every token has zero probability")
    shifted = logits - peak
    return shifted - float(np.log(np.exp(shifted).sum()))


def _masked_top_k(logits: np.ndarray, k: int) -> np.ndarray:
    """Keep the k largest logits, set the rest to -inf."""
    if k >= logits.size:
        return logits
    order = np.argsort(-logits)
    output = np.full_like(logits, -np.inf)
    output[order[:k]] = logits[order[:k]]
    return output


def _masked_top_p(logits: np.ndarray, p: float) -> np.ndarray:
    """Keep the smallest set of tokens whose probability reaches p."""
    probabilities = np.exp(_log_softmax(logits))
    order = np.argsort(-probabilities)
    cumulative = np.cumsum(probabilities[order])
    keep = int(np.searchsorted(cumulative, p) + 1)
    keep = min(max(keep, 1), logits.size)
    output = np.full_like(logits, -np.inf)
    output[order[:keep]] = logits[order[:keep]]
    return output


class Decoder:
    """Decode sequences from a next-token model.

    Parameters
    ----------
    variant : str
        One of `VARIANTS`.
    beam : int
        Width of the beam, or one for greedy.
    k : int
        Number of tokens kept by `top_k`.
    p : float
        Mass kept by `top_p`, in ``(0, 1]``.
    temperature : float
        Divisor applied to the logits by `temperature`.
    penalty : float
        Divisor for repeated tokens, at least one.
    alpha : float
        Weight of the amateur model in `contrastive`.
    max_length : int
        Number of tokens generated after the prompt.
    seed : int
        Seed for the sampling variants.

    Attributes
    ----------
    logprob_ : float
        Log probability of the returned continuation under the model.
    """

    def __init__(self, *, variant: str = "beam", beam: int = 4, k: int = 5,
                 p: float = 0.9, temperature: float = 1.0, penalty: float = 1.2,
                 alpha: float = 0.5, max_length: int = 32, seed: int = 0) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
        for name, value, low in (("beam", beam, 1), ("k", k, 1),
                                 ("max_length", max_length, 1)):
            if not isinstance(value, (int, np.integer)) or value < low:
                raise ValueError(f"{name} must be an integer of at least {low}")
        if not np.isfinite(p) or not 0.0 < p <= 1.0:
            raise ValueError("p must lie in (0, 1]")
        if not np.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")
        if not np.isfinite(penalty) or penalty < 1.0:
            raise ValueError("penalty must be finite and at least one")
        if not np.isfinite(alpha) or alpha < 0.0:
            raise ValueError("alpha must be finite and non-negative")
        self.variant = variant
        self.beam = int(beam)
        self.k = int(k)
        self.p = float(p)
        self.temperature = float(temperature)
        self.penalty = float(penalty)
        self.alpha = float(alpha)
        self.max_length = int(max_length)
        self.seed = int(seed)
        self.logprob_ = 0.0

    # ------------------------------------------------------------------ #
    # input handling
    # ------------------------------------------------------------------ #
    @staticmethod
    def _prompt(prompt) -> tuple:
        array = np.asarray(prompt)
        if array.ndim == 0:
            array = array[None]
        if array.ndim != 1:
            raise ValueError("the prompt must be one-dimensional")
        if not np.issubdtype(array.dtype, np.integer):
            numeric = array.astype(float)
            if not np.all(numeric == np.round(numeric)):
                raise ValueError("prompt tokens must be integers")
            array = numeric.astype(int)
        if array.size and array.min() < 0:
            raise ValueError("prompt tokens must be non-negative")
        return tuple(int(token) for token in array)

    @staticmethod
    def _logits(scores, context: tuple) -> np.ndarray:
        if not callable(scores):
            raise ValueError("scores must be callable")
        values = np.asarray(scores(context), dtype=float)
        if values.ndim != 1:
            raise ValueError("scores must return one logit per token")
        if values.size < 2:
            raise ValueError("the vocabulary needs at least two tokens")
        if np.any(np.isnan(values)) or np.any(values == np.inf):
            raise ValueError("logits must be finite or negative infinity")
        return values

    # ------------------------------------------------------------------ #
    # shaping the distribution
    # ------------------------------------------------------------------ #
    def _penalise(self, logits: np.ndarray, context: tuple) -> np.ndarray:
        if self.variant != "repetition_penalty":
            return logits
        output = logits.copy()
        for token in set(context):
            if 0 <= token < output.size:
                output[token] = (output[token] / self.penalty if output[token] > 0.0
                                 else output[token] * self.penalty)
        return output

    def _distribution(self, logits: np.ndarray, context: tuple,
                      amateur=None) -> np.ndarray:
        """Log probabilities after the variant's shaping."""
        shaped = self._penalise(logits, context)
        if self.variant == "contrastive":
            if amateur is None:
                raise ValueError("contrastive decoding needs an amateur model")
            expert = _log_softmax(shaped)
            weak = _log_softmax(self._logits(amateur, context))
            shaped = (1.0 + self.alpha) * expert - self.alpha * weak
            return _log_softmax(shaped)
        if self.variant == "temperature":
            shaped = shaped / self.temperature
        elif self.variant == "top_k":
            shaped = _masked_top_k(shaped, self.k)
        elif self.variant == "top_p":
            shaped = _masked_top_p(shaped, self.p)
        return _log_softmax(shaped)

    # ------------------------------------------------------------------ #
    # searches
    # ------------------------------------------------------------------ #
    def _argmax(self, scores, prompt: tuple, eos, amateur) -> tuple:
        tokens = prompt
        logprob = 0.0
        for _ in range(self.max_length):
            logits = self._logits(scores, tokens)
            log_probs = self._distribution(logits, tokens, amateur)
            token = int(log_probs.argmax())
            logprob += float(log_probs[token])
            tokens = tokens + (token,)
            if eos is not None and token == eos:
                break
        return tokens, logprob

    def _beam(self, scores, prompt: tuple, eos, amateur) -> tuple:
        beams = [(prompt, 0.0)]
        finished: list = []
        for _ in range(self.max_length):
            candidates = []
            for tokens, logprob in beams:
                log_probs = self._distribution(self._logits(scores, tokens), tokens,
                                               amateur)
                for token in range(log_probs.size):
                    value = logprob + float(log_probs[token])
                    if not np.isfinite(value):
                        continue
                    if eos is not None and token == eos:
                        finished.append((tokens + (token,), value))
                    else:
                        candidates.append((tokens + (token,), value))
            if not candidates:
                break
            candidates.sort(key=lambda item: -item[1])
            beams = candidates[:self.beam]
            if len(finished) >= self.beam:
                best_finished = max(score for _, score in finished)
                if candidates[0][1] <= best_finished:
                    break
        pool = finished + beams
        best = max(pool, key=lambda item: item[1])
        return best[0], best[1]

    def _sample(self, scores, prompt: tuple, eos, amateur) -> tuple:
        rng = np.random.default_rng(self.seed)
        tokens = prompt
        logprob = 0.0
        for _ in range(self.max_length):
            logits = self._logits(scores, tokens)
            log_probs = self._distribution(logits, tokens, amateur)
            probabilities = np.exp(log_probs)
            probabilities = probabilities / probabilities.sum()
            token = int(rng.choice(probabilities.size, p=probabilities))
            logprob += float(log_probs[token])
            tokens = tokens + (token,)
            if eos is not None and token == eos:
                break
        return tokens, logprob

    def decode(self, scores, prompt=(), eos=None, amateur=None) -> np.ndarray:
        """Generate a continuation and return the whole sequence."""
        prompt = self._prompt(prompt)
        if eos is not None:
            if not isinstance(eos, (int, np.integer)) or eos < 0:
                raise ValueError("eos must be a non-negative integer")
            channels = self._logits(scores, prompt).size
            if eos >= channels:
                raise ValueError("eos is outside the vocabulary")
        if self.variant == "beam":
            tokens, logprob = self._beam(scores, prompt, eos, amateur)
        elif self.variant in ("top_k", "top_p", "temperature"):
            tokens, logprob = self._sample(scores, prompt, eos, amateur)
        else:
            tokens, logprob = self._argmax(scores, prompt, eos, amateur)
        self.logprob_ = logprob
        return np.array(tokens, dtype=int)

    def sequence_logprob(self, scores, sequence, prompt=()) -> float:
        """Log probability of a full sequence after the prompt, under the model."""
        prompt = self._prompt(prompt)
        sequence = self._prompt(sequence)
        if sequence[:len(prompt)] != prompt:
            raise ValueError("the sequence must start with the prompt")
        logprob = 0.0
        for index in range(len(prompt), len(sequence)):
            log_probs = _log_softmax(self._logits(scores, sequence[:index]))
            logprob += float(log_probs[sequence[index]])
        return logprob


# ---------------------------------------------------------------------- #
# demo
# ---------------------------------------------------------------------- #
def _table(rows: dict):
    """A model whose logits are a fixed table keyed by the token prefix."""

    def scores(context: tuple) -> np.ndarray:
        key = tuple(context)
        if key not in rows:
            raise KeyError(f"no row for {key}")
        return np.asarray(rows[key], dtype=float)

    return scores


def _two_step() -> dict:
    """The table where the locally best first token is not the best start."""
    log = np.log
    return {
        (): [log(0.6), log(0.4)],
        (0,): [log(0.5), log(0.5)],
        (1,): [log(0.1), log(0.9)],
    }


def _demo_greedy_and_beam() -> None:
    """A two-step table where greedy takes the wrong first token."""
    scores = _table(_two_step())
    greedy = Decoder(variant="greedy", max_length=2).decode(scores)
    beam = Decoder(variant="beam", beam=2, max_length=2).decode(scores)
    greedy_log = Decoder(variant="greedy", max_length=2)
    greedy_log.decode(scores)
    beam_log = Decoder(variant="beam", beam=2, max_length=2)
    beam_log.decode(scores)
    print("two steps, two tokens, p(0) = 0.6 and then a 50/50 split:")
    print(f"  greedy takes {greedy.tolist()} at log probability "
          f"{greedy_log.logprob_:.4f}, probability {np.exp(greedy_log.logprob_):.4f}")
    print(f"  a beam of two takes {beam.tolist()} at log probability "
          f"{beam_log.logprob_:.4f}, probability {np.exp(beam_log.logprob_):.4f}")
    print("  Greedy is exact for one step and wrong for the sequence: 0.6 x 0.5")
    print("  loses to 0.4 x 0.9, and the better first token is the rare one.")
    assert np.array_equal(beam, np.array([1, 1]))
    assert np.array_equal(greedy, np.array([0, 0]))
    assert beam_log.logprob_ > greedy_log.logprob_ + 0.15


def _demo_exhaustive() -> None:
    """Beam wide enough to keep everything must equal brute force."""
    rng = np.random.default_rng(10)
    rows = {}
    for length in range(4):
        for prefix in np.ndindex(*([3] * length)):
            rows[tuple(int(x) for x in prefix)] = (
                rng.standard_normal(3) * 1.5).tolist()
    scores = _table(rows)
    length = 3
    best_log, best = -np.inf, None
    for candidate in np.ndindex(*([3] * length)):
        sequence = np.array(candidate, dtype=int)
        value = Decoder(variant="greedy", max_length=length).sequence_logprob(
            scores, sequence)
        if value > best_log:
            best_log, best = value, sequence
    exact = Decoder(variant="beam", beam=3 ** length, max_length=length)
    found = exact.decode(scores)
    print(f"all {3 ** length} sequences of length {length} under a random table:")
    print(f"  brute force picks {best.tolist()} at {best_log:.6f}")
    print(f"  a beam of {3 ** length} picks {found.tolist()} at "
          f"{exact.logprob_:.6f}")
    narrow = Decoder(variant="beam", beam=1, max_length=length)
    narrow_tokens = narrow.decode(scores)
    print(f"  a beam of one picks {narrow_tokens.tolist()} at {narrow.logprob_:.6f}")
    assert np.array_equal(found, best)
    assert abs(exact.logprob_ - best_log) < 1e-12


def _demo_temperature() -> None:
    """Temperature T turns p into p^(1/T), renormalised."""
    logits = np.array([2.0, 1.0, 0.5, 0.0])
    base = np.exp(_log_softmax(logits))
    print("a four-token distribution and the effect of temperature:")
    for temperature in (0.5, 1.0, 2.0):
        decoder = Decoder(variant="temperature", temperature=temperature)
        shaped = np.exp(decoder._distribution(logits, ()))
        expected = np.exp(_log_softmax(logits / temperature))
        print(f"  T = {temperature:.1f}   {np.round(shaped, 4).tolist()}   "
              f"matches softmax(z/T) to {np.abs(shaped - expected).max():.1e}")
        assert np.abs(shaped - expected).max() < 1e-12
        assert np.abs(shaped.sum() - 1.0) < 1e-12
    cold = np.exp(Decoder(variant="temperature", temperature=0.5)._distribution(
        logits, ()))
    print(f"  T = 0.5 moves mass from {base[1]:.4f} to {cold[1]:.4f} on token 1")
    print("  low temperature sharpens toward the argmax; high temperature flattens")
    assert cold[0] > base[0] and cold[-1] < base[-1]


def _demo_top_k_and_p() -> None:
    """Truncation keeps a fixed count or a fixed mass."""
    logits = np.log(np.array([0.5, 0.25, 0.15, 0.1]))
    print("a distribution of 0.5, 0.25, 0.15, 0.1:")
    for k in (1, 2, 3):
        kept = np.isfinite(_masked_top_k(logits, k))
        print(f"  top-k with k = {k} keeps {int(kept.sum())} token(s), "
              f"mass {np.exp(_log_softmax(logits))[kept].sum():.2f}")
    for p in (0.5, 0.7, 0.9):
        kept = np.isfinite(_masked_top_p(logits, p))
        mass = np.exp(_log_softmax(logits))[kept].sum()
        print(f"  top-p with p = {p:.1f} keeps {int(kept.sum())} token(s), "
              f"mass {mass:.2f}")
    kept = np.isfinite(_masked_top_p(logits, 0.7))
    print("  top-p is the smallest set that reaches the mass, so 0.75 is enough")
    print("  for 0.7 and the answer is two tokens, not a fixed count")
    assert kept.sum() == 2
    assert np.isclose(np.exp(_log_softmax(logits))[kept].sum(), 0.75)
    assert np.isfinite(_masked_top_k(logits, 2)).sum() == 2


def _demo_repetition() -> None:
    """A penalty turns a one-token loop into a cycle through the vocabulary."""
    fixed = np.array([4.0, 3.0, 2.5, 2.2])
    scores = lambda context: fixed  # noqa: E731
    plain = Decoder(variant="greedy", max_length=8).decode(scores)
    penalised = Decoder(variant="repetition_penalty", penalty=2.0,
                        max_length=8).decode(scores)
    print("a model that always prefers token 0, eight steps:")
    print(f"  no penalty        {plain.tolist()}")
    print(f"  penalty of two    {penalised.tolist()}")
    print("  Dividing a used token's logit by two drops it below the next best")
    print("  unused one, so the loop visits four tokens before it locks on. The")
    print("  penalty reads the set of tokens in the context, not their recency,")
    print("  so once all four are used the ranking is fixed and it repeats.")
    assert np.array_equal(plain, np.zeros(8, dtype=int))
    assert np.array_equal(penalised, np.array([0, 1, 2, 3, 0, 0, 0, 0]))


def _demo_contrastive() -> None:
    """Subtracting an amateur model suppresses generic tokens."""
    expert = np.array([1.0, 0.9, 0.9])
    amateur = np.array([3.0, 0.0, 0.0])
    plain = Decoder(variant="greedy", max_length=1)
    plain_tokens = plain.decode(lambda context: expert)
    contrast = Decoder(variant="contrastive", alpha=0.5, max_length=1)
    contrast_tokens = contrast.decode(lambda context: expert,
                                      amateur=lambda context: amateur)
    print("an expert that mildly prefers token 0, an amateur that loves it:")
    print(f"  the expert alone picks token {int(plain_tokens[0])}")
    print(f"  contrastive decoding picks token {int(contrast_tokens[0])}")
    print(f"  at alpha 0.5 the expert's margin of "
          f"{expert[0] - expert[1]:.2f} is outweighed by the amateur's "
          f"{amateur[0] - amateur[1]:.2f}")
    assert plain_tokens[0] == 0
    assert contrast_tokens[0] == 1
    neutral = Decoder(variant="contrastive", alpha=0.0, max_length=1)
    assert neutral.decode(lambda context: expert,
                          amateur=lambda context: amateur)[0] == 0


def _demo_beam_eos() -> None:
    """Ending early beats a locally better continuation."""
    log = np.log
    rows = {
        (): [log(0.6), log(0.4)],
        (0,): [log(0.5), log(0.5)],
    }
    scores = _table(rows)
    greedy = Decoder(variant="greedy", max_length=2)
    greedy_tokens = greedy.decode(scores, eos=1)
    print("two tokens, the second is the end token, two steps at most:")
    print(f"  greedy never emits it: {greedy_tokens.tolist()} at "
          f"{greedy.logprob_:.4f}")
    for width in (2, 4):
        decoder = Decoder(variant="beam", beam=width, max_length=2)
        tokens = decoder.decode(scores, eos=1)
        print(f"  beam of {width} stops:      {tokens.tolist()} at "
              f"{decoder.logprob_:.4f}")
    print("  Ending at once pays 0.4; the 0.6 start leads to a 0.5 split, so")
    print("  both continuations are worth 0.3 and the short answer wins.")
    print("  Finished sequences leave the beam but stay in the comparison,")
    print("  and the search stops once the best active prefix cannot beat them.")
    assert np.array_equal(greedy_tokens, np.array([0, 0]))
    decoder = Decoder(variant="beam", beam=2, max_length=2)
    assert np.array_equal(decoder.decode(scores, eos=1), np.array([1]))


def _demo_guards() -> None:
    """Malformed decoders, prompts and tables, all refused."""
    scores = _table(_two_step())
    attempts = [
        lambda: Decoder(variant="nope"),
        lambda: Decoder(beam=0),
        lambda: Decoder(k=0),
        lambda: Decoder(p=0.0),
        lambda: Decoder(p=1.5),
        lambda: Decoder(temperature=0.0),
        lambda: Decoder(temperature=np.inf),
        lambda: Decoder(penalty=0.5),
        lambda: Decoder(alpha=-1.0),
        lambda: Decoder(max_length=0),
        lambda: Decoder().decode(scores, prompt=[0.5]),
        lambda: Decoder().decode(scores, prompt=[[0]]),
        lambda: Decoder().decode(scores, prompt=[-1]),
        lambda: Decoder(max_length=1).decode(scores, eos=2),
        lambda: Decoder(max_length=1).decode(scores, eos=-1),
        lambda: Decoder(max_length=1).decode("not callable"),
        lambda: Decoder(max_length=1).decode(lambda context: np.zeros((2, 2))),
        lambda: Decoder(max_length=1).decode(lambda context: np.array([0.0])),
        lambda: Decoder(max_length=1).decode(lambda context: np.array([0.0, np.nan])),
        lambda: Decoder(max_length=1).decode(lambda context: np.array([0.0, np.inf])),
        lambda: Decoder(variant="contrastive").decode(scores),
        lambda: Decoder(max_length=1).sequence_logprob(scores, [1, 1], prompt=[0]),
        lambda: Decoder(max_length=2).decode(_table({(): [0.0, 0.0]})),
    ]
    refused = 0
    for attempt in attempts:
        try:
            attempt()
        except (ValueError, KeyError):
            refused += 1
    print(f"{refused} malformed calls refused out of {len(attempts)}: an unknown")
    print("  variant, a zero width, a probability outside (0, 1], a zero or")
    print("  infinite temperature, a penalty below one, a negative alpha, a")
    print("  fractional or negative or nested prompt, an end token outside the")
    print("  vocabulary, a non-callable or ill-shaped table, a non-finite")
    print("  logit, a contrastive decoder with no amateur, a sequence that does")
    print("  not start with the prompt, and a table with no row for a prefix.")
    assert refused == len(attempts)


def main() -> None:
    _demo_greedy_and_beam()
    _demo_exhaustive()
    _demo_temperature()
    _demo_top_k_and_p()
    _demo_repetition()
    _demo_contrastive()
    _demo_beam_eos()
    _demo_guards()
    print("beam_search demo passed")


if __name__ == "__main__":
    main()
