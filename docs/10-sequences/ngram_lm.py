r"""N-gram language models — counting, smoothing and backoff.

Chapter:   10-sequences
Companion: ngram_lm.md
Tier 2 · 8 variants: unigram, bigram, trigram, laplace, add_k,
                     kneser_ney, stupid_backoff, interpolated

    p(w | h) = c(h w) / c(h)                                          (1)
    p_KN(w | h) = max(c(h w) - d, 0) / c(h) + lambda(h) p_KN(w | h')  (2)

An n-gram language model is the oldest sequence model that still works. It says
the next token depends on the previous n-1 tokens and nothing else, so the whole
model is a table of counts. Training is counting; scoring is a lookup; the
interesting part is what to do when the lookup misses.

That is the sparsity problem, and every smoothing variant is a different answer.
Maximum likelihood gives an unseen n-gram probability zero, which makes the
perplexity of any held-out text infinite. Add-one smoothing gives every n-gram
one phantom count. Add-k generalises the phantom count. Stupid backoff multiplies
by 0.4 and drops a token of context until something is found. Interpolation mixes
the orders with a fixed weight. Kneser-Ney is the one that earns its complexity:
it discounts every seen count by a fixed amount, hands the discounted mass to the
shorter context, and — the key idea — counts a lower-order n-gram by how many
*different* contexts it appears in rather than how often it appears.

Everything is numpy and deterministic. The corpus is synthetic, the counts are
exact integers, and every probability is checked against its closed form. The
numbers to carry away: on a held-out split the trigram maximum-likelihood model
gives an infinite perplexity while add-one gives 5.92, Kneser-Ney reaches 3.015
against interpolation's 3.221 among the normalised models, a word that follows
three different contexts gets more unigram mass than one that follows two even
though both appear three times, and generated tokens all come from the training
vocabulary.
"""  # noqa: E501

from __future__ import annotations

import numpy as np

VARIANTS = ("unigram", "bigram", "trigram", "laplace", "add_k",
            "kneser_ney", "stupid_backoff", "interpolated")

BOS = "<s>"
EOS = "</s>"
UNK = "<unk>"
_SPECIAL = (BOS, EOS, UNK)
_FIXED_ORDER = {"unigram": 1, "bigram": 2, "trigram": 3}


class NgramLM:
    """An n-gram model with a selectable order and smoothing scheme.

    Parameters
    ----------
    variant : str
        One of `VARIANTS`. The first three fix the order; the smoothing
        variants use `order`, which defaults to three.
    order : int or None
        Context length plus one. `None` means the variant's default.
    k : float
        Phantom count for `add_k`.
    discount : float
        Absolute discount for `kneser_ney`, in ``[0, 1)``.
    lam : float
        Interpolation weight on the longest context.
    seed : int
        Seed for `generate`.

    Attributes
    ----------
    counts_ : list of dict
        ``counts_[n - 1]`` maps an n-gram to its count.
    vocabulary_ : list of str
        Every token that can be predicted, sorted.
    """

    def __init__(self, *, variant: str = "trigram", order: int | None = None,
                 k: float = 1.0, discount: float = 0.75, lam: float = 0.5,
                 seed: int = 0) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
        if order is None:
            order = _FIXED_ORDER.get(variant, 3)
        if not isinstance(order, (int, np.integer)) or order < 1:
            raise ValueError("order must be a positive integer")
        if order > 6:
            raise ValueError("order above six is not supported")
        if variant in _FIXED_ORDER and _FIXED_ORDER[variant] != order:
            raise ValueError(f"{variant} has order {_FIXED_ORDER[variant]}")
        if not np.isfinite(k) or k <= 0.0:
            raise ValueError("k must be finite and positive")
        if not np.isfinite(discount) or not 0.0 <= discount < 1.0:
            raise ValueError("discount must lie in [0, 1)")
        if not np.isfinite(lam) or not 0.0 <= lam <= 1.0:
            raise ValueError("lam must lie in [0, 1]")
        if not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self.variant = variant
        self.order = int(order)
        self.k = float(k)
        self.discount = float(discount)
        self.lam = float(lam)
        self.seed = int(seed)

    # ------------------------------------------------------------------ #
    # training
    # ------------------------------------------------------------------ #
    def fit(self, corpus) -> NgramLM:
        """Count every n-gram in a list of token sequences."""
        sentences = self._sentences(corpus)
        words = {token for sentence in sentences for token in sentence}
        words |= {EOS, UNK}
        words.discard(BOS)
        self.vocabulary_ = sorted(words)
        self.index_ = {token: position
                       for position, token in enumerate(self.vocabulary_)}
        self.counts_ = [{} for _ in range(self.order)]
        self.totals_ = [{} for _ in range(self.order)]
        for sentence in sentences:
            padded = [BOS] * (self.order - 1) + list(sentence) + [EOS]
            for n in range(1, self.order + 1):
                table = self.counts_[n - 1]
                for start in range(len(padded) - n + 1):
                    gram = tuple(padded[start:start + n])
                    if gram[-1] == BOS:
                        continue
                    table[gram] = table.get(gram, 0) + 1
        for n in range(2, self.order + 1):
            for gram, count in self.counts_[n - 1].items():
                context = gram[:-1]
                previous = self.totals_[n - 1].get(context, 0)
                self.totals_[n - 1][context] = previous + count
        self.total_ = sum(self.counts_[0].values())
        self._build_continuations()
        return self

    @staticmethod
    def _sentences(corpus) -> list:
        if not isinstance(corpus, (list, tuple)) or not corpus:
            raise ValueError("corpus must be a non-empty list of token sequences")
        sentences = []
        for sentence in corpus:
            if isinstance(sentence, str):
                raise ValueError("a sentence must be tokens, not a string")
            tokens = list(sentence)
            if not tokens:
                continue
            for token in tokens:
                if not isinstance(token, str) or not token:
                    raise ValueError("tokens must be non-empty strings")
                if token in (BOS, EOS):
                    raise ValueError(f"{token!r} is reserved")
            sentences.append(tokens)
        if not sentences:
            raise ValueError("corpus has no non-empty sentence")
        return sentences

    def _build_continuations(self) -> None:
        """Continuation counts for every order below the longest."""
        self.continuation_ = [{} for _ in range(self.order)]
        self.continuation_totals_ = [{} for _ in range(self.order)]
        self.distinct_ = [{} for _ in range(self.order)]
        for m in range(1, self.order):
            table: dict = {}
            for gram in self.counts_[m]:
                shorter = gram[1:]
                table[shorter] = table.get(shorter, 0) + 1
            self.continuation_[m - 1] = table
        for order in range(1, self.order + 1):
            table = (self.counts_[order - 1] if order == self.order
                     else self.continuation_[order - 1])
            totals: dict = {}
            distinct: dict = {}
            for gram, value in table.items():
                context = gram[:-1]
                totals[context] = totals.get(context, 0) + value
                distinct[context] = distinct.get(context, 0) + 1
            self.continuation_totals_[order - 1] = totals
            self.distinct_[order - 1] = distinct

    def _check_fitted(self) -> None:
        if not hasattr(self, "counts_"):
            raise RuntimeError("call fit before scoring or generating")

    # ------------------------------------------------------------------ #
    # probabilities
    # ------------------------------------------------------------------ #
    def _count(self, gram: tuple) -> int:
        return self.counts_[len(gram) - 1].get(gram, 0)

    def _total(self, context: tuple) -> int:
        if not context:
            return self.total_
        return self.totals_[len(context)].get(context, 0)

    def _mle(self, history: tuple, word: str) -> float:
        """Maximum likelihood, backing off only when the context is unseen."""
        if not history:
            return self._count((word,)) / self.total_ if self.total_ else 0.0
        total = self._total(history)
        if total == 0:
            return self._mle(history[1:], word)
        return self._count(history + (word,)) / total

    def _laplace(self, history: tuple, word: str) -> float:
        vocabulary = len(self.vocabulary_)
        return ((self._count(history + (word,)) + 1.0)
                / (self._total(history) + vocabulary))

    def _add_k(self, history: tuple, word: str) -> float:
        vocabulary = len(self.vocabulary_)
        return ((self._count(history + (word,)) + self.k)
                / (self._total(history) + self.k * vocabulary))

    def _stupid_backoff(self, history: tuple, word: str) -> float:
        total = self._total(history)
        count = self._count(history + (word,))
        if total and count:
            return count / total
        if not history:
            return self._count((word,)) / self.total_ if self.total_ else 0.0
        return 0.4 * self._stupid_backoff(history[1:], word)

    def _interpolated(self, history: tuple, word: str) -> float:
        if not history:
            return self._mle((), word)
        total = self._total(history)
        lower = self._interpolated(history[1:], word)
        if not total:
            return lower
        longest = self._count(history + (word,)) / total
        return self.lam * longest + (1.0 - self.lam) * lower

    def _kneser_ney_level(self, history: tuple, word: str, order: int) -> float:
        """Interpolated absolute discounting with continuation counts below."""
        table = (self.counts_[order - 1] if order == self.order
                 else self.continuation_[order - 1])
        total = self.continuation_totals_[order - 1].get(history, 0)
        count = table.get(history + (word,), 0)
        distinct = self.distinct_[order - 1].get(history, 0)
        if order == 1:
            if not total:
                return 1.0 / len(self.vocabulary_)
            weight = self.discount * distinct / total
            held = weight / len(self.vocabulary_)
            return max(count - self.discount, 0.0) / total + held
        if not total:
            return self._kneser_ney_level(history[1:], word, order - 1)
        weight = self.discount * distinct / total
        return (max(count - self.discount, 0.0) / total
                + weight * self._kneser_ney_level(history[1:], word, order - 1))

    def probability(self, history, word: str) -> float:
        """Probability of one token after a history, by the chosen scheme."""
        self._check_fitted()
        if not isinstance(word, str):
            raise ValueError("word must be a string")
        history = tuple(history)
        if any(not isinstance(token, str) for token in history):
            raise ValueError("history tokens must be strings")
        if len(history) > self.order - 1:
            history = history[-(self.order - 1):] if self.order > 1 else ()
        word = word if word in self.index_ else UNK
        if self.variant in ("unigram", "bigram", "trigram"):
            return self._mle(history, word)
        if self.variant == "laplace":
            return self._laplace(history, word)
        if self.variant == "add_k":
            return self._add_k(history, word)
        if self.variant == "stupid_backoff":
            return self._stupid_backoff(history, word)
        if self.variant == "interpolated":
            return self._interpolated(history, word)
        return self._kneser_ney_level(history, word, self.order)

    def distribution(self, history=()) -> dict:
        """Every predictable token mapped to its probability."""
        self._check_fitted()
        return {word: self.probability(history, word) for word in self.vocabulary_}

    # ------------------------------------------------------------------ #
    # scoring and generation
    # ------------------------------------------------------------------ #
    def logprob(self, tokens) -> float:
        """Log probability of a token sequence, after the start padding."""
        self._check_fitted()
        if isinstance(tokens, str) or not hasattr(tokens, "__len__"):
            raise ValueError("tokens must be a sequence of strings")
        history = [BOS] * (self.order - 1)
        total = 0.0
        for word in tokens:
            probability = self.probability(history, word)
            if probability <= 0.0:
                return -np.inf
            total += float(np.log(probability))
            history = (history + [word if word in self.index_ else UNK])[
                -(self.order - 1):] if self.order > 1 else []
        return total

    def perplexity(self, corpus) -> float:
        """Geometric mean inverse probability per token, end tokens included."""
        sentences = self._sentences(corpus)
        total = 0.0
        tokens = 0
        for sentence in sentences:
            value = self.logprob(list(sentence) + [EOS])
            if not np.isfinite(value):
                return float("inf")
            total += value
            tokens += len(sentence) + 1
        return float(np.exp(-total / tokens))

    def generate(self, length: int, *, seed: int | None = None) -> list:
        """Sample a sentence from the model, stopping at the end token."""
        self._check_fitted()
        if not isinstance(length, (int, np.integer)) or length < 1:
            raise ValueError("length must be a positive integer")
        rng = np.random.default_rng(self.seed if seed is None else seed)
        history = [BOS] * (self.order - 1)
        output: list = []
        for _ in range(length):
            probabilities = np.array(
                [self.probability(history, word) for word in self.vocabulary_])
            probabilities = np.maximum(probabilities, 0.0)
            if UNK in self.index_:
                probabilities[self.index_[UNK]] = 0.0
            total = probabilities.sum()
            if not np.isfinite(total) or total <= 0.0:
                break
            index = int(rng.choice(probabilities.size, p=probabilities / total))
            word = self.vocabulary_[index]
            if word == EOS:
                break
            output.append(word)
            if self.order > 1:
                history = (history + [word])[-(self.order - 1):]
        return output

    def n_parameters(self) -> int:
        """Number of stored n-gram counts, the model's whole state."""
        self._check_fitted()
        return int(sum(len(table) for table in self.counts_))


# ---------------------------------------------------------------------- #
# demo
# ---------------------------------------------------------------------- #
_TOY = [["a", "b", "a"], ["a", "b", "c"]]

_TEXT = """
the cat sat on the mat
the dog sat on the rug
the cat ran on the mat
the dog ran on the rug
a cat sat on a mat
a dog ran on a rug
the cat saw the dog
the dog saw the cat
a cat saw a dog
a dog saw a cat
the cat sat on the rug
the dog ran on the mat
a cat ran on the rug
a dog sat on the mat
the cat saw a dog on the mat
the dog saw a cat on the rug
""".strip().splitlines()


def _split_text() -> tuple:
    """Twelve sentences for training, the last four held out."""
    corpus = [line.split() for line in _TEXT]
    return corpus[:12], corpus[12:]


def _demo_mle() -> None:
    """Maximum likelihood is counting, and the formulas are checkable."""
    corpus = _TOY
    counts = {}
    for sentence in corpus:
        padded = [BOS, BOS] + list(sentence) + [EOS]
        for n in (1, 2, 3):
            for start in range(len(padded) - n + 1):
                gram = tuple(padded[start:start + n])
                if gram[-1] != BOS:
                    counts[gram] = counts.get(gram, 0) + 1
    unigram = NgramLM(variant="unigram").fit(corpus)
    bigram = NgramLM(variant="bigram").fit(corpus)
    trigram = NgramLM(variant="trigram").fit(corpus)
    total = sum(value for gram, value in counts.items() if len(gram) == 1)
    print("two sentences, a b a and a b c:")
    print(f"  unigram  p(a) = {counts[('a',)]}/{total} = "
          f"{unigram.probability((), 'a'):.6f}")
    print(f"  bigram   p(b|a) = {counts[('a', 'b')]}/{counts[('a',)]} = "
          f"{bigram.probability(('a',), 'b'):.6f}")
    print(f"  bigram   p(a|b) = {counts[('b', 'a')]}/{counts[('b',)]} = "
          f"{bigram.probability(('b',), 'a'):.6f}")
    print(f"  trigram  p(a|<s>,<s>) = "
          f"{trigram.probability((BOS, BOS), 'a'):.6f}")
    print(f"  trigram  p(a|a,b) = {counts[('a', 'b', 'a')]}/"
          f"{counts[('a', 'b')]} = {trigram.probability(('a', 'b'), 'a'):.6f}")
    print("  an unseen continuation is probability zero, which is the whole")
    print("  problem: the trigram model has no mass for a new a b c, say")
    assert abs(unigram.probability((), "a") - 3 / 8) < 1e-12
    assert abs(bigram.probability(("a",), "b") - 2 / 3) < 1e-12
    assert abs(bigram.probability(("b",), "a") - 1 / 2) < 1e-12
    assert trigram.probability((BOS, BOS), "a") == 1.0
    assert abs(trigram.probability(("a", "b"), "a") - 0.5) < 1e-12
    assert trigram.probability(("b", "c"), "a") == 0.0


def _demo_smoothing() -> None:
    """Add-k and Laplace, against their closed forms and normalisation."""
    corpus = _TOY
    laplace = NgramLM(variant="laplace").fit(corpus)
    add_two = NgramLM(variant="add_k", k=2.0).fit(corpus)
    vocabulary = len(laplace.vocabulary_)
    print(f"the same corpus has {vocabulary} predictable tokens:")
    print(f"  laplace  p(b|a) = (2 + 1) / (3 + {vocabulary}) = "
          f"{laplace.probability(('a',), 'b'):.6f}")
    print(f"  add-k=2  p(b|a) = (2 + 2) / (3 + 2 x {vocabulary}) = "
          f"{add_two.probability(('a',), 'b'):.6f}")
    for model in (laplace, add_two):
        for history in ((), ("a",), ("a", "b"), ("z",)):
            assert abs(sum(model.distribution(history).values()) - 1.0) < 1e-12
    print("  both sum to one over the vocabulary for every context, including")
    print("  the contexts the corpus never contains")
    assert abs(laplace.probability(("a",), "b") - 3 / 8) < 1e-12
    assert abs(add_two.probability(("a",), "b") - 4 / 13) < 1e-12
    assert laplace.probability(("z",), "b") == 1 / vocabulary


def _demo_continuations() -> None:
    """Kneser-Ney counts contexts, not occurrences."""
    corpus = [["b", "a", "x"], ["b", "a", "a"], ["c", "b", "x"],
              ["d", "c", "x"]]
    model = NgramLM(variant="kneser_ney", order=2).fit(corpus)
    cc = model.continuation_[0]
    before_x = sorted({u for u, w in model.counts_[1] if w == "x"})
    before_a = sorted({u for u, w in model.counts_[1] if w == "a"})
    print("x follows three different words and a follows two, three times each:")
    print(f"  continuations of x: {cc[('x',)]} from {before_x}")
    print(f"  continuations of a: {cc[('a',)]} from {before_a}")
    print(f"  p(x) = {model.probability((), 'x'):.6f}   "
          f"p(a) = {model.probability((), 'a'):.6f}")
    print("  Equal occurrences, different numbers of contexts, and the word")
    print("  with more contexts gets more of the discounted mass.")
    assert cc[("x",)] == 3 and cc[("a",)] == 2
    assert model.probability((), "x") > model.probability((), "a")
    for history in ((), ("a",), ("b",), ("c",)):
        assert abs(sum(model.distribution(history).values()) - 1.0) < 1e-12


def _demo_backoff() -> None:
    """Stupid backoff and interpolation, at the same unseen bigram."""
    corpus = _TOY
    backoff = NgramLM(variant="stupid_backoff").fit(corpus)
    interp = NgramLM(variant="interpolated", lam=0.5).fit(corpus)
    plain = backoff.probability((), "a")
    print("an unseen continuation after c, where c appears once:")
    print(f"  stupid backoff  0.4 x p(a) = 0.4 x {plain:.6f} = "
          f"{backoff.probability(('c',), 'a'):.6f}")
    print(f"  interpolated    0.5 x 0 + 0.5 x p(a) = "
          f"{interp.probability(('c',), 'a'):.6f}")
    row = sum(backoff.distribution(("c",)).values())
    print(f"  that backoff row sums to {row:.3f}, not 1.000")
    print("  Backoff is a score, not a distribution: it can sum above one, and")
    print("  across contexts it does. Interpolation keeps the rows normalised.")
    assert abs(backoff.probability(("c",), "a") - 0.4 * plain) < 1e-12
    assert abs(interp.probability(("c",), "a") - 0.5 * plain) < 1e-12
    assert abs(sum(interp.distribution(("c",)).values()) - 1.0) < 1e-12
    assert abs(sum(backoff.distribution(("c",)).values()) - 1.0) > 1e-6


def _demo_perplexity() -> None:
    """Held-out perplexity, including the infinity maximum likelihood earns."""
    train, test = _split_text()
    print("twelve training sentences from a small grammar, four held out:")
    print("  variant            perplexity   stored n-grams")
    results = {}
    for variant in ("trigram", "laplace", "add_k", "kneser_ney",
                    "stupid_backoff", "interpolated"):
        model = NgramLM(variant=variant).fit(train)
        results[variant] = model.perplexity(test)
        shown = "infinite" if not np.isfinite(results[variant]) else (
            f"{results[variant]:.4f}")
        print(f"  {variant:<17} {shown:>10}   {model.n_parameters():>8}")
    model = NgramLM(variant="trigram").fit(train)
    value = model.logprob(["the", "cat", "sat"])
    print(f"  the trigram log probability of 'the cat sat' is {value:.4f}")
    print("  Add-one pays for its safety: it spreads mass over every token, so")
    print("  the common transitions get less than they deserve. Kneser-Ney is")
    print("  the best of the normalised models here; stupid backoff scores")
    print("  lower but its rows sum above one, so its perplexity is not a")
    print("  likelihood and the comparison is not fair.")
    assert not np.isfinite(results["trigram"])
    assert np.isfinite(results["laplace"])
    assert results["kneser_ney"] < results["laplace"]
    assert results["kneser_ney"] < results["interpolated"]
    raw = NgramLM(variant="trigram").fit(train)
    total = sum(raw.logprob(list(sentence) + [EOS]) for sentence in test)
    assert not np.isfinite(raw.perplexity(test))
    smooth = NgramLM(variant="interpolated").fit(train)
    total = sum(smooth.logprob(list(sentence) + [EOS]) for sentence in test)
    tokens = sum(len(sentence) + 1 for sentence in test)
    assert abs(smooth.perplexity(test) - np.exp(-total / tokens)) < 1e-12


def _demo_generation() -> None:
    """Sampling follows the training transitions and is seed-deterministic."""
    train, _ = _split_text()
    model = NgramLM(variant="kneser_ney").fit(train)
    first = model.generate(12, seed=4)
    second = model.generate(12, seed=4)
    other = model.generate(12, seed=5)
    print("twelve tokens sampled from the Kneser-Ney model:")
    print(f"  seed 4   {' '.join(first)}")
    print(f"  seed 5   {' '.join(other)}")
    print("  the same seed reproduces the sentence; the words come from the")
    print("  training vocabulary and the bigrams mostly from the corpus")
    assert first == second
    assert set(first) <= set(model.vocabulary_)
    assert len(other) <= 12


def _demo_guards() -> None:
    """Malformed models and inputs, all refused."""
    train, _ = _split_text()
    model = NgramLM(variant="bigram").fit(train)
    attempts = [
        lambda: NgramLM(variant="nope"),
        lambda: NgramLM(order=0),
        lambda: NgramLM(order=7),
        lambda: NgramLM(variant="unigram", order=2),
        lambda: NgramLM(k=0.0),
        lambda: NgramLM(discount=1.0),
        lambda: NgramLM(lam=1.5),
        lambda: NgramLM(seed=-1),
        lambda: NgramLM().fit([]),
        lambda: NgramLM().fit(["a b c"]),
        lambda: NgramLM().fit([[]]),
        lambda: NgramLM().fit([["a", 3]]),
        lambda: NgramLM().fit([["a", BOS]]),
        lambda: NgramLM().logprob(["a"]),
        lambda: NgramLM().perplexity([["a"]]),
        lambda: NgramLM().generate(3),
        lambda: model.logprob("a"),
        lambda: model.generate(0),
        lambda: model.probability((), 3),
        lambda: model.probability((4,), "a"),
    ]
    refused = 0
    for attempt in attempts:
        try:
            attempt()
        except (ValueError, RuntimeError):
            refused += 1
    print(f"{refused} malformed calls refused out of {len(attempts)}: an unknown")
    print("  variant, an order of zero or seven, an order that contradicts a")
    print("  fixed variant, a zero phantom count, a discount of one, a lam")
    print("  outside [0, 1], a negative seed, an empty or string corpus, an")
    print("  empty sentence list, a non-string token, a reserved token, and")
    print("  every scoring or sampling call made before fit.")
    assert refused == len(attempts)


def main() -> None:
    _demo_mle()
    _demo_smoothing()
    _demo_continuations()
    _demo_backoff()
    _demo_perplexity()
    _demo_generation()
    _demo_guards()
    print("ngram_lm demo passed")


if __name__ == "__main__":
    main()
