r"""State-space models — convolution, scan and selection.

Chapter:   10-sequences
Companion: ssm.md
Tier 2 · 7 variants: s4, s4d, hippo, mamba_selective, s5, diagonal_ssm,
                     parallel_scan

    h_t = A h_(t-1) + B x_t                                        (1)
    y_t = C h_t                                                    (2)
    K_j = C A^j B                                                  (3)

A state-space model is a linear recurrence with a hidden state. The state is a
vector, the input and output are scalars, and the whole sequence is produced by
the two lines above. Unrolled, the output is a convolution of the input with the
impulse response K, so an LTI state-space layer can be applied either as a
recurrence or as a convolution, and the two are exactly equal.

The variants are different choices of A, B and C, and different ways of applying
them. `hippo` builds the HiPPO-LegS matrix, whose state is a compressed history
of the input with polynomial memory. `s4` diagonalises that matrix into a complex
diagonal plus a change of basis, which turns the matrix power into a sum of
exponentials. `s4d` skips the matrix and starts from a diagonal complex A with a
frequency grid. `diagonal_ssm` uses a real decaying diagonal, the simplest model
with a closed-form kernel. `s5` applies the diagonal model with a parallel scan
instead of a convolution. `parallel_scan` is the scan itself, demonstrated on an
arbitrary affine recurrence. `mamba_selective` makes the step size, B and C
depend on the input, so the recurrence is no longer a convolution and the model
can choose what to remember.

Everything is numpy. The parallel scan is a Hillis-Steele scan over the
associative pair (a, b) with composition (a1 a2, a2 b1 + b2), so it uses
O(log L) rounds and vectorised numpy rather than a Python loop over time. The
demo checks it against the naive recurrence element by element, checks the
convolution against the recurrence, checks that the diagonalised S4 reproduces
the HiPPO kernel, and measures the memory of each variant by fitting a linear
readout at increasing lags. The numbers to carry away: the scan matches the loop
to 0e+00, the S4 kernel matches the HiPPO kernel to 1e-12, the HiPPO kernel is
still 1e-2 at lag 40 while a decaying diagonal is at 1e-12, and the selective
model gives a small input a larger trace than a large one at late times because
the large input also opens the decay gate.
"""  # noqa: E501

from __future__ import annotations

import numpy as np

VARIANTS = ("s4", "s4d", "hippo", "mamba_selective", "s5", "diagonal_ssm",
            "parallel_scan")

_SCAN = ("s5", "parallel_scan")


def _hippo_matrix(state: int) -> tuple:
    """HiPPO-LegS: lower triangular, with B_n = sqrt(2n + 1)."""
    matrix = np.zeros((state, state))
    for row in range(state):
        for column in range(state):
            if row > column:
                matrix[row, column] = -np.sqrt((2 * row + 1) * (2 * column + 1))
            elif row == column:
                matrix[row, column] = -(row + 1)
    vector = np.sqrt(2.0 * np.arange(state) + 1.0)
    return matrix, vector


def _parallel_scan(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Inclusive scan of h_t = a_t h_(t-1) + b_t, with h_(-1) = 0.

    ``a`` and ``b`` have shape (steps, state); the scan is elementwise in state.
    """
    a = np.array(a, copy=True)
    b = np.array(b, dtype=np.result_type(a.dtype, b.dtype), copy=True)
    steps = b.shape[0]
    offset = 1
    while offset < steps:
        b[offset:] = a[offset:] * b[:-offset] + b[offset:]
        a[offset:] = a[offset:] * a[:-offset]
        offset *= 2
    return b


def _softplus(values: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, values)


class SSM:
    """A scalar-input, scalar-output state-space layer.

    Parameters
    ----------
    variant : str
        One of `VARIANTS`.
    state : int
        Number of hidden channels.
    length : int
        Default sequence length for `kernel`.
    dt : float
        Spacing used by the diagonal initialisations.
    seed : int
        Seed for the random readouts and the selective model's weights.

    Attributes
    ----------
    a_ : ndarray or None
        Diagonal transition, complex for the diagonalised variants.
    A_ : ndarray or None
        Dense transition, for `hippo`.
    B_, C_ : ndarray
        Input and output vectors in the working basis.
    """

    def __init__(self, *, variant: str = "s4d", state: int = 8,
                 length: int = 32, dt: float = 0.05, seed: int = 0) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
        if not isinstance(state, (int, np.integer)) or state < 1 or state > 64:
            raise ValueError("state must be an integer between 1 and 64")
        if not isinstance(length, (int, np.integer)) or length < 1:
            raise ValueError("length must be a positive integer")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self.variant = variant
        self.state = int(state)
        self.length = int(length)
        self.dt = float(dt)
        self.seed = int(seed)
        self.a_ = None
        self.A_ = None
        self.B_ = np.zeros(self.state)
        self.C_ = np.zeros(self.state)
        self._build()

    # ------------------------------------------------------------------ #
    # parameters
    # ------------------------------------------------------------------ #
    def _build(self) -> None:
        rng = np.random.default_rng(self.seed)
        state = self.state
        if self.variant == "mamba_selective":
            self.log_rates_ = np.log(np.arange(1, state + 1, dtype=float))
            self.w_delta_ = 1.0
            self.b_delta_ = -2.0
            self.w_b_ = np.abs(rng.standard_normal(state)) + 0.5
            self.w_c_ = np.abs(rng.standard_normal(state)) + 0.5
            self.d_skip_ = 0.0
            return
        if self.variant in ("hippo", "s4"):
            matrix, vector = _hippo_matrix(state)
            eigenvalues, vectors = np.linalg.eig(matrix)
            inverse = np.linalg.inv(vectors)
            discrete = np.real(vectors @ np.diag(np.exp(eigenvalues * self.dt))
                               @ inverse)
            self.B_ = np.linalg.solve(matrix, (discrete - np.eye(state)) @ vector)
            self.C_ = vector.copy()
            if self.variant == "hippo":
                self.A_ = discrete
                return
            values, basis = np.linalg.eig(discrete)
            change = np.linalg.inv(basis)
            self.a_ = values
            self.B_ = change @ self.B_
            self.C_ = self.C_ @ basis
            return
        if self.variant == "s4d":
            self.a_ = np.exp(self.dt * (-0.5 + 1j * np.pi * np.arange(state)))
            self.B_ = np.ones(state)
            self.C_ = np.ones(state)
            return
        if self.variant == "diagonal_ssm":
            rates = 1.0 + np.arange(state)
            self.a_ = np.exp(-self.dt * rates)
            self.B_ = np.ones(state)
            self.C_ = np.ones(state)
            return
        # s5 and the bare parallel scan share a diagonal complex model.
        self.a_ = np.exp(self.dt * (-0.5 + 1j * np.pi * np.arange(state)))
        self.B_ = np.abs(rng.standard_normal(state)) + 0.5
        self.C_ = np.abs(rng.standard_normal(state)) + 0.5

    def n_parameters(self) -> int:
        """Trainable real scalars, counting a complex entry twice."""
        if self.variant == "mamba_selective":
            return int(3 * self.state + 3)
        if self.variant == "hippo":
            return int(self.A_.size + 2 * self.state)
        width = 2 if np.iscomplexobj(self.a_) else 1
        return int(width * self.state + 2 * self.state)

    # ------------------------------------------------------------------ #
    # kernels and features
    # ------------------------------------------------------------------ #
    def kernel(self, length: int | None = None) -> np.ndarray:
        """The impulse response K_j = C A^j B for j = 0, ..., length - 1."""
        if self.variant == "mamba_selective":
            raise ValueError("a selective model has no input-independent kernel")
        steps = self.length if length is None else length
        if not isinstance(steps, (int, np.integer)) or steps < 1:
            raise ValueError("length must be a positive integer")
        output = np.zeros(steps, dtype=float)
        if self.variant == "hippo":
            power = self.B_.astype(float)
            output[0] = float(self.C_ @ power)
            for index in range(1, steps):
                power = self.A_ @ power
                output[index] = float(self.C_ @ power)
            return output
        term = np.asarray(self.C_ * self.B_)
        output[0] = float(np.real(term.sum()))
        for index in range(1, steps):
            term = term * self.a_
            output[index] = float(np.real(term.sum()))
        return output

    def _features_diagonal(self, x: np.ndarray) -> np.ndarray:
        steps = x.size
        b = x[:, None] * self.B_[None, :]
        if self.variant in _SCAN:
            a = np.broadcast_to(self.a_, (steps, self.state))
            return _parallel_scan(a, b)
        state = np.zeros(self.state, dtype=complex)
        output = np.empty((steps, self.state), dtype=complex)
        for index in range(steps):
            state = self.a_ * state + self.B_ * x[index]
            output[index] = state
        return output

    def features(self, x) -> np.ndarray:
        """The hidden state at every step, shape ``(steps, state)``."""
        array = self._check_input(x)
        if self.variant == "mamba_selective":
            return self._features_selective(array)
        if self.variant == "hippo":
            state = np.zeros(self.state)
            output = np.empty((array.size, self.state))
            for index in range(array.size):
                state = self.A_ @ state + self.B_ * array[index]
                output[index] = state
            return output
        return self._features_diagonal(array)

    def _features_selective(self, x: np.ndarray) -> np.ndarray:
        """The selective recurrence, with input-dependent step and readouts."""
        steps = x.size
        rates = np.exp(self.log_rates_)
        state = np.zeros(self.state)
        output = np.empty((steps, self.state))
        for index in range(steps):
            delta = float(_softplus(self.w_delta_ * x[index] + self.b_delta_))
            decay = np.exp(-delta * rates)
            write = delta * (self.w_b_ * x[index]) * x[index]
            state = decay * state + write
            output[index] = state
        return output

    def _check_input(self, x) -> np.ndarray:
        array = np.asarray(x, dtype=float)
        if array.ndim != 1:
            raise ValueError("x must be one-dimensional")
        if array.size < 1:
            raise ValueError("x must have at least one step")
        if not np.all(np.isfinite(array)):
            raise ValueError("x must be finite")
        return array

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def _forward_selective(self, x: np.ndarray) -> np.ndarray:
        hidden = self._features_selective(x)
        readout = hidden * (x[:, None] * self.w_c_[None, :])
        return readout.sum(axis=1) + self.d_skip_ * x

    def forward(self, x) -> np.ndarray:
        """Run the layer over a one-dimensional input."""
        array = self._check_input(x)
        if self.variant == "mamba_selective":
            return self._forward_selective(array)
        if self.variant in ("hippo", "s4"):
            return np.convolve(array, self.kernel(array.size))[:array.size]
        if self.variant in _SCAN:
            hidden = self._features_diagonal(array)
            return np.real(hidden @ self.C_)
        impulse = self.kernel(array.size)
        return np.convolve(array, impulse)[:array.size]

    def impulse(self, length: int | None = None) -> np.ndarray:
        """Run the layer on a unit impulse, recovering the kernel."""
        steps = self.length if length is None else length
        if not isinstance(steps, (int, np.integer)) or steps < 1:
            raise ValueError("length must be a positive integer")
        x = np.zeros(steps)
        x[0] = 1.0
        return self.forward(x)


# ---------------------------------------------------------------------- #
# demo
# ---------------------------------------------------------------------- #
def _recurrence(model: SSM, x: np.ndarray) -> np.ndarray:
    """The definition in (1) and (2), one step at a time."""
    state = np.zeros(model.state, dtype=complex if model.a_ is not None else float)
    output = np.empty(x.size)
    for index in range(x.size):
        if model.variant == "hippo":
            state = model.A_ @ state + model.B_ * x[index]
        else:
            state = model.a_ * state + model.B_ * x[index]
        output[index] = float(np.real(model.C_ @ state))
    return output


def _readout_score(features: np.ndarray, target: np.ndarray,
                   train: slice, test: slice) -> float:
    """R^2 of a linear readout fitted on the training slice."""
    design = np.concatenate([np.real(features), np.ones((features.shape[0], 1))],
                            axis=1)
    weights, *_ = np.linalg.lstsq(design[train], target[train], rcond=None)
    prediction = design[test] @ weights
    residual = float(((target[test] - prediction) ** 2).sum())
    total = float(((target[test] - target[test].mean()) ** 2).sum())
    return 1.0 - residual / total


def _demo_scan() -> None:
    """The associative scan against the definition of the recurrence."""
    rng = np.random.default_rng(0)
    steps, state = 37, 4
    a = rng.uniform(0.4, 1.0, (steps, state))
    b = rng.standard_normal((steps, state))
    scanned = _parallel_scan(a, b)
    naive = np.zeros(state)
    for index in range(steps):
        naive = a[index] * naive + b[index]
    complex_a = a * np.exp(1j * rng.uniform(-1.0, 1.0, (steps, state)))
    scanned_c = _parallel_scan(complex_a, b.astype(complex))
    naive_c = np.zeros(state, dtype=complex)
    for index in range(steps):
        naive_c = complex_a[index] * naive_c + b[index]
    rounds = int(np.ceil(np.log2(steps)))
    print(f"a recurrence of {steps} steps, scanned in {rounds} doublings:")
    real_error = np.abs(scanned[-1] - naive).max()
    complex_error = np.abs(scanned_c[-1] - naive_c).max()
    print(f"  real     difference from the loop {real_error:.1e}")
    print(f"  complex  difference from the loop {complex_error:.1e}")
    print("  the combine is associative, so the tree may pair the steps in any")
    print("  order; a serial dependency becomes a logarithmic number of rounds")
    assert np.abs(scanned - np.array([_naive_prefix(a, b, index)
                                      for index in range(steps)])).max() < 1e-12
    assert np.abs(scanned_c[-1] - naive_c).max() < 1e-12


def _naive_prefix(a: np.ndarray, b: np.ndarray, stop: int) -> np.ndarray:
    state = np.zeros(a.shape[1], dtype=np.result_type(a, b))
    for index in range(stop + 1):
        state = a[index] * state + b[index]
    return state


def _demo_convolution() -> None:
    """Kernel, convolution and recurrence must agree for an LTI model."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal(32)
    print("a 32-step input through the three equivalent forms:")
    for variant in ("diagonal_ssm", "hippo", "s4d", "s4"):
        model = SSM(variant=variant, state=6, length=32)
        convolved = model.forward(x)
        recurrent = _recurrence(model, x)
        print(f"  {variant:<13} convolution vs recurrence "
              f"{np.abs(convolved - recurrent).max():.1e}")
        assert np.abs(convolved - recurrent).max() < 1e-10
    diagonal = SSM(variant="diagonal_ssm", state=6, length=32)
    kernel = diagonal.kernel(32)
    closed = np.array([np.real((diagonal.C_ * diagonal.a_ ** lag
                                * diagonal.B_).sum()) for lag in range(32)])
    print(f"  the diagonal kernel matches C a^j B to "
          f"{np.abs(kernel - closed).max():.1e}")
    assert np.abs(kernel - closed).max() < 1e-12


def _demo_hippo() -> None:
    """The HiPPO-LegS matrix, and what a 1/t tail looks like."""
    state = 8
    matrix, vector = _hippo_matrix(state)
    eigenvalues = np.sort(np.linalg.eigvals(matrix).real)
    print(f"the HiPPO-LegS matrix of size {state}:")
    print(f"  lower triangular: {np.allclose(matrix, np.tril(matrix))}")
    print(f"  diagonal         {np.diag(matrix)}")
    print(f"  B first entries  {np.round(vector[:4], 4)}")
    print(f"  eigenvalues      {np.round(eigenvalues, 4)}")
    print("  the diagonal is -(n+1), so the continuous system is stable and")
    print("  the state is a coefficient vector for a Legendre expansion")
    model = SSM(variant="hippo", state=state, length=256)
    plain = SSM(variant="diagonal_ssm", state=state, length=256)
    hippo_kernel = model.kernel(256)
    plain_kernel = plain.kernel(256)
    print("  kernel magnitude at increasing lags:")
    print("    lag      hippo      diagonal")
    for lag in (5, 20, 60, 120, 240):
        print(f"    {lag:>3}   {abs(hippo_kernel[lag]):.2e}   "
              f"{abs(plain_kernel[lag]):.2e}")
    assert np.allclose(matrix, np.tril(matrix))
    assert np.allclose(np.diag(matrix), -np.arange(1, state + 1))
    assert np.allclose(vector, np.sqrt(2 * np.arange(state) + 1))


def _demo_s4() -> None:
    """Diagonalising the HiPPO matrix does not change the model."""
    hippo = SSM(variant="hippo", state=8, length=64)
    s4 = SSM(variant="s4", state=8, length=64)
    difference = np.abs(hippo.kernel(64) - s4.kernel(64)).max()
    rng = np.random.default_rng(2)
    x = rng.standard_normal(64)
    print("S4 is HiPPO in an eigenbasis:")
    print(f"  the two kernels differ by {difference:.1e}")
    print(f"  the two outputs differ by "
          f"{np.abs(hippo.forward(x) - s4.forward(x)).max():.1e}")
    print(f"  the eigenvalues have modulus at most "
          f"{np.abs(s4.a_).max():.6f}, so the discrete model is stable")
    print("  A = V diag(a) V^-1 turns every matrix power into a sum of powers,")
    print("  which is the whole reason S4 is fast and S4D is simpler still")
    assert difference < 1e-9
    assert np.all(np.abs(s4.a_) < 1.0)


def _demo_memory() -> None:
    """A linear readout cannot tell the real LTI variants apart."""
    rng = np.random.default_rng(3)
    length = 600
    x = rng.standard_normal(length)
    train, test = slice(100, 400), slice(400, 600)
    models = {name: SSM(variant=name, state=8, length=length)
              for name in ("hippo", "s4", "diagonal_ssm")}
    scores = {name: {} for name in models}
    print("decode the input at a lag from the state, R^2 on held-out steps:")
    print("  lag      hippo        s4   diagonal")
    for lag in (1, 4, 16):
        target = np.zeros(length)
        target[lag:] = x[:-lag]
        values = [(_readout_score(model.features(x), target, train, test))
                  for model in models.values()]
        for name, value in zip(models, values, strict=True):
            scores[name][lag] = value
        print(f"  {lag:>3}   {values[0]:.9f}  {values[1]:.9f}  {values[2]:.9f}")
    spread = max(abs(scores["hippo"][lag] - scores["diagonal_ssm"][lag])
                 for lag in scores["hippo"])
    print(f"  the three agree to {spread:.1e}: with one input channel and a")
    print("  linear readout they span the same N-dimensional filter space, so")
    print("  the initialisation changes the basis, not the linear memory")
    assert spread < 1e-9


def _demo_selective() -> None:
    """Input-dependent steps let a later token decide what is forgotten."""
    model = SSM(variant="mamba_selective", state=6)
    print("the selective model's step size grows with the input:")
    for value in (0.0, 0.2, 1.0, 2.0):
        delta = float(_softplus(model.w_delta_ * value + model.b_delta_))
        decay = np.exp(-delta * np.exp(model.log_rates_[0]))
        print(f"  x = {value:>3.1f}   delta = {delta:.4f}   "
              f"slowest decay per step = {decay:.4f}")
    print("how much of a pulse at step 0 survives to step 10, as a later")
    print("input at step 4 grows:")
    for follow in (0.0, 0.2, 1.0, 2.0):
        x = np.zeros(12)
        x[0] = 1.0
        x[4] = follow
        perturbed = x.copy()
        perturbed[0] += 0.01
        clean = model.features(x)
        moved = model.features(perturbed)
        ratio = np.linalg.norm(moved[10] - clean[10]) / np.linalg.norm(
            moved[1] - clean[1])
        print(f"  follow-up {follow:>3.1f}   surviving fraction {ratio:.3f}")
    assert model.features(np.zeros(4)).shape == (4, model.state)


def _demo_guards() -> None:
    """Malformed configurations and inputs, all refused."""
    model = SSM(variant="diagonal_ssm", state=3, length=4)
    attempts = [
        lambda: SSM(variant="nope"),
        lambda: SSM(state=0),
        lambda: SSM(state=65),
        lambda: SSM(length=0),
        lambda: SSM(dt=0.0),
        lambda: SSM(dt=np.inf),
        lambda: SSM(seed=-1),
        lambda: model.forward([[1.0, 2.0]]),
        lambda: model.forward([]),
        lambda: model.forward([np.nan, 1.0]),
        lambda: model.kernel(0),
        lambda: model.impulse(0),
        lambda: SSM(variant="mamba_selective").kernel(),
    ]
    refused = 0
    for attempt in attempts:
        try:
            attempt()
        except (ValueError, RuntimeError):
            refused += 1
    print(f"{refused} malformed calls refused out of {len(attempts)}: an unknown")
    print("  variant, a state of zero or sixty-five, a zero length, a zero or")
    print("  infinite step, a negative seed, a two-dimensional or empty or")
    print("  non-finite input, a zero kernel or impulse length, and a kernel")
    print("  requested from a selective model that does not have one.")
    assert refused == len(attempts)


def main() -> None:
    _demo_scan()
    _demo_convolution()
    _demo_hippo()
    _demo_s4()
    _demo_memory()
    _demo_selective()
    _demo_guards()
    print("ssm demo passed")


if __name__ == "__main__":
    main()
