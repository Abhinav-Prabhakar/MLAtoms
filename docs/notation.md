# Notation

Used consistently across every `.md` in the repository. When an atom needs a
symbol outside this table, it defines it locally and says so.

| Symbol | Meaning |
|:--|:--|
| `n` | number of samples (rows, tokens, nodes) |
| `d` | number of features (columns, embedding dimensions) |
| `k` | number of components, clusters, neighbours, or latent dimensions |
| `m` | number of outputs or classes |
| `T` | number of time steps, iterations, or trees |
| `X ∈ ℝ^{n×d}` | design matrix; `x_i` is row `i`, `x_{ij}` an entry |
| `y ∈ ℝ^n` | targets; `Y ∈ ℝ^{n×m}` when multi-output |
| `w ∈ ℝ^d`, `b` | weights and bias |
| `W ∈ ℝ^{d×k}` | matrix of parameters or components |
| `θ` | the full parameter vector of a model |
| `η` | learning rate, or a step size |
| `λ` | regularisation strength, or an eigenvalue |
| `σ(·)` | logistic sigmoid |
| `softmax(·)` | normalised exponential, per row |
| `‖·‖_p` | L_p norm; `‖·‖_F` Frobenius; `‖·‖_*` nuclear |
| `⟨·,·⟩` | inner product |
| `⊙` | elementwise product |
| `⊤` | transpose |
| `∇_θ L` | gradient of loss with respect to `θ` |
| `𝔼[·]`, `𝕍[·]` | expectation, variance |
| `p(x)`, `q(x)` | true and approximate distributions |
| `KL(p‖q)` | Kullback-Leibler divergence |
| `𝒩(μ, Σ)` | Gaussian with mean `μ` and covariance `Σ` |
| `𝟙[·]` | indicator, 1 when the condition holds |
| `O(·)` | asymptotic complexity, worst case unless stated |

## Conventions

- Vectors are columns. `x_i` is the `i`-th sample (a row of `X`); `x_{:j}` is the
  `j`-th feature (a column).
- Indices start at 0 in code and at 1 in mathematics. Equations in `.md` use
  1-based indices; code comments translate.
- `log` is natural unless stated; `log₂` appears only in information-theoretic
  derivations and is marked.
- A "hat" means estimated: `ŷ`, `Σ̂`.
- A star means optimal: `θ*`, `w*`.
