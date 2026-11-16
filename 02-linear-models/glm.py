"""Generalised linear models — 9 variants of one IRLS loop.

Chapter:   02-linear-models
Companion: glm.md
Tier 1 · 9 variants: poisson, gamma, binomial, tweedie, negative_binomial,
log_link, identity_link, offsets, dispersion

A GLM keeps the linear predictor and replaces the Gaussian assumption:

    E[y] = mu = g^-1(X beta + offset),      Var[y] = phi * V(mu)

Two functions define the model — the link `g` and the variance function `V`.
Everything else (the IRLS update, the deviance, the standard errors) follows from
them, which is why one loop covers all nine variants.

IRLS is Fisher scoring:

    w_i = 1 / (V(mu_i) * g'(mu_i)^2)
    z_i = eta_i + (y_i - mu_i) * g'(mu_i)      (the working response)
    beta <- argmin sum w_i (z_i - x_i^T beta - offset_i)^2

The offset is not a covariate: it enters the linear predictor with coefficient 1,
which is how exposure, effort and population size are handled.
"""

from __future__ import annotations

import numpy as np

VARIANTS = frozenset(
    ["poisson", "gamma", "binomial", "tweedie", "negative_binomial",
     "log_link", "identity_link", "offsets", "dispersion"]
)

FAMILIES = frozenset(
    ["poisson", "gamma", "binomial", "gaussian", "tweedie", "negative_binomial"]
)
LINKS = frozenset(["log", "identity", "logit"])

EPS = 1e-12


# --------------------------------------------------------------------------- #
# Stable primitives
# --------------------------------------------------------------------------- #
def sigmoid(z: np.ndarray) -> np.ndarray:
    """1/(1+e^-z), evaluated without overflow at either end."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _inv_link(eta: np.ndarray, link: str) -> np.ndarray:
    if link == "log":
        return np.exp(eta)
    if link == "identity":
        return eta
    return sigmoid(eta)


def _link(mu: np.ndarray, link: str) -> np.ndarray:
    if link == "log":
        return np.log(mu)
    if link == "identity":
        return mu
    return np.log(mu / (1.0 - mu))


def _link_grad(mu: np.ndarray, link: str) -> np.ndarray:
    """g'(mu) = d eta / d mu."""
    if link == "log":
        return 1.0 / mu
    if link == "identity":
        return np.ones_like(mu)
    return 1.0 / (mu * (1.0 - mu))


def _variance(mu, family: str, power: float, theta: float) -> np.ndarray:
    if family == "poisson":
        return mu
    if family == "gamma":
        return mu * mu
    if family == "binomial":
        return mu * (1.0 - mu)
    if family == "gaussian":
        return np.ones_like(mu)
    if family == "tweedie":
        return mu**power
    return mu + mu * mu / theta          # negative binomial


def _deviance(y, mu, family: str, power: float, theta: float) -> float:
    """Twice the log-likelihood ratio against the saturated model."""
    if family == "gaussian":
        return float(np.sum((y - mu) ** 2))
    if family == "poisson":
        t = np.where(y > 0.0, y * np.log(np.maximum(y, EPS) / mu), 0.0) - (y - mu)
        return float(2.0 * np.sum(t))
    if family == "gamma":
        return float(2.0 * np.sum(-np.log(y / mu) + (y - mu) / mu))
    if family == "binomial":
        t = (np.where(y > 0.0, y * np.log(np.maximum(y, EPS) / mu), 0.0)
             + np.where(y < 1.0, (1.0 - y) * np.log(np.maximum(1.0 - y, EPS) / (1.0 - mu)), 0.0))
        return float(2.0 * np.sum(t))
    if family == "negative_binomial":
        t = (np.where(y > 0.0, y * np.log(np.maximum(y, EPS) / mu), 0.0)
             - (y + theta) * np.log1p((y - mu) / (mu + theta)))
        return float(2.0 * np.sum(t))
    # tweedie: the Poisson (p=1) and Gamma (p=2) cases are the limits
    if power == 1.0:
        return _deviance(y, mu, "poisson", power, theta)
    if power == 2.0:
        return _deviance(y, mu, "gamma", power, theta)
    t = (y ** (2.0 - power) / ((1.0 - power) * (2.0 - power))
         - y * mu ** (1.0 - power) / (1.0 - power)
         + mu ** (2.0 - power) / (2.0 - power))
    return float(2.0 * np.sum(t))


def _clip_mu(mu: np.ndarray, family: str) -> np.ndarray:
    if family == "binomial":
        return np.clip(mu, 1e-10, 1.0 - 1e-10)
    if family == "gaussian":
        return mu
    return np.maximum(mu, EPS)


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
def glm(
    X: np.ndarray,
    y: np.ndarray,
    variant: str = "poisson",
    *,
    offset: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    power: float = 1.5,
    theta: float = 1.0,
    fit_intercept: bool = True,
    iters: int = 100,
    tol: float = 1e-12,
) -> dict:
    """Fit a GLM by iteratively reweighted least squares.

    X       : (n, d) design, without an intercept column.
    y       : (n,) response. Binomial takes proportions in [0, 1].
    offset  : (n,) known term added to the linear predictor (e.g. log exposure).
    weights : (n,) prior weights, not the IRLS weights.
    power   : Tweedie variance exponent, V(mu) = mu**power.
    theta   : negative-binomial size, Var = mu + mu^2/theta.

    Returns a dict: coef, intercept, mu, eta, se, cov, dispersion, deviance,
    pearson, n_iter, converged, family, link, variant.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    family, link = {
        "poisson": ("poisson", "log"),
        "gamma": ("gamma", "log"),
        "binomial": ("binomial", "logit"),
        "tweedie": ("tweedie", "log"),
        "negative_binomial": ("negative_binomial", "log"),
        "log_link": ("gaussian", "log"),
        "identity_link": ("gaussian", "identity"),
        "offsets": ("poisson", "log"),
        "dispersion": ("poisson", "log"),
    }[variant]
    phi_known = family in ("poisson", "binomial", "negative_binomial")
    if variant == "dispersion":
        phi_known = False

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = X.shape[0]
    Xa = np.column_stack([X, np.ones(n)]) if fit_intercept else X
    off = np.zeros(n) if offset is None else np.asarray(offset, dtype=float)
    sw = np.ones(n) if weights is None else np.asarray(weights, dtype=float)

    # Start from a smoothed response, mapped through the link. This keeps mu
    # positive (log) or interior (logit); for the identity link the first IRLS
    # step is exactly OLS.
    mu0 = _clip_mu(0.5 * (y + float(np.mean(y))), family)
    beta = np.linalg.lstsq(Xa, _link(mu0, link) - off, rcond=None)[0]

    dev = _deviance(y, _clip_mu(_inv_link(Xa @ beta + off, link), family),
                    family, power, theta)
    history = [dev]
    converged = False
    n_iter = 0

    for _ in range(iters):
        eta = Xa @ beta + off
        mu = _clip_mu(_inv_link(eta, link), family)
        gp = _link_grad(mu, link)
        w = sw / (_variance(mu, family, power, theta) * gp * gp)
        w = np.maximum(w, 0.0)
        z = eta - off + (y - mu) * gp          # working response for beta only
        sqrt_w = np.sqrt(w)
        beta_new = np.linalg.lstsq(Xa * sqrt_w[:, None], z * sqrt_w, rcond=None)[0]

        # Backtracking keeps the deviance monotone for non-canonical links.
        step = 1.0
        for _ in range(40):
            cand = beta + step * (beta_new - beta)
            mu_c = _clip_mu(_inv_link(Xa @ cand + off, link), family)
            dev_c = _deviance(y, mu_c, family, power, theta)
            if np.isfinite(dev_c) and dev_c <= dev + 1e-12 * abs(dev):
                break
            step *= 0.5
        else:
            cand, dev_c = beta, dev

        dev_old = dev
        beta, dev = cand, dev_c
        n_iter += 1
        history.append(dev)
        if abs(dev - dev_old) <= tol * (abs(dev_old) + 0.1):
            converged = True
            break

    eta = Xa @ beta + off
    mu = _clip_mu(_inv_link(eta, link), family)
    gp = _link_grad(mu, link)
    var = _variance(mu, family, power, theta)
    w = np.maximum(sw / (var * gp * gp), 0.0)
    info = Xa.T @ (Xa * w[:, None])
    rank = int(np.linalg.matrix_rank(Xa))
    pearson = float(np.sum(sw * (y - mu) ** 2 / var))
    dispersion = 1.0 if phi_known else pearson / max(n - rank, 1)
    cov = dispersion * np.linalg.pinv(info)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))

    coef = beta[:-1] if fit_intercept else beta
    intercept = float(beta[-1]) if fit_intercept else 0.0
    return {
        "coef": coef,
        "intercept": intercept,
        "mu": mu,
        "eta": eta,
        "se": se,
        "cov": cov,
        "dispersion": float(dispersion),
        "deviance": float(dev),
        "pearson": pearson,
        "n_iter": n_iter,
        "converged": converged,
        "history": history,
        "family": family,
        "link": link,
        "variant": variant,
    }


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def main() -> None:
    rng = np.random.default_rng(0)
    n, d = 400, 3
    X = rng.normal(size=(n, d))
    beta_true = np.array([0.40, -0.30, 0.20])
    b0 = 0.50

    # --- Poisson: canonical log link, so the score is exactly X^T (y - mu) ---
    mu_true = np.exp(X @ beta_true + b0)
    y = rng.poisson(mu_true).astype(float)
    fit = glm(X, y, "poisson")
    Xa = np.column_stack([X, np.ones(n)])
    score = Xa.T @ (y - fit["mu"])
    assert np.max(np.abs(score)) < 1e-6, score
    assert np.allclose(fit["mu"], np.exp(fit["eta"]))
    assert np.linalg.norm(fit["coef"] - beta_true) < 0.15, fit["coef"]
    assert fit["dispersion"] == 1.0
    print(f"poisson: beta = {np.round(fit['coef'], 3)}, "
          f"intercept = {fit['intercept']:.3f}, iters = {fit['n_iter']}")

    # --- Gamma: log link is non-canonical, score is X^T (y/mu - 1) ---
    mu_g = np.exp(X @ beta_true + b0)
    y_g = rng.gamma(shape=2.0, scale=mu_g / 2.0)
    fit_g = glm(X, y_g, "gamma")
    score_g = Xa.T @ (y_g / fit_g["mu"] - 1.0)
    assert np.max(np.abs(score_g)) < 1e-5, score_g
    assert fit_g["dispersion"] > 0.0
    print(f"gamma: beta = {np.round(fit_g['coef'], 3)}, "
          f"phi = {fit_g['dispersion']:.3f}")

    # --- Binomial: logit link, score X^T (y - p) ---
    p_true = sigmoid(X @ np.array([1.0, -0.8, 0.5]) - 0.2)
    y_b = (rng.random(n) < p_true).astype(float)
    fit_b = glm(X, y_b, "binomial")
    assert np.max(np.abs(Xa.T @ (y_b - fit_b["mu"]))) < 1e-6
    assert np.all((fit_b["mu"] > 0.0) & (fit_b["mu"] < 1.0))
    acc = float(np.mean((fit_b["mu"] > 0.5) == y_b))
    assert acc > 0.65, acc
    print(f"binomial: beta = {np.round(fit_b['coef'], 3)}, train acc = {acc:.3f}")

    # --- Tweedie limits: power 1 is Poisson, power 2 is Gamma ---
    fit_t1 = glm(X, y, "tweedie", power=1.0)
    fit_t2 = glm(X, y_g, "tweedie", power=2.0)
    # Same coefficients; the tweedie family treats phi as unknown, so its SEs are
    # the Poisson ones scaled by sqrt(phi_hat).
    assert np.allclose(fit_t1["coef"], fit["coef"])
    assert np.allclose(fit_t1["se"], fit["se"] * np.sqrt(fit_t1["dispersion"]))
    assert np.allclose(fit_t2["coef"], fit_g["coef"])
    assert np.allclose(fit_t2["se"], fit_g["se"])
    print(f"tweedie: power 1 == poisson ({np.max(np.abs(fit_t1['coef'] - fit['coef'])):.1e}), "
          f"power 2 == gamma ({np.max(np.abs(fit_t2['coef'] - fit_g['coef'])):.1e})")

    # --- Negative binomial: large theta collapses to Poisson ---
    fit_nb = glm(X, y, "negative_binomial", theta=1e7)
    gap = np.max(np.abs(fit_nb["coef"] - fit["coef"]))
    assert gap < 1e-4, gap
    print(f"negative_binomial: theta=1e7 matches poisson to {gap:.1e}")

    # --- Log link on a Gaussian family: score X^T (mu * (mu - y)) ---
    y_ll = np.exp(X @ beta_true + b0) + rng.normal(0.0, 0.3, size=n)
    fit_ll = glm(X, y_ll, "log_link")
    score_ll = Xa.T @ (fit_ll["mu"] * (fit_ll["mu"] - y_ll))
    assert np.max(np.abs(score_ll)) < 1e-6, score_ll
    ols_log = np.linalg.lstsq(Xa, np.log(np.maximum(y_ll, 1e-6)), rcond=None)[0]
    gap_ll = np.max(np.abs(fit_ll["coef"] - ols_log[:-1]))
    assert gap_ll > 1e-3, gap_ll
    print(f"log_link: score ok, differs from OLS-on-log(y) by {gap_ll:.3f}")

    # --- Identity link on a Gaussian family is exactly OLS ---
    y_i = X @ beta_true + b0 + rng.normal(0.0, 0.2, size=n)
    fit_i = glm(X, y_i, "identity_link")
    ols = np.linalg.lstsq(Xa, y_i, rcond=None)[0]
    gap_i = np.max(np.abs(np.concatenate([fit_i["coef"], [fit_i["intercept"]]]) - ols))
    assert gap_i < 1e-8, gap_i
    print(f"identity_link: == OLS to {gap_i:.1e}")

    # --- Offsets: exposure is known, not a coefficient ---
    exposure = np.exp(0.6 * X[:, 0])
    y_off = rng.poisson(exposure * np.exp(X @ beta_true + b0)).astype(float)
    fit_off = glm(X, y_off, "offsets", offset=np.log(exposure))
    fit_no = glm(X, y_off, "poisson")
    err_off = np.max(np.abs(fit_off["coef"] - beta_true))
    err_no = np.max(np.abs(fit_no["coef"] - beta_true))
    assert err_off < 0.15, err_off
    assert abs(fit_no["coef"][0] - beta_true[0]) > 0.3, fit_no["coef"]
    print(f"offsets: with offset err {err_off:.3f}, without {err_no:.3f}")

    # --- Quasi-Poisson: overdispersion inflates SEs by sqrt(phi) ---
    eps = rng.normal(0.0, 0.5, size=n)
    y_od = rng.poisson(mu_true * np.exp(eps)).astype(float)
    fit_p = glm(X, y_od, "poisson")
    fit_d = glm(X, y_od, "dispersion")
    assert np.allclose(fit_p["coef"], fit_d["coef"])
    phi = fit_d["dispersion"]
    assert phi > 1.2, phi
    ratio = np.max(np.abs(fit_d["se"] / fit_p["se"] - np.sqrt(phi)))
    assert ratio < 1e-8, ratio
    print(f"dispersion: phi = {phi:.3f}, SE ratio = {np.sqrt(phi):.3f}")

    print("glm demo passed")


if __name__ == "__main__":
    main()
