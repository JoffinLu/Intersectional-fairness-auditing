"""Finite-group estimators and audit decisions with explicit inferential targets.

The reference for each mutually exclusive atomic group is its complement.
Bayesian/posterior diagnostics are NOT labelled frequency-valid p-values.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr
from scipy.stats import beta, binom


def _counts(k, n):
    k, n = np.broadcast_arrays(np.asarray(k, float), np.asarray(n, float))
    if k.ndim not in (1, 2) or np.any(~np.isfinite(k)) or np.any(~np.isfinite(n)):
        raise ValueError("Expected finite counts with final axis = mutually exclusive groups.")
    if np.any(k < 0) or np.any(n < k) or np.any(n != np.floor(n)) or np.any(k != np.floor(k)):
        raise ValueError("Counts must be integers satisfying 0 <= k <= n.")
    return k, n


def observed_rates(k, n):
    k, n = _counts(k, n)
    return np.divide(k, n, out=np.full_like(k, np.nan), where=n > 0)


def complement_counts(k, n):
    k, n = _counts(k, n)
    return k.sum(axis=-1, keepdims=True) - k, n.sum(axis=-1, keepdims=True) - n


def complement_rates(k, n):
    kc, nc = complement_counts(k, n)
    return observed_rates(kc, nc)


def true_disparities(p, n):
    p, n = np.broadcast_arrays(np.asarray(p, float), np.asarray(n, float))
    denom = n.sum(axis=-1, keepdims=True) - n
    ref = np.divide((p * n).sum(axis=-1, keepdims=True) - p * n, denom,
                    out=np.full_like(p, np.nan), where=denom > 0)
    return p - ref


def posterior_mean(k, n, mu, kappa):
    k, n = _counts(k, n)
    mu, kappa = np.asarray(mu, float), np.asarray(kappa, float)
    if np.any((mu <= 0) | (mu >= 1)) or np.any(kappa <= 0):
        raise ValueError("Proper beta prior requires 0<mu<1 and kappa>0.")
    return (k + kappa * mu) / (n + kappa)


def four_cell_dirichlet_mean(k, n):
    """Group conditional error mean from a counts+1 four-cell Dirichlet.

    The group's conditional ratio has Beta(k+1,n-k+1) distribution;
    the other group's two pseudo-counts do not enter this denominator.
    This is the point mean only, not the complete SAFT procedure.
    """
    k, n = _counts(k, n)
    return (k+1)/(n+2)


def fit_eb_moments(k, n, kappa_min=2.0, kappa_max=2000.0):
    """Equal-group moment EB, with sampling variance subtracted.

    This is a local, fully specified estimator, NOT Herlihy SR or SAFT.
    Its all-data fitted prior is not covered by fixed-prior masking theorems.
    """
    k, n = _counts(k, n)
    raw = observed_rates(k, n)
    count = (n > 0).sum(axis=-1, keepdims=True)
    mu = np.divide(np.nansum(raw, axis=-1, keepdims=True), count,
                   out=np.full_like(count, .5, dtype=float), where=count > 0)
    mu = np.clip(mu, 1e-4, 1 - 1e-4)
    total_var = np.nanmean((raw - mu) ** 2, axis=-1, keepdims=True)
    noise = np.nanmean(raw * (1 - raw) / np.maximum(n - 1, 1), axis=-1, keepdims=True)
    between = np.maximum(total_var - noise, 1e-8)
    concentration = np.clip(mu * (1 - mu) / between - 1, kappa_min, kappa_max)
    return {"estimate": posterior_mean(k, n, mu, concentration), "mu": mu, "kappa": concentration}


def weighted_reference(estimate, n):
    estimate, n = np.broadcast_arrays(np.asarray(estimate, float), np.asarray(n, float))
    weighted = np.where(n > 0, estimate * n, 0)
    denom = n.sum(axis=-1, keepdims=True) - n
    return np.divide(weighted.sum(axis=-1, keepdims=True) - weighted, denom,
                     out=np.full_like(estimate, np.nan), where=denom > 0)


def hoeffding_difference_pvalues(k, n, margin=.05):
    """Valid for independent, possibly nonidentical Bernoulli observations.

    Target: difference of mean success probabilities for group and complement.
    Zero denominators yield p=1 (inconclusive).
    """
    k, n = _counts(k, n)
    kc, nc = complement_counts(k, n)
    valid = (n > 0) & (nc > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        gap = k / n - kc / nc
        variance_proxy = 1 / n + 1 / nc
        p = np.exp(-2 * np.maximum(gap - margin, 0) ** 2 / variance_proxy)
    return np.where(valid, np.minimum(1, p), 1.)


def reference_upper_binomial_pvalues(k, n, margin=.05, eta=1e-6):
    """Finite-sample valid Hoeffding-reference / exact-target-binomial p.

    Reference observations may be independent heterogeneous Bernoulli.
    Target count must be binomial (iid within that atomic group). eta is fixed
    before seeing confirmation data. Counts for a group and its complement
    are disjoint, so the reference bound is independent of the target count.
    """
    if not 0 < eta < 1 or not 0 <= margin < 1:
        raise ValueError("Require eta in (0,1), margin in [0,1).")
    k, n = _counts(k, n)
    kc, nc = complement_counts(k, n)
    valid = (n > 0) & (nc > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        uref = np.minimum(1, kc / nc + np.sqrt(np.log(1 / eta) / (2 * nc)))
    cutoff = np.minimum(1, uref + margin)
    p = np.minimum(1, eta + binom.sf(k - 1, n, cutoff))
    return np.where(valid, p, 1.)


def weighted_holm(pvalues, weights=None, alpha=.05):
    """Weighted Holm with weights fixed independently of all confirmation p's.

    Arbitrary dependence among the p-values is permitted. First test uses
    alpha*w; after each rejection, remaining weights are renormalized.
    """
    p = np.asarray(pvalues, float)
    if p.ndim not in (1, 2) or np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("Expected finite p-values in [0,1].")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1).")
    g = p.shape[-1]
    w = np.full_like(p, 1/g) if weights is None else np.broadcast_to(np.asarray(weights, float), p.shape).copy()
    if np.any(w <= 0) or np.any(~np.isfinite(w)):
        raise ValueError("Weights must be finite and strictly positive.")
    w /= w.sum(axis=-1, keepdims=True)
    order = np.argsort(p / w, axis=-1, kind="stable")
    po = np.take_along_axis(p, order, axis=-1)
    wo = np.take_along_axis(w, order, axis=-1)
    remaining = np.maximum(1 - np.cumsum(wo, axis=-1) + wo, np.finfo(float).eps)
    passed = po <= alpha * wo / remaining
    rejected_sorted = np.logical_and.accumulate(passed, axis=-1)
    rejected = np.zeros_like(rejected_sorted)
    np.put_along_axis(rejected, order, rejected_sorted, axis=-1)
    return rejected


def discovery_weights(k, n, *, margin=.05, floor=.1, mode="conflict"):
    if not 0 < floor <= 1:
        raise ValueError("floor must be in (0,1].")
    k, n = _counts(k, n)
    g = k.shape[-1]
    raw = np.nan_to_num(observed_rates(k, n), nan=0.)
    ref = np.nan_to_num(complement_rates(k, n), nan=0.)
    eb = fit_eb_moments(k, n)
    shrunk = eb["estimate"]
    sref = weighted_reference(shrunk, n)
    if mode == "uniform":
        score = np.ones_like(k)
    elif mode == "size":
        score = np.sqrt(n)
    elif mode == "raw":
        score = np.maximum(raw - ref - margin, 0.)
    elif mode == "eb":
        score = np.maximum(shrunk - sref - margin, 0.)
    elif mode == "conflict":
        # Preserve raw excess when the shared model suppresses it; no truth is
        # used here. The rule is an application of independent weighting.
        excess = np.maximum(raw - ref - margin, 0.)
        disagreement = np.maximum((raw - ref) - (shrunk - sref), 0.)
        score = excess + disagreement
    else:
        raise ValueError(mode)
    score = np.where(n > 0, score, 0.)
    total = score.sum(axis=-1, keepdims=True)
    normalized = np.divide(score, total, out=np.full_like(score, 1/g), where=total > 0)
    return floor/g + (1-floor) * normalized


def naive_wald_pvalues(k, n, margin=.05):
    """Unprotected Wald diagnostic; NOT promised valid in sparse groups."""
    k, n = _counts(k, n)
    kc, nc = complement_counts(k, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        p, q = k/n, kc/nc
        se = np.sqrt(p*(1-p)/n + q*(1-q)/nc)
        z = (p-q-margin)/se
    # No observations or a zero estimated variance -> inconclusive. Do not
    # manufacture certainty by dividing a tiny number into a positive gap.
    return np.where((n > 0) & (nc > 0) & (se > 0), ndtr(-z), 1.)


def beta_posterior_interval(k, n, mu, kappa, level=.95):
    k, n = _counts(k, n)
    a = k + np.asarray(kappa)*np.asarray(mu)
    b = n-k + np.asarray(kappa)*(1-np.asarray(mu))
    return beta.ppf((1-level)/2, a, b), beta.ppf(1-(1-level)/2, a, b)


def fixed_prior_mse(p, n, mu, kappa):
    p, n = np.asarray(p, float), np.asarray(n, float)
    naive = p*(1-p)/n
    shrink = (n*p*(1-p) + kappa*kappa*(mu-p)**2)/(n+kappa)**2
    return naive, shrink


def fixed_prior_screen_power(p, n, mu, kappa, threshold):
    """Exact probability that posterior mean exceeds a FIXED threshold.

    This is a mechanism screen, not a level-alpha test or a fitted-EB theorem.
    """
    boundary = n*threshold + kappa*(threshold-mu)
    minimum = np.floor(boundary).astype(int)+1
    return binom.sf(minimum-1, n, p)

