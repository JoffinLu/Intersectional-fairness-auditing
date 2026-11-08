"""Local paper-specified SR point adapter; not upstream code or rBLPR inference.

Herlihy et al. (FAccT 2024), Eq. (2): inverse-variance weighted lasso
with unpenalized intercept, attribute indicators and saturated cell indicators.
The shared pooled-variance factor cancels into the selected penalty scale.
This adapter uses binomial sufficient statistics and individual-stratified CV.
It deliberately returns no confidence intervals or audit decisions.
"""
from __future__ import annotations

import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso


def _fit(x, k, n, alpha):
    observed = n > 0
    if not np.any(observed):
        return np.full(len(n), np.nan), False
    y = k[observed] / n[observed]
    if alpha == 0:
        # The saturated design interpolates all observed cell means. Minimum
        # weighted least squares defines predictions for absent training cells.
        design = np.column_stack((np.ones(len(n)), x))
        sw = np.sqrt(n[observed])
        coef = np.linalg.lstsq(design[observed] * sw[:, None], y * sw,
                               rcond=None)[0]
        return design @ coef, False
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always', ConvergenceWarning)
        model = Lasso(alpha=float(alpha), fit_intercept=True, max_iter=30000,
                      tol=1e-8, selection='cyclic')
        model.fit(x[observed], y, sample_weight=n[observed])
    return model.predict(x), any(isinstance(w.message, ConvergenceWarning)
                                 for w in captured)


def _individual_folds(k, n, folds, rng):
    """Random group-stratified individual folds without expanding the sample."""
    test_n = np.zeros((folds, len(n)), dtype=np.int64)
    test_k = np.zeros_like(test_n)
    for g in range(len(n)):
        capacities = np.full(folds, n[g] // folds, dtype=np.int64)
        capacities[rng.permutation(folds)[:int(n[g] % folds)]] += 1
        remaining_n, remaining_k = int(n[g]), int(k[g])
        for f, capacity in enumerate(capacities):
            capacity = int(capacity)
            drawn = (remaining_k if capacity == remaining_n else
                     int(rng.hypergeometric(remaining_k,
                         remaining_n - remaining_k, capacity))) if capacity else 0
            test_n[f, g], test_k[f, g] = capacity, drawn
            remaining_n -= capacity
            remaining_k -= drawn
    return test_k, test_n


def estimate_sr(k, n, features, seed, folds=5):
    """Estimate FNR cell means from 1D failure/positive counts and group features.

    ``features`` must contain the prespecified sensitive-attribute one-hot
    columns. All G cell-identity columns are appended, including absent cells.
    No feature scaling, probability clipping or prior pseudo-counts are used.
    ``alpha`` uses scikit-learn's weighted-lasso normalization, not the paper's
    raw lambda scale. A 33-point relative path includes the saturated alpha=0.
    Zero-positive cells remain NaN in the returned estimate and are not audited.
    """
    k, n = np.asarray(k, dtype=float), np.asarray(n, dtype=float)
    attributes = np.asarray(features, dtype=float)
    if k.ndim != 1 or n.ndim != 1 or k.shape != n.shape:
        raise ValueError('k and n must be matching one-dimensional counts')
    if attributes.ndim != 2 or attributes.shape[0] != len(n):
        raise ValueError('features must have shape (G, d)')
    if (not np.all(np.isfinite(k)) or not np.all(np.isfinite(n)) or
            not np.all(np.isfinite(attributes)) or np.any(k < 0) or
            np.any(n < k) or np.any(n != np.floor(n)) or
            np.any(k != np.floor(k))):
        raise ValueError('finite integer counts with 0 <= k <= n are required')
    if not isinstance(folds, (int, np.integer)) or folds < 2:
        raise ValueError('folds must be an integer >= 2')
    ni, ki = n.astype(np.int64), k.astype(np.int64)
    x = np.column_stack((attributes, np.eye(len(n))))
    observed = n > 0
    total_n = int(ni.sum())
    if total_n == 0:
        return dict(estimate=np.full(len(n), np.nan), alpha=np.nan, cv_folds=0,
                    status='no_positive_outcomes', inference='none')
    cv_folds = min(int(folds), total_n)
    global_mean = float(ki.sum() / total_n)
    centered_y = ki[observed] / ni[observed] - global_mean
    x_mean = np.average(x[observed], axis=0, weights=ni[observed])
    gradient = (x[observed] - x_mean).T @ (ni[observed] * centered_y) / total_n
    alpha_max = float(np.max(np.abs(gradient), initial=0))
    alphas = np.r_[0., alpha_max * np.geomspace(1e-4, 1., 32)]
    losses = np.zeros(len(alphas))
    convergence_warnings = 0
    if cv_folds >= 2:
        tk, tn = _individual_folds(ki, ni, cv_folds,
                                   np.random.default_rng(int(seed)))
        validation_count = 0
        for fold in range(cv_folds):
            valid = tn[fold] > 0
            if not np.any(valid) or np.sum(ni - tn[fold]) == 0:
                continue
            validation_count += int(tn[fold].sum())
            for j, alpha in enumerate(alphas):
                pred, warned = _fit(x, ki - tk[fold], ni - tn[fold], alpha)
                convergence_warnings += int(warned)
                losses[j] += np.sum(tn[fold, valid] *
                    (pred[valid] - tk[fold, valid] / tn[fold, valid]) ** 2)
        losses /= max(validation_count, 1)
        selected = int(np.argmin(losses))
    else:
        selected = 0
        losses[:] = np.nan
    estimate, warned = _fit(x, ki, ni, alphas[selected])
    convergence_warnings += int(warned)
    estimate[~observed] = np.nan
    return dict(estimate=estimate, alpha=float(alphas[selected]),
                cv_folds=cv_folds, requested_folds=int(folds),
                alpha_grid=alphas, cv_losses=losses,
                n_attribute_features=int(attributes.shape[1]),
                n_identity_features=len(n), n_zero_cells=int((~observed).sum()),
                n_cells_smaller_than_folds=int(((ni > 0) & (ni < cv_folds)).sum()),
                convergence_warnings=convergence_warnings,
                status='point_adapter', inference='none',
                deviations=['local implementation; no author SR source found',
                            'binomial sufficient-statistic input',
                            'default 5 folds rather than paper 10',
                            'count-proportional loss (common variance factor absorbed)',
                            'no explanatory covariates beyond supplied features',
                            'no rBLPR confidence intervals'])
