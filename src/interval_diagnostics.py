"""Exploratory interval diagnostics on retained outcomes, not new audit data.

The SAFT-style branch implements published mathematical constructions locally.
It neither imports unlicensed inspection code nor claims the original SP study.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import beta, norm
from artifacts import ROOT, now, sha256, write_json
from audit_methods import complement_counts, true_disparities, weighted_reference


def wald_intervals(k, n, alpha=.05):
    k = np.asarray(k, float)
    n = np.broadcast_to(np.asarray(n, float), k.shape)
    kc, nc = complement_counts(k, n)
    valid = (n > 0) & (nc > 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        p, q = k/n, kc/nc
        gap = p-q
        se = np.sqrt(p*(1-p)/n + q*(1-q)/nc)
    lo, hi = gap-norm.ppf(1-alpha/2)*se, gap+norm.ppf(1-alpha/2)*se
    return np.where(valid, lo, -np.inf), np.where(valid, hi, np.inf)


def saft_style_intervals(k, n, seed, draws=10000, alpha=.05):
    """Published four-outcome-count switch; local FNR interval adaptation.

    Four counts+1 Dirichlet coordinates imply independent Beta conditional
    target/complement rates. Sampling those two ratios is distributionally
    identical to sampling the four-cell posterior then forming its ratios.
    The returned equal-tail intervals are pointwise, not simultaneous.
    """
    k = np.asarray(k, int)
    n = np.broadcast_to(np.asarray(n, int), k.shape)
    lo, hi = wald_intervals(k, n, alpha)
    kc, nc = complement_counts(k, n)
    rng = np.random.default_rng(seed)
    for index in np.ndindex(k.shape):
        if n[index] > 0 and nc[index] > 0 and min(k[index], n[index]-k[index], kc[index], nc[index]-kc[index]) < 30:
            target = rng.beta(k[index]+1, n[index]-k[index]+1, draws)
            reference = rng.beta(kc[index]+1, nc[index]-kc[index]+1, draws)
            lo[index], hi[index] = np.quantile(target-reference, [alpha/2, 1-alpha/2], method='inverted_cdf')
    return lo, hi


def cp_simultaneous_intervals(k, n, alpha=.05):
    """Propagate a Bonferroni exact group-rate rectangle to all contrasts.

    At most alpha/G noncoverage for each binomial rate implies simultaneous
    contrast coverage >=1-alpha by the union bound and nonnegative weights.
    No binomial model is imposed on a heterogeneous complement count.
    """
    k = np.asarray(k, float)
    n = np.broadcast_to(np.asarray(n, float), k.shape)
    tail = alpha/(2*k.shape[-1])
    lower = np.where(k > 0, beta.ppf(tail, k, n-k+1), 0.)
    upper = np.where(k < n, beta.ppf(1-tail, k+1, n-k), 1.)
    # Sum the complement directly to avoid cancellation when a target cell
    # dominates the count vector. Round endpoints outward to protect exact
    # boundary values such as delta=-1 from floating-point comparison error.
    lo, hi = np.empty_like(lower), np.empty_like(upper)
    for g in range(k.shape[-1]):
        other = np.arange(k.shape[-1]) != g
        denominator = n[..., other].sum(axis=-1)
        with np.errstate(divide='ignore', invalid='ignore'):
            lo[..., g] = lower[..., g]-(upper[..., other]*n[..., other]).sum(axis=-1)/denominator
            hi[..., g] = upper[..., g]-(lower[..., other]*n[..., other]).sum(axis=-1)/denominator
    guard = 8*k.shape[-1]*np.finfo(float).eps
    lo, hi = np.maximum(-1., lo-guard), np.minimum(1., hi+guard)
    valid = (n > 0) & (n.sum(axis=-1, keepdims=True) > n)
    return np.where(valid, lo, -np.inf), np.where(valid, hi, np.inf)


def mean_mcse(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(x.std(ddof=1)/np.sqrt(len(x)))


def summarize(lo, hi, p, n, target, margin=.05):
    truth = true_disparities(p, n)
    coverage = (lo <= truth) & (truth <= hi)
    flags = lo > margin
    null = truth <= margin
    arrays = {'target_coverage': coverage[:, target],
              'family_coverage': coverage.all(axis=1),
              'mean_width': (hi-lo).mean(axis=1),
              'target_width': (hi-lo)[:, target],
              'target_tolerance_rule': flags[:, target],
              'any_null_tolerance_rule': flags[:, null].any(axis=1),
              'any_zero_null_rejection': ((lo > 0) | (hi < 0))[:, np.isclose(truth, 0, atol=1e-12)].any(axis=1)}
    result = {}
    for key, values in arrays.items():
        result[key], result[key+'_mcse'] = mean_mcse(values)
    result.update(target_is_harmed=bool(truth[target] > margin),
                  zero_disparity_cells=int(np.isclose(truth, 0, atol=1e-12).sum()),
                  target_true_disparity=float(truth[target]))
    return result


def main(output_name="interval_diagnostics"):
    out = ROOT/'analysis'/output_name
    if out.exists():
        raise FileExistsError('Retain the actual run; choose a fresh output for a new run.')
    out.mkdir()
    plan = ROOT/'protocols/interval_diagnostics.json'
    assert json.loads(plan.read_text(encoding='utf-8'))['status'] == 'frozen_before_execution'
    inputs = {plan.relative_to(ROOT).as_posix(): sha256(plan)}
    start = time.perf_counter()
    rows, sensitivity = [], []
    for i in range(7):
        source = ROOT/'results/newer_baselines_v1'/f'scenario_{i:03}.npz'
        configpath = source.with_suffix('.json')
        inputs[source.relative_to(ROOT).as_posix()] = sha256(source)
        inputs[configpath.relative_to(ROOT).as_posix()] = sha256(configpath)
        a = np.load(source, allow_pickle=False)
        cfg = json.loads(configpath.read_text(encoding='utf-8'))
        k, n, p = a['k'][:120], a['n'], a['p']
        target = cfg['target_group']
        stored = {'k': k, 'n': n, 'p': p, 'truth': true_disparities(p, n)}
        methods = {
            'wald_pointwise': wald_intervals(k, n),
            'saft_style_pointwise': saft_style_intervals(k, n, np.random.SeedSequence([20261026, i])),
            'cp_simultaneous': cp_simultaneous_intervals(k, n)}
        for method, (lo, hi) in methods.items():
            assert np.isfinite(lo).all() and np.isfinite(hi).all() and np.all(lo <= hi)
            stored['lower__'+method], stored['upper__'+method] = lo, hi
            rows.append({**cfg, 'method': method, 'repetitions': len(k),
                         **summarize(lo, hi, p, n, target),
                         'guarantee': 'simultaneous finite-sample' if method == 'cp_simultaneous' else 'no familywise guarantee'})
        alt_lo, alt_hi = saft_style_intervals(k[:20], n, np.random.SeedSequence([20261027, i]))
        lo, hi = methods['saft_style_pointwise']
        sensitivity.append({'scenario': i, 'repetitions': 20,
            'max_endpoint_difference': float(max(np.max(np.abs(alt_lo-lo[:20])), np.max(np.abs(alt_hi-hi[:20])))),
            'target_tolerance_decisions_changed': int(((alt_lo[:, target] > .05) != (lo[:20, target] > .05)).sum()),
            'family_tolerance_decisions_changed': int(((alt_lo > .05).any(axis=1) != (lo[:20] > .05).any(axis=1)).sum())})
        stored['sensitivity_lower'], stored['sensitivity_upper'] = alt_lo, alt_hi
        np.savez_compressed(out/f'scenario_{i:03}.npz', **stored)
        print(f'Intervals {i+1}/7; elapsed {time.perf_counter()-start:.1f}s', flush=True)
    pd.DataFrame(rows).to_csv(out/'summary.csv', index=False)
    write_json(out/'posterior_mc_sensitivity.json', sensitivity)
    write_json(out/'run_metadata.json', {'completed_at': now(), 'elapsed_seconds': time.perf_counter()-start,
        'plan_sha256': sha256(plan), 'source_sha256': sha256(Path(__file__)), 'input_hashes': inputs,
        'settings': 7, 'reused_audit_datasets': 840, 'new_generated_audit_datasets': 0,
        'group_contrasts_per_method': 26880, 'posterior_draws': 10000,
        'quantile_method': 'inverted_cdf', 'posterior_seed': 20261026,
        'sensitivity_seed': 20261027, 'upstream_code_imported_or_executed': False,
        'scope': 'exploratory diagnostics; published interval ingredients adapted to conditional FNR, not a reproduction of the original SAFT study or guaranteed frequentist SAFT family control'} )
    print(pd.DataFrame(rows)[['scenario_id','method','target_coverage','family_coverage','target_tolerance_rule','any_null_tolerance_rule']].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-name', default='interval_diagnostics')
    args = parser.parse_args()
    if not args.output_name or Path(args.output_name).name != args.output_name:
        parser.error('--output-name must be a directory name, not a path')
    main(args.output_name)
