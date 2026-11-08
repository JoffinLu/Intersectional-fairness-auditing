"""Reviewed CPU import bridge to the MIT-licensed author's SureMap estimator.

The upstream files remain untouched. This bridge compiles their exact estimator
function bodies and MultiTaskStatistics class, omitting unused dataset/model
imports and the optional Numba decorator. No statistical formula is changed.
It is an import adaptation, not an exact upstream environment reproduction.
"""
from __future__ import annotations

import ast
import hashlib
from functools import cache, cached_property, lru_cache
from itertools import chain, combinations, product
from pathlib import Path
import sys
import numpy as np
import scipy as sp

UPSTREAM_COMMIT = 'ccd80ae74e20a044fd415a65f9946536c3b5fe4a'


@lru_cache(maxsize=1)
def _load_upstream():
    vendor = Path(__file__).resolve().parents[1] / 'vendor' / 'SureMap'
    expected = {
        'data.py': '9bfed17a01ebcbfbc72a9be6ce44e0a9f5faa7b055cce01e7b2bdb64206a4be5',
        'suremap.py': '692163e07cada321df9474f38b402745743660f6ff4da78b41facea2f2bd8fb7',
    }
    for filename, digest in expected.items():
        if hashlib.sha256((vendor / filename).read_bytes()).hexdigest() != digest:
            raise RuntimeError('SureMap source hash differs from reviewed commit')
    namespace = dict(np=np, sp=sp, sys=sys, cache=cache,
                     cached_property=cached_property, chain=chain,
                     combinations=combinations, product=product)
    parsed_data = ast.parse((vendor / 'data.py').read_text(encoding='utf-8'))
    names = {'MultiTaskStatistics', 'error_metric', 'write'}
    nodes = [node for node in parsed_data.body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef))
             and node.name in names]
    if {node.name for node in nodes} != names:
        raise RuntimeError('Reviewed SureMap data interface changed')
    exec(compile(ast.Module(body=nodes, type_ignores=[]),
                 str(vendor / 'data.py'), 'exec'), namespace)
    parsed_estimator = ast.parse((vendor / 'suremap.py').read_text(encoding='utf-8'))
    nodes = [node for node in parsed_estimator.body
             if isinstance(node, ast.FunctionDef)]
    for node in nodes:
        if node.name == 'inplace_batch_inv':
            # Optional compilation only; the body is the upstream Python body.
            node.decorator_list = []
    exec(compile(ast.Module(body=nodes, type_ignores=[]),
                 str(vendor / 'suremap.py'), 'exec'), namespace)
    return namespace


def estimate_suremap(k, n, group_codes):
    """Run author single_task on literal Bernoulli error arrays from counts.

    group_codes: G tuples of the original attribute categories, with no None.
    Returns G point estimates; empty cells are explicitly returned as NaN.
    Upstream can fail for zero shared variance; this error is not replaced by
    another method. Upstream clips error estimates below zero but not above one.
    """
    k, n = np.asarray(k), np.asarray(n)
    if k.ndim != 1 or n.shape != k.shape:
        raise ValueError('matching one-dimensional counts required')
    if (not np.all(np.isfinite(k)) or not np.all(np.isfinite(n)) or
            np.any(k != np.floor(k)) or np.any(n != np.floor(n)) or
            np.any(k < 0) or np.any(n < k)):
        raise ValueError('integer counts satisfying 0 <= k <= n required')
    groups = [tuple(row) for row in group_codes]
    if len(groups) != len(n) or not groups or len({len(g) for g in groups}) != 1:
        raise ValueError('one equal-length attribute tuple per group required')
    if any(any(v is None for v in g) for g in groups) or len(set(groups)) != len(groups):
        raise ValueError('unique attribute tuples without None required')
    if np.sum(n) <= 1:
        return np.full(len(n), np.nan)
    errors = [np.r_[np.ones(int(failures)), np.zeros(int(count-failures))]
              for failures, count in zip(k, n)]
    author = _load_upstream()
    with np.errstate(all='raise'):
        mts = author['MultiTaskStatistics']([errors], groups=groups)
        output = author['single_task'](mts)[0]
    output[np.asarray(n) == 0] = np.nan
    return output
