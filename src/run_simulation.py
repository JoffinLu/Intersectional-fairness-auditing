"""Paired, known-truth finite-group Monte Carlo benchmark.

Main files retain integer counts and all decisions. No existing results are
overwritten: choose a new --name for a changed protocol.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from artifacts import ROOT, manifest, now, sha256, write_json
from audit_methods import (
    complement_rates, discovery_weights, fit_eb_moments, fixed_prior_mse, four_cell_dirichlet_mean,
    fixed_prior_screen_power, hoeffding_difference_pvalues, observed_rates,
    posterior_mean, reference_upper_binomial_pvalues, true_disparities,
    weighted_holm, weighted_reference,
)


def evaluate_counts(p, n, repetitions, rng, *, confirmation=.8, alpha=.05, margin=.05, floor=.1):
    n = np.asarray(n, int)
    p = np.asarray(p, float)
    g = len(n)
    nd = np.rint((1-confirmation)*n).astype(int)
    nc = n-nd
    if np.any(nd < 1) or np.any(nc < 1):
        raise ValueError("Both split parts need support in every planned group.")
    fraction = Fraction(str(confirmation))
    if np.any(n % fraction.denominator != 0) or np.any(nc != (n//fraction.denominator)*fraction.numerator):
        raise ValueError("Exact equal split fractions required to keep complement target fixed.")
    kd = rng.binomial(nd, p, size=(repetitions,g))
    kc = rng.binomial(nc, p, size=(repetitions,g))
    k = kd+kc
    truth = true_disparities(p,n)
    harmful = truth > margin+1e-12
    null = ~harmful
    # Fixed across the floor ablation (smallest planned floor is .05), so a
    # change in discovery weights never also changes the confirmation test.
    eta = alpha*.05/(20*g)
    p_full = reference_upper_binomial_pvalues(k,n,margin,eta)
    p_confirmation = reference_upper_binomial_pvalues(kc,nc,margin,eta)
    p_hoeff = hoeffding_difference_pvalues(k,n,margin)
    decisions = {"full_hb_holm": weighted_holm(p_full,alpha=alpha),
                 "full_hoeffding_holm": weighted_holm(p_hoeff,alpha=alpha)}
    weights = {}
    for mode in ("uniform","size","raw","eb","conflict"):
        w = discovery_weights(kd,nd,margin=margin,floor=floor,mode=mode)
        weights[mode] = w
        decisions[f"split_{mode}_holm"] = weighted_holm(p_confirmation,w,alpha)
    raw = observed_rates(k,n)
    eb = fit_eb_moments(k,n)
    estimates = {"raw":raw, "pooled":np.broadcast_to(k.sum(axis=1,keepdims=True)/n.sum(),k.shape),
                 "fixed_beta_100":posterior_mean(k,n,.12,100), "moment_eb":eb["estimate"],
                 "dirichlet_four_cell_mean":four_cell_dirichlet_mean(k,n)}
    screens = {name:(estimate-weighted_reference(estimate,n)>margin) for name,estimate in estimates.items()}
    return {"p":p,"n":n,"n_discovery":nd,"n_confirmation":nc,"k":k,"k_discovery":kd,
            "k_confirmation":kc,"true_delta":truth,"harmful":harmful,"null":null,
            "p_full":p_full,"p_confirmation":p_confirmation,"eb_mu":eb["mu"],"eb_kappa":eb["kappa"],
            **{f"estimate__{key}":val for key,val in estimates.items()},
            **{f"flag__{key}":val for key,val in decisions.items()},
            **{f"screen__{key}":val for key,val in screens.items()},
            **{f"weight__{key}":val for key,val in weights.items()}}


def summarize(result, config):
    p,n,truth=result["p"],result["n"],result["true_delta"]
    harmful,null=result["harmful"],result["null"]
    target=int(config["target_group"])
    rows=[]
    for key,values in result.items():
        if key.startswith("flag__") or key.startswith("screen__"):
            kind,method=key.split("__",1)
            targets=values[:,target].astype(float)
            fwer=np.any(values[:,null],axis=1).astype(float) if null.any() else np.zeros(len(values))
            group_power=values[:,harmful].mean(axis=0) if harmful.any() else np.array([np.nan])
            row={**config,"kind":kind,"method":method,"repetitions":len(values),
                 "target_true_delta":float(truth[target]),"harmful_groups":int(harmful.sum()),
                 "target_flag_rate":float(targets.mean()),
                 "target_mcse":float(targets.std(ddof=1)/np.sqrt(len(values))),
                 "fwer":float(fwer.mean()),"fwer_mcse":float(fwer.std(ddof=1)/np.sqrt(len(values))),
                 "mean_harm_power":float(np.nanmean(group_power)) if harmful.any() else None,
                 "worst_harm_power":float(np.nanmin(group_power)) if harmful.any() else None}
            rows.append(row)
        elif key.startswith("estimate__"):
            method=key.split("__",1)[1]
            losses=(values-p)**2
            avg=losses.mean(axis=1)
            row={**config,"kind":"estimate","method":method,"repetitions":len(values),
                 "target_true_delta":float(truth[target]),"harmful_groups":int(harmful.sum()),
                 "mean_group_mse":float(avg.mean()),"mse_mcse":float(avg.std(ddof=1)/np.sqrt(len(values))),
                 "target_mse":float(losses[:,target].mean()),
                 "population_weighted_mse":float(np.average(losses,weights=n,axis=1).mean())}
            rows.append(row)
    return rows


def configurations(small=False):
    gs=(32,) if small else (16,32,64)
    ns=(20,80) if small else (10,20,40,80,160)
    gaps=(.10,.25) if small else (.10,.20,.30)
    for g in gs:
        n=np.resize(np.array([10,20,40,80,160,320]),g)
        for target_n in ns:
            for gap in gaps:
                nn=n.copy();nn[0]=target_n
                p=np.full(g,.12);p[0]+=.05+gap
                yield {"family":"sparse_rare","groups":g,"target_n":target_n,"effect_above_margin":gap,
                       "target_group":0,"confirmation":.8,"floor":.1},p,nn
        # Stress controls share the same inferential target and integer budget.
        for family in ("global_null","boundary_null","common_harm","dense_harm","structured_additive"):
            nn=n.copy();p=np.full(g,.12);target=0
            if family=="boundary_null":p[0]=.17
            if family=="common_harm":target=int(np.argmax(nn));p[target]=.37
            if family=="dense_harm":p[:max(2,g//8)]=.37
            if family=="structured_additive":
                index=np.arange(g);p=.08+.035*(index%2)+.02*((index//2)%4)+.015*((index//8)%4)
            yield {"family":family,"groups":g,"target_n":int(nn[target]),"effect_above_margin":float(p[target]-.17),
                   "target_group":target,"confirmation":.8,"floor":.1},p,nn


def run_main(args):
    out=ROOT/"results"/args.name
    if out.exists():raise FileExistsError(f"Use a fresh --name, not {out}")
    out.mkdir(parents=True)
    start=time.perf_counter()
    configs=list(configurations(args.pilot))
    meta={"started_at":now(),"seed":args.seed,"repetitions":args.repetitions,"pilot":args.pilot,
          "margin":.05,"alpha":.05,"configurations":len(configs),"gpu_hours":0,
          "python":platform.python_version(),"numpy":np.__version__,"code_sha256":{p.name:sha256(p) for p in (Path(__file__),ROOT/"src/audit_methods.py")}}
    write_json(out/"run_metadata.json",meta)
    rows=[]
    seeds=np.random.SeedSequence(args.seed).spawn(len(configs))
    for i,((cfg,p,n),seed) in enumerate(zip(configs,seeds)):
        result=evaluate_counts(p,n,args.repetitions,np.random.default_rng(seed),confirmation=cfg["confirmation"],floor=cfg["floor"])
        resultfile=out/f"scenario_{i:03}.npz"
        np.savez_compressed(resultfile,**result)
        config={"scenario_id":f"scenario_{i:03}",**cfg,"seed_spawn_key":list(seed.spawn_key)}
        write_json(out/f"scenario_{i:03}.json",config)
        rows.extend(summarize(result,config))
        print(f"[{i+1}/{len(configs)}] {cfg['family']} G={len(n)} n_target={n[cfg['target_group']]} elapsed={time.perf_counter()-start:.1f}s",flush=True)
    frame=pd.DataFrame(rows)
    frame.to_csv(out/"summary.csv",index=False)
    meta.update(completed_at=now(),elapsed_seconds=time.perf_counter()-start,raw_npz_files=len(configs),summary_rows=len(frame))
    write_json(out/"run_metadata.json",meta)
    manifest(out/"summary.csv","Paired finite-group benchmark: estimates, screens, valid flags, MC errors")
    manifest(out/"run_metadata.json","Completed run metadata with parameters, execution duration and source code hashes")
    print(json.dumps(meta,indent=2,ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--name",required=True)
    parser.add_argument("--repetitions",type=int,default=2000)
    parser.add_argument("--seed",type=int,default=20261022)
    parser.add_argument("--pilot",action="store_true")
    run_main(parser.parse_args())
