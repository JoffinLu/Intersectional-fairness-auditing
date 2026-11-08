"""Ablations, real-support semi-synthetic experiments and newer estimators."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from artifacts import ROOT, manifest, now, sha256, write_json
from audit_methods import observed_rates, true_disparities, weighted_reference
from run_simulation import configurations, evaluate_counts, summarize


def group_features(codes):
    codes=np.asarray(codes)
    columns=[]
    for j in range(codes.shape[1]):
        for value in np.unique(codes[:,j]):columns.append((codes[:,j]==value).astype(float))
    return np.column_stack(columns)


def synthetic_codes(g):
    return np.column_stack((np.arange(g)%2,(np.arange(g)//2)%4,np.arange(g)//8))


def auxiliary_estimators(result, config, out, repetitions, group_codes):
    from sr_adapter import estimate_sr
    from suremap_adapter import estimate_suremap
    count=min(repetitions,len(result["k"]))
    n,p=result["n"],result["p"]
    features=group_features(group_codes)
    predictions={name:[] for name in ("sr_point_adapter","suremap_author_adapter")}
    diagnostics=[]
    begin=time.perf_counter()
    for r in range(count):
        for method in predictions:
            try:
                if method=="sr_point_adapter":
                    fitted=estimate_sr(result["k"][r],n,features,seed=400000+r,folds=5)
                    estimate=fitted["estimate"]
                    diagnostics.append({"replicate":r,"method":method,"status":"ok",
                                        "alpha":fitted["alpha"],"convergence_warnings":fitted["convergence_warnings"]})
                else:
                    estimate=estimate_suremap(result["k"][r],n,group_codes)
                    diagnostics.append({"replicate":r,"method":method,"status":"ok"})
                if not np.all(np.isfinite(estimate)):
                    raise FloatingPointError("Estimator returned nonfinite value for a positive-count group.")
            except Exception as error:
                estimate=np.full(len(n),np.nan)
                diagnostics.append({"replicate":r,"method":method,"status":"failed",
                                    "error":f"{type(error).__name__}: {error}"})
            predictions[method].append(estimate)
        if (r+1)%20==0:print(f"  newer estimators {r+1}/{count}, elapsed={time.perf_counter()-begin:.1f}s",flush=True)
    arrays={f"estimate__{method}":np.asarray(values) for method,values in predictions.items()}
    # All point baselines are compared on the same repeat indices. Failure
    # counts and valid paired sets are explicit, never quietly replaced.
    for key,values in result.items():
        if key.startswith("estimate__"):arrays[key]=values[:count]
    np.savez_compressed(out.with_suffix(".npz"),p=p,n=n,**arrays)
    write_json(out.with_suffix(".diagnostics.json"),diagnostics)
    rows=[]
    target=int(config["target_group"])
    pair_valid=np.logical_and.reduce([np.all(np.isfinite(v),axis=1) for v in arrays.values()])
    for methodkey,values in arrays.items():
        method=methodkey.split("__",1)[1]
        valid=np.all(np.isfinite(values),axis=1)
        for scope,selection in (("method_available",valid),("paired_all_methods",pair_valid)):
            vals=values[selection]
            if len(vals):
                loss=(vals-p)**2
                mse=float(loss.mean())
                target_mse=float(loss[:,target].mean())
                screens=vals-weighted_reference(vals,n)>.05
                target_rate=float(screens[:,target].mean())
            else:mse=target_mse=target_rate=None
            rows.append({**config,"method":method,"scope":scope,"attempted_repetitions":count,
                         "valid_repetitions":int(selection.sum()),"failed_repetitions":int((~valid).sum()),
                         "mean_group_mse":mse,"target_mse":target_mse,"target_screen_rate":target_rate,
                         "kind":"point_estimate_only"})
    return rows


def run(args):
    out=ROOT/"results"/args.name
    if out.exists():raise FileExistsError("Choose a fresh --name.")
    out.mkdir(parents=True)
    rng=np.random.default_rng(args.seed)
    start=time.perf_counter()
    rows=[];baseline_rows=[];configs=[]
    if args.mode=="ablation":
        n=np.resize(np.array([10,20,40,80,160,320]),32)
        for c in (.5,.8,.9):
            for floor in (.05,.1,.5,1.):
                for nt in (20,80):
                    nn=n.copy();nn[0]=nt;p=np.full(32,.12);p[0]=.37
                    cfg={"family":"split_ablation","groups":32,"target_n":nt,"effect_above_margin":.20,
                         "target_group":0,"confirmation":c,"floor":floor}
                    configs.append((cfg,p,nn,synthetic_codes(32)))
    elif args.mode=="semisynthetic":
        template_rows=[]
        for dataset in ("adult","diabetes","acs_income","acs_employment"):
            cache=np.load(ROOT/"data/processed"/f"{dataset}.npz",allow_pickle=False)
            group=cache["group_id"];g=len(cache["groupcodes"]);y=cache["y"]
            masks={s:(cache["split"]==s)&(y==1) for s in (1,2)}
            ns={s:np.bincount(group[masks[s]],minlength=g) for s in (1,2)}
            # Support selection is deterministic from covariates/positive
            # counts, prior to generating any semi-synthetic error outcomes.
            use=(ns[2]>=10)&(ns[1]>=5)
            nn=np.minimum((ns[2][use]//10)*10,1000)
            if len(nn)<3:raise ValueError(f"Too few prespecified supported groups for {dataset}.")
            codes=cache["groupcodes"][use]
            original=np.flatnonzero(use)
            for model in ("lr","hgb"):
                fn=(cache[f"{model}_prediction"]==0)
                kc=np.bincount(group[masks[1]&fn],minlength=g)
                baseline=np.clip((kc[use]+.5)/(ns[1][use]+1),.02,.8)
                for geometry in ("homogeneous","calibration_heterogeneous"):
                    p=baseline.copy() if geometry=="calibration_heterogeneous" else np.full(len(nn),np.average(baseline,weights=nn))
                    target=int(np.argmin(nn))
                    reference=float(np.average(np.delete(p,target),weights=np.delete(nn,target)))
                    p[target]=min(.95,reference+.25)
                    cfg={"family":"real_support_semisynthetic","dataset":dataset,"model":model,"geometry":geometry,
                         "groups":len(nn),"target_n":int(nn[target]),"effect_above_margin":float(p[target]-reference-.05),
                         "target_group":target,"confirmation":.8,"floor":.1}
                    configs.append((cfg,p,nn,codes))
                    for j,orig in enumerate(original):
                        template_rows.append({**cfg,"group":j,"original_group":int(orig),"key":str(cache["groupkeys"][orig]),
                                              "positive_audit_count":int(ns[2][orig]),"simulation_count":int(nn[j]),
                                              "calibration_error_rate_template":float(baseline[j]),"true_simulation_p":float(p[j])})
        pd.DataFrame(template_rows).to_csv(out/"templates.csv",index=False)
    elif args.mode=="newer_baselines":
        for cfg,p,n in configurations(False):
            if cfg["groups"]!=32:continue
            selected=(cfg["family"]=="sparse_rare" and cfg["target_n"] in (20,80) and cfg["effect_above_margin"] in (.10,.30)) or cfg["family"] in ("global_null","common_harm","structured_additive")
            if selected:configs.append((cfg,p,n,synthetic_codes(32)))
    for i,(cfg,p,n,codes) in enumerate(configs):
        cfg={"scenario_id":f"scenario_{i:03}",**cfg}
        result=evaluate_counts(p,n,args.repetitions,rng,confirmation=cfg["confirmation"],floor=cfg["floor"])
        np.savez_compressed(out/f"scenario_{i:03}.npz",**result,group_codes=codes)
        write_json(out/f"scenario_{i:03}.json",cfg)
        rows.extend(summarize(result,cfg))
        if args.baseline_repetitions and (args.mode=="newer_baselines" or cfg.get("geometry")=="calibration_heterogeneous"):
            baseline_rows.extend(auxiliary_estimators(result,cfg,out/f"baseline_{i:03}",args.baseline_repetitions,codes))
        print(f"[{i+1}/{len(configs)}] {cfg} elapsed={time.perf_counter()-start:.1f}s",flush=True)
    pd.DataFrame(rows).to_csv(out/"summary.csv",index=False)
    if baseline_rows:pd.DataFrame(baseline_rows).to_csv(out/"newer_baselines_summary.csv",index=False)
    meta={"mode":args.mode,"started_seed":args.seed,"completed_at":now(),"elapsed_seconds":time.perf_counter()-start,
          "repetitions":args.repetitions,"baseline_repetitions":args.baseline_repetitions,"configs":len(configs),
          "code_sha256":{name:sha256(ROOT/'src'/name) for name in ('run_extended.py','run_simulation.py','audit_methods.py','sr_adapter.py','suremap_adapter.py')},
          "data_sha256":{path.name:sha256(path) for path in (ROOT/'data/processed').glob('*_metadata.json')} if args.mode=='semisynthetic' else {},
          "scope":"known independent Bernoulli semi-synthetic losses, not real population truths","gpu_hours":0}
    write_json(out/"run_metadata.json",meta)
    manifest(out/"summary.csv",f"Completed {args.mode} experiment summary with raw per-repeat artifacts")
    manifest(out/"run_metadata.json","Recorded actual repeat counts, runtime and generation scope")
    print(json.dumps(meta,indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=("ablation","semisynthetic","newer_baselines"),required=True)
    parser.add_argument("--name",required=True)
    parser.add_argument("--seed",type=int,default=20261023)
    parser.add_argument("--repetitions",type=int,default=1000)
    parser.add_argument("--baseline-repetitions",type=int,default=0)
    run(parser.parse_args())
