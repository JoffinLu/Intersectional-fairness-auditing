"""Reproducible public-data preparation and frozen CPU prediction baselines.

All preprocessing is fit on train only. Calibration chooses one global threshold;
audit data never select features, model parameters, or thresholds. Observational
audit estimates are descriptions, never known population truths.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import time
import zipfile

# Avoid oversubscribing CPU when multiple experiment workers coexist.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from folktables import ACSIncome, ACSEmployment

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
SEED = 20261022
AGE_LEVELS = ["<40", "40-69", "70+"]

def read_zip(archive, filename, **kwargs):
    with zipfile.ZipFile(RAW / archive) as z:
        return pd.read_csv(z.open(filename), **kwargs)

def age_bins(values):
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError('Missing or unparsed age cannot be silently assigned to an age band.')
    return np.where(values < 40, "<40", np.where(values < 70, "40-69", "70+"))

def datasets():
    cols = ["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country","income"]
    parts=[]
    for filename,prefix in [("adult.data","original_train"),("adult.test","original_test")]:
        df=read_zip("adult.zip",filename,names=cols,skipinitialspace=True,comment="|",na_values="?",dtype={"income":str})
        df["record_id"] = [f"{prefix}:{i}" for i in range(len(df))]
        parts.append(df)
    df=pd.concat(parts,ignore_index=True)
    a=pd.DataFrame({"sex":df["sex"].fillna("Unknown"),"race":df["race"].fillna("Unknown"),"age":age_bins(df["age"])})
    y=df["income"].str.rstrip(".").eq(">50K").astype(np.int8).to_numpy()
    X=df.drop(columns=["income","record_id","sex","race","age","fnlwgt"])
    yield "adult", X, y, a, df.record_id.astype(str).to_numpy(), None, {"raw_records":len(df),"eligible_records":len(df),"target":"income > $50,000; trailing test-label dot removed", "selection":"Combine UCI supplied train/test then seeded split; original person identifiers are unavailable, so row IDs identify records only. fnlwgt excluded because analysis is unweighted and it is a survey weight."}

    df=read_zip("diabetes_130.zip","diabetic_data.csv",na_values="?",low_memory=False)
    raw_n=len(df)
    # encounter_id order is a deterministic encounter selection rule, not a
    # claim that an encounter timestamp is available in this public file.
    df=df.sort_values("encounter_id",kind="stable").drop_duplicates("patient_nbr",keep="first").reset_index(drop=True)
    eligible=len(df)
    if len(df)>50000:
        chosen=np.sort(np.random.default_rng(SEED).choice(len(df),50000,replace=False))
        df=df.iloc[chosen].reset_index(drop=True)
    lower=df.age.str.extract(r"^\[(\d+)-\d+\)$",expand=False).astype(float)
    a=pd.DataFrame({"sex":df.gender.fillna("Unknown"),"race":df.race.fillna("Unknown"),"age":age_bins(lower)})
    y=df.readmitted.eq("<30").astype(np.int8).to_numpy()
    X=df.drop(columns=["encounter_id","patient_nbr","readmitted","gender","race","age"])
    for c in ["admission_type_id","discharge_disposition_id","admission_source_id"]:
        X[c]=X[c].astype(str)
    yield "diabetes", X, y, a, df.encounter_id.astype(str).to_numpy(), df.patient_nbr.astype(str).to_numpy(), {"raw_records":raw_n,"eligible_records":eligible,"target":"readmission <30 days; >30 and NO are negative", "selection":"Select minimum encounter_id per patient_nbr, then seeded uniform cap of 50,000 patients. Minimum encounter ID is a deterministic choice, not verified chronological ordering. Prediction is at discharge; discharge disposition and in-hospital treatments are included. No clinical utility or prospective deployment claim."}

    df=read_zip("acs_2018_1y_al_person.zip","psam_p01.csv",low_memory=False,dtype={"SERIALNO":str})
    for name,task in [("acs_income",ACSIncome),("acs_employment",ACSEmployment)]:
        sub=task._preprocess(df).copy().reset_index(drop=True)
        if len(sub)>50000:
            chosen=np.sort(np.random.default_rng(SEED).choice(len(sub),50000,replace=False))
            sub=sub.iloc[chosen].reset_index(drop=True)
        labels={1:"Male",2:"Female"}
        races={1:"White",2:"Black",3:"AmericanIndian",4:"AlaskaNative",5:"AIAN_combined",6:"Asian",7:"NativeHawaiianPacific",8:"Other",9:"Multiracial"}
        a=pd.DataFrame({"sex":sub.SEX.map(labels).fillna("Unknown"),"race":sub.RAC1P.map(races).fillna("Unknown"),"age":age_bins(sub.AGEP)})
        y=np.asarray(task._target_transform(sub[task.target]),dtype=np.int8)
        X=sub[[f for f in task.features if f not in ["AGEP","SEX","RAC1P"]]].copy()
        numeric=["WKHP"]
        for c in X.columns:
            if c not in numeric:
                X[c]=X[c].fillna(-999).astype(str)
        ids=(sub.SERIALNO.astype(str)+":"+sub.SPORDER.astype(str)).to_numpy()
        yield name,X,y,a,ids,sub.SERIALNO.astype(str).to_numpy(), {"raw_records":len(df),"eligible_records":len(task._preprocess(df)),"target":"PINCP > $50,000" if name=="acs_income" else "ESR == 1 (civilian employed at work)","selection":"Folktables 0.0.12 official task preprocessing. ACSIncome: AGEP>16, PINCP>100, WKHP>0, PWGTP>=1. ACSEmployment: no prefilter (includes minors/nonworking people); military ESR categories are negative under the official target. Alabama, 2018, 1-year person PUMS. Split by household SERIALNO. Unweighted person-record analysis; no US-population estimand."}

def split_rows(y, unit_id, name):
    ix=np.arange(len(y))
    if name.startswith("acs_"):
        train,tmp=next(GroupShuffleSplit(n_splits=1,test_size=.4,random_state=SEED).split(ix,y,unit_id))
        cal_rel,aud_rel=next(GroupShuffleSplit(n_splits=1,test_size=.5,random_state=SEED+1).split(tmp,y[tmp],unit_id[tmp]))
        cal,audit=tmp[cal_rel],tmp[aud_rel]
    else:
        train,tmp=train_test_split(ix,test_size=.4,stratify=y,random_state=SEED)
        cal,audit=train_test_split(tmp,test_size=.5,stratify=y[tmp],random_state=SEED+1)
    split=np.zeros(len(y),np.int8); split[cal]=1; split[audit]=2
    for p,q in itertools.combinations([train,cal,audit],2):
        assert len(np.intersect1d(p,q))==0
        if unit_id is not None:
            assert len(set(unit_id[p])&set(unit_id[q]))==0
    return split,train,cal,audit

def train_baselines(X,y,train,cal,audit):
    cat=[c for c in X if not pd.api.types.is_numeric_dtype(X[c])]
    num=[c for c in X if c not in cat]
    # Use ordinary Python object arrays for portable sklearn/pandas handling.
    for c in cat:
        X[c]=X[c].fillna("__MISSING__").astype(str).astype(object)
    num_pipe=Pipeline([("impute",SimpleImputer(strategy="median",keep_empty_features=True)),("scale",StandardScaler())])
    prep=ColumnTransformer([("num",num_pipe,num),("cat",OneHotEncoder(handle_unknown="ignore",min_frequency=10,sparse_output=True),cat)],sparse_threshold=1.0)
    t0=time.perf_counter()
    Xt=prep.fit_transform(X.iloc[train]); Xall=sparse.csr_matrix(prep.transform(X),dtype=np.float32)
    lr=LogisticRegression(C=1.0,max_iter=400,solver="lbfgs",random_state=SEED)
    lr.fit(Xt,y[train]); lr_prob=lr.predict_proba(Xall)[:,1]
    lr_seconds=time.perf_counter()-t0
    # Fit compact categorical vocabulary on train, and never use y for grouping.
    hgprep=ColumnTransformer([("num",SimpleImputer(strategy="median",keep_empty_features=True),num),("cat",OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1,min_frequency=10,max_categories=32),cat)])
    t0=time.perf_counter()
    Htrain=hgprep.fit_transform(X.iloc[train]); Hall=hgprep.transform(X)
    categorical=np.array([False]*len(num)+[True]*len(cat))
    hgb=HistGradientBoostingClassifier(learning_rate=.08,max_iter=100,max_leaf_nodes=15,max_depth=5,min_samples_leaf=30,l2_regularization=1.0,categorical_features=categorical,early_stopping=False,random_state=SEED)
    hgb.fit(Htrain,y[train]); hg_prob=hgb.predict_proba(Hall)[:,1]
    hg_seconds=time.perf_counter()-t0
    results={}; details={}
    for name,prob,seconds in [("lr",lr_prob,lr_seconds),("hgb",hg_prob,hg_seconds)]:
        # At most 20% empirical misses on calibration positives. Ties predict
        # positive, making calibration FNR conservative at the chosen order stat.
        scores=np.sort(prob[cal][y[cal]==1])
        if len(scores)==0:
            raise ValueError("No calibration positives")
        threshold=float(scores[int(np.floor(.2*len(scores)))])
        pred=(prob>=threshold).astype(np.int8)
        results[name+"_probability"]=prob.astype(np.float64)
        results[name+"_prediction"]=pred
        details[name]={"threshold":threshold,"threshold_rule":"calibration positive 20th-percentile order statistic, ties positive", "fit_seconds":seconds,"audit_auc":float(roc_auc_score(y[audit],prob[audit])),"audit_accuracy":float(accuracy_score(y[audit],pred[audit])),"calibration_fnr":float(np.mean(pred[cal][y[cal]==1]==0)),"audit_fnr":float(np.mean(pred[audit][y[audit]==1]==0)),"hyperparameters": {"C":1.,"max_iter":400,"solver":"lbfgs"} if name=="lr" else {"learning_rate":.08,"max_iter":100,"max_leaf_nodes":15,"max_depth":5,"min_samples_leaf":30,"l2_regularization":1.,"early_stopping":False,"categorical_max_categories":32}}
    return Xall,prep.get_feature_names_out().astype(str),results,details

def prepare_one(name,X,y,attrs,ids,units,description):
    print(f"Preparing {name}: {len(y)} records, {int(np.sum(y, dtype=np.int64))} positives",flush=True)
    if units is None: units=ids.copy()
    assert len(set(ids))==len(ids)
    split,train,cal,audit=split_rows(y,units,name)
    sex=sorted(attrs.sex.unique()); race=sorted(attrs.race.unique()); levels=[sex,race,AGE_LEVELS]
    groupcodes=np.array(list(itertools.product(*[range(len(v)) for v in levels])),np.int16)
    a=np.column_stack([attrs[c].map({label:i for i,label in enumerate(lev)}).to_numpy(dtype=np.int16) for c,lev in zip(["sex","race","age"],levels)])
    group_id=np.ravel_multi_index(a.T,tuple(map(len,levels))).astype(np.int16)
    groupkeys=np.array([f"sex={sex[s]}|race={race[r]}|age={AGE_LEVELS[b]}" for s,r,b in groupcodes])
    Xa,feature_names,predictions,models=train_baselines(X,y,train,cal,audit)
    sparse.save_npz(OUT/f"{name}_X.npz",Xa)
    np.savez_compressed(OUT/f"{name}.npz",y=y,a=a,group_id=group_id,groupcodes=groupcodes,groupkeys=groupkeys,id=np.asarray(ids,dtype=str),unit_id=np.asarray(units,dtype=str),split=split,feature_names=feature_names,**predictions)
    counts=[]
    for gi,key in enumerate(groupkeys):
        row={"dataset":name,"group_id":gi,"groupkey":key,**{c:levels[i][groupcodes[gi,i]] for i,c in enumerate(["sex","race","age"])}}
        for sid,sname in [(0,"train"),(1,"calibration"),(2,"audit")]:
            mask=(split==sid)&(group_id==gi); positives=mask&(y==1)
            row[sname+"_rows"]=int(mask.sum()); row[sname+"_positive_n"]=int(positives.sum())
            for model in ["lr","hgb"]:
                fn=int(((predictions[model+"_prediction"]==0)&positives).sum())
                row[sname+"_"+model+"_fn"]=fn
                row[sname+"_"+model+"_fnr"]=fn/int(positives.sum()) if positives.sum() else np.nan
        counts.append(row)
    pd.DataFrame(counts).to_csv(OUT/f"{name}_group_counts.csv",index=False)
    meta={"dataset":name,"created_utc":datetime.now(timezone.utc).isoformat(),"seed":SEED,"rows":len(y),"positives":int(np.sum(y, dtype=np.int64)),"split_rows":{s:int(sum(split==v)) for v,s in enumerate(["train","calibration","audit"])},"split_positive_n":{s:int(np.sum(y[split==v], dtype=np.int64)) for v,s in enumerate(["train","calibration","audit"])},"predictor_columns":list(X.columns),"excluded_sensitive_predictors":["sex/gender/SEX","race/RAC1P","age/AGEP"],"group_levels":dict(zip(["sex","race","age"],levels)),"group_count":len(groupkeys),"nonempty_audit_groups":int(len(np.unique(group_id[audit]))),"groups_with_audit_positives":int(len(np.unique(group_id[audit][y[audit]==1]))),"X_shape":list(Xa.shape),"X_nnz":int(Xa.nnz),"X_format":"CSR, transform fit on train only; row order matches main npz","group_n_definition":"Number of y=1 records in the group and relevant split (FNR denominator), not total group rows.","unit_split":"household SERIALNO" if name.startswith("acs_") else ("patient_nbr (deduplicated)" if name=="diabetes" else "record ID; public Adult has no original person identifier"),"models":models,**description}
    (OUT/f"{name}_metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print(json.dumps({"dataset":name,"rows":len(y),"splits":meta["split_rows"],"groups":len(groupkeys),"models":models}),flush=True)
    return meta

def provenance(metas):
    manifest=json.loads((RAW/"download_manifest.json").read_text())
    versions={p:importlib.metadata.version(p) for p in ["numpy","pandas","scipy","scikit-learn","folktables","requests"]}
    lines=["# Public-data provenance and frozen prediction protocol","", "All sources are public and require no login. Raw archives are retained unchanged. Download times are actual UTC times. A supplied archive is verified against the recorded SHA256 before preprocessing. These data support observational descriptions and covariate-based semi-synthetic experiments; the full observed sample is never a population ground truth.","", "## Raw sources","", "| File | Downloaded UTC | Bytes | SHA256 | Source |","|---|---|---:|---|---|"]
    for filename,m in manifest.items(): lines.append(f"| {filename} | {m['downloaded_utc']} | {m['bytes']} | `{m['sha256']}` | [official archive]({m['source_url']}) |")
    lines += ["","UCI documentation: [Adult](https://archive.ics.uci.edu/dataset/2/adult), DOI 10.24432/C5XW20; [Diabetes 130-US Hospitals](https://archive.ics.uci.edu/dataset/296/diabetes%2B130-us%2Bhospitals%2Bfor%2Byears%2B1999-2008), DOI 10.24432/C5230J. UCI pages state CC BY 4.0. ACS source: [Census 2018 1-year PUMS](https://www.census.gov/programs-surveys/acs/microdata/access/2018.html). Task definitions follow [Folktables](https://github.com/socialfoundations/folktables), package version 0.0.12 (MIT code). Raw ACS archive also contains ACS2018_PUMS_README.pdf.","","## Protocol fixed before audit","","- Random seed 20261022. Approximate 60% train, 20% calibration, 20% audit. Adult/Diabetes split stratified on y; ACS split by household SERIALNO. Disjoint row IDs and unit IDs are checked in code. Adult provides no original person identifier; its synthetic IDs identify records only. Sex/race/age groups are never used to train the predictor or choose its threshold.","- Intersectional groups use sex × race × age (<40, 40–69, 70+). All source race categories are retained; missing values become Unknown. Full Cartesian group universes retain empty cells. Diabetes decade intervals align exactly with these boundaries. ACS code RAC1P=5 is retained with the label AIAN_combined. Source categories differ between datasets and are not asserted to be exchangeable.","- `n_g` counts audit positives y=1. A cell without positives has undefined FNR, preserved as missing in CSV; it is not assigned an FNR of zero.","- Both fixed CPU predictors exclude sex/race/age and their code equivalents. Other proxies remain: omission of protected features does not guarantee fairness. Adult fnlwgt is excluded. Preprocessing fits train only; no audit-dependent vocabulary, imputation, feature selection, or tuning.","- LR: median-impute/standardize numerical features, train-fit one-hot categorical features with min_frequency=10; C=1, lbfgs, max_iter=400. HGB: train-fit median and ordinal categorical preprocessing, min_frequency=10, max_categories=32, native categorical handling, learning rate .08, 100 iterations, 15 leaves, depth 5, minimum leaf 30, L2=1, early_stopping=False. Random states fixed.","- Each predictor chooses a single threshold from calibration positive scores, the floor(0.2*n) zero-based order statistic; scores tied at threshold predict positive. This gives empirical calibration FNR ≤0.2, but is no guarantee on unseen groups. All audit probabilities and decisions are frozen.","- Diabetes keeps the minimum encounter_id record for each patient, then uniformly subsamples 50,000 people without using labels. This is a deterministic deduplication rule, not verified temporal ordering. Features represent information available by discharge, including disposition and treatments. No prospective clinical usefulness is claimed.","- ACS uses official Folktables tasks. Employment includes all ages with no prefilter. Income uses the package's adult_filter. The analysis is unweighted, and makes no claim to represent US-population rates. ACS audit records may remain dependent within households: group-disjoint train splits address train leakage, not within-audit sampling dependence. Exact binomial calibration for these observational records is not claimed; independent semi-synthetic losses can be generated conditional on covariates.","","## Prepared datasets","","| Dataset | Raw | Eligible | Used | Train | Calibration | Audit | Audit positives | Groups | Positive audit groups |","|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m in metas:
        s=m["split_rows"]; lines.append(f"| {m['dataset']} | {m['raw_records']} | {m['eligible_records']} | {m['rows']} | {s['train']} | {s['calibration']} | {s['audit']} | {m['split_positive_n']['audit']} | {m['group_count']} | {m['groups_with_audit_positives']} |")
    lines += ["","## Cache contract","", "`data/processed/{dataset}.npz`: `y` integer labels; `a` N×3 sensitive-category integer codes in sex/race/age order; `group_id` scalar cell ID per row; `groupcodes` K×3 code tuples; `groupkeys` K readable strings; `id` unique record ID; `unit_id` patient/person or household ID; `split` 0=train, 1=calibration, 2=audit; `feature_names`; `lr_probability`, `hgb_probability`, `lr_prediction`, `hgb_prediction`. No object arrays or pickle are required. `*_X.npz` is the train-fitted transformed CSR feature matrix in matching row order. `*_group_counts.csv` preserves every group, both row counts and positive counts, and model-specific false-negative counts for all splits. `*_metadata.json` records target, selection, predictor inputs, group levels, parameters, times and metrics. `audit_group_counts.csv` combines all datasets.","","Reproduce: `python src/prepare_data.py` from the project directory (offline once raw archives exist). Download only: `python scripts/download_data.py`. Environment: "+json.dumps(versions)+".",""]
    (ROOT/"DATA_PROVENANCE.md").write_text("\n".join(lines),encoding="utf-8")

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--dataset",choices=["adult","diabetes","acs_income","acs_employment","all"],default="all"); args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((RAW/"download_manifest.json").read_text())
    for filename,m in manifest.items():
        assert hashlib.sha256((RAW/filename).read_bytes()).hexdigest()==m["sha256"],f"Raw archive changed: {filename}"
    metas=[]
    for values in datasets():
        if args.dataset in ["all",values[0]]: metas.append(prepare_one(*values))
    all_metas=[json.loads(p.read_text()) for p in sorted(OUT.glob("*_metadata.json"))]
    provenance(all_metas)
    frames=[pd.read_csv(p) for p in sorted(OUT.glob("*_group_counts.csv")) if p.name!="audit_group_counts.csv"]
    pd.concat(frames,ignore_index=True).to_csv(OUT/"audit_group_counts.csv",index=False)

if __name__=="__main__": main()
