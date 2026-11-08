"""Frozen, supplemental support/target-size sensitivity; no main-run edits.

Prepare first (no random outcomes), freeze the code/configurations, then
run. All methods reuse evaluate_counts and its paired full/split counts.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from artifacts import ROOT, now, sha256, write_json
from audit_methods import true_disparities, weighted_reference
from run_simulation import evaluate_counts, summarize


OUT = ROOT / "results/support_sensitivity_v1"
PLAN = ROOT / "protocols/support_sensitivity.md"
TASKS = ("adult", "diabetes", "acs_income", "acs_employment")
MODELS = ("lr", "hgb")
COUNT_GEOMETRIES = ("capped_1000", "uncapped_rounded")
TARGET_RULES = ("smallest_eligible", "smallest_at_least_80")
SEED = 20261024
REPETITIONS = 1000
CONFIRMATION, FLOOR, ALPHA, MARGIN = .8, .1, .05, .05


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def frozen_inputs():
    """Hash actual NPZ inputs, not only their metadata sidecars."""
    code = ("run_support_sensitivity.py", "run_simulation.py",
            "audit_methods.py", "artifacts.py", "run_extended.py")
    data = [ROOT / "data/processed" / f"{task}{suffix}"
            for task in TASKS for suffix in (".npz", "_metadata.json", "_group_counts.csv")]
    data.append(ROOT / "data/processed/validation_report.json")
    return {
        "code_sha256": {f"src/{name}": sha256(ROOT / "src" / name) for name in code},
        "data_sha256": {path.relative_to(ROOT).as_posix(): sha256(path) for path in data},
        "protocol_sha256": sha256(PLAN),
    }


def select_target(n, rule):
    candidates = np.arange(len(n)) if rule == "smallest_eligible" else np.flatnonzero(n >= 80)
    if not len(candidates):
        raise ValueError(f"No target satisfies prespecified rule {rule}.")
    # n and candidates retain original cache order; argmin breaks ties first.
    return int(candidates[np.argmin(n[candidates])])


def prepare():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite frozen configuration directory: {OUT}")
    prepared_at = now()
    hashes = frozen_inputs()
    configurations, template_rows, eligibility_rows = [], [], []
    seeds = np.random.SeedSequence(SEED).spawn(32)
    for task in TASKS:
        with np.load(ROOT / "data/processed" / f"{task}.npz", allow_pickle=False) as cache:
            group, y, split = cache["group_id"], cache["y"], cache["split"]
            codes, keys = cache["groupcodes"], cache["groupkeys"]
            g = len(codes)
            audit = (split == 2) & (y == 1)
            calibration = (split == 1) & (y == 1)
            audit_n = np.bincount(group[audit], minlength=g)
            cal_n = np.bincount(group[calibration], minlength=g)
            eligible = (audit_n >= 10) & (cal_n >= 5)
            original = np.flatnonzero(eligible)
            if len(original) < 3:
                raise ValueError(f"Insufficient eligible family for {task}.")
            # Uncapped counts derive directly from original audit counts.
            rounded = (audit_n[eligible] // 10) * 10
            for j in range(g):
                eligibility_rows.append({"dataset": task, "original_group": j,
                    "key": str(keys[j]), "positive_audit_count": int(audit_n[j]),
                    "positive_calibration_count": int(cal_n[j]), "eligible": bool(eligible[j])})
            for model in MODELS:
                false_negative = cache[f"{model}_prediction"] == 0
                cal_k = np.bincount(group[calibration & false_negative], minlength=g)
                baseline = np.clip((cal_k[eligible] + .5) / (cal_n[eligible] + 1), .02, .8)
                for geometry in COUNT_GEOMETRIES:
                    n = np.minimum(rounded, 1000) if geometry == "capped_1000" else rounded.copy()
                    if np.any(n < 10) or np.any(n % 10):
                        raise ValueError("Every transformed count must be a positive multiple of ten.")
                    for rule in TARGET_RULES:
                        target = select_target(n, rule)
                        reference = float(weighted_reference(baseline, n)[target])
                        p = baseline.copy()
                        p[target] = min(.95, reference + .25)
                        truth = true_disparities(p, n)
                        if not np.all(np.isfinite(truth)) or not truth[target] > MARGIN + 1e-12:
                            raise ValueError("Injected target must have a finite harmful disparity.")
                        i = len(configurations)
                        cfg = {
                            "scenario_id": f"scenario_{i:03}",
                            "family": "real_support_sensitivity", "dataset": task, "model": model,
                            "geometry": "calibration_heterogeneous", "count_geometry": geometry,
                            "target_rule": rule, "groups": len(n), "target_group": target,
                            "target_original_group": int(original[target]), "target_key": str(keys[original[target]]),
                            "target_n": int(n[target]), "total_n": int(n.sum()),
                            "target_original_audit_n": int(audit_n[original[target]]),
                            "target_calibration_n": int(cal_n[original[target]]),
                            "target_template_reference": reference, "target_probability": float(p[target]),
                            "injection_capped": bool(reference + .25 > .95),
                            "target_true_delta": float(truth[target]),
                            "effect_above_margin": float(truth[target] - MARGIN),
                            "harmful_groups": int((truth > MARGIN + 1e-12).sum()),
                            "confirmation": CONFIRMATION, "floor": FLOOR, "alpha": ALPHA,
                            "margin": MARGIN, "eta": ALPHA * .05 / (20 * len(n)),
                            "planned_repetitions": REPETITIONS, "seed": SEED,
                            "seed_spawn_key": list(seeds[i].spawn_key),
                        }
                        configurations.append({"config": cfg, "p": p.tolist(), "n": n.tolist(),
                            "baseline_p": baseline.tolist(), "true_delta": truth.tolist(),
                            "original_groups": original.tolist(), "group_codes": codes[eligible].tolist()})
                        for j, orig in enumerate(original):
                            template_rows.append({**cfg, "group": j, "original_group": int(orig),
                                "key": str(keys[orig]), "is_target": j == target,
                                "positive_audit_count": int(audit_n[orig]),
                                "positive_calibration_count": int(cal_n[orig]),
                                "calibration_false_negatives": int(cal_k[orig]),
                                "rounded_count_before_cap": int(rounded[j]), "simulation_count": int(n[j]),
                                "calibration_error_rate_template": float(baseline[j]),
                                "true_simulation_p": float(p[j]), "true_group_delta": float(truth[j])})
    expected = {(task, model, geometry, rule) for task in TASKS for model in MODELS
                for geometry in COUNT_GEOMETRIES for rule in TARGET_RULES}
    actual = {(a["config"]["dataset"], a["config"]["model"], a["config"]["count_geometry"],
               a["config"]["target_rule"]) for a in configurations}
    if len(configurations) != 32 or actual != expected:
        raise ValueError("The frozen grid must contain exactly the 32 prespecified configurations.")
    if hashes != frozen_inputs():
        raise RuntimeError("Source or data changed during configuration preparation.")
    OUT.mkdir(parents=True)
    write_json(OUT / "predeclared_configurations.json", configurations)
    pd.DataFrame([a["config"] for a in configurations]).to_csv(OUT / "configuration_summary.csv", index=False)
    pd.DataFrame(template_rows).to_csv(OUT / "templates.csv", index=False)
    pd.DataFrame(eligibility_rows).to_csv(OUT / "support_eligibility.csv", index=False)
    for item in configurations:
        write_json(OUT / f"{item['config']['scenario_id']}.json", item["config"])
    artifacts = ["predeclared_configurations.json", "configuration_summary.csv", "templates.csv",
                 "support_eligibility.csv"] + [f"scenario_{i:03}.json" for i in range(32)]
    write_json(OUT / "prelaunch_metadata.json", {
        "status": "prepared_no_generated_outcomes", "prepared_at": prepared_at,
        "configuration_count": len(configurations), "seed": SEED, "repetitions": REPETITIONS,
        "confirmation": CONFIRMATION, "floor": FLOOR, "alpha": ALPHA, "margin": MARGIN,
        "stream_rule": "SeedSequence(20261024).spawn(32), frozen configuration order",
        "scope": "Known truths of generated independent Bernoulli errors conditional on fixed templates only.",
        **hashes, "frozen_artifact_sha256": {name: sha256(OUT / name) for name in artifacts},
    })
    print(f"Prepared exactly 32 configurations at {OUT}; no random outcomes generated.", flush=True)


def wilson(events):
    events = np.asarray(events, dtype=float)
    n, phat, z = len(events), float(events.mean()), 1.959963984540054
    denominator = 1 + z*z/n
    center = (phat + z*z/(2*n))/denominator
    half = z*np.sqrt(phat*(1-phat)/n + z*z/(4*n*n))/denominator
    return max(0., float(center-half)), min(1., float(center+half))


def fwer_events(flags, null):
    return np.any(flags[:, null], axis=1).astype(float)


def summarize_with_intervals(result, cfg):
    rows = summarize(result, cfg)
    for row in rows:
        if row["kind"] not in ("flag", "screen"):
            continue
        values = result[f"{row['kind']}__{row['method']}"]
        target = values[:, cfg["target_group"]].astype(float)
        false_flags = fwer_events(values, result["null"])
        row.update(target_events=int(target.sum()), false_flag_family_events=int(false_flags.sum()),
                   null_groups=int(result["null"].sum()),
                   target_wilson_lower=wilson(target)[0], target_wilson_upper=wilson(target)[1],
                   fwer_wilson_lower=wilson(false_flags)[0], fwer_wilson_upper=wilson(false_flags)[1],
                   inferential_status="calibrated_flag" if row["kind"] == "flag" else "uncalibrated_screen")
    return rows


def paired_comparisons(result, cfg):
    target, null = cfg["target_group"], result["null"]
    full = result["flag__full_hb_holm"]
    full_target = full[:, target].astype(float)
    full_fwer = fwer_events(full, null)
    rows = []
    for key, flags in result.items():
        if not key.startswith("flag__") or key == "flag__full_hb_holm":
            continue
        other_target = flags[:, target].astype(float)
        other_fwer = fwer_events(flags, null)
        power_difference, fwer_difference = other_target-full_target, other_fwer-full_fwer
        rows.append({**cfg, "reference_method": "full_hb_holm", "method": key.split("__", 1)[1],
            "repetitions": len(flags), "full_power": float(full_target.mean()),
            "method_power": float(other_target.mean()),
            "paired_difference": float(power_difference.mean()),
            "paired_mcse": float(power_difference.std(ddof=1)/np.sqrt(len(flags))),
            "full_fwer": float(full_fwer.mean()), "method_fwer": float(other_fwer.mean()),
            "fwer_paired_difference": float(fwer_difference.mean()),
            "fwer_paired_mcse": float(fwer_difference.std(ddof=1)/np.sqrt(len(flags)))})
    return rows


def run():
    frozen = read_json(OUT / "prelaunch_metadata.json")
    if frozen["status"] != "prepared_no_generated_outcomes":
        raise RuntimeError("Expected a prepared protocol with no generated outcomes.")
    if (OUT / "run_metadata.json").exists() or list(OUT.glob("scenario_*.npz")):
        raise FileExistsError("Refusing to restart or overwrite a started run.")
    for key, value in frozen_inputs().items():
        if frozen[key] != value:
            raise RuntimeError(f"Frozen input hash changed: {key}. Do not execute a changed protocol.")
    for name, digest in frozen["frozen_artifact_sha256"].items():
        if sha256(OUT / name) != digest:
            raise RuntimeError(f"Frozen configuration artifact changed: {name}.")
    configurations = read_json(OUT / "predeclared_configurations.json")
    if len(configurations) != 32 or frozen["seed"] != SEED or frozen["repetitions"] != REPETITIONS:
        raise RuntimeError("Frozen configuration count/seed/repetitions disagree with the protocol.")
    start = time.perf_counter()
    metadata = {**frozen, "status": "running", "started_at": now(), "gpu_hours": 0,
                "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                "scipy": scipy.__version__, "completed_configurations": 0, "actual_repetitions": 0}
    write_json(OUT / "run_metadata.json", metadata)
    seeds = np.random.SeedSequence(SEED).spawn(32)
    rows, paired_rows, runtime_rows = [], [], []
    try:
        for item, seed in zip(configurations, seeds):
            cfg = item["config"]
            if cfg["seed_spawn_key"] != list(seed.spawn_key):
                raise RuntimeError("Configuration seed order differs from the frozen declaration.")
            begin = time.perf_counter()
            p, n = np.asarray(item["p"], float), np.asarray(item["n"], int)
            result = evaluate_counts(p, n, REPETITIONS, np.random.default_rng(seed),
                confirmation=CONFIRMATION, floor=FLOOR, alpha=ALPHA, margin=MARGIN)
            if not np.array_equal(result["true_delta"], np.asarray(item["true_delta"], float)):
                raise RuntimeError("Generated experiment truth differs from frozen declaration.")
            if not np.array_equal(result["n_discovery"] + result["n_confirmation"], n):
                raise RuntimeError("Full and split total data budgets differ.")
            if not np.allclose(result["true_delta"], true_disparities(p, result["n_confirmation"]),
                               rtol=0, atol=1e-12):
                raise RuntimeError("Full and confirmation complement targets differ.")
            file = OUT / f"{cfg['scenario_id']}.npz"
            np.savez_compressed(file, **result, group_codes=np.asarray(item["group_codes"], int),
                                original_groups=np.asarray(item["original_groups"], int),
                                baseline_p=np.asarray(item["baseline_p"], float))
            rows.extend(summarize_with_intervals(result, cfg))
            paired_rows.extend(paired_comparisons(result, cfg))
            runtime_rows.append({"scenario_id": cfg["scenario_id"], "repetitions": len(result["k"]),
                "elapsed_seconds": time.perf_counter()-begin, "raw_npz_sha256": sha256(file)})
            metadata.update(completed_configurations=len(runtime_rows),
                            actual_repetitions=sum(row["repetitions"] for row in runtime_rows),
                            elapsed_seconds=time.perf_counter()-start)
            write_json(OUT / "run_metadata.json", metadata)
            print(f"[{len(runtime_rows)}/32] {cfg['dataset']} {cfg['model']} "
                  f"{cfg['count_geometry']} {cfg['target_rule']} n*={cfg['target_n']} "
                  f"elapsed={time.perf_counter()-start:.1f}s", flush=True)
        if frozen_inputs() != {key: frozen[key] for key in frozen_inputs()}:
            raise RuntimeError("Source or data changed while the run was in progress.")
        pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
        pd.DataFrame(paired_rows).to_csv(OUT / "paired_comparisons.csv", index=False)
        pd.DataFrame(runtime_rows).to_csv(OUT / "scenario_runtime.csv", index=False)
        metadata.update(status="completed", completed_at=now(), elapsed_seconds=time.perf_counter()-start,
                        raw_npz_files=len(runtime_rows), summary_rows=len(rows), paired_rows=len(paired_rows),
                        scenario_runtime=runtime_rows)
        metadata["summary_sha256"] = {name: sha256(OUT/name) for name in
                                    ("summary.csv", "paired_comparisons.csv", "scenario_runtime.csv")}
        write_json(OUT / "run_metadata.json", metadata)
    except Exception as error:
        metadata.update(status="failed_preserved", failed_at=now(), elapsed_seconds=time.perf_counter()-start,
                        error=f"{type(error).__name__}: {error}", scenario_runtime=runtime_rows)
        write_json(OUT / "run_metadata.json", metadata)
        raise
    print(json.dumps({key: metadata[key] for key in
        ("status", "actual_repetitions", "raw_npz_files", "elapsed_seconds")}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true", help="Freeze all 32 configurations without simulation.")
    mode.add_argument("--run", action="store_true", help="Execute the frozen protocol after code review.")
    parser.add_argument("--name", default="support_sensitivity_v1", help="Choose a fresh results directory for a replication.")
    args = parser.parse_args()
    if not args.name or Path(args.name).name != args.name:
        parser.error("--name must be a directory name, not a path")
    OUT = ROOT / "results" / args.name
    prepare() if args.prepare else run()
