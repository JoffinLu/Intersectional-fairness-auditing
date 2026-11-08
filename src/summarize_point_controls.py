"""Exploratory diagnostic of retained point screens; no new outcomes or fits.

All seven stored synthetic baseline settings and all seven estimators are
reported on the same first 120 repetitions. This is an exploratory appendix
to the delivery, separate from the frozen manuscript and its numeric audit.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from artifacts import ROOT, now, sha256, write_json


def mcse(values):
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def main():
    source = ROOT / "results/newer_baselines_v1"
    rows, inputs = [], {}
    for i in range(7):
        paths = {"baseline": source / f"baseline_{i:03}.npz",
                 "scenario": source / f"scenario_{i:03}.npz",
                 "configuration": source / f"scenario_{i:03}.json"}
        for path in paths.values():
            inputs[path.relative_to(ROOT).as_posix()] = "sha256:" + sha256(path)
        cfg = json.loads(paths["configuration"].read_text(encoding="utf-8"))
        with np.load(paths["baseline"], allow_pickle=False) as baseline, np.load(paths["scenario"], allow_pickle=False) as scenario:
            n, p = baseline["n"], baseline["p"]
            np.testing.assert_array_equal(n, scenario["n"])
            np.testing.assert_array_equal(p, scenario["p"])
            ref_truth = (np.dot(p, n) - p * n) / (n.sum() - n)
            truth = p - ref_truth
            np.testing.assert_allclose(truth, scenario["true_delta"], rtol=0, atol=2e-16)
            harm = truth > .05 + 1e-12
            null = ~harm
            np.testing.assert_array_equal(harm, scenario["harmful"])
            count = len(baseline["estimate__raw"])
            assert count == 120
            np.testing.assert_allclose(baseline["estimate__raw"], scenario["k"][:count] / n, rtol=0, atol=0)
            screens = {}
            for key in baseline.files:
                if key.startswith("estimate__"):
                    estimates = baseline[key]
                    assert estimates.shape == (count, len(n)) and np.isfinite(estimates).all()
                    reference = (np.sum(estimates * n, axis=1, keepdims=True) - estimates * n) / (n.sum() - n)
                    screens[key.removeprefix("estimate__")] = estimates - reference > .05
            np.testing.assert_array_equal(screens["raw"], scenario["screen__raw"][:count])
            raw_false = np.any(screens["raw"][:, null], axis=1).astype(float)
            for method, screen in screens.items():
                estimate = baseline["estimate__" + method]
                false_screen = np.any(screen[:, null], axis=1).astype(float)
                null_average = screen[:, null].mean(axis=1)
                harmed_average = screen[:, harm].mean(axis=1) if harm.any() else None
                target = screen[:, cfg["target_group"]].astype(float)
                difference = false_screen - raw_false
                macro_loss = ((estimate - p) ** 2).mean(axis=1)
                target_loss = (estimate[:, cfg["target_group"]] - p[cfg["target_group"]]) ** 2
                rows.append({"scenario_id": cfg["scenario_id"], "family": cfg["family"],
                    "target_n": cfg["target_n"], "effect_above_margin": cfg["effect_above_margin"],
                    "method": method, "repetitions": count, "null_cells": int(null.sum()),
                    "harmful_cells": int(harm.sum()), "macro_mse": float(macro_loss.mean()),
                    "macro_mse_mcse": mcse(macro_loss), "target_mse": float(target_loss.mean()),
                    "target_mse_mcse": mcse(target_loss),
                    "target_is_harmful": bool(harm[cfg["target_group"]]),
                    "target_screen_frequency": float(target.mean()), "target_screen_mcse": mcse(target),
                    "any_null_screen_frequency": float(false_screen.mean()), "any_null_screen_mcse": mcse(false_screen),
                    "mean_null_cell_screen_frequency": float(null_average.mean()), "mean_null_cell_screen_mcse": mcse(null_average),
                    "mean_harmed_cell_screen_frequency": float(harmed_average.mean()) if harm.any() else None,
                    "mean_harmed_cell_screen_mcse": mcse(harmed_average) if harm.any() else None,
                    "any_null_screen_difference_vs_raw": float(difference.mean()),
                    "any_null_screen_difference_mcse": mcse(difference)})
    output = ROOT / "analysis/point_screen_controls.csv"
    frame = pd.DataFrame(rows)
    assert len(frame) == 49 and frame.groupby("scenario_id").size().eq(7).all()
    frame.to_csv(output, index=False)
    # A standalone diagnostic figure, deliberately outside the frozen paper.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    chosen = frame[(frame.scenario_id == "scenario_000") &
                   frame.method.isin(["raw", "sr_point_adapter", "suremap_author_adapter"])].set_index("method")
    methods = ["raw", "sr_point_adapter", "suremap_author_adapter"]
    labels = ["Raw", "SR point adapter", "SureMap point adapter"]
    def wilson_errors(rates):
        rates = np.asarray(rates)
        z, size = 1.959963984540054, 120
        center = (rates + z*z/(2*size))/(1 + z*z/size)
        half = z*np.sqrt(rates*(1-rates)/size + z*z/(4*size*size))/(1 + z*z/size)
        return np.vstack((rates-(center-half), (center+half)-rates))*100
    plt.rcParams.update({"font.size": 11, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    positions = np.arange(3)
    for offset, metric, color, label in [(-.19, "target_screen_frequency", "#346b9a", "Harmed target screened"),
            (.19, "any_null_screen_frequency", "#d28938", "At least one null cell screened")]:
        rates = chosen.loc[methods, metric].to_numpy()
        errors = wilson_errors(rates)
        bars = ax.bar(positions+offset, rates*100, .36, color=color, label=label,
                      yerr=errors, error_kw={"capsize": 3, "elinewidth": 1})
        for bar, rate, upper_error in zip(bars, rates, errors[1]):
            ax.text(bar.get_x()+bar.get_width()/2, rate*100+upper_error+2, f"{100*rate:.1f}",
                    ha="center", va="bottom", fontsize=10)
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Screen frequency (%)")
    ax.set_ylim(0, 119)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.set_title("Uncalibrated sensitivity and false screens", loc="left", fontsize=13, pad=12)
    fig.text(.12, .015, "32 cells; target n=20; excess=0.10; 120 paired repeats. Bars: pointwise 95% Wilson intervals.", fontsize=8.5)
    fig.tight_layout(rect=(0, .05, 1, 1))
    figure_paths = [ROOT / "analysis/point_screen_tradeoff.pdf", ROOT / "analysis/point_screen_tradeoff.png"]
    for path in figure_paths:
        fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    record = {"generated_at": now(), "analysis_status": "exploratory_not_prespecified",
              "new_datasets_generated": 0, "new_models_fitted": 0, "rows": len(rows),
              "settings": 7, "paired_repetitions_per_setting": 120,
              "raw_estimate_and_screen_crosschecks": "all seven settings exactly match retained full-run subset",
              "input_hashes": inputs, "output_sha256": sha256(output),
              "analysis_code_sha256": sha256(ROOT / "src/summarize_point_controls.py"),
              "figure_sha256": {p.relative_to(ROOT).as_posix():sha256(p) for p in figure_paths}}
    write_json(ROOT / "analysis/point_screen_controls.json", record)
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
