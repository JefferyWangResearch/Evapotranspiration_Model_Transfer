#!/usr/bin/env python3
"""Run prespecified cohort sensitivities for within-site and strict transfer tests."""

from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_experiments import fit_predict, metric_row, tune  # noqa: E402


DATA_PATH = ROOT / "data/processed/site_daily_features.csv"
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
SITES = ["US-Ne2", "US-Ne3"]
MODELS = ["climatology", "ridge", "random_forest"]
N_BOOT = 10_000
SEED = 20260912


def ci(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(SEED)
    samples = rng.choice(values, size=(N_BOOT, len(values)), replace=True)
    estimates = np.median(samples, axis=1)
    return float(np.median(values)), *np.percentile(estimates, [2.5, 97.5]).tolist()


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_PATH, parse_dates=["TIMESTAMP"])
    cohorts = {
        "growing_qc050": data["growing_season"] & (data["QC"] >= 0.50),
        "growing_qc090": data["growing_season"] & (data["QC"] >= 0.90),
        "all_year_qc075": data["QC"] >= 0.75,
        "crop_matched_growing_qc075": data["growing_season"] & (data["QC"] >= 0.75) & data["crop_matched"],
        "exclude2013_growing_qc075": data["growing_season"] & (data["QC"] >= 0.75) & (data["year"] != 2013),
    }

    metric_rows = []
    tuning_rows = []
    for cohort_name, mask in cohorts.items():
        cohort = data[mask].copy()
        years = sorted(cohort.groupby("year")["site"].nunique().loc[lambda x: x == 2].index.tolist())
        for source in SITES:
            other = SITES[1] if source == SITES[0] else SITES[0]
            for year in years:
                train = cohort[(cohort["site"] == source) & (cohort["year"] != year)].copy()
                if train["year"].nunique() < 4:
                    continue
                tuned = {"climatology": {}}
                for model in ["ridge", "random_forest"]:
                    params, inner_rmse = tune(model, train, year)
                    tuned[model] = params
                    tuning_rows.append(
                        {
                            "cohort": cohort_name,
                            "source_site": source,
                            "outer_year": year,
                            "model": model,
                            "n_train": len(train),
                            "best_params": json.dumps(params, sort_keys=True),
                            "inner_group_rmse": inner_rmse,
                        }
                    )
                for model in MODELS:
                    for target, regime in [(source, "V1_within_year"), (other, "V3_strict_cross")]:
                        test = cohort[(cohort["site"] == target) & (cohort["year"] == year)].copy()
                        if len(test) < 3:
                            continue
                        pred = fit_predict(model, tuned[model], train, test, seed=year + 29)
                        metric_rows.append(
                            {
                                "cohort": cohort_name,
                                "regime": regime,
                                "source_site": source,
                                "target_site": target,
                                "model": model,
                                "outer_year": year,
                                "n_train": len(train),
                                **metric_row(test["LE"].to_numpy(), pred),
                            }
                        )
        print(f"Sensitivity complete: {cohort_name}", flush=True)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(RESULTS / "metrics_sensitivity.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(RESULTS / "tuning_sensitivity.csv", index=False)

    # Within target/year: other-site V3 minus target-site V1.
    wide = metrics.pivot_table(
        index=["cohort", "target_site", "model", "outer_year"],
        columns="regime",
        values="rmse",
    ).reset_index()
    wide["strict_transfer_penalty"] = wide["V3_strict_cross"] - wide["V1_within_year"]
    wide.to_csv(TABLES / "sensitivity_yearly_penalties.csv", index=False)

    rows = []
    for key, group in wide.groupby(["cohort", "target_site", "model"], sort=True):
        estimate, lo, hi = ci(group["strict_transfer_penalty"].to_numpy())
        rows.append(
            {
                "cohort": key[0],
                "target_site": key[1],
                "model": key[2],
                "median_strict_transfer_penalty": estimate,
                "ci_low": lo,
                "ci_high": hi,
                "positive_years": int((group["strict_transfer_penalty"] > 0).sum()),
                "n_years": len(group),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(TABLES / "sensitivity_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
