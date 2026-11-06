#!/usr/bin/env python3
"""Re-evaluate V2 without giving it an ensemble advantage over V3."""

from __future__ import annotations

from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_experiments import fit_predict, metric_row  # noqa: E402


RESULTS = ROOT / "results"
DATA_PATH = ROOT / "data/processed/site_daily_features.csv"
SITES = ["US-Ne2", "US-Ne3"]
YEARS = list(range(2001, 2014))
MODELS = ["climatology", "ridge", "random_forest"]


def main() -> None:
    data = pd.read_csv(DATA_PATH, parse_dates=["TIMESTAMP"])
    data = data[data["growing_season"] & (data["QC"] >= 0.75)].copy()
    tuning = pd.read_csv(RESULTS / "tuning_main.csv")
    best = {
        (row.site, int(row.outer_year), row.model): json.loads(row.best_params)
        for row in tuning.itertuples()
    }

    rows = []
    for source in SITES:
        target = SITES[1] if source == SITES[0] else SITES[0]
        for year in YEARS:
            test = data[(data["site"] == target) & (data["year"] == year)].copy()
            for excluded_year in [x for x in YEARS if x != year]:
                train = data[(data["site"] == source) & (data["year"] != excluded_year)].copy()
                for model in MODELS:
                    params = {} if model == "climatology" else best[(source, year, model)]
                    pred = fit_predict(model, params, train, test, seed=year * 100 + excluded_year)
                    rows.append(
                        {
                            "regime": "V2_same_year_cross",
                            "source_site": source,
                            "target_site": target,
                            "model": model,
                            "outer_year": year,
                            "excluded_source_year": excluded_year,
                            "n_train": len(train),
                            **metric_row(test["LE"].to_numpy(), pred),
                        }
                    )
        print(f"Corrected V2 complete: {source}", flush=True)

    variants = pd.DataFrame(rows)
    variants.to_csv(RESULTS / "metrics_v2_variants.csv", index=False)
    corrected = (
        variants.groupby(
            ["regime", "source_site", "target_site", "model", "outer_year"], as_index=False
        )
        .agg(
            n_train=("n_train", "mean"),
            n_test=("n_test", "first"),
            rmse=("rmse", "mean"),
            mae=("mae", "mean"),
            bias=("bias", "mean"),
            r2=("r2", "mean"),
            v2_variant_sd_rmse=("rmse", "std"),
        )
    )
    corrected["repeat"] = 0
    corrected.to_csv(RESULTS / "metrics_v2_corrected.csv", index=False)
    print(corrected.groupby(["target_site", "model"])["rmse"].median().to_string())


if __name__ == "__main__":
    main()
