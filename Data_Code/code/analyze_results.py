#!/usr/bin/env python3
"""Summarize the main validation audit with year-paired uncertainty."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
SEED = 20260911
N_BOOT = 10_000


def percentile_ci(values: np.ndarray, statistic=np.median) -> tuple[float, float, float]:
    """Return statistic and a year-resampling percentile interval."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(SEED)
    sampled = rng.choice(values, size=(N_BOOT, len(values)), replace=True)
    estimates = np.apply_along_axis(statistic, 1, sampled)
    return float(statistic(values)), *np.percentile(estimates, [2.5, 97.5]).tolist()


def metric_summary(frame: pd.DataFrame, value: str, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(group_cols, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        estimate, lo, hi = percentile_ci(group[value].to_numpy())
        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "metric": value,
                "median": estimate,
                "ci_low": lo,
                "ci_high": hi,
                "n_years": int(group["outer_year"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def rmse(observed: pd.Series, predicted: pd.Series) -> float:
    return float(np.sqrt(np.mean((predicted.to_numpy() - observed.to_numpy()) ** 2)))


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(RESULTS / "metrics_raw_main.csv")

    # Ten V0 repeats are reduced to a median within each target/model/year first.
    v0 = (
        metrics[metrics["regime"] == "V0_random_day"]
        .groupby(["target_site", "model", "outer_year"], as_index=False)
        .agg(
            rmse=("rmse", "median"),
            mae=("mae", "median"),
            bias=("bias", "median"),
            r2=("r2", "median"),
            n_test=("n_test", "median"),
        )
    )
    # Replace the original prediction-averaged V2 audit output with the corrected
    # average-of-variant-metrics comparator. This avoids an ensemble advantage.
    other = metrics[~metrics["regime"].isin(["V0_random_day", "V2_same_year_cross"])].copy()
    v2_corrected = pd.read_csv(RESULTS / "metrics_v2_corrected.csv")
    other = pd.concat([other, v2_corrected], ignore_index=True)
    annual = pd.concat([v0.assign(regime="V0_random_day"), other], ignore_index=True)
    annual.to_csv(TABLES / "yearly_validation_metrics.csv", index=False)

    summaries = pd.concat(
        [metric_summary(annual, metric, ["target_site", "model", "regime"])
         for metric in ["rmse", "mae", "bias", "r2"]],
        ignore_index=True,
    )
    summaries.to_csv(TABLES / "validation_summary.csv", index=False)

    wide = annual.pivot_table(
        index=["target_site", "model", "outer_year"], columns="regime", values="rmse"
    ).reset_index()
    wide["random_split_optimism"] = wide["V1_within_year"] - wide["V0_random_day"]
    wide["strict_transfer_penalty"] = wide["V3_strict_cross"] - wide["V1_within_year"]
    wide["same_year_transfer_optimism"] = wide["V3_strict_cross"] - wide["V2_same_year_cross"]
    wide.to_csv(TABLES / "yearly_penalties.csv", index=False)

    penalty_rows = []
    for (site, model), group in wide.groupby(["target_site", "model"], sort=True):
        for effect in [
            "random_split_optimism",
            "strict_transfer_penalty",
            "same_year_transfer_optimism",
        ]:
            estimate, lo, hi = percentile_ci(group[effect].to_numpy())
            penalty_rows.append(
                {
                    "target_site": site,
                    "model": model,
                    "effect": effect,
                    "median_rmse_difference": estimate,
                    "ci_low": lo,
                    "ci_high": hi,
                    "positive_years": int((group[effect] > 0).sum()),
                    "n_years": int(len(group)),
                }
            )
    penalties = pd.DataFrame(penalty_rows)
    penalties.to_csv(TABLES / "penalty_summary.csv", index=False)

    # RQ3 diagnostic: thresholds are calculated only from each strict source-training fold.
    predictions = pd.read_csv(RESULTS / "predictions_main.csv", parse_dates=["TIMESTAMP"])
    features = pd.read_csv(ROOT / "data/processed/site_daily_features.csv", parse_dates=["TIMESTAMP"])
    main = features[features["growing_season"] & (features["QC"] >= 0.75)].copy()
    v3 = predictions[predictions["regime"] == "V3_strict_cross"].copy()
    threshold_rows = []
    for source in sorted(v3["source_site"].unique()):
        for year in sorted(v3["outer_year"].unique()):
            train = main[(main["site"] == source) & (main["year"] != year)]
            threshold_rows.append(
                {
                    "source_site": source,
                    "outer_year": year,
                    "vpd_upper_tercile": float(train["VPD"].quantile(2 / 3)),
                }
            )
    thresholds = pd.DataFrame(threshold_rows)
    v3 = v3.merge(thresholds, on=["source_site", "outer_year"], how="left", validate="many_to_one")
    v3["vpd_state"] = np.where(v3["VPD"] >= v3["vpd_upper_tercile"], "high_vpd", "lower_vpd")
    v3["canopy_state"] = np.where(v3["NDVI"] >= 0.40, "active_canopy", "lower_ndvi")
    v3["diagnostic_cell"] = v3["vpd_state"] + "__" + v3["canopy_state"]

    diag_rows = []
    keys = ["source_site", "target_site", "model", "outer_year", "diagnostic_cell"]
    for key, group in v3.groupby(keys, sort=True):
        if len(group) < 3:
            continue
        diag_rows.append(
            {
                **dict(zip(keys, key)),
                "n_days": len(group),
                "rmse": rmse(group["observed_LE"], group["predicted_LE"]),
                "bias": float((group["predicted_LE"] - group["observed_LE"]).mean()),
            }
        )
    diagnostics = pd.DataFrame(diag_rows)
    diagnostics.to_csv(TABLES / "diagnostic_by_year.csv", index=False)

    diag_summaries = []
    for metric in ["rmse", "bias"]:
        diag_summaries.append(
            metric_summary(
                diagnostics,
                metric,
                ["source_site", "target_site", "model", "diagnostic_cell"],
            )
        )
    pd.concat(diag_summaries, ignore_index=True).to_csv(
        TABLES / "diagnostic_summary.csv", index=False
    )

    # Explicit directional check: high-VPD active canopy versus all remaining days.
    contrast_rows = []
    for key, group in v3.groupby(["source_site", "target_site", "model", "outer_year"], sort=True):
        high_active = group[group["diagnostic_cell"] == "high_vpd__active_canopy"]
        other_days = group[group["diagnostic_cell"] != "high_vpd__active_canopy"]
        if len(high_active) < 3 or len(other_days) < 3:
            continue
        high_bias = float((high_active["predicted_LE"] - high_active["observed_LE"]).mean())
        other_bias = float((other_days["predicted_LE"] - other_days["observed_LE"]).mean())
        contrast_rows.append(
            {
                "source_site": key[0],
                "target_site": key[1],
                "model": key[2],
                "outer_year": key[3],
                "n_high_active": len(high_active),
                "bias_high_active": high_bias,
                "bias_other_days": other_bias,
                "bias_contrast_high_minus_other": high_bias - other_bias,
            }
        )
    contrasts = pd.DataFrame(contrast_rows)
    contrasts.to_csv(TABLES / "directional_diagnostic_by_year.csv", index=False)
    contrast_summary = []
    for key, group in contrasts.groupby(["source_site", "target_site", "model"], sort=True):
        estimate, lo, hi = percentile_ci(group["bias_contrast_high_minus_other"].to_numpy())
        contrast_summary.append(
            {
                "source_site": key[0],
                "target_site": key[1],
                "model": key[2],
                "median_bias_contrast": estimate,
                "ci_low": lo,
                "ci_high": hi,
                "negative_years": int((group["bias_contrast_high_minus_other"] < 0).sum()),
                "n_years": len(group),
            }
        )
    pd.DataFrame(contrast_summary).to_csv(TABLES / "directional_diagnostic_summary.csv", index=False)

    print(penalties.to_string(index=False))
    print("\nDirectional diagnostic")
    print(pd.DataFrame(contrast_summary).to_string(index=False))


if __name__ == "__main__":
    main()
