#!/usr/bin/env python3
"""Final reproducibility and leakage-invariant checks for generated outputs."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from run_experiments import FEATURES, RANDOM_SEEDS, SITES, YEARS


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main() -> None:
    checks: list[dict] = []

    def check(name: str, condition: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(condition), "detail": detail})
        if not condition:
            raise AssertionError(f"{name}: {detail}")

    data = pd.read_csv(ROOT / "data/processed/site_daily_features.csv")
    main = data[data["growing_season"] & (data["QC"] >= 0.75)]
    check("target_not_in_features", "LE" not in FEATURES and "QC" not in FEATURES, str(FEATURES))
    check("two_sites", set(main["site"]) == set(SITES), str(sorted(main["site"].unique())))
    check("thirteen_years", set(main["year"]) == set(YEARS), f"{main.year.min()}-{main.year.max()}")
    check("finite_main_inputs", np.isfinite(main[FEATURES + ["LE"]].to_numpy()).all(), f"n={len(main)}")

    metrics = pd.read_csv(RESULTS / "metrics_raw_main.csv")
    v0 = metrics[metrics["regime"] == "V0_random_day"]
    check("v0_ten_repeats", v0.groupby(["target_site", "model", "outer_year"])["repeat"].nunique().eq(len(RANDOM_SEEDS)).all(), "10 per site/model/year")
    v1 = metrics[metrics["regime"] == "V1_within_year"]
    v3 = metrics[metrics["regime"] == "V3_strict_cross"]
    check("v1_same_site", (v1["source_site"] == v1["target_site"]).all(), "source equals target")
    check("v3_other_site", (v3["source_site"] != v3["target_site"]).all(), "source differs from target")
    check("v1_v3_complete_grid", len(v1) == len(v3) == 2 * 13 * 3, f"V1={len(v1)}, V3={len(v3)}")

    v2 = pd.read_csv(RESULTS / "metrics_v2_variants.csv")
    check("v2_includes_target_year", (v2["excluded_source_year"] != v2["outer_year"]).all(), "excluded year never equals test year")
    check("v2_twelve_variants", v2.groupby(["source_site", "target_site", "model", "outer_year"])["excluded_source_year"].nunique().eq(12).all(), "12 per direction/model/year")

    expected = [
        RESULTS / "tables/penalty_summary.csv",
        RESULTS / "tables/sensitivity_summary.csv",
        RESULTS / "tables/directional_diagnostic_summary.csv",
        RESULTS / "figures/figure1_validation_design.pdf",
        RESULTS / "figures/figure2_main_results.pdf",
        RESULTS / "figures/figure3_directional_diagnostic.pdf",
    ]
    check("required_outputs_exist", all(path.exists() and path.stat().st_size > 0 for path in expected), ", ".join(path.name for path in expected))

    report = {"status": "PASS", "n_checks": len(checks), "checks": checks}
    (RESULTS / "verification_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
