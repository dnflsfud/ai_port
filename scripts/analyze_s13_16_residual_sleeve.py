#!/usr/bin/env python
"""Independent arithmetic checks and compact report data for S13.16.

The script only reads the already-completed formal A/B/C experiment.  It does
not refit the model, alter the score, change its sign, or search a risk budget.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


METRIC_KEYS = (
    "information_ratio",
    "active_return",
    "tracking_error",
    "avg_annual_turnover",
    "realized_beta",
    "max_drawdown",
)
PERIOD_KEYS = ("P1_ir", "P2_ir", "P3_ir")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _base_metrics(payload: dict) -> dict:
    return payload.get("metrics", payload)


def _close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(left), float(right), abs_tol=tolerance, rel_tol=0.0)


def _check_delta(
    higher: dict, lower: dict, stored: dict, *, label: str
) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for key in METRIC_KEYS:
        checks[f"{label}.{key}"] = _close(
            float(higher[key]) - float(lower[key]), float(stored[key])
        )
    for key in PERIOD_KEYS:
        checks[f"{label}.sub_periods.{key}"] = _close(
            float(higher["sub_periods"][key])
            - float(lower["sub_periods"][key]),
            float(stored["sub_periods"][key]),
        )
    return checks


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows supplied for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("outputs/arm_s13_16_residual_nonlinear_sleeve/metrics.json"),
    )
    parser.add_argument(
        "--baseline-metrics",
        type=Path,
        default=Path("outputs/codex_causal_rank_65/metrics.json"),
    )
    parser.add_argument(
        "--inventory", type=Path, default=Path("experiment_inventory.json")
    )
    parser.add_argument(
        "--selection-bias-csv",
        type=Path,
        default=Path("outputs/csv/selection_bias_metrics.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/arm_s13_16_residual_nonlinear_sleeve"),
    )
    args = parser.parse_args()

    payload = _read_json(args.metrics)
    production_payload = _read_json(args.baseline_metrics)
    inventory = _read_json(args.inventory)
    metrics = _base_metrics(payload)
    baseline = _base_metrics(production_payload)
    attribution = metrics["residual_sleeve_attribution"]

    a = attribution["A_production"]
    b = attribution["B_reserve_only"]
    c = attribution["C_residual_sleeve"]
    b_minus_a = attribution["B_minus_A_reserve_cost"]
    c_minus_b = attribution["C_minus_B_pure_sleeve"]
    c_minus_a = attribution["C_minus_A_net"]
    diagnostics = attribution["diagnostics"]

    checks: dict[str, bool] = {}
    for key in METRIC_KEYS:
        checks[f"production_parity.{key}"] = _close(a[key], baseline[key])
    for key in PERIOD_KEYS:
        checks[f"production_parity.sub_periods.{key}"] = _close(
            a["sub_periods"][key], baseline["sub_periods"][key]
        )
    checks.update(_check_delta(b, a, b_minus_a, label="B_minus_A"))
    checks.update(_check_delta(c, b, c_minus_b, label="C_minus_B"))
    checks.update(_check_delta(c, a, c_minus_a, label="C_minus_A"))

    arm_specs = (
        (1, "A", "Production", a),
        (2, "B", "Reserve only", b),
        (3, "C", "Residual sleeve", c),
    )
    arm_rows: list[dict] = []
    for order, arm, role, values in arm_specs:
        arm_rows.append(
            {
                "display_order": order,
                "arm": arm,
                "role": role,
                "information_ratio": values["information_ratio"],
                "delta_ir_vs_a": values["information_ratio"]
                - a["information_ratio"],
                "annual_active_return": values["active_return"],
                "delta_active_return_vs_a": values["active_return"]
                - a["active_return"],
                "tracking_error": values["tracking_error"],
                "annual_turnover": values["avg_annual_turnover"],
                "realized_beta": values["realized_beta"],
                "P1_ir": values["sub_periods"]["P1_ir"],
                "P2_ir": values["sub_periods"]["P2_ir"],
                "P3_ir": values["sub_periods"]["P3_ir"],
                "trial_inventory_n": int(inventory["n_trials_total"]),
            }
        )

    period_rows: list[dict] = []
    for period_label, period_key in (
        ("Full", "information_ratio"),
        ("P1", "P1_ir"),
        ("P2", "P2_ir"),
        ("P3", "P3_ir"),
    ):
        for order, arm, role, values in arm_specs:
            ir = (
                values[period_key]
                if period_label == "Full"
                else values["sub_periods"][period_key]
            )
            a_ir = (
                a[period_key]
                if period_label == "Full"
                else a["sub_periods"][period_key]
            )
            period_rows.append(
                {
                    "period_order": ("Full", "P1", "P2", "P3").index(
                        period_label
                    )
                    + 1,
                    "period": period_label,
                    "arm_order": order,
                    "arm": arm,
                    "role": role,
                    "information_ratio": ir,
                    "delta_ir_vs_a": ir - a_ir,
                    "annual_active_return": values["active_return"],
                    "tracking_error": values["tracking_error"],
                    "annual_turnover": values["avg_annual_turnover"],
                }
            )

    risk = diagnostics["post_projection_risk"]
    orth = diagnostics["orthogonalization"]
    structural_rows = [
        {
            "metric": "Residual score IC",
            "value": diagnostics["mean_residual_score_ic"],
            "target": "> 0",
            "pass": diagnostics["mean_residual_score_ic"] > 0,
            "interpretation": "Direct out-of-sample residual alpha evidence",
        },
        {
            "metric": "Median realized residual variance share",
            "value": risk["median_residual_variance_share"],
            "target": "0.08 to 0.12",
            "pass": diagnostics["gates"][
                "median_residual_risk_share_8_to_12pct"
            ],
            "interpretation": "Whether the intended 10% risk sleeve was delivered",
        },
        {
            "metric": "Residual optimizer failure rate",
            "value": diagnostics["optimizer_failure_rate_c"],
            "target": "<= production + 0.01",
            "pass": diagnostics["gates"]["optimizer_failure_within_1pp"],
            "interpretation": "Implementation feasibility at active rebalances",
        },
        {
            "metric": "21-day score persistence",
            "value": orth["score_persistence_21d"],
            "target": ">= 0.60",
            "pass": diagnostics["gates"]["score_persistence_at_least_0_60"],
            "interpretation": "Signal stability after causal smoothing",
        },
        {
            "metric": "Between-ticker variance share",
            "value": orth["between_ticker_variance_share"],
            "target": "<= 0.30",
            "pass": diagnostics["gates"][
                "between_ticker_variance_at_most_0_30"
            ],
            "interpretation": "Protection against static name-identity leakage",
        },
        {
            "metric": "Maximum control correlation after orthogonalization",
            "value": orth["max_abs_control_corr_after"],
            "target": "<= 0.05",
            "pass": diagnostics["gates"]["control_corr_within_0_05"],
            "interpretation": "Separation from base and style controls",
        },
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "residual_sleeve_arm_comparison.csv", arm_rows)
    _write_csv(args.output_dir / "residual_sleeve_period_ir.csv", period_rows)
    _write_csv(
        args.output_dir / "residual_sleeve_structural_diagnostics.csv",
        structural_rows,
    )

    failed_checks = [name for name, passed in checks.items() if not passed]
    selection_bias: dict[str, float | int | str] = {}
    if args.selection_bias_csv.exists():
        with args.selection_bias_csv.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                raw_value = row["value"]
                try:
                    selection_bias[row["metric"]] = float(raw_value)
                except ValueError:
                    selection_bias[row["metric"]] = raw_value
        if "N_trials" in selection_bias:
            checks["selection_bias.trial_count_matches_inventory"] = _close(
                float(selection_bias["N_trials"]), float(inventory["n_trials_total"])
            )
            failed_checks = [name for name, passed in checks.items() if not passed]
    summary = {
        "decision": "REJECT_KEEP_DEFAULT_OFF",
        "reason": (
            "The pure residual sleeve reduced IR and annual active return, while "
            "the score had negative out-of-sample residual IC. The intended 10% "
            "risk allocation was also not delivered because the residual optimizer "
            "was frequently infeasible."
        ),
        "trial_inventory_n": int(inventory["n_trials_total"]),
        "formal_results": {
            "A_production_ir": a["information_ratio"],
            "B_reserve_only_ir": b["information_ratio"],
            "C_residual_sleeve_ir": c["information_ratio"],
            "B_minus_A_ir": b_minus_a["information_ratio"],
            "C_minus_B_ir": c_minus_b["information_ratio"],
            "C_minus_A_ir": c_minus_a["information_ratio"],
            "C_minus_B_active_return": c_minus_b["active_return"],
            "C_minus_A_active_return": c_minus_a["active_return"],
        },
        "diagnostics": {
            "mean_residual_score_ic": diagnostics["mean_residual_score_ic"],
            "active_rebalances": diagnostics["active_rebalances"],
            "composed_rebalances": diagnostics["composed_rebalances"],
            "unexplained_passthrough_rebalances": diagnostics[
                "unexplained_passthrough_rebalances"
            ],
            "optimizer_failure_rate_c": diagnostics["optimizer_failure_rate_c"],
            "median_realized_residual_variance_share": risk[
                "median_residual_variance_share"
            ],
            "median_risk_correlation": risk["median_risk_correlation"],
            "score_persistence_21d": orth["score_persistence_21d"],
            "between_ticker_variance_share": orth[
                "between_ticker_variance_share"
            ],
            "max_abs_control_corr_after": orth["max_abs_control_corr_after"],
        },
        "gates": diagnostics["gates"],
        "selection_bias": selection_bias,
        "arithmetic_validation": {
            "checks": len(checks),
            "all_pass": not failed_checks,
            "failed": failed_checks,
        },
    }
    (args.output_dir / "residual_sleeve_analysis.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failed_checks:
        raise AssertionError(f"Arithmetic validation failed: {failed_checks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
