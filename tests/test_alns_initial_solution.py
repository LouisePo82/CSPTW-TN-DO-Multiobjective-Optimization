from __future__ import annotations

from pathlib import Path
import csv
import json
import statistics

from core.instance_loader import load_instance
from alns_solver.initial_solution import construct_initial_solution

SEEDS = list(range(100))
EMISSION_FACTORS = (3.0, 1.0)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    instance = load_instance(project_root / "data" / "small" / "instance_001")

    rows = []
    failures = []

    for seed in SEEDS:
        try:
            state = construct_initial_solution(instance, seed=seed)
            solution = state.to_core_solution(
                instance=instance,
                lambda_value=0.0,
                objective_mode="cost",
                emission_factors=EMISSION_FACTORS,
                require_complete=True,
                metadata={"seed": seed, "test": "gate2_initial_solution"},
            )
            if not solution.validator_pass:
                failures.append({"seed": seed, "errors": solution.validation_errors})
                continue

            rows.append({
                "seed": seed,
                "status": solution.status,
                "cost": solution.cost,
                "emission": solution.emission,
                "dv_distance": solution.dv_distance,
                "od_extra_distance": solution.od_extra_distance,
                "used_dvs": sum(bool(r) for r in solution.dv_routes.values()),
                "used_ods": sum(bool(r) for r in solution.od_routes.values()),
                "uses_tn": any(
                    tn in route
                    for tn in instance["tns"]
                    for route in solution.od_routes.values()
                ),
                "validator_pass": solution.validator_pass,
            })
        except Exception as exc:
            failures.append({"seed": seed, "errors": [f"{type(exc).__name__}: {exc}"]})

    output_dir = project_root / "outputs" / "alns_initial_solution_tests"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "initial_solution_100_seeds.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    report = {
        "total_seeds": len(SEEDS),
        "valid_solutions": len(rows),
        "failures": len(failures),
        "pass_rate": len(rows) / len(SEEDS),
        "cost_statistics": {
            "min": min(r["cost"] for r in rows) if rows else None,
            "mean": statistics.mean(r["cost"] for r in rows) if rows else None,
            "max": max(r["cost"] for r in rows) if rows else None,
            "std": statistics.pstdev(r["cost"] for r in rows) if len(rows) > 1 else 0.0,
        },
        "emission_statistics": {
            "min": min(r["emission"] for r in rows) if rows else None,
            "mean": statistics.mean(r["emission"] for r in rows) if rows else None,
            "max": max(r["emission"] for r in rows) if rows else None,
            "std": statistics.pstdev(r["emission"] for r in rows) if len(rows) > 1 else 0.0,
        },
        "tn_usage_count": sum(1 for r in rows if r["uses_tn"]),
        "failure_details": failures,
    }

    report_path = output_dir / "initial_solution_gate2_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nCSV saved to: {csv_path}")
    print(f"Report saved to: {report_path}")

    if failures:
        raise SystemExit(f"GATE 2 FAILED: {len(failures)} of {len(SEEDS)} seeds were invalid.")

    print("\n[PASS] 100/100 initial solutions are complete.")
    print("[PASS] 100/100 initial solutions pass shared validator.")
    print("[PASS] All customers are assigned exactly once.")
    print("[PASS] Initial-solution objective uses shared core.")
    print("\nALNS INITIAL SOLUTION GATE PASSED")


if __name__ == "__main__":
    main()
