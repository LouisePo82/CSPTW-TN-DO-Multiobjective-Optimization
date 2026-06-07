from __future__ import annotations

from pathlib import Path
import csv
import json
import math

from core.instance_loader import load_instance
from exact_solver import ExactSolver
from reporting.common_artifact_exporter import (
    export_exact_common_artifacts,
)


EMISSION_FACTORS = (3.0, 1.0)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    instance = load_instance(
        project_root
        / "data"
        / "small"
        / "instance_001"
    )

    solver = ExactSolver()
    solver_config = {
        "time_limit_sec": 300,
        "mip_gap": 0.0,
        "enable_output": False,
    }
    base = {
        "emission_factor_dv": 3.0,
        "emission_factor_od": 1.0,
    }

    cost_anchor = solver.solve(
        instance,
        {
            **base,
            "mode": "cost",
            "lambda_value": 0.0,
        },
        solver_config,
    )
    emission_anchor = solver.solve(
        instance,
        {
            **base,
            "mode": "emission",
            "lambda_value": 1.0,
        },
        solver_config,
    )

    cost_bounds = (
        float(cost_anchor.cost),
        float(emission_anchor.cost),
    )
    emission_bounds = (
        float(emission_anchor.emission),
        float(cost_anchor.emission),
    )

    solution = solver.solve(
        instance,
        {
            **base,
            "mode": "weighted",
            "lambda_value": 0.0,
            "cost_bounds": cost_bounds,
            "emission_bounds": emission_bounds,
        },
        solver_config,
    )

    if solution.status != "OPTIMAL":
        raise AssertionError(
            f"Expected OPTIMAL, got {solution.status}"
        )
    if not solution.validator_pass:
        raise AssertionError(
            f"Validation failed: {solution.validation_errors}"
        )

    output_dir = (
        project_root
        / "outputs"
        / "exact_compatibility_tests"
        / "ex3_common_artifacts"
    )

    artifacts = export_exact_common_artifacts(
        instance=instance,
        solution=solution,
        output_dir=output_dir,
        instance_id="instance_001",
        lambda_value=0.0,
        cost_bounds=cost_bounds,
        emission_bounds=emission_bounds,
        emission_factors=EMISSION_FACTORS,
    )

    required_files = {
        "run_config.json",
        "run_results.json",
        "run_results.csv",
        "best_solution.json",
        "best_route_map.png",
        "artifact_manifest.json",
    }

    actual_files = {
        path.name
        for path in output_dir.iterdir()
        if path.is_file()
    }

    missing = required_files - actual_files
    if missing:
        raise AssertionError(
            f"Missing common artifacts: {sorted(missing)}"
        )

    for filename in required_files:
        path = output_dir / filename
        if path.stat().st_size <= 0:
            raise AssertionError(
                f"Empty artifact: {path}"
            )

    run_results = json.loads(
        (
            output_dir
            / "run_results.json"
        ).read_text(encoding="utf-8")
    )
    summary = run_results["summary"]

    expected_keys = {
        "instance_id",
        "solver",
        "method",
        "objective_mode",
        "lambda",
        "runtime_seconds",
        "best_cost",
        "best_emission",
        "best_F_lambda",
        "dv_distance",
        "od_extra_distance",
        "validation_pass",
        "status",
        "termination_reason",
        "optimality_gap",
    }

    absent = expected_keys - set(summary)
    if absent:
        raise AssertionError(
            "Common summary keys missing: "
            f"{sorted(absent)}"
        )

    if summary["solver"] != "exact":
        raise AssertionError(
            "Exact solver identity was not preserved."
        )
    if summary["validation_pass"] is not True:
        raise AssertionError(
            "Exact validation status was not preserved."
        )
    if not math.isclose(
        float(summary["optimality_gap"]),
        0.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise AssertionError(
            "Exact optimality gap is not zero."
        )

    with (
        output_dir
        / "run_results.csv"
    ).open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        csv_rows = list(
            csv.DictReader(file)
        )

    if len(csv_rows) != 1:
        raise AssertionError(
            "run_results.csv must contain exactly one row."
        )

    manifest = json.loads(
        (
            output_dir
            / "artifact_manifest.json"
        ).read_text(encoding="utf-8")
    )

    if manifest[
        "common_artifact_contract"
    ] is not True:
        raise AssertionError(
            "Common artifact contract flag is missing."
        )

    if (
        manifest["solver_specific"][
            "iteration_history"
        ]
        is not None
    ):
        raise AssertionError(
            "Exact export must not fabricate "
            "ALNS iteration history."
        )

    vehicle_maps = artifacts[
        "vehicle_route_maps"
    ]
    if not vehicle_maps:
        raise AssertionError(
            "No exact per-vehicle route maps were exported."
        )

    for path in vehicle_maps:
        if not path.exists() or path.stat().st_size <= 0:
            raise AssertionError(
                f"Invalid vehicle map: {path}"
            )

    print(
        "[PASS] Exact exports the common run artifact set"
    )
    print(
        "[PASS] Exact run_results JSON and CSV use the common summary keys"
    )
    print(
        "[PASS] Exact best_solution uses the shared route/assignment schema"
    )
    print(
        "[PASS] Exact combined and per-vehicle maps use shared visualization"
    )
    print(
        "[PASS] Exact-specific gap/status metadata is preserved"
    )
    print(
        "[PASS] ALNS-only iteration/operator artifacts are not fabricated"
    )
    print(
        "\nEX-3 — EXACT COMMON ARTIFACT "
        "EXPORT PASSED"
    )


if __name__ == "__main__":
    main()
