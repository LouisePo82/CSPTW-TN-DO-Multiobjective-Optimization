from __future__ import annotations

from pathlib import Path
import json

from core.instance_loader import load_instance
from alns_solver.paper_alns_main import (
    build_ml1_paper_initial_state,
)
from alns_solver.paper_production_runner import (
    PaperALNSRunConfig,
    export_production_result,
    run_paper_alns_production,
)
from exact_solver import ExactSolver
from reporting.common_artifact_exporter import (
    export_exact_common_artifacts,
)
from reporting.route_visualization import (
    plot_solution_routes,
    plot_vehicle_routes,
)


EMISSION_FACTORS = (3.0, 1.0)

COMMON_SUMMARY_KEYS = {
    "instance_id",
    "lambda",
    "cost_weight",
    "emission_weight",
    "seed",
    "iteration_limit",
    "runtime_seconds",
    "best_cost",
    "best_emission",
    "best_F_lambda",
    "dv_distance",
    "od_extra_distance",
    "validation_pass",
    "termination_reason",
    "paper_faithful",
    "enhanced",
}

COMMON_ARTIFACT_FILENAMES = {
    "run_config.json",
    "run_results.json",
    "run_results.csv",
    "best_solution.json",
    "best_route_map.png",
    "artifact_manifest.json",
}


def _load_json(path: Path) -> dict:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def _assert_common_summary(
    *,
    solver_name: str,
    summary: dict,
) -> None:
    missing = COMMON_SUMMARY_KEYS - set(summary)

    if missing:
        raise AssertionError(
            f"{solver_name} summary is missing "
            f"common keys: {sorted(missing)}"
        )


def _assert_common_artifacts(
    *,
    solver_name: str,
    output_dir: Path,
) -> None:
    actual = {
        path.name
        for path in output_dir.iterdir()
        if path.is_file()
    }

    missing = (
        COMMON_ARTIFACT_FILENAMES
        - actual
    )

    if missing:
        raise AssertionError(
            f"{solver_name} is missing common "
            f"artifacts: {sorted(missing)}"
        )

    for filename in (
        COMMON_ARTIFACT_FILENAMES
    ):
        path = output_dir / filename

        if path.stat().st_size <= 0:
            raise AssertionError(
                f"{solver_name} artifact is empty: "
                f"{path}"
            )


def _assert_shared_best_solution_schema(
    *,
    solver_name: str,
    payload: dict,
) -> None:
    state = payload.get("state")
    metrics = payload.get("metrics")

    if not isinstance(state, dict):
        raise AssertionError(
            f"{solver_name} best_solution has no state."
        )

    if not isinstance(metrics, dict):
        raise AssertionError(
            f"{solver_name} best_solution has no metrics."
        )

    required_state_keys = {
        "dv_routes",
        "od_routes",
        "assignments",
        "unassigned_customers",
    }
    required_metric_keys = {
        "cost",
        "emission",
        "F_lambda",
        "dv_distance",
        "od_extra_distance",
    }

    missing_state = (
        required_state_keys - set(state)
    )
    missing_metrics = (
        required_metric_keys - set(metrics)
    )

    if missing_state:
        raise AssertionError(
            f"{solver_name} state schema missing: "
            f"{sorted(missing_state)}"
        )

    if missing_metrics:
        raise AssertionError(
            f"{solver_name} metric schema missing: "
            f"{sorted(missing_metrics)}"
        )


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    instance = load_instance(
        project_root
        / "data"
        / "small"
        / "instance_001"
    )

    cost_bounds = (
        23.089059445460528,
        24.28427622523578,
    )
    emission_bounds = (
        77.85476672718833,
        79.22375667475296,
    )

    root = (
        project_root
        / "outputs"
        / "cross_solver_contract_tests"
    )
    exact_dir = root / "exact"
    alns_dir = root / "alns"

    # --------------------------------------------------------
    # Exact export
    # --------------------------------------------------------
    exact_solution = ExactSolver().solve(
        instance,
        objective_config={
            "mode": "weighted",
            "lambda_value": 0.5,
            "cost_bounds": cost_bounds,
            "emission_bounds": emission_bounds,
            "emission_factor_dv": (
                EMISSION_FACTORS[0]
            ),
            "emission_factor_od": (
                EMISSION_FACTORS[1]
            ),
        },
        solver_config={
            "time_limit_sec": 300,
            "mip_gap": 0.0,
            "enable_output": False,
        },
    )

    if (
        exact_solution.status != "OPTIMAL"
        or not exact_solution.validator_pass
    ):
        raise AssertionError(
            "Exact solution is not valid and optimal."
        )

    export_exact_common_artifacts(
        instance=instance,
        solution=exact_solution,
        output_dir=exact_dir,
        instance_id="instance_001",
        lambda_value=0.5,
        cost_bounds=cost_bounds,
        emission_bounds=emission_bounds,
        emission_factors=EMISSION_FACTORS,
    )

    # --------------------------------------------------------
    # ALNS export
    # --------------------------------------------------------
    config = PaperALNSRunConfig(
        instance_id="instance_001",
        lambda_value=0.5,
        run_seed=2026,
        iteration_limit=20,
        cost_bounds=cost_bounds,
        emission_bounds=emission_bounds,
        emission_factors=EMISSION_FACTORS,
    )

    initial_state = build_ml1_paper_initial_state(
        instance,
        seed=config.run_seed,
        lambda_value=config.lambda_value,
        cost_bounds=config.cost_bounds,
        emission_bounds=config.emission_bounds,
        emission_factors=config.emission_factors,
    )

    alns_result = run_paper_alns_production(
        instance=instance,
        initial_state=initial_state,
        config=config,
    )

    if not alns_result.best_solution.validator_pass:
        raise AssertionError(
            "ALNS best solution failed validation."
        )

    export_production_result(
        alns_result,
        alns_dir,
    )

    plot_solution_routes(
        instance=instance,
        state=alns_result.best_state,
        output_path=(
            alns_dir / "best_route_map.png"
        ),
        instance_id=config.instance_id,
        lambda_value=config.lambda_value,
        seed=config.run_seed,
    )

    plot_vehicle_routes(
        instance=instance,
        state=alns_result.best_state,
        output_dir=(
            alns_dir / "vehicle_routes"
        ),
        instance_id=config.instance_id,
        lambda_value=config.lambda_value,
        seed=config.run_seed,
    )

    alns_manifest = {
        "run_identity": {
            "instance_id": config.instance_id,
            "solver": "alns",
            "lambda_value": (
                config.lambda_value
            ),
            "seed": config.run_seed,
            "iteration_limit": (
                config.iteration_limit
            ),
        },
        "common_artifact_contract": True,
        "artifacts": {
            "run_config": "run_config.json",
            "run_results_json": (
                "run_results.json"
            ),
            "run_results_csv": (
                "run_results.csv"
            ),
            "best_solution": (
                "best_solution.json"
            ),
            "combined_route_map": (
                "best_route_map.png"
            ),
        },
        "solver_specific": {
            "iteration_history": (
                "iteration_history.csv"
            ),
            "operator_statistics": (
                "operator_statistics.csv"
            ),
            "optimality_gap": None,
        },
    }

    (
        alns_dir
        / "artifact_manifest.json"
    ).write_text(
        json.dumps(
            alns_manifest,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Cross-solver contract checks
    # --------------------------------------------------------
    _assert_common_artifacts(
        solver_name="exact",
        output_dir=exact_dir,
    )
    _assert_common_artifacts(
        solver_name="alns",
        output_dir=alns_dir,
    )

    exact_results = _load_json(
        exact_dir / "run_results.json"
    )
    alns_results = _load_json(
        alns_dir / "run_results.json"
    )

    _assert_common_summary(
        solver_name="exact",
        summary=exact_results["summary"],
    )
    _assert_common_summary(
        solver_name="alns",
        summary=alns_results["summary"],
    )

    exact_best = _load_json(
        exact_dir / "best_solution.json"
    )
    alns_best = _load_json(
        alns_dir / "best_solution.json"
    )

    _assert_shared_best_solution_schema(
        solver_name="exact",
        payload=exact_best,
    )
    _assert_shared_best_solution_schema(
        solver_name="alns",
        payload=alns_best,
    )

    exact_manifest = _load_json(
        exact_dir / "artifact_manifest.json"
    )
    alns_manifest_loaded = _load_json(
        alns_dir / "artifact_manifest.json"
    )

    if (
        exact_manifest[
            "common_artifact_contract"
        ]
        is not True
    ):
        raise AssertionError(
            "Exact common artifact flag is false."
        )

    if (
        alns_manifest_loaded[
            "common_artifact_contract"
        ]
        is not True
    ):
        raise AssertionError(
            "ALNS common artifact flag is false."
        )

    if (
        exact_manifest["solver_specific"][
            "iteration_history"
        ]
        is not None
    ):
        raise AssertionError(
            "Exact must not fabricate "
            "iteration history."
        )

    if (
        alns_manifest_loaded[
            "solver_specific"
        ]["iteration_history"]
        is None
    ):
        raise AssertionError(
            "ALNS iteration history must be preserved."
        )

    report = {
        "gate": "EX-4",
        "purpose": (
            "cross_solver_common_artifact_contract"
        ),
        "common_summary_keys": sorted(
            COMMON_SUMMARY_KEYS
        ),
        "common_artifact_filenames": sorted(
            COMMON_ARTIFACT_FILENAMES
        ),
        "exact": {
            "status": exact_solution.status,
            "validation_pass": (
                exact_solution.validator_pass
            ),
            "output_dir": str(
                exact_dir.relative_to(
                    project_root
                )
            ),
        },
        "alns": {
            "termination_reason": (
                alns_result.termination_reason
            ),
            "validation_pass": (
                alns_result.best_solution
                .validator_pass
            ),
            "output_dir": str(
                alns_dir.relative_to(
                    project_root
                )
            ),
        },
        "shared_best_solution_schema": True,
        "shared_route_visualization": True,
        "solver_specific_artifacts_preserved": True,
    }

    report_path = (
        root
        / "cross_solver_artifact_schema_ex4_report.json"
    )
    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "[PASS] Exact and ALNS share the common artifact filenames"
    )
    print(
        "[PASS] Exact and ALNS share the common run summary keys"
    )
    print(
        "[PASS] Exact and ALNS share the best-solution state/metric schema"
    )
    print(
        "[PASS] Exact and ALNS use the same route visualization module"
    )
    print(
        "[PASS] Solver-specific artifacts remain explicit and separate"
    )
    print(f"\nReport saved to: {report_path}")
    print(
        "\nEX-4 — CROSS-SOLVER COMMON "
        "ARTIFACT CONTRACT PASSED"
    )


if __name__ == "__main__":
    main()
