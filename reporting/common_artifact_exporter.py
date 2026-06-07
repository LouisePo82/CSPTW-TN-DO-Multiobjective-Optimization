from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import json

from reporting.route_visualization import (
    plot_solution_routes,
    plot_vehicle_routes,
)


def _json_dump(
    path: Path,
    payload: Any,
) -> Path:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _csv_dump(
    path: Path,
    row: dict[str, Any],
) -> Path:
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(row),
        )
        writer.writeheader()
        writer.writerow(row)

    return path


def export_exact_common_artifacts(
    *,
    instance: dict,
    solution,
    output_dir: Path,
    instance_id: str,
    lambda_value: float,
    cost_bounds: tuple[float, float],
    emission_bounds: tuple[float, float],
    emission_factors: tuple[float, float],
    time_limit_seconds: float | None = None,
    mip_gap_target: float | None = None,
) -> dict[str, Path | list[Path]]:
    """
    Export one exact weighted-sum solution using the common
    exact/ALNS artifact contract.

    Exact-only information is preserved. ALNS-only artifacts
    are intentionally not fabricated.
    """
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_config = {
        "instance_id": instance_id,
        "solver": "exact",
        "objective_mode": (
            solution.objective_mode
        ),
        "lambda_value": float(
            lambda_value
        ),
        "cost_bounds": [
            float(cost_bounds[0]),
            float(cost_bounds[1]),
        ],
        "emission_bounds": [
            float(emission_bounds[0]),
            float(emission_bounds[1]),
        ],
        "emission_factors": [
            float(emission_factors[0]),
            float(emission_factors[1]),
        ],
        "time_limit_seconds": (
            None
            if time_limit_seconds is None
            else float(time_limit_seconds)
        ),
        "mip_gap_target": (
            None
            if mip_gap_target is None
            else float(mip_gap_target)
        ),
        "seed": None,
        "iteration_limit": None,
    }

    summary = {
        "instance_id": instance_id,
        "solver": "exact",
        "method": "weighted_sum",
        "objective_mode": (
            solution.objective_mode
        ),
        "lambda": float(
            lambda_value
        ),
        "cost_weight": float(
            1.0 - lambda_value
        ),
        "emission_weight": float(
            lambda_value
        ),
        "seed": None,
        "iteration_limit": None,
        "time_limit_seconds": (
            None
            if time_limit_seconds is None
            else float(time_limit_seconds)
        ),
        "mip_gap_target": (
            None
            if mip_gap_target is None
            else float(mip_gap_target)
        ),
        "runtime_seconds": float(
            solution.runtime_sec
        ),
        "best_cost": float(
            solution.cost
        ),
        "best_emission": float(
            solution.emission
        ),
        "best_F_lambda": float(
            solution.objective
        ),
        "dv_distance": float(
            solution.dv_distance
        ),
        "od_extra_distance": float(
            solution.od_extra_distance
        ),
        "validation_pass": bool(
            solution.validator_pass
        ),
        "status": solution.status,
        "termination_reason": (
            solution.status
        ),
        "optimality_gap": (
            None
            if solution.optimality_gap is None
            else float(
                solution.optimality_gap
            )
        ),
        "paper_faithful": None,
        "enhanced": None,
    }

    best_solution = {
        "solver": "exact",
        "status": solution.status,
        "objective_mode": (
            solution.objective_mode
        ),
        "lambda": float(
            lambda_value
        ),
        "state": {
            "dv_routes": {
                key: list(value)
                for key, value
                in solution.dv_routes.items()
            },
            "od_routes": {
                key: list(value)
                for key, value
                in solution.od_routes.items()
            },
            "assignments": {
                key: dict(value)
                for key, value
                in solution.assignments.items()
            },
            "unassigned_customers": [],
        },
        "metrics": {
            "cost": float(
                solution.cost
            ),
            "emission": float(
                solution.emission
            ),
            "F_lambda": float(
                solution.objective
            ),
            "dv_distance": float(
                solution.dv_distance
            ),
            "od_extra_distance": float(
                solution.od_extra_distance
            ),
        },
        "arrival_times": (
            solution.arrival_times
        ),
        "validator_pass": bool(
            solution.validator_pass
        ),
        "validation_errors": list(
            solution.validation_errors
        ),
        "solver_metadata": {
            "runtime_seconds": float(
                solution.runtime_sec
            ),
            "optimality_gap": (
                None
                if solution.optimality_gap
                is None
                else float(
                    solution.optimality_gap
                )
            ),
            "time_limit_seconds": (
                None
                if time_limit_seconds is None
                else float(
                    time_limit_seconds
                )
            ),
            "mip_gap_target": (
                None
                if mip_gap_target is None
                else float(
                    mip_gap_target
                )
            ),
            "metadata": (
                solution.metadata
            ),
        },
    }

    artifacts: dict[
        str,
        Path | list[Path],
    ] = {}

    artifacts["run_config"] = (
        _json_dump(
            output_dir
            / "run_config.json",
            run_config,
        )
    )

    artifacts["run_results_json"] = (
        _json_dump(
            output_dir
            / "run_results.json",
            {
                "summary": summary,
                "metadata": {
                    "solver": "exact",
                    "shared_solution_schema": True,
                    "shared_route_visualization": True,
                    "alns_specific_artifacts": False,
                },
            },
        )
    )

    artifacts["run_results_csv"] = (
        _csv_dump(
            output_dir
            / "run_results.csv",
            summary,
        )
    )

    artifacts["best_solution"] = (
        _json_dump(
            output_dir
            / "best_solution.json",
            best_solution,
        )
    )

    artifacts["combined_route_map"] = (
        plot_solution_routes(
            instance=instance,
            state=solution,
            output_path=(
                output_dir
                / "best_route_map.png"
            ),
            instance_id=instance_id,
            lambda_value=lambda_value,
            seed=None,
            title=(
                "Crowd-Shipping Route Map — "
                f"Exact weighted solution, "
                f"λ = {lambda_value:g}\n"
                f"{instance_id} | Exact MILP | "
                f"{solution.status}"
            ),
        )
    )

    vehicle_maps = (
        plot_vehicle_routes(
            instance=instance,
            state=solution,
            output_dir=(
                output_dir
                / "vehicle_routes"
            ),
            instance_id=instance_id,
            lambda_value=lambda_value,
            seed=None,
        )
    )

    artifacts[
        "vehicle_route_maps"
    ] = vehicle_maps

    manifest = {
        "run_identity": {
            "instance_id": instance_id,
            "solver": "exact",
            "lambda_value": float(
                lambda_value
            ),
            "seed": None,
            "iteration_limit": None,
        },
        "common_artifact_contract": True,
        "artifacts": {
            key: str(
                value.relative_to(
                    output_dir
                )
            )
            for key, value
            in artifacts.items()
            if isinstance(value, Path)
        },
        "vehicle_route_maps": [
            str(
                path.relative_to(
                    output_dir
                )
            )
            for path in vehicle_maps
        ],
        "solver_specific": {
            "optimality_gap": (
                summary[
                    "optimality_gap"
                ]
            ),
            "status": (
                solution.status
            ),
            "time_limit_seconds": (
                summary[
                    "time_limit_seconds"
                ]
            ),
            "mip_gap_target": (
                summary[
                    "mip_gap_target"
                ]
            ),
            "iteration_history": None,
            "operator_statistics": None,
        },
    }

    artifacts[
        "artifact_manifest"
    ] = _json_dump(
        output_dir
        / "artifact_manifest.json",
        manifest,
    )

    return artifacts
