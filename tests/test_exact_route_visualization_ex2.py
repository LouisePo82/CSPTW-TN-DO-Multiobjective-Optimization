from __future__ import annotations

from pathlib import Path
import json

from core.instance_loader import load_instance
from exact_solver import ExactSolver
from reporting.route_visualization import (
    plot_solution_routes,
    plot_vehicle_routes,
)


EMISSION_FACTORS = (3.0, 1.0)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    instance = load_instance(
        project_root / "data" / "small" / "instance_001"
    )

    solution = ExactSolver().solve(
        instance,
        objective_config={
            "mode": "cost",
            "lambda_value": 0.0,
            "emission_factor_dv": EMISSION_FACTORS[0],
            "emission_factor_od": EMISSION_FACTORS[1],
        },
        solver_config={
            "time_limit_sec": 300,
            "mip_gap": 0.0,
            "enable_output": False,
        },
    )

    if solution.status != "OPTIMAL":
        raise AssertionError(
            f"Expected OPTIMAL exact solution, got {solution.status}."
        )

    if not solution.validator_pass:
        raise AssertionError(
            "Exact solution failed validation: "
            f"{solution.validation_errors}"
        )

    output_dir = (
        project_root
        / "outputs"
        / "exact_compatibility_tests"
        / "ex2_visualization"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined_map = plot_solution_routes(
        instance=instance,
        state=solution,
        output_path=output_dir / "best_route_map.png",
        instance_id="instance_001",
        lambda_value=0.0,
        seed=None,
        title=(
            "Crowd-Shipping Route Map — Exact cost-optimal solution\n"
            "instance_001 | Exact MILP | OPTIMAL"
        ),
    )

    vehicle_maps = plot_vehicle_routes(
        instance=instance,
        state=solution,
        output_dir=output_dir / "vehicle_routes",
        instance_id="instance_001",
        lambda_value=0.0,
        seed=None,
    )

    if not combined_map.exists():
        raise AssertionError(
            "Combined exact route map was not created."
        )

    if combined_map.stat().st_size <= 0:
        raise AssertionError(
            "Combined exact route map is empty."
        )

    active_vehicle_count = sum(
        1
        for route in solution.dv_routes.values()
        if route
    ) + sum(
        1
        for route in solution.od_routes.values()
        if route
    )

    if len(vehicle_maps) != active_vehicle_count:
        raise AssertionError(
            "Per-vehicle map count does not match active exact routes: "
            f"maps={len(vehicle_maps)}, "
            f"active_routes={active_vehicle_count}"
        )

    for path in vehicle_maps:
        if not path.exists():
            raise AssertionError(
                f"Missing per-vehicle map: {path}"
            )

        if path.stat().st_size <= 0:
            raise AssertionError(
                f"Empty per-vehicle map: {path}"
            )

    report = {
        "gate": "EX-2",
        "purpose": (
            "exact_shared_route_visualization_compatibility"
        ),
        "instance_id": "instance_001",
        "solver": "exact",
        "status": solution.status,
        "validator_pass": solution.validator_pass,
        "combined_route_map": str(
            combined_map.relative_to(project_root)
        ),
        "combined_route_map_size_bytes": (
            combined_map.stat().st_size
        ),
        "active_vehicle_count": active_vehicle_count,
        "vehicle_route_maps": [
            {
                "path": str(
                    path.relative_to(project_root)
                ),
                "size_bytes": path.stat().st_size,
            }
            for path in vehicle_maps
        ],
        "shared_visualization_module": (
            "reporting.route_visualization"
        ),
        "exact_solution_used_directly": True,
        "conversion_to_alns_state": False,
    }

    report_path = (
        project_root
        / "outputs"
        / "exact_compatibility_tests"
        / "exact_route_visualization_ex2_report.json"
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
        "[PASS] Exact core.Solution is accepted directly "
        "by shared route visualization"
    )
    print(
        "[PASS] Combined exact route map is created"
    )
    print(
        "[PASS] Per-vehicle exact route maps are created"
    )
    print(
        "[PASS] Exact visualization uses the same reporting "
        "module as ALNS"
    )
    print(
        "[PASS] No conversion to ALNSSolutionState is required"
    )
    print(f"\nReport saved to: {report_path}")
    print(
        "\nEX-2 — EXACT SHARED ROUTE "
        "VISUALIZATION PASSED"
    )


if __name__ == "__main__":
    main()
