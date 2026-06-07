from __future__ import annotations

from pathlib import Path
import json
import math

from core.instance_loader import load_instance
from core.objective import recompute_objectives
from core.solution import Solution
from exact_solver import ExactSolver

EMISSION_FACTORS = (3.0, 1.0)
EXPECTED_COST_MIN = 23.089059445460528
EXPECTED_COST_MIN_EMISSION = 79.22375667475296
TOLERANCE = 1.0e-7


def assert_close(label: str, actual: float, expected: float) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=TOLERANCE):
        raise AssertionError(f"{label} mismatch: actual={actual}, expected={expected}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    instance = load_instance(project_root / "data" / "small" / "instance_001")

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

    if not isinstance(solution, Solution):
        raise AssertionError("ExactSolver did not return core.Solution.")
    if solution.status != "OPTIMAL":
        raise AssertionError(f"Expected OPTIMAL, received {solution.status}.")
    if solution.solver_name != "exact":
        raise AssertionError(f"Unexpected solver name: {solution.solver_name}")
    if not solution.validator_pass:
        raise AssertionError(
            f"Exact solution failed shared validation: {solution.validation_errors}"
        )

    required_numeric_fields = {
        "cost": solution.cost,
        "emission": solution.emission,
        "objective": solution.objective,
        "dv_distance": solution.dv_distance,
        "od_extra_distance": solution.od_extra_distance,
        "runtime_sec": solution.runtime_sec,
    }
    for field_name, value in required_numeric_fields.items():
        if value is None or not math.isfinite(float(value)):
            raise AssertionError(f"Invalid numeric field {field_name}: {value}")

    if set(solution.assignments) != set(instance["customers"]):
        raise AssertionError("Assignment coverage does not match customer set.")
    if set(solution.dv_routes) != set(instance["dvs"]):
        raise AssertionError("DV route keys do not match instance DVs.")
    if set(solution.od_routes) != set(instance["ods"]):
        raise AssertionError("OD route keys do not match instance ODs.")

    for vehicle, route in solution.dv_routes.items():
        if route and (
            route[0] != instance["start_depot"]
            or route[-1] != instance["end_depot"]
        ):
            raise AssertionError(f"Invalid DV route endpoints for {vehicle}: {route}")

    for driver, route in solution.od_routes.items():
        if route:
            info = instance["vehicles"][driver]
            if route[0] != info["origin"] or route[-1] != info["destination"]:
                raise AssertionError(f"Invalid OD route endpoints for {driver}: {route}")

    recomputed = recompute_objectives(
        instance,
        solution.dv_routes,
        solution.od_routes,
        emission_factors=EMISSION_FACTORS,
    )
    assert_close("recomputed cost", recomputed["cost"], solution.cost)
    assert_close("recomputed emission", recomputed["emission"], solution.emission)
    assert_close("recomputed DV distance", recomputed["dv_distance"], solution.dv_distance)
    assert_close(
        "recomputed OD extra distance",
        recomputed["od_extra_distance"],
        solution.od_extra_distance,
    )
    assert_close("known minimum cost", solution.cost, EXPECTED_COST_MIN)
    assert_close(
        "emission at cost anchor",
        solution.emission,
        EXPECTED_COST_MIN_EMISSION,
    )

    if solution.optimality_gap is None:
        raise AssertionError("Exact solution did not expose optimality gap.")
    assert_close("optimality gap", solution.optimality_gap, 0.0)

    report = {
        "gate": "EX-1",
        "purpose": "exact_core_solution_schema",
        "status": solution.status,
        "solver_name": solution.solver_name,
        "objective_mode": solution.objective_mode,
        "lambda_value": solution.lambda_value,
        "metrics": {
            "cost": solution.cost,
            "emission": solution.emission,
            "objective": solution.objective,
            "dv_distance": solution.dv_distance,
            "od_extra_distance": solution.od_extra_distance,
            "runtime_sec": solution.runtime_sec,
            "optimality_gap": solution.optimality_gap,
        },
        "schema": {
            "dv_route_keys": sorted(solution.dv_routes),
            "od_route_keys": sorted(solution.od_routes),
            "assignment_keys": sorted(solution.assignments),
            "arrival_times_present": bool(solution.arrival_times),
        },
        "validator_pass": solution.validator_pass,
        "validation_errors": list(solution.validation_errors),
    }

    output_dir = project_root / "outputs" / "exact_compatibility_tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "exact_solution_schema_ex1_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("[PASS] ExactSolver returns core.Solution")
    print("[PASS] Exact status is OPTIMAL with zero gap")
    print("[PASS] Shared validator accepts the exact solution")
    print("[PASS] Customer assignments cover the full instance")
    print("[PASS] DV and OD routes follow the shared route schema")
    print("[PASS] Shared objective recomputation matches exact metrics")
    print("[PASS] Known small-instance cost anchor is recovered")
    print(f"\nReport saved to: {report_path}")
    print("\nEX-1 — EXACT CORE SOLUTION SCHEMA PASSED")


if __name__ == "__main__":
    main()
