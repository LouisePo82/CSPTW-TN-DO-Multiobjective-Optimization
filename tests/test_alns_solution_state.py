from __future__ import annotations

from pathlib import Path
import json

from core.instance_loader import load_instance
from alns_solver.solution_state import ALNSSolutionState


EXPECTED_COST = 23.089059445460528
EXPECTED_EMISSION = 79.22375667475296
TOLERANCE = 1e-6


def assert_close(label: str, actual: float, expected: float) -> None:
    difference = abs(actual - expected)
    if difference > TOLERANCE:
        raise AssertionError(
            f"{label}: actual={actual}, expected={expected}, "
            f"difference={difference}"
        )
    print(f"[PASS] {label}: {actual:.12f}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    instance = load_instance(
        project_root / "data" / "small" / "instance_001"
    )

    # Manually reconstruct the validated exact cost-anchor solution.
    state = ALNSSolutionState(
        dv_routes={
            "DV1": [],
            "DV2": ["S", "A1", "T"],
        },
        od_routes={
            "OD1": ["O1", "S", "C1", "D1"],
            "OD2": ["O2", "S", "C2", "C5", "D2"],
        },
        assignments={
            "C1": {
                "mode": "OD_HOME",
                "driver": "OD1",
                "pickup": "S",
            },
            "C2": {
                "mode": "OD_HOME",
                "driver": "OD2",
                "pickup": "S",
            },
            "C3": {
                "mode": "ADP",
                "vehicle": "DV2",
                "adp": "A1",
            },
            "C4": {
                "mode": "ADP",
                "vehicle": "DV2",
                "adp": "A1",
            },
            "C5": {
                "mode": "OD_HOME",
                "driver": "OD2",
                "pickup": "S",
            },
            "C6": {
                "mode": "ADP",
                "vehicle": "DV2",
                "adp": "A1",
            },
        },
    )

    solution = state.to_core_solution(
        instance=instance,
        lambda_value=0.0,
        objective_mode="weighted",
        cost_bounds=(
            23.089059445460528,
            24.28427622523578,
        ),
        emission_bounds=(
            77.85476672718833,
            79.22375667475296,
        ),
        emission_factors=(3.0, 1.0),
        metadata={
            "test_name": "manual_cost_anchor_state",
        },
    )

    if solution.status != "FEASIBLE":
        raise AssertionError(
            f"Expected FEASIBLE, received {solution.status}: "
            f"{solution.validation_errors}"
        )

    if not solution.validator_pass:
        raise AssertionError(
            f"Shared validator failed: {solution.validation_errors}"
        )

    assert_close("cost", solution.cost, EXPECTED_COST)
    assert_close("emission", solution.emission, EXPECTED_EMISSION)
    assert_close("normalized objective at lambda=0", solution.objective, 0.0)

    if solution.vehicle_loads != {
        "DV1": 0.0,
        "DV2": 3.0,
    }:
        raise AssertionError(
            f"Unexpected DV loads: {solution.vehicle_loads}"
        )

    if solution.tn_demands != {
        "S": 3.0,
        "TN1": 0.0,
    }:
        raise AssertionError(
            f"Unexpected pickup demands: {solution.tn_demands}"
        )

    output_dir = project_root / "outputs" / "alns_state_tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "manual_cost_anchor_solution.json"
    output_path.write_text(
        json.dumps(solution.to_dict(), indent=2),
        encoding="utf-8",
    )

    print("[PASS] shared validator")
    print("[PASS] shared objective")
    print("[PASS] shared schedule")
    print("[PASS] ALNS state -> core Solution conversion")
    print(f"\nOutput saved to: {output_path}")
    print("\nALNS SOLUTION STATE GATE PASSED")


if __name__ == "__main__":
    main()
