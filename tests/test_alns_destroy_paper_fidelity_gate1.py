from __future__ import annotations

from pathlib import Path
import json

from core.instance_loader import load_instance
from alns_solver.solution_state import ALNSSolutionState
from alns_solver.paper_destroy_operators import (
    paper_route_removal,
    paper_worst_adp_removal,
    score_adp_pair_eq46,
)
from alns_solver.repair_operators import (
    regret_3_repair,
)


COST_BOUNDS = (
    23.089059445460528,
    24.28427622523578,
)

EMISSION_BOUNDS = (
    77.85476672718833,
    79.22375667475296,
)

EMISSION_FACTORS = (3.0, 1.0)


def cost_anchor_state() -> ALNSSolutionState:
    return ALNSSolutionState(
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


def emission_anchor_state() -> ALNSSolutionState:
    return ALNSSolutionState(
        dv_routes={
            "DV1": [],
            "DV2": ["S", "TN1", "A1", "T"],
        },
        od_routes={
            "OD1": ["O1", "TN1", "C1", "D1"],
            "OD2": ["O2", "S", "C2", "C5", "D2"],
        },
        assignments={
            "C1": {
                "mode": "OD_HOME",
                "driver": "OD1",
                "pickup": "TN1",
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


def assert_partial_state(
    result,
    expected_customers: set[str],
) -> None:
    actual = set(result.removed_customers)

    if actual != expected_customers:
        raise AssertionError(
            f"Removed customers mismatch. "
            f"Actual={sorted(actual)}, "
            f"expected={sorted(expected_customers)}"
        )

    for customer in expected_customers:
        if customer in result.state.assignments:
            raise AssertionError(
                f"{customer} remains assigned."
            )

        if customer not in result.state.unassigned_customers:
            raise AssertionError(
                f"{customer} is not tracked as unassigned."
            )


def repair_and_validate(
    partial_state: ALNSSolutionState,
    instance: dict,
    *,
    lambda_value: float,
):
    result = regret_3_repair(
        partial_state,
        instance,
        lambda_value=lambda_value,
        cost_bounds=COST_BOUNDS,
        emission_bounds=EMISSION_BOUNDS,
        emission_factors=EMISSION_FACTORS,
        strategy_2_mode="paper_random_dv",
        strategy_2_seed=7,
    )

    if not result.validator_pass:
        raise AssertionError(
            f"Repair failed: {result.validation_errors}"
        )

    return result


def main() -> None:
    root = Path(__file__).resolve().parents[1]

    instance = load_instance(
        root
        / "data"
        / "small"
        / "instance_001"
    )

    report = {}

    # =========================================================
    # Test 1 — Route removal selects a DV route only.
    # =========================================================
    route_result = paper_route_removal(
        state=emission_anchor_state(),
        instance=instance,
        vehicle="DV2",
        seed=0,
    )

    expected_removed = {
        "C1",
        "C3",
        "C4",
        "C6",
    }

    assert_partial_state(
        route_result,
        expected_removed,
    )

    if route_result.removed_route["route_type"] != "DV":
        raise AssertionError(
            "Paper route removal selected a non-DV route."
        )

    if route_result.removed_route["route_id"] != "DV2":
        raise AssertionError(
            "Unexpected DV route selected."
        )

    if route_result.state.dv_routes["DV2"] != []:
        raise AssertionError(
            "Selected DV route was not deactivated."
        )

    if route_result.state.od_routes["OD1"] != []:
        raise AssertionError(
            "OD1 should be inactive after its TN customer is removed."
        )

    repaired_route = repair_and_validate(
        route_result.state,
        instance,
        lambda_value=0.5,
    )

    print("[PASS] Paper route removal selects DV route only")
    print("[PASS] DV route removal propagates through affected TN")
    print("[PASS] Route-removal partial state repaired and validated")

    report["paper_route_removal"] = {
        "removed_route": route_result.removed_route,
        "removed_customers": route_result.removed_customers,
        "repair_cost": repaired_route.final_cost,
        "repair_emission": repaired_route.final_emission,
        "validator_pass": repaired_route.validator_pass,
    }

    # =========================================================
    # Test 2 — Eq. (46) is explicitly average saving.
    # =========================================================
    base = cost_anchor_state()

    score = score_adp_pair_eq46(
        base,
        instance,
        vehicle="DV2",
        adp="A1",
        lambda_value=0.0,
        cost_bounds=COST_BOUNDS,
        emission_bounds=EMISSION_BOUNDS,
        emission_factors=EMISSION_FACTORS,
    )

    expected_count = 3

    if len(score.removed_customers) != expected_count:
        raise AssertionError(
            "Unexpected number of customers at A1."
        )

    expected_average = (
        score.total_saving / expected_count
    )

    if abs(
        score.average_saving - expected_average
    ) > 1e-12:
        raise AssertionError(
            "Eq. (46) average saving was computed incorrectly."
        )

    worst_adp_result = paper_worst_adp_removal(
        base,
        instance,
        lambda_value=0.0,
        cost_bounds=COST_BOUNDS,
        emission_bounds=EMISSION_BOUNDS,
        emission_factors=EMISSION_FACTORS,
    )

    assert_partial_state(
        worst_adp_result,
        {"C3", "C4", "C6"},
    )

    actual_average = worst_adp_result.removed_route[
        "average_saving_eq46"
    ]

    if abs(
        actual_average - score.average_saving
    ) > 1e-12:
        raise AssertionError(
            "Worst ADP result does not use Eq. (46)."
        )

    repaired_adp = repair_and_validate(
        worst_adp_result.state,
        instance,
        lambda_value=0.0,
    )

    print("[PASS] Worst ADP score follows Eq. (46)")
    print("[PASS] Worst ADP uses average saving per customer")
    print("[PASS] Worst-ADP partial state repaired and validated")

    report["paper_worst_adp_removal"] = {
        "removed_facility": worst_adp_result.removed_route,
        "removed_customers": worst_adp_result.removed_customers,
        "objective_before": score.objective_before,
        "objective_after": score.objective_after,
        "total_saving": score.total_saving,
        "average_saving_eq46": score.average_saving,
        "repair_cost": repaired_adp.final_cost,
        "repair_emission": repaired_adp.final_emission,
        "validator_pass": repaired_adp.validator_pass,
    }

    output_dir = (
        root
        / "outputs"
        / "alns_destroy_fidelity_tests"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "destroy_fidelity_gate1_report.json"
    )
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nReport saved to: {output_path}")
    print(
        "\nPAPER FIDELITY GATE 1 — "
        "ROUTE AND WORST ADP REMOVAL PASSED"
    )


if __name__ == "__main__":
    main()
