from __future__ import annotations

from pathlib import Path
import json

from core.instance_loader import load_instance
from alns_solver.solution_state import ALNSSolutionState
from alns_solver.destroy_operators import (
    random_adp_removal,
    worst_adp_removal,
    random_tn_removal,
)
from alns_solver.od_insertion import (
    od_insertion_strategy_2,
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


def assert_partial_result(
    result,
    expected_customers: set[str],
) -> None:
    actual = set(result.removed_customers)

    if actual != expected_customers:
        raise AssertionError(
            f"{result.operator_name}: removed customers mismatch.\n"
            f"Actual: {sorted(actual)}\n"
            f"Expected: {sorted(expected_customers)}"
        )

    for customer in expected_customers:
        if customer in result.state.assignments:
            raise AssertionError(
                f"{customer} remains assigned after destroy."
            )
        if customer not in result.state.unassigned_customers:
            raise AssertionError(
                f"{customer} is not tracked as unassigned."
            )


def restore_adp_group(
    state: ALNSSolutionState,
    instance: dict,
    *,
    vehicle: str,
    adp: str,
    customers: list[str],
) -> ALNSSolutionState:
    repaired = state.copy()

    route = repaired.dv_routes.get(vehicle, [])

    if not route:
        route = [
            instance["start_depot"],
            instance["end_depot"],
        ]
        repaired.dv_routes[vehicle] = route

    if adp not in route:
        repaired.dv_routes[vehicle].insert(
            len(route) - 1,
            adp,
        )

    for customer in customers:
        repaired.assign_customer(
            customer,
            {
                "mode": "ADP",
                "vehicle": vehicle,
                "adp": adp,
            },
        )

    return repaired


def assert_complete_valid(
    state: ALNSSolutionState,
    instance: dict,
    *,
    lambda_value: float,
    label: str,
):
    solution = state.to_core_solution(
        instance=instance,
        lambda_value=lambda_value,
        objective_mode="weighted",
        cost_bounds=COST_BOUNDS,
        emission_bounds=EMISSION_BOUNDS,
        emission_factors=EMISSION_FACTORS,
        require_complete=True,
    )

    if not solution.validator_pass:
        raise AssertionError(
            f"{label}: shared validator failed:\n"
            + "\n".join(solution.validation_errors)
        )

    return solution


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    instance = load_instance(
        project_root
        / "data"
        / "small"
        / "instance_001"
    )

    report = {}

    # =========================================================
    # Test A — Random ADP Removal
    # =========================================================
    random_adp = random_adp_removal(
        state=cost_anchor_state(),
        instance=instance,
        seed=0,
        vehicle="DV2",
        adp="A1",
    )

    expected_adp_customers = {"C3", "C4", "C6"}

    assert_partial_result(
        random_adp,
        expected_adp_customers,
    )

    if "A1" in random_adp.state.dv_routes["DV2"]:
        raise AssertionError(
            "A1 should be removed after its last parcel is destroyed."
        )

    repaired_random_adp = restore_adp_group(
        random_adp.state,
        instance,
        vehicle="DV2",
        adp="A1",
        customers=sorted(expected_adp_customers),
    )

    random_adp_solution = assert_complete_valid(
        repaired_random_adp,
        instance,
        lambda_value=0.0,
        label="Random ADP repair",
    )

    print("[PASS] Random ADP removal — all ADP parcels removed")
    print("[PASS] Random ADP removal — orphan ADP node cleaned")
    print("[PASS] Random ADP removal — repaired state valid")

    report["random_adp_removal"] = {
        "removed_customers": random_adp.removed_customers,
        "removed_facility": random_adp.removed_route,
        "repaired_route": random_adp_solution.dv_routes["DV2"],
        "repaired_cost": random_adp_solution.cost,
        "repaired_emission": random_adp_solution.emission,
        "validator_pass": random_adp_solution.validator_pass,
    }

    # =========================================================
    # Test B — Worst ADP Removal
    # =========================================================
    worst_adp = worst_adp_removal(
        state=cost_anchor_state(),
        instance=instance,
        lambda_value=0.0,
        cost_bounds=COST_BOUNDS,
        emission_bounds=EMISSION_BOUNDS,
        emission_factors=EMISSION_FACTORS,
    )

    assert_partial_result(
        worst_adp,
        expected_adp_customers,
    )

    if worst_adp.removed_route["facility_id"] != "A1":
        raise AssertionError(
            "Worst ADP removal should select A1 in this instance."
        )

    repaired_worst_adp = restore_adp_group(
        worst_adp.state,
        instance,
        vehicle="DV2",
        adp="A1",
        customers=sorted(expected_adp_customers),
    )

    worst_adp_solution = assert_complete_valid(
        repaired_worst_adp,
        instance,
        lambda_value=0.0,
        label="Worst ADP repair",
    )

    print("[PASS] Worst ADP removal — marginal scoring")
    print("[PASS] Worst ADP removal — repaired state valid")

    report["worst_adp_removal"] = {
        "removed_customers": worst_adp.removed_customers,
        "removed_facility": worst_adp.removed_route,
        "repaired_cost": worst_adp_solution.cost,
        "repaired_emission": worst_adp_solution.emission,
        "validator_pass": worst_adp_solution.validator_pass,
    }

    # =========================================================
    # Test C — Random TN Removal
    # =========================================================
    random_tn = random_tn_removal(
        state=emission_anchor_state(),
        instance=instance,
        seed=0,
        tn="TN1",
    )

    assert_partial_result(
        random_tn,
        {"C1"},
    )

    if random_tn.state.od_routes["OD1"] != []:
        raise AssertionError(
            "OD1 must be inactive after removing its only TN parcel."
        )

    if any(
        "TN1" in route
        for route in random_tn.state.dv_routes.values()
    ):
        raise AssertionError(
            "TN1 must be removed from all DV routes."
        )

    repaired_tn_result = od_insertion_strategy_2(
        state=random_tn.state,
        instance=instance,
        customer_id="C1",
        lambda_value=1.0,
        cost_bounds=COST_BOUNDS,
        emission_bounds=EMISSION_BOUNDS,
        emission_factors=EMISSION_FACTORS,
        candidate_drivers=["OD1"],
        candidate_tns=["TN1"],
    )

    if repaired_tn_result is None:
        raise AssertionError(
            "Strategy II could not repair the TN removal state."
        )

    repaired_tn_solution = assert_complete_valid(
        repaired_tn_result.state,
        instance,
        lambda_value=1.0,
        label="Random TN repair",
    )

    tn_completion = repaired_tn_solution.arrival_times[
        "tn_completion"
    ]["TN1"]

    od_pickup = repaired_tn_solution.arrival_times[
        "od_pickup"
    ]["OD1"]["TN1"]

    if od_pickup + 1e-6 < tn_completion:
        raise AssertionError(
            "TN synchronization failed after repair."
        )

    print("[PASS] Random TN removal — all TN parcels removed")
    print("[PASS] Random TN removal — affected OD deactivated")
    print("[PASS] Random TN removal — orphan TN cleaned")
    print("[PASS] Random TN removal — Strategy II repair valid")
    print("[PASS] Random TN removal — synchronization preserved")

    report["random_tn_removal"] = {
        "removed_customers": random_tn.removed_customers,
        "removed_facility": random_tn.removed_route,
        "repaired_dv_routes": repaired_tn_solution.dv_routes,
        "repaired_od_route": repaired_tn_solution.od_routes["OD1"],
        "tn_completion": tn_completion,
        "od_pickup": od_pickup,
        "repaired_cost": repaired_tn_solution.cost,
        "repaired_emission": repaired_tn_solution.emission,
        "validator_pass": repaired_tn_solution.validator_pass,
    }

    output_dir = (
        project_root
        / "outputs"
        / "alns_destroy_operator_tests"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "destroy_operators_gate4b_report.json"
    )

    output_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nReport saved to: {output_path}")
    print("\nALNS DESTROY OPERATORS GATE 4B PASSED")


if __name__ == "__main__":
    main()
