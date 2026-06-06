from __future__ import annotations

from pathlib import Path
import json

from core.instance_loader import load_instance
from alns_solver.solution_state import ALNSSolutionState
from alns_solver.od_insertion import (
    od_insertion_strategy_1,
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


def remove_customer_from_od(
    state: ALNSSolutionState,
    customer: str,
    driver: str,
) -> None:
    state.od_routes[driver] = [
        node
        for node in state.od_routes[driver]
        if node != customer
    ]
    state.mark_customer_unassigned(customer)


def assert_valid_complete(instance, result, label):
    if result is None:
        raise AssertionError(f"{label}: no feasible insertion found.")

    solution = result.state.to_core_solution(
        instance=instance,
        lambda_value=0.0,
        objective_mode="weighted",
        cost_bounds=COST_BOUNDS,
        emission_bounds=EMISSION_BOUNDS,
        emission_factors=EMISSION_FACTORS,
        require_complete=True,
    )

    if not solution.validator_pass:
        raise AssertionError(
            f"{label}: shared validator failed: "
            f"{solution.validation_errors}"
        )

    return solution


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    instance = load_instance(
        project_root / "data" / "small" / "instance_001"
    )

    report = {}

    # ---------------------------------------------------------
    # Test A: Strategy I with an existing depot pickup.
    # ---------------------------------------------------------
    state_a = cost_anchor_state()
    remove_customer_from_od(
        state=state_a,
        customer="C5",
        driver="OD2",
    )

    result_a = od_insertion_strategy_1(
        state=state_a,
        instance=instance,
        customer_id="C5",
        lambda_value=0.0,
        cost_bounds=COST_BOUNDS,
        emission_bounds=EMISSION_BOUNDS,
        emission_factors=EMISSION_FACTORS,
        candidate_drivers=["OD2"],
    )

    solution_a = assert_valid_complete(
        instance,
        result_a,
        "Strategy I depot pickup",
    )

    if result_a.pickup != "S":
        raise AssertionError(
            f"Strategy I should preserve depot pickup S, "
            f"received {result_a.pickup}."
        )

    if solution_a.od_routes["OD2"][1] != "S":
        raise AssertionError(
            "OD2 route no longer has depot S as its pickup."
        )

    print("[PASS] Strategy I — existing depot pickup")
    report["strategy_1_depot"] = {
        "driver": result_a.driver,
        "pickup": result_a.pickup,
        "route": solution_a.od_routes[result_a.driver],
        "cost": solution_a.cost,
        "emission": solution_a.emission,
    }

    # ---------------------------------------------------------
    # Test B: Strategy I with an existing TN pickup.
    # ---------------------------------------------------------
    state_b = emission_anchor_state()
    remove_customer_from_od(
        state=state_b,
        customer="C1",
        driver="OD1",
    )

    result_b = od_insertion_strategy_1(
        state=state_b,
        instance=instance,
        customer_id="C1",
        lambda_value=1.0,
        cost_bounds=COST_BOUNDS,
        emission_bounds=EMISSION_BOUNDS,
        emission_factors=EMISSION_FACTORS,
        candidate_drivers=["OD1"],
    )

    solution_b = assert_valid_complete(
        instance,
        result_b,
        "Strategy I TN pickup",
    )

    if result_b.pickup != "TN1":
        raise AssertionError(
            f"Strategy I should preserve TN1, "
            f"received {result_b.pickup}."
        )

    tn_completion = solution_b.arrival_times[
        "tn_completion"
    ]["TN1"]
    od_pickup = solution_b.arrival_times[
        "od_pickup"
    ]["OD1"]["TN1"]

    if od_pickup + 1e-6 < tn_completion:
        raise AssertionError(
            "Strategy I TN synchronization failed."
        )

    print("[PASS] Strategy I — existing TN pickup")
    print("[PASS] Strategy I — TN synchronization")
    report["strategy_1_tn"] = {
        "driver": result_b.driver,
        "pickup": result_b.pickup,
        "route": solution_b.od_routes[result_b.driver],
        "tn_completion": tn_completion,
        "od_pickup": od_pickup,
        "cost": solution_b.cost,
        "emission": solution_b.emission,
    }

    # ---------------------------------------------------------
    # Test C: Strategy II creates a new TN pickup and inserts TN
    # into a DV route.
    # ---------------------------------------------------------
    state_c = cost_anchor_state()
    remove_customer_from_od(
        state=state_c,
        customer="C1",
        driver="OD1",
    )
    state_c.od_routes["OD1"] = []

    result_c = od_insertion_strategy_2(
        state=state_c,
        instance=instance,
        customer_id="C1",
        lambda_value=1.0,
        cost_bounds=COST_BOUNDS,
        emission_bounds=EMISSION_BOUNDS,
        emission_factors=EMISSION_FACTORS,
        candidate_drivers=["OD1"],
        candidate_tns=["TN1"],
    )

    solution_c = assert_valid_complete(
        instance,
        result_c,
        "Strategy II new TN",
    )

    if result_c.pickup != "TN1":
        raise AssertionError(
            f"Strategy II should select TN1, "
            f"received {result_c.pickup}."
        )

    if "TN1" not in solution_c.od_routes["OD1"]:
        raise AssertionError(
            "Strategy II did not insert TN1 in OD1 route."
        )

    if not any(
        "TN1" in route
        for route in solution_c.dv_routes.values()
    ):
        raise AssertionError(
            "Strategy II did not insert TN1 in a DV route."
        )

    tn_completion_c = solution_c.arrival_times[
        "tn_completion"
    ]["TN1"]
    od_pickup_c = solution_c.arrival_times[
        "od_pickup"
    ]["OD1"]["TN1"]

    if od_pickup_c + 1e-6 < tn_completion_c:
        raise AssertionError(
            "Strategy II TN synchronization failed."
        )

    print("[PASS] Strategy II — new TN pickup")
    print("[PASS] Strategy II — TN inserted into DV route")
    print("[PASS] Strategy II — TN synchronization")
    report["strategy_2_new_tn"] = {
        "driver": result_c.driver,
        "pickup": result_c.pickup,
        "dv_vehicle": result_c.dv_vehicle,
        "dv_route": solution_c.dv_routes[result_c.dv_vehicle],
        "od_route": solution_c.od_routes[result_c.driver],
        "tn_completion": tn_completion_c,
        "od_pickup": od_pickup_c,
        "cost": solution_c.cost,
        "emission": solution_c.emission,
    }

    output_dir = (
        project_root
        / "outputs"
        / "alns_od_insertion_tests"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "od_insertion_gate3_report.json"
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nReport saved to: {output_path}")
    print("\nALNS OD INSERTION GATE PASSED")


if __name__ == "__main__":
    main()
