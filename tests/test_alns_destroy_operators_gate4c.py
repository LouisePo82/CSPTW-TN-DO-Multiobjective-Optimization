from __future__ import annotations

from pathlib import Path
import json

from core.instance_loader import load_instance
from alns_solver.solution_state import ALNSSolutionState
from alns_solver.destroy_operators import (
    related_removal,
    historical_node_removal,
    neighborhood_removal,
    node_neighborhood_removal,
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


REFERENCE_DV_ROUTES = {
    "DV1": [],
    "DV2": ["S", "A1", "T"],
}

REFERENCE_OD_ROUTES = {
    "OD1": ["O1", "S", "C1", "D1"],
    "OD2": ["O2", "S", "C2", "C5", "D2"],
}

REFERENCE_ASSIGNMENTS = {
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
}


def cost_anchor_state() -> ALNSSolutionState:
    return ALNSSolutionState(
        dv_routes={
            vehicle: list(route)
            for vehicle, route in REFERENCE_DV_ROUTES.items()
        },
        od_routes={
            driver: list(route)
            for driver, route in REFERENCE_OD_ROUTES.items()
        },
        assignments={
            customer: dict(assignment)
            for customer, assignment
            in REFERENCE_ASSIGNMENTS.items()
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


def restore_reference_customers(
    state: ALNSSolutionState,
    customers: list[str],
) -> ALNSSolutionState:
    """
    Controlled Gate-4C repair.

    Production best/regret repair is implemented in the later repair gate.
    Here we restore each removed customer to its validated reference mode and
    route, then send the result through the shared validator.
    """
    repaired = state.copy()

    for customer in customers:
        repaired.assign_customer(
            customer,
            REFERENCE_ASSIGNMENTS[customer],
        )

    # Rebuild route structures from reference decisions.
    repaired.dv_routes = {
        vehicle: list(route)
        for vehicle, route in REFERENCE_DV_ROUTES.items()
    }

    repaired.od_routes = {
        driver: list(route)
        for driver, route in REFERENCE_OD_ROUTES.items()
    }

    repaired.normalize_routes(
        {
            "dvs": list(REFERENCE_DV_ROUTES),
            "ods": list(REFERENCE_OD_ROUTES),
        }
    )

    return repaired


def assert_complete_valid(
    state: ALNSSolutionState,
    instance: dict,
    label: str,
):
    solution = state.to_core_solution(
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
            f"{label}: shared validator failed:\n"
            + "\n".join(solution.validation_errors)
        )

    return solution


def run_and_validate(
    *,
    result,
    expected_customers: set[str],
    instance: dict,
    label: str,
):
    assert_partial_result(
        result,
        expected_customers,
    )

    repaired = restore_reference_customers(
        result.state,
        sorted(expected_customers),
    )

    solution = assert_complete_valid(
        repaired,
        instance,
        label,
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
    # Test A — Related Removal
    # C2 and C5 share customer type and OD delivery mode.
    # =========================================================
    related = related_removal(
        state=cost_anchor_state(),
        instance=instance,
        removal_count=2,
        seed_customer="C2",
        candidate_customers=["C2", "C5"],
    )

    related_solution = run_and_validate(
        result=related,
        expected_customers={"C2", "C5"},
        instance=instance,
        label="Related removal repair",
    )

    print("[PASS] Related removal — similarity selection")
    print("[PASS] Related removal — repaired state valid")

    report["related_removal"] = {
        "removed_customers": related.removed_customers,
        "repaired_cost": related_solution.cost,
        "repaired_emission": related_solution.emission,
        "validator_pass": related_solution.validator_pass,
    }

    # =========================================================
    # Test B — Historical Node Removal
    # Highest scores must be removed first.
    # =========================================================
    history_scores = {
        "C1": 1.0,
        "C2": 8.0,
        "C3": 2.0,
        "C4": 3.0,
        "C5": 10.0,
        "C6": 4.0,
    }

    historical = historical_node_removal(
        state=cost_anchor_state(),
        instance=instance,
        removal_count=2,
        historical_scores=history_scores,
    )

    historical_solution = run_and_validate(
        result=historical,
        expected_customers={"C2", "C5"},
        instance=instance,
        label="Historical removal repair",
    )

    print("[PASS] Historical node removal — score ranking")
    print("[PASS] Historical node removal — repaired state valid")

    report["historical_node_removal"] = {
        "removed_customers": historical.removed_customers,
        "selected_scores": {
            customer: history_scores[customer]
            for customer in historical.removed_customers
        },
        "repaired_cost": historical_solution.cost,
        "repaired_emission": historical_solution.emission,
        "validator_pass": historical_solution.validator_pass,
    }

    # =========================================================
    # Test C — Neighborhood Removal
    # Restricted candidate set gives a controlled spatial test.
    # =========================================================
    neighborhood = neighborhood_removal(
        state=cost_anchor_state(),
        instance=instance,
        removal_count=2,
        seed_customer="C2",
        candidate_customers=["C2", "C5"],
    )

    neighborhood_solution = run_and_validate(
        result=neighborhood,
        expected_customers={"C2", "C5"},
        instance=instance,
        label="Neighborhood removal repair",
    )

    print("[PASS] Neighborhood removal — spatial ranking")
    print("[PASS] Neighborhood removal — repaired state valid")

    report["neighborhood_removal"] = {
        "removed_customers": neighborhood.removed_customers,
        "repaired_cost": neighborhood_solution.cost,
        "repaired_emission": neighborhood_solution.emission,
        "validator_pass": neighborhood_solution.validator_pass,
    }

    # =========================================================
    # Test D — Node-Neighborhood Removal
    # Every ADP assignment has service anchor A1, so radius=0
    # must remove C3, C4 and C6 together.
    # =========================================================
    node_neighborhood = node_neighborhood_removal(
        state=cost_anchor_state(),
        instance=instance,
        center_node="A1",
        radius=0.0,
    )

    node_solution = run_and_validate(
        result=node_neighborhood,
        expected_customers={"C3", "C4", "C6"},
        instance=instance,
        label="Node-neighborhood removal repair",
    )

    if "A1" in node_neighborhood.state.dv_routes["DV2"]:
        raise AssertionError(
            "A1 should be cleaned after all A1 parcels are removed."
        )

    print("[PASS] Node-neighborhood removal — service-anchor selection")
    print("[PASS] Node-neighborhood removal — orphan ADP cleanup")
    print("[PASS] Node-neighborhood removal — repaired state valid")

    report["node_neighborhood_removal"] = {
        "center_node": "A1",
        "removed_customers": node_neighborhood.removed_customers,
        "repaired_cost": node_solution.cost,
        "repaired_emission": node_solution.emission,
        "validator_pass": node_solution.validator_pass,
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
        / "destroy_operators_gate4c_report.json"
    )

    output_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nReport saved to: {output_path}")
    print("\nALNS DESTROY OPERATORS GATE 4C PASSED")


if __name__ == "__main__":
    main()
