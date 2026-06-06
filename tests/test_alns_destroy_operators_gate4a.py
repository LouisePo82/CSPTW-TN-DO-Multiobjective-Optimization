from __future__ import annotations

from pathlib import Path
import json

from core.instance_loader import load_instance
from alns_solver.solution_state import ALNSSolutionState
from alns_solver.destroy_operators import (
    random_customer_removal,
    worst_customer_removal,
    route_removal,
)
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


def assert_partial_destroy_state(
    result,
    expected_count: int,
) -> None:
    if len(result.removed_customers) != expected_count:
        raise AssertionError(
            f"{result.operator_name}: expected "
            f"{expected_count} removed customers, received "
            f"{result.removed_customers}."
        )

    for customer in result.removed_customers:
        if customer in result.state.assignments:
            raise AssertionError(
                f"{result.operator_name}: customer {customer} is still assigned."
            )

        if customer not in result.state.unassigned_customers:
            raise AssertionError(
                f"{result.operator_name}: customer {customer} "
                "was not added to unassigned_customers."
            )


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


def repair_with_existing_pickup(
    state: ALNSSolutionState,
    instance: dict,
    customer_id: str,
    driver: str,
) -> ALNSSolutionState:
    result = od_insertion_strategy_1(
        state=state,
        instance=instance,
        customer_id=customer_id,
        lambda_value=0.0,
        cost_bounds=COST_BOUNDS,
        emission_bounds=EMISSION_BOUNDS,
        emission_factors=EMISSION_FACTORS,
        candidate_drivers=[driver],
    )

    if result is None:
        raise RuntimeError(
            f"Strategy I could not repair {customer_id} through driver {driver}."
        )

    return result.state


def repair_with_new_tn(
    state: ALNSSolutionState,
    instance: dict,
    customer_id: str,
    driver: str,
) -> ALNSSolutionState:
    result = od_insertion_strategy_2(
        state=state,
        instance=instance,
        customer_id=customer_id,
        lambda_value=1.0,
        cost_bounds=COST_BOUNDS,
        emission_bounds=EMISSION_BOUNDS,
        emission_factors=EMISSION_FACTORS,
        candidate_drivers=[driver],
        candidate_tns=["TN1"],
    )

    if result is None:
        raise RuntimeError(
            f"Strategy II could not repair {customer_id} "
            f"through driver {driver} and TN1."
        )

    return result.state


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    instance = load_instance(project_root / "data" / "small" / "instance_001")

    report = {
        "random_customer_removal": [],
        "worst_customer_removal": {},
        "route_removal": {},
    }

    # =========================================================
    # Test A — Random Customer Removal
    #
    # Restrict to C2/C5 because both belong to OD2.
    # Removing one leaves OD2 active, allowing Strategy I repair.
    # =========================================================
    random_failures = []

    for seed in range(100):
        result = random_customer_removal(
            state=cost_anchor_state(),
            instance=instance,
            removal_count=1,
            seed=seed,
            candidate_customers=["C2", "C5"],
        )

        assert_partial_destroy_state(
            result=result,
            expected_count=1,
        )

        removed_customer = result.removed_customers[0]

        try:
            repaired_state = repair_with_existing_pickup(
                state=result.state,
                instance=instance,
                customer_id=removed_customer,
                driver="OD2",
            )

            solution = assert_complete_valid(
                state=repaired_state,
                instance=instance,
                label=f"Random removal seed {seed}",
            )

            report["random_customer_removal"].append(
                {
                    "seed": seed,
                    "removed_customer": removed_customer,
                    "repaired_route": solution.od_routes["OD2"],
                    "cost": solution.cost,
                    "emission": solution.emission,
                    "validator_pass": solution.validator_pass,
                }
            )

        except Exception as exc:
            random_failures.append(
                {
                    "seed": seed,
                    "removed_customer": removed_customer,
                    "error": (f"{type(exc).__name__}: {exc}"),
                }
            )

    if random_failures:
        raise AssertionError(
            "Random customer removal had repair failures:\n"
            + json.dumps(
                random_failures,
                indent=2,
            )
        )

    print("[PASS] Random customer removal — 100/100 partial states")
    print("[PASS] Random customer removal — 100/100 repaired states")

    # =========================================================
    # Test B — Worst Customer Removal
    #
    # Restrict to C2/C5 so the selected customer can be repaired
    # through OD2's existing depot pickup.
    # =========================================================
    worst_result = worst_customer_removal(
        state=cost_anchor_state(),
        instance=instance,
        removal_count=1,
        lambda_value=0.0,
        cost_bounds=COST_BOUNDS,
        emission_bounds=EMISSION_BOUNDS,
        emission_factors=EMISSION_FACTORS,
        candidate_customers=["C2", "C5"],
    )

    assert_partial_destroy_state(
        result=worst_result,
        expected_count=1,
    )

    worst_customer = worst_result.removed_customers[0]

    worst_repaired_state = repair_with_existing_pickup(
        state=worst_result.state,
        instance=instance,
        customer_id=worst_customer,
        driver="OD2",
    )

    worst_solution = assert_complete_valid(
        state=worst_repaired_state,
        instance=instance,
        label="Worst customer removal repair",
    )

    report["worst_customer_removal"] = {
        "removed_customer": worst_customer,
        "repaired_route": worst_solution.od_routes["OD2"],
        "repaired_cost": worst_solution.cost,
        "repaired_emission": worst_solution.emission,
        "validator_pass": worst_solution.validator_pass,
    }

    print("[PASS] Worst customer removal — marginal objective scoring")
    print("[PASS] Worst customer removal — repaired state")

    # =========================================================
    # Test C — Route Removal
    #
    # Remove OD1 entirely. Since OD1 becomes inactive, repair C1
    # through Strategy II using a new TN pickup.
    # =========================================================
    route_result = route_removal(
        state=cost_anchor_state(),
        instance=instance,
        route_type="OD",
        route_id="OD1",
        seed=0,
    )

    assert_partial_destroy_state(
        result=route_result,
        expected_count=1,
    )

    expected_removed_route = {
        "route_type": "OD",
        "route_id": "OD1",
    }

    if route_result.removed_route != expected_removed_route:
        raise AssertionError(
            "Unexpected route removal metadata:\n"
            f"Actual: {route_result.removed_route}\n"
            f"Expected: {expected_removed_route}"
        )

    if route_result.state.od_routes["OD1"] != []:
        raise AssertionError("OD1 must be inactive after route removal.")

    removed_customer = route_result.removed_customers[0]

    route_repaired_state = repair_with_new_tn(
        state=route_result.state,
        instance=instance,
        customer_id=removed_customer,
        driver="OD1",
    )

    route_solution = assert_complete_valid(
        state=route_repaired_state,
        instance=instance,
        label="OD1 route removal repair",
    )

    if "TN1" not in route_solution.od_routes["OD1"]:
        raise AssertionError("Repaired OD1 route must use TN1.")

    if not any("TN1" in route for route in route_solution.dv_routes.values()):
        raise AssertionError("A DV route must visit TN1 after Strategy II repair.")

    report["route_removal"] = {
        "removed_route": route_result.removed_route,
        "removed_customers": route_result.removed_customers,
        "repaired_od_route": route_solution.od_routes["OD1"],
        "repaired_dv_routes": route_solution.dv_routes,
        "repaired_cost": route_solution.cost,
        "repaired_emission": route_solution.emission,
        "validator_pass": route_solution.validator_pass,
    }

    print("[PASS] Route removal — OD route deactivated")
    print("[PASS] Route removal — all route customers unassigned")
    print("[PASS] Route removal — repaired through Strategy II")
    print("[PASS] Route removal — TN synchronization preserved")

    # =========================================================
    # Save report
    # =========================================================
    output_dir = project_root / "outputs" / "alns_destroy_operator_tests"

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_dir / "destroy_operators_gate4a_report.json"

    output_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nReport saved to: {output_path}")

    print("\nALNS DESTROY OPERATORS GATE 4A PASSED")


if __name__ == "__main__":
    main()
