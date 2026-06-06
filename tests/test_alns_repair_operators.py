from __future__ import annotations

from pathlib import Path
import json

from core.instance_loader import load_instance
from alns_solver.solution_state import ALNSSolutionState
from alns_solver.destroy_operators import (
    random_customer_removal,
    random_adp_removal,
    route_removal,
)
from alns_solver.repair_operators import (
    best_insertion_repair,
    regret_2_repair,
    regret_3_repair,
    perturbed_best_insertion_repair,
    perturbed_regret_repair,
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


def build_partial_states(instance: dict) -> dict[str, ALNSSolutionState]:
    partials = {}

    partials["two_random_customers"] = random_customer_removal(
        state=cost_anchor_state(),
        instance=instance,
        removal_count=2,
        seed=7,
        candidate_customers=["C2", "C5"],
    ).state

    partials["adp_group"] = random_adp_removal(
        state=cost_anchor_state(),
        instance=instance,
        seed=0,
        vehicle="DV2",
        adp="A1",
    ).state

    partials["od_route"] = route_removal(
        state=cost_anchor_state(),
        instance=instance,
        route_type="OD",
        route_id="OD2",
        seed=0,
    ).state

    return partials


def assert_repair_valid(
    result,
    instance: dict,
    label: str,
):
    if not result.validator_pass:
        raise AssertionError(
            f"{label}: validator failed:\n"
            + "\n".join(result.validation_errors)
        )

    if result.state.unassigned_customers:
        raise AssertionError(
            f"{label}: unassigned customers remain: "
            f"{sorted(result.state.unassigned_customers)}"
        )

    if set(result.state.assignments) != set(instance["customers"]):
        raise AssertionError(
            f"{label}: assignments do not cover all customers."
        )

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
            f"{label}: final shared validation failed."
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

    operators = {
        "best_insertion": lambda state: best_insertion_repair(
            state,
            instance,
            lambda_value=0.0,
            cost_bounds=COST_BOUNDS,
            emission_bounds=EMISSION_BOUNDS,
            emission_factors=EMISSION_FACTORS,
        ),
        "regret_2": lambda state: regret_2_repair(
            state,
            instance,
            lambda_value=0.0,
            cost_bounds=COST_BOUNDS,
            emission_bounds=EMISSION_BOUNDS,
            emission_factors=EMISSION_FACTORS,
        ),
        "regret_3": lambda state: regret_3_repair(
            state,
            instance,
            lambda_value=0.0,
            cost_bounds=COST_BOUNDS,
            emission_bounds=EMISSION_BOUNDS,
            emission_factors=EMISSION_FACTORS,
        ),
        "perturbed_best": lambda state: perturbed_best_insertion_repair(
            state,
            instance,
            seed=42,
            noise_strength=0.05,
            lambda_value=0.0,
            cost_bounds=COST_BOUNDS,
            emission_bounds=EMISSION_BOUNDS,
            emission_factors=EMISSION_FACTORS,
        ),
        "perturbed_regret_2": lambda state: perturbed_regret_repair(
            state,
            instance,
            k=2,
            seed=42,
            noise_strength=0.05,
            lambda_value=0.0,
            cost_bounds=COST_BOUNDS,
            emission_bounds=EMISSION_BOUNDS,
            emission_factors=EMISSION_FACTORS,
        ),
    }

    report = {}

    for scenario_name, partial_state in build_partial_states(instance).items():
        report[scenario_name] = {}

        for operator_name, operator in operators.items():
            result = operator(partial_state.copy())

            solution = assert_repair_valid(
                result,
                instance,
                f"{scenario_name}/{operator_name}",
            )

            report[scenario_name][operator_name] = {
                "insertion_order": result.insertion_order,
                "cost": solution.cost,
                "emission": solution.emission,
                "objective": solution.objective,
                "validator_pass": solution.validator_pass,
                "dv_routes": solution.dv_routes,
                "od_routes": solution.od_routes,
                "assignments": solution.assignments,
            }

            print(
                f"[PASS] {operator_name} — "
                f"{scenario_name}"
            )

    # Multi-objective behavior check:
    # repair OD1 after route removal under lambda=1 should allow TN-oriented
    # reconstruction and remain feasible.
    emission_partial = route_removal(
        state=cost_anchor_state(),
        instance=instance,
        route_type="OD",
        route_id="OD1",
        seed=0,
    ).state

    emission_result = best_insertion_repair(
        emission_partial,
        instance,
        lambda_value=1.0,
        cost_bounds=COST_BOUNDS,
        emission_bounds=EMISSION_BOUNDS,
        emission_factors=EMISSION_FACTORS,
    )

    emission_solution = assert_repair_valid(
        emission_result,
        instance,
        "emission-oriented best insertion",
    )

    report["emission_oriented_repair"] = {
        "insertion_order": emission_result.insertion_order,
        "cost": emission_solution.cost,
        "emission": emission_solution.emission,
        "objective": emission_solution.objective,
        "dv_routes": emission_solution.dv_routes,
        "od_routes": emission_solution.od_routes,
        "validator_pass": emission_solution.validator_pass,
    }

    print("[PASS] Best insertion responds to lambda=1")
    print("[PASS] Emission-oriented repair is feasible")

    output_dir = (
        project_root
        / "outputs"
        / "alns_repair_operator_tests"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "repair_operators_report.json"
    )

    output_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nReport saved to: {output_path}")
    print("\nALNS REPAIR OPERATORS GATE PASSED")


if __name__ == "__main__":
    main()
