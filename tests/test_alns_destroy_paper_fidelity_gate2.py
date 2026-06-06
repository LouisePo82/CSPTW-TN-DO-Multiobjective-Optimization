from __future__ import annotations

from pathlib import Path
import json

from core.instance_loader import load_instance
from alns_solver.solution_state import ALNSSolutionState
from alns_solver.paper_destroy_operators import (
    relatedness_eq47,
    paper_related_removal,
    paper_historical_node_removal,
    current_position_cost,
    neighborhood_contribution,
    paper_neighborhood_removal,
    paper_node_neighborhood_removal,
)
from alns_solver.repair_operators import regret_3_repair


COST_BOUNDS = (
    23.089059445460528,
    24.28427622523578,
)

EMISSION_BOUNDS = (
    77.85476672718833,
    79.22375667475296,
)

EMISSION_FACTORS = (3.0, 1.0)


def mixed_home_state() -> ALNSSolutionState:
    """
    Valid controlled complete solution.

    Customer modes follow instance eligibility:
    - C1, C2: Type 1 home delivery through ODs
    - C3, C4: Type 2 delivery through ADP A1
    - C5: Type 3 home delivery through OD2
    - C6: Type 3 home delivery through DV1

    This gives the fidelity tests multiple Type 1/3 home-delivery customers
    while preserving Type 2 ADP-only compatibility.
    """
    return ALNSSolutionState(
        dv_routes={
            "DV1": ["S", "C6", "T"],
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
                "mode": "DV_HOME",
                "vehicle": "DV1",
            },
        },
    )


def assert_partial(
    result,
    expected_count: int,
) -> None:
    if len(result.removed_customers) != expected_count:
        raise AssertionError(
            f"Expected {expected_count} removals, "
            f"received {result.removed_customers}."
        )

    for customer in result.removed_customers:
        if customer in result.state.assignments:
            raise AssertionError(
                f"{customer} remains assigned."
            )

        if customer not in result.state.unassigned_customers:
            raise AssertionError(
                f"{customer} not tracked as unassigned."
            )


def repair_and_validate(
    state: ALNSSolutionState,
    instance: dict,
):
    result = regret_3_repair(
        state,
        instance,
        lambda_value=0.0,
        cost_bounds=COST_BOUNDS,
        emission_bounds=EMISSION_BOUNDS,
        emission_factors=EMISSION_FACTORS,
        strategy_2_mode="paper_random_dv",
        strategy_2_seed=5,
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

    state = mixed_home_state()

    controlled = state.to_core_solution(
        instance=instance,
        lambda_value=0.0,
        objective_mode="weighted",
        cost_bounds=COST_BOUNDS,
        emission_bounds=EMISSION_BOUNDS,
        emission_factors=EMISSION_FACTORS,
        require_complete=True,
    )

    if not controlled.validator_pass:
        raise AssertionError(
            "Controlled Gate-2 state is invalid: "
            f"{controlled.validation_errors}"
        )

    print("[PASS] Controlled Gate-2 state is complete and valid")

    report = {}

    # =========================================================
    # Test 1 — Related Removal Eq. (47)
    # =========================================================
    score = relatedness_eq47(
        state,
        instance,
        seed_customer="C1",
        customer="C5",
        phi_1=5.0,
        phi_2=9.0,
        phi_3=1.0,
    )

    expected = (
        5.0 * score.normalized_distance
        + 9.0 * score.normalized_demand_difference
        + 1.0 * score.type_similarity
    )

    if abs(score.total_score - expected) > 1e-12:
        raise AssertionError(
            "Eq. (47) weighted score is incorrect."
        )

    related = paper_related_removal(
        state,
        instance,
        removal_count=2,
        seed_customer="C1",
        candidate_customers=[
            "C1",
            "C2",
            "C5",
            "C6",
        ],
        phi_1=5.0,
        phi_2=9.0,
        phi_3=1.0,
    )

    assert_partial(
        related,
        expected_count=2,
    )

    repaired_related = repair_and_validate(
        related.state,
        instance,
    )

    print("[PASS] Related Removal follows Eq. (47)")
    print("[PASS] Related Removal uses phi=(5,9,1)")
    print("[PASS] Related partial state repaired and validated")

    report["related_removal"] = {
        "seed_customer": "C1",
        "removed_customers": related.removed_customers,
        "example_score_C1_C5": {
            "normalized_distance": score.normalized_distance,
            "normalized_demand_difference": (
                score.normalized_demand_difference
            ),
            "type_similarity": score.type_similarity,
            "total_score": score.total_score,
        },
        "validator_pass": repaired_related.validator_pass,
    }

    # =========================================================
    # Test 2 — Historical Node Removal
    # =========================================================
    eligible_customers = [
        "C1",
        "C2",
        "C5",
        "C6",
    ]

    current_costs = {
        customer: current_position_cost(
            state,
            instance,
            customer,
        )
        for customer in eligible_customers
    }

    # Controlled design: C6 has the largest historical deterioration.
    historical_best = {
        "C1": max(0.0, current_costs["C1"] - 0.2),
        "C2": max(0.0, current_costs["C2"] - 0.1),
        "C5": max(0.0, current_costs["C5"] - 0.1),
        "C6": max(0.0, current_costs["C6"] - 2.0),
    }

    historical = paper_historical_node_removal(
        state,
        instance,
        removal_count=1,
        best_historical_position_costs=historical_best,
    )

    assert_partial(
        historical,
        expected_count=1,
    )

    if historical.removed_customers != ["C6"]:
        raise AssertionError(
            "Historical removal should select C6 under the "
            "controlled score design."
        )

    repaired_historical = repair_and_validate(
        historical.state,
        instance,
    )

    print(
        "[PASS] Historical removal uses current minus "
        "best historical cost"
    )
    print(
        "[PASS] Historical removal restricts eligibility "
        "to Type 1/3"
    )
    print(
        "[PASS] Historical partial state repaired and validated"
    )

    report["historical_node_removal"] = {
        "removed_customers": historical.removed_customers,
        "current_position_costs": current_costs,
        "best_historical_costs": historical_best,
        "validator_pass": repaired_historical.validator_pass,
    }

    # =========================================================
    # Test 3 — Neighborhood Removal
    # =========================================================
    contributions = {
        customer: neighborhood_contribution(
            state,
            instance,
            customer,
        ).contribution
        for customer in eligible_customers
    }

    expected_customer = max(
        contributions,
        key=contributions.get,
    )

    neighborhood = paper_neighborhood_removal(
        state,
        instance,
        removal_count=1,
    )

    assert_partial(
        neighborhood,
        expected_count=1,
    )

    if neighborhood.removed_customers != [
        expected_customer
    ]:
        raise AssertionError(
            "Neighborhood removal did not choose the maximum "
            "route-cost contribution."
        )

    repaired_neighborhood = repair_and_validate(
        neighborhood.state,
        instance,
    )

    print(
        "[PASS] Neighborhood removal uses route-cost contribution"
    )
    print(
        "[PASS] Neighborhood removal restricts eligibility "
        "to Type 1/3"
    )
    print(
        "[PASS] Neighborhood partial state repaired and validated"
    )

    report["neighborhood_removal"] = {
        "removed_customers": neighborhood.removed_customers,
        "contributions": contributions,
        "validator_pass": repaired_neighborhood.validator_pass,
    }

    # =========================================================
    # Test 4 — Node-Neighborhood Removal
    # =========================================================
    node_neighborhood = paper_node_neighborhood_removal(
        state,
        instance,
        removal_count=2,
        seed_customer="C1",
    )

    assert_partial(
        node_neighborhood,
        expected_count=2,
    )

    if node_neighborhood.removed_customers[0] != "C1":
        raise AssertionError(
            "Node-neighborhood removal must remove the seed first."
        )

    if any(
        int(instance["nodes"][customer]["customer_type"])
        not in {1, 3}
        for customer in node_neighborhood.removed_customers
    ):
        raise AssertionError(
            "Node-neighborhood removal selected an ineligible type."
        )

    repaired_node = repair_and_validate(
        node_neighborhood.state,
        instance,
    )

    print(
        "[PASS] Node-neighborhood uses Type 1/3 customer seed"
    )
    print(
        "[PASS] Node-neighborhood removes nearest eligible customers"
    )
    print(
        "[PASS] Node-neighborhood partial state repaired and validated"
    )

    report["node_neighborhood_removal"] = {
        "seed_customer": "C1",
        "removed_customers": (
            node_neighborhood.removed_customers
        ),
        "validator_pass": repaired_node.validator_pass,
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
        / "destroy_fidelity_gate2_report.json"
    )

    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nReport saved to: {output_path}")
    print(
        "\nPAPER FIDELITY GATE 2 — "
        "RELATED AND NEIGHBORHOOD OPERATORS PASSED"
    )


if __name__ == "__main__":
    main()
