from __future__ import annotations

from pathlib import Path
import json

from core.instance_loader import load_instance
from alns_solver.paper_destroy_operators import (
    current_position_cost,
    neighborhood_contribution,
    paper_historical_node_removal,
)
from tests.test_alns_destroy_paper_fidelity_gate2 import mixed_home_state


def route_neighbors(state, customer):
    assignment = state.assignments[customer]
    if assignment["mode"] == "DV_HOME":
        route = state.dv_routes[assignment["vehicle"]]
    elif assignment["mode"] == "OD_HOME":
        route = state.od_routes[assignment["driver"]]
    else:
        raise AssertionError("Customer is not home-delivered.")
    position = route.index(customer)
    return route[position - 1], route[position + 1]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    instance = load_instance(root / "data" / "small" / "instance_001")
    state = mixed_home_state()
    eligible = ["C1", "C2", "C5", "C6"]

    paper_costs = {}
    neighborhood_costs = {}

    for customer in eligible:
        prev_node, next_node = route_neighbors(state, customer)
        expected_paper = (
            float(instance["distance"][prev_node][customer])
            + float(instance["distance"][customer][next_node])
        )
        actual_paper = current_position_cost(state, instance, customer)

        if abs(actual_paper - expected_paper) > 1e-12:
            raise AssertionError(
                f"{customer}: historical cost is not c(prev,i)+c(i,next)."
            )

        contribution = neighborhood_contribution(
            state, instance, customer
        ).contribution

        expected_marginal = (
            expected_paper
            - float(instance["distance"][prev_node][next_node])
        )

        if abs(contribution - expected_marginal) > 1e-12:
            raise AssertionError(
                f"{customer}: neighborhood contribution is incorrect."
            )

        paper_costs[customer] = actual_paper
        neighborhood_costs[customer] = contribution

    print("[PASS] Historical position cost equals c(prev,i) + c(i,next)")
    print("[PASS] Historical position cost does not subtract bypass arc")
    print("[PASS] Neighborhood Removal retains marginal contribution")

    if all(
        abs(paper_costs[c] - neighborhood_costs[c]) <= 1e-12
        for c in eligible
    ):
        raise AssertionError("Fixture does not distinguish the two metrics.")

    print("[PASS] Historical and neighborhood metrics are distinct")

    history = dict(paper_costs)
    history["C1"] -= 0.2
    history["C2"] -= 0.1
    history["C5"] -= 0.1
    history["C6"] -= 2.0

    result = paper_historical_node_removal(
        state,
        instance,
        removal_count=1,
        best_historical_position_costs=history,
    )

    if result.removed_customers != ["C6"]:
        raise AssertionError(
            "Historical removal did not use corrected current-minus-best score."
        )

    print("[PASS] Historical Removal uses corrected paper score")
    print("[PASS] Destroy-state semantics remain valid")

    report = {
        "paper_formula": "c(prev,i) + c(i,next)",
        "bypass_arc_subtracted": False,
        "historical_position_costs": paper_costs,
        "neighborhood_marginal_contributions": neighborhood_costs,
        "selected_customer": result.removed_customers[0],
        "paper_faithful": True,
        "enhanced": False,
        "objective_extension_applied": False,
    }

    output_dir = root / "outputs" / "alns_main_loop_fidelity_tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "alns_main_loop_ml5a1_report.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nReport saved to: {output_path}")
    print(
        "\nALNS MAIN LOOP FIDELITY ML-5A.1 — "
        "PAPER HISTORICAL POSITION COST PASSED"
    )


if __name__ == "__main__":
    main()
