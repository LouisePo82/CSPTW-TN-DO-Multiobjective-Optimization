from __future__ import annotations

from pathlib import Path
import json

from core.instance_loader import load_instance
from alns_solver.solution_state import ALNSSolutionState
from alns_solver.repair_operators import enumerate_insertion_candidates
from alns_solver.paper_local_search import (
    move_intra_classic_classic,
    move_inter_classic_classic,
    move_inter_classic_crowd,
    move_inter_crowd_classic,
)

EMISSION_FACTORS = (3.0, 1.0)
EPSILON = 1e-10


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



def intra_fixture_state(
    route: list[str],
) -> ALNSSolutionState:
    """
    Complete controlled state for intra-DV relocation.

    DV2 contains:
    - C1 as a DV-home customer;
    - A1 serving Type-2 customers C3 and C4;
    - TN1 supplying OD1 customer C2.

    C6 is assigned to OD2 together with C5, keeping DV2 load within capacity.
    The movable paper nodes are C1 and TN1; A1 remains fixed.
    """
    return ALNSSolutionState(
        dv_routes={
            "DV1": [],
            "DV2": list(route),
        },
        od_routes={
            "OD1": ["O1", "TN1", "C2", "D1"],
            "OD2": ["O2", "S", "C5", "C6", "D2"],
        },
        assignments={
            "C1": {
                "mode": "DV_HOME",
                "vehicle": "DV2",
            },
            "C2": {
                "mode": "OD_HOME",
                "driver": "OD1",
                "pickup": "TN1",
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
                "mode": "OD_HOME",
                "driver": "OD2",
                "pickup": "S",
            },
        },
    )


def objective(
    state: ALNSSolutionState,
    instance: dict,
) -> float:
    return float(
        state.evaluate(
            instance=instance,
            lambda_value=0.0,
            cost_bounds=None,
            emission_bounds=None,
            emission_factors=EMISSION_FACTORS,
        )["objective"]
    )


def core_solution(
    state: ALNSSolutionState,
    instance: dict,
):
    return state.to_core_solution(
        instance=instance,
        lambda_value=0.0,
        objective_mode="cost",
        emission_factors=EMISSION_FACTORS,
        require_complete=True,
    )


def is_valid(
    state: ALNSSolutionState,
    instance: dict,
) -> bool:
    return bool(
        core_solution(
            state,
            instance,
        ).validator_pass
    )



def controlled_intra_pair(
    instance: dict,
) -> tuple[ALNSSolutionState, ALNSSolutionState]:
    """
    Return one valid state and one strictly better valid state reachable by
    a single intra-route relocation of C1 or TN1.

    The fixture searches all permutations of [C1, A1, TN1], validates them
    through the shared core, and independently checks one-node relocation
    adjacency. This avoids assuming that a particular pair of route orders
    must differ on the current geometry.
    """
    from itertools import permutations

    valid_states: dict[
        tuple[str, ...],
        tuple[float, ALNSSolutionState],
    ] = {}

    for middle_nodes in permutations(
        ["C1", "A1", "TN1"]
    ):
        route = (
            "S",
            *middle_nodes,
            "T",
        )
        state = intra_fixture_state(
            list(route)
        )
        solution = core_solution(
            state,
            instance,
        )

        if solution.validator_pass:
            valid_states[route] = (
                float(solution.objective),
                state,
            )

    if len(valid_states) < 2:
        raise AssertionError(
            "Controlled intra fixture produced fewer than two "
            f"valid route permutations: {list(valid_states)}"
        )

    movable_nodes = {"C1", "TN1"}
    improving_pairs = []

    for high_route, (
        high_objective,
        high_state,
    ) in valid_states.items():
        route_list = list(high_route)

        for source_position in range(
            1,
            len(route_list) - 1,
        ):
            node = route_list[source_position]

            if node not in movable_nodes:
                continue

            reduced = (
                route_list[:source_position]
                + route_list[source_position + 1 :]
            )

            for target_position in range(
                1,
                len(reduced),
            ):
                neighbor = list(reduced)
                neighbor.insert(
                    target_position,
                    node,
                )
                neighbor_key = tuple(neighbor)

                if neighbor_key == high_route:
                    continue

                if neighbor_key not in valid_states:
                    continue

                low_objective, low_state = (
                    valid_states[neighbor_key]
                )

                if (
                    low_objective
                    < high_objective - EPSILON
                ):
                    improving_pairs.append(
                        (
                            high_objective
                            - low_objective,
                            high_state,
                            low_state,
                            node,
                        )
                    )

    if not improving_pairs:
        diagnostic = {
            "|".join(route): objective_value
            for route, (
                objective_value,
                _,
            ) in valid_states.items()
        }
        raise AssertionError(
            "No valid strictly improving one-node relocation exists "
            "for C1 or TN1 in the controlled fixture. "
            f"Valid objectives={diagnostic}"
        )

    improving_pairs.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    _, high_state, low_state, _ = (
        improving_pairs[0]
    )

    return high_state, low_state


def detach_customer(
    state: ALNSSolutionState,
    customer: str,
) -> ALNSSolutionState:
    result = state.copy()
    assignment = result.assignments.pop(
        customer
    )

    if assignment["mode"] == "OD_HOME":
        driver = assignment["driver"]
        result.od_routes[driver] = [
            node
            for node in result.od_routes[driver]
            if node != customer
        ]

        if not any(
            current_assignment.get("mode")
            == "OD_HOME"
            and current_assignment.get("driver")
            == driver
            for current_assignment
            in result.assignments.values()
        ):
            result.od_routes[driver] = []

    elif assignment["mode"] == "DV_HOME":
        vehicle = assignment["vehicle"]
        result.dv_routes[vehicle] = [
            node
            for node in result.dv_routes[vehicle]
            if node != customer
        ]

    result.unassigned_customers.add(
        customer
    )
    result.invalidate_cache()
    return result


def candidates_for(
    instance: dict,
    customer: str,
):
    partial = detach_customer(
        cost_anchor_state(),
        customer,
    )

    return enumerate_insertion_candidates(
        partial,
        instance,
        customer,
        lambda_value=0.0,
        cost_bounds=None,
        emission_bounds=None,
        emission_factors=EMISSION_FACTORS,
        strategy_2_mode="exhaustive_best_dv",
        strategy_2_seed=0,
    )


def find_pair(
    instance: dict,
    customer: str,
    high_mode: str,
    low_mode: str,
    same_vehicle: bool | None = None,
):
    candidates = candidates_for(
        instance,
        customer,
    )

    highs = [
        candidate
        for candidate in candidates
        if candidate.mode == high_mode
    ]
    lows = [
        candidate
        for candidate in candidates
        if candidate.mode == low_mode
    ]

    pairs = []

    for high in highs:
        for low in lows:
            if (
                same_vehicle is True
                and high.details.get("vehicle")
                != low.details.get("vehicle")
            ):
                continue

            if (
                same_vehicle is False
                and high.details.get("vehicle")
                == low.details.get("vehicle")
            ):
                continue

            if (
                low.objective
                < high.objective - EPSILON
            ):
                pairs.append(
                    (
                        high,
                        low,
                    )
                )

    if not pairs:
        diagnostic = [
            {
                "mode": candidate.mode,
                "objective": candidate.objective,
                "details": candidate.details,
            }
            for candidate in candidates
        ]

        raise AssertionError(
            f"No controlled pair for {customer}: "
            f"{high_mode}->{low_mode}, "
            f"same_vehicle={same_vehicle}. "
            f"Candidates={diagnostic}"
        )

    return max(
        pairs,
        key=lambda pair: (
            pair[0].objective
            - pair[1].objective
        ),
    )


def assert_improvement(
    result,
    instance: dict,
    name: str,
) -> None:
    if not result.improved:
        raise AssertionError(
            f"{name}: no improvement accepted"
        )

    if not (
        result.final_objective
        < result.base_objective - EPSILON
    ):
        raise AssertionError(
            f"{name}: objective did not strictly improve"
        )

    solution = core_solution(
        result.state,
        instance,
    )

    if not solution.validator_pass:
        raise AssertionError(
            f"{name}: accepted state is invalid: "
            f"{solution.validation_errors}"
        )


def main() -> None:
    root = Path(__file__).resolve().parents[1]

    instance = load_instance(
        root
        / "data"
        / "small"
        / "instance_001"
    )

    report = {}

    # ---------------------------------------------------------
    # LS-1A — Intra-DV move using TN relocation.
    # ---------------------------------------------------------
    intra_high, intra_low = (
        controlled_intra_pair(
            instance
        )
    )

    result = move_intra_classic_classic(
        intra_high,
        instance,
        lambda_value=0.0,
        cost_bounds=None,
        emission_bounds=None,
        emission_factors=EMISSION_FACTORS,
    )

    assert_improvement(
        result,
        instance,
        "move_intra_classic_classic",
    )

    if result.details["vehicle"] != "DV2":
        raise AssertionError(
            "Intra move changed or selected the wrong vehicle."
        )

    if result.details["node"] not in {"C1", "TN1"}:
        raise AssertionError(
            "Intra paper move relocated an ineligible node. "
            f"Moved={result.details['node']}"
        )

    if (
        result.state.dv_routes["DV2"]
        != intra_low.dv_routes["DV2"]
    ):
        raise AssertionError(
            "Intra operator did not recover the controlled "
            "lower-objective route."
        )

    print(
        "[PASS] Intra classic-classic uses one DV route only"
    )
    print(
        "[PASS] Intra classic-classic relocates a paper-eligible "
        "DV-home customer or TN"
    )
    print(
        "[PASS] Intra classic-classic accepts strict improvement"
    )
    print(
        "[PASS] Intra classic-classic accepted state is valid"
    )

    report[result.operator_name] = {
        "base_objective": (
            result.base_objective
        ),
        "final_objective": (
            result.final_objective
        ),
        "base_route": (
            intra_high.dv_routes["DV2"]
        ),
        "expected_route": (
            intra_low.dv_routes["DV2"]
        ),
        "final_route": (
            result.state.dv_routes["DV2"]
        ),
        "details": result.details,
    }

    # ---------------------------------------------------------
    # LS-1B — Inter-DV move.
    # ---------------------------------------------------------
    high, _ = find_pair(
        instance,
        "C1",
        "DV_HOME",
        "DV_HOME",
        same_vehicle=False,
    )

    result = move_inter_classic_classic(
        high.state,
        instance,
        lambda_value=0.0,
        cost_bounds=None,
        emission_bounds=None,
        emission_factors=EMISSION_FACTORS,
    )

    assert_improvement(
        result,
        instance,
        "move_inter_classic_classic",
    )

    if (
        result.details["source_vehicle"]
        == result.details["target_vehicle"]
    ):
        raise AssertionError(
            "Inter classic-classic did not change DV"
        )

    print(
        "[PASS] Inter classic-classic transfers between DVs"
    )
    print(
        "[PASS] Inter classic-classic accepts strict improvement"
    )
    print(
        "[PASS] Inter classic-classic accepted state is valid"
    )

    report[result.operator_name] = {
        "base_objective": (
            result.base_objective
        ),
        "final_objective": (
            result.final_objective
        ),
        "details": result.details,
    }

    # ---------------------------------------------------------
    # LS-1C — DV to OD, Type 1 only.
    # ---------------------------------------------------------
    high, _ = find_pair(
        instance,
        "C1",
        "DV_HOME",
        "OD_HOME",
    )

    result = move_inter_classic_crowd(
        high.state,
        instance,
        lambda_value=0.0,
        cost_bounds=None,
        emission_bounds=None,
        emission_factors=EMISSION_FACTORS,
        strategy_2_seed=0,
    )

    assert_improvement(
        result,
        instance,
        "move_inter_classic_crowd",
    )

    moved = result.details["customer"]

    if (
        result.state.assignments[moved]["mode"]
        != "OD_HOME"
    ):
        raise AssertionError(
            "Classic-to-crowd did not create OD_HOME"
        )

    if (
        int(
            instance["nodes"][moved][
                "customer_type"
            ]
        )
        != 1
    ):
        raise AssertionError(
            "Classic-to-crowd moved a non-Type-1 customer."
        )

    print(
        "[PASS] Classic-to-crowd moves Type 1 DV customer to OD"
    )
    print(
        "[PASS] Classic-to-crowd accepts strict improvement"
    )
    print(
        "[PASS] Classic-to-crowd accepted state is valid"
    )

    report[result.operator_name] = {
        "base_objective": (
            result.base_objective
        ),
        "final_objective": (
            result.final_objective
        ),
        "details": result.details,
    }

    # ---------------------------------------------------------
    # LS-1D — OD to DV, Type 1 only.
    # ---------------------------------------------------------
    od_dv_pair = None
    selected_customer = None

    for customer in instance["type1"]:
        try:
            od_dv_pair = find_pair(
                instance,
                customer,
                "OD_HOME",
                "DV_HOME",
            )
            selected_customer = customer
            break
        except AssertionError:
            continue

    if od_dv_pair is None:
        raise AssertionError(
            "No controlled OD->DV improving pair found"
        )

    high, _ = od_dv_pair

    result = move_inter_crowd_classic(
        high.state,
        instance,
        lambda_value=0.0,
        cost_bounds=None,
        emission_bounds=None,
        emission_factors=EMISSION_FACTORS,
    )

    assert_improvement(
        result,
        instance,
        "move_inter_crowd_classic",
    )

    moved = result.details["customer"]

    if (
        result.state.assignments[moved]["mode"]
        != "DV_HOME"
    ):
        raise AssertionError(
            "Crowd-to-classic did not create DV_HOME"
        )

    if (
        int(
            instance["nodes"][moved][
                "customer_type"
            ]
        )
        != 1
    ):
        raise AssertionError(
            "Crowd-to-classic moved a non-Type-1 customer."
        )

    active_tns = {
        assignment["pickup"]
        for assignment
        in result.state.assignments.values()
        if (
            assignment.get("mode") == "OD_HOME"
            and assignment.get("pickup")
            in instance["tns"]
        )
    }

    for vehicle, route in (
        result.state.dv_routes.items()
    ):
        orphan = [
            tn
            for tn in instance["tns"]
            if (
                tn in route
                and tn not in active_tns
            )
        ]

        if orphan:
            raise AssertionError(
                f"Orphan TN remains on {vehicle}: "
                f"{orphan}"
            )

    print(
        "[PASS] Crowd-to-classic moves Type 1 OD customer to DV"
    )
    print(
        "[PASS] Crowd-to-classic cleans empty OD/orphan TN structure"
    )
    print(
        "[PASS] Crowd-to-classic accepted state is valid"
    )

    report[result.operator_name] = {
        "base_objective": (
            result.base_objective
        ),
        "final_objective": (
            result.final_objective
        ),
        "details": result.details,
        "controlled_customer": (
            selected_customer
        ),
    }

    # ---------------------------------------------------------
    # LS-1E — No-improvement/non-applicable contract.
    # ---------------------------------------------------------
    baseline = cost_anchor_state()

    no_change = move_inter_classic_crowd(
        baseline,
        instance,
        lambda_value=0.0,
        cost_bounds=None,
        emission_bounds=None,
        emission_factors=EMISSION_FACTORS,
        strategy_2_seed=0,
    )

    if no_change.improved:
        raise AssertionError(
            "Non-applicable/non-improving move was accepted"
        )

    if (
        objective(
            no_change.state,
            instance,
        )
        != objective(
            baseline,
            instance,
        )
    ):
        raise AssertionError(
            "No-improvement result changed the objective."
        )

    print(
        "[PASS] Non-improving or non-applicable candidates "
        "are rejected"
    )

    output_dir = (
        root
        / "outputs"
        / "alns_local_search_fidelity_tests"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "local_search_ls1_report.json"
    )

    output_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nReport saved to: {output_path}")
    print(
        "\nLOCAL SEARCH FIDELITY LS-1 — "
        "MOVE OPERATORS PASSED"
    )


if __name__ == "__main__":
    main()
