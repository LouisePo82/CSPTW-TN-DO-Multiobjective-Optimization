from __future__ import annotations

from itertools import permutations
from pathlib import Path
import json

from core.instance_loader import load_instance
from alns_solver.solution_state import ALNSSolutionState
from alns_solver.paper_local_search import (
    swap_intra_classic_classic,
    swap_inter_classic_classic,
    swap_inter_classic_crowd,
    swap_inter_crowd_crowd,
)

EMISSION_FACTORS = (3.0, 1.0)
EPSILON = 1e-10


def solution(
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


def assert_improvement(
    result,
    instance: dict,
    label: str,
) -> None:
    if not result.improved:
        raise AssertionError(
            f"{label}: no improving swap was accepted."
        )

    if not (
        result.final_objective
        < result.base_objective - EPSILON
    ):
        raise AssertionError(
            f"{label}: objective did not strictly improve."
        )

    checked = solution(
        result.state,
        instance,
    )

    if not checked.validator_pass:
        raise AssertionError(
            f"{label}: accepted state is invalid: "
            f"{checked.validation_errors}"
        )


def complete_state(
    *,
    dv1: list[str],
    dv2: list[str],
    od1: list[str],
    od2: list[str],
    assignments: dict,
) -> ALNSSolutionState:
    return ALNSSolutionState(
        dv_routes={
            "DV1": list(dv1),
            "DV2": list(dv2),
        },
        od_routes={
            "OD1": list(od1),
            "OD2": list(od2),
        },
        assignments=assignments,
    )


def intra_base_assignments() -> dict:
    return {
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
    }


def find_intra_fixture(
    instance: dict,
) -> tuple[ALNSSolutionState, ALNSSolutionState]:
    valid = {}

    for middle in permutations(
        ["C1", "TN1", "A1"]
    ):
        route = (
            "S",
            *middle,
            "T",
        )
        state = complete_state(
            dv1=[],
            dv2=list(route),
            od1=["O1", "TN1", "C2", "D1"],
            od2=["O2", "S", "C5", "C6", "D2"],
            assignments=intra_base_assignments(),
        )
        checked = solution(
            state,
            instance,
        )
        if checked.validator_pass:
            valid[route] = (
                float(checked.objective),
                state,
            )

    improving = []

    for route, (
        current_objective,
        current_state,
    ) in valid.items():
        route_list = list(route)
        eligible_positions = [
            index
            for index in range(1, len(route_list) - 1)
            if route_list[index] in {"C1", "TN1"}
        ]

        for first_index, first in enumerate(
            eligible_positions
        ):
            for second in eligible_positions[
                first_index + 1 :
            ]:
                swapped = list(route_list)
                swapped[first], swapped[second] = (
                    swapped[second],
                    swapped[first],
                )
                swapped_key = tuple(swapped)

                if swapped_key not in valid:
                    continue

                next_objective, next_state = valid[
                    swapped_key
                ]

                if (
                    next_objective
                    < current_objective - EPSILON
                ):
                    improving.append(
                        (
                            current_objective
                            - next_objective,
                            current_state,
                            next_state,
                        )
                    )

    if not improving:
        diagnostics = {
            "|".join(route): objective
            for route, (
                objective,
                _,
            ) in valid.items()
        }
        raise AssertionError(
            "No controlled improving intra-DV swap exists. "
            f"Valid objectives={diagnostics}"
        )

    improving.sort(
        key=lambda item: item[0],
        reverse=True,
    )
    _, high, low = improving[0]
    return high, low


def inter_dv_state(
    *,
    c1_vehicle: str,
    c2_vehicle: str,
) -> ALNSSolutionState:
    routes = {
        "DV1": ["S", "T"],
        "DV2": ["S", "A1", "T"],
    }

    routes[c1_vehicle].insert(
        len(routes[c1_vehicle]) - 1,
        "C1",
    )
    routes[c2_vehicle].insert(
        len(routes[c2_vehicle]) - 1,
        "C2",
    )

    return complete_state(
        dv1=routes["DV1"],
        dv2=routes["DV2"],
        od1=["O1", "S", "C5", "D1"],
        od2=["O2", "S", "C6", "D2"],
        assignments={
            "C1": {
                "mode": "DV_HOME",
                "vehicle": c1_vehicle,
            },
            "C2": {
                "mode": "DV_HOME",
                "vehicle": c2_vehicle,
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
                "driver": "OD1",
                "pickup": "S",
            },
            "C6": {
                "mode": "OD_HOME",
                "driver": "OD2",
                "pickup": "S",
            },
        },
    )


def cross_fleet_state(
    *,
    dv_customer: str,
    od_customer: str,
) -> ALNSSolutionState:
    return complete_state(
        dv1=[],
        dv2=["S", dv_customer, "A1", "T"],
        od1=["O1", "S", od_customer, "D1"],
        od2=["O2", "S", "C5", "C6", "D2"],
        assignments={
            dv_customer: {
                "mode": "DV_HOME",
                "vehicle": "DV2",
            },
            od_customer: {
                "mode": "OD_HOME",
                "driver": "OD1",
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
                "mode": "OD_HOME",
                "driver": "OD2",
                "pickup": "S",
            },
        },
    )


def crowd_crowd_state(
    *,
    od1_customer: str,
    od2_customer: str,
) -> ALNSSolutionState:
    return complete_state(
        dv1=[],
        dv2=["S", "A1", "T"],
        od1=[
            "O1",
            "S",
            od1_customer,
            "C5",
            "D1",
        ],
        od2=[
            "O2",
            "S",
            od2_customer,
            "C6",
            "D2",
        ],
        assignments={
            od1_customer: {
                "mode": "OD_HOME",
                "driver": "OD1",
                "pickup": "S",
            },
            od2_customer: {
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
                "driver": "OD1",
                "pickup": "S",
            },
            "C6": {
                "mode": "OD_HOME",
                "driver": "OD2",
                "pickup": "S",
            },
        },
    )


def choose_high_low(
    first: ALNSSolutionState,
    second: ALNSSolutionState,
    instance: dict,
    label: str,
) -> tuple[ALNSSolutionState, ALNSSolutionState]:
    first_solution = solution(
        first,
        instance,
    )
    second_solution = solution(
        second,
        instance,
    )

    if not first_solution.validator_pass:
        raise AssertionError(
            f"{label} first fixture invalid: "
            f"{first_solution.validation_errors}"
        )

    if not second_solution.validator_pass:
        raise AssertionError(
            f"{label} second fixture invalid: "
            f"{second_solution.validation_errors}"
        )

    first_objective = float(
        first_solution.objective
    )
    second_objective = float(
        second_solution.objective
    )

    if abs(
        first_objective - second_objective
    ) <= EPSILON:
        raise AssertionError(
            f"{label} controlled states have equal objective: "
            f"{first_objective}"
        )

    if first_objective > second_objective:
        return first, second

    return second, first


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    instance = load_instance(
        root / "data" / "small" / "instance_001"
    )
    report = {}

    # ---------------------------------------------------------
    # LS-2A — Intra DV swap
    # ---------------------------------------------------------
    high, low = find_intra_fixture(
        instance
    )

    result = swap_intra_classic_classic(
        high,
        instance,
        lambda_value=0.0,
        cost_bounds=None,
        emission_bounds=None,
        emission_factors=EMISSION_FACTORS,
    )
    assert_improvement(
        result,
        instance,
        "swap_intra_classic_classic",
    )

    if (
        result.state.dv_routes["DV2"]
        != low.dv_routes["DV2"]
    ):
        raise AssertionError(
            "Intra-DV swap did not recover the controlled "
            "lower-objective route."
        )

    print("[PASS] Intra classic-classic swaps two eligible DV nodes")
    print("[PASS] Intra classic-classic accepts strict improvement")
    print("[PASS] Intra classic-classic accepted state is valid")

    report[result.operator_name] = {
        "base_objective": result.base_objective,
        "final_objective": result.final_objective,
        "base_route": high.dv_routes["DV2"],
        "final_route": result.state.dv_routes["DV2"],
        "details": result.details,
    }

    # ---------------------------------------------------------
    # LS-2B — Inter DV swap
    # ---------------------------------------------------------
    first = inter_dv_state(
        c1_vehicle="DV1",
        c2_vehicle="DV2",
    )
    second = inter_dv_state(
        c1_vehicle="DV2",
        c2_vehicle="DV1",
    )
    high, _ = choose_high_low(
        first,
        second,
        instance,
        "Inter-DV",
    )

    result = swap_inter_classic_classic(
        high,
        instance,
        lambda_value=0.0,
        cost_bounds=None,
        emission_bounds=None,
        emission_factors=EMISSION_FACTORS,
    )
    assert_improvement(
        result,
        instance,
        "swap_inter_classic_classic",
    )

    if (
        result.details["first_vehicle"]
        == result.details["second_vehicle"]
    ):
        raise AssertionError(
            "Inter-DV swap used the same DV twice."
        )

    print("[PASS] Inter classic-classic swaps nodes between two DVs")
    print("[PASS] Inter classic-classic updates customer ownership")
    print("[PASS] Inter classic-classic accepted state is valid")

    report[result.operator_name] = {
        "base_objective": result.base_objective,
        "final_objective": result.final_objective,
        "details": result.details,
    }

    # ---------------------------------------------------------
    # LS-2C — DV/OD swap
    # ---------------------------------------------------------
    first = cross_fleet_state(
        dv_customer="C1",
        od_customer="C2",
    )
    second = cross_fleet_state(
        dv_customer="C2",
        od_customer="C1",
    )
    high, _ = choose_high_low(
        first,
        second,
        instance,
        "DV-OD",
    )

    result = swap_inter_classic_crowd(
        high,
        instance,
        lambda_value=0.0,
        cost_bounds=None,
        emission_bounds=None,
        emission_factors=EMISSION_FACTORS,
    )
    assert_improvement(
        result,
        instance,
        "swap_inter_classic_crowd",
    )

    dv_customer = result.details["dv_customer"]
    od_customer = result.details["od_customer"]

    if (
        result.state.assignments[dv_customer]["mode"]
        != "OD_HOME"
    ):
        raise AssertionError(
            "DV customer did not become OD_HOME."
        )

    if (
        result.state.assignments[od_customer]["mode"]
        != "DV_HOME"
    ):
        raise AssertionError(
            "OD customer did not become DV_HOME."
        )

    print("[PASS] Classic-crowd swaps two Type-1 customers")
    print("[PASS] Classic-crowd exchanges delivery modes")
    print("[PASS] Classic-crowd accepted state is valid")

    report[result.operator_name] = {
        "base_objective": result.base_objective,
        "final_objective": result.final_objective,
        "details": result.details,
    }

    # ---------------------------------------------------------
    # LS-2D — OD/OD swap
    # ---------------------------------------------------------
    first = crowd_crowd_state(
        od1_customer="C1",
        od2_customer="C2",
    )
    second = crowd_crowd_state(
        od1_customer="C2",
        od2_customer="C1",
    )
    high, _ = choose_high_low(
        first,
        second,
        instance,
        "OD-OD",
    )

    result = swap_inter_crowd_crowd(
        high,
        instance,
        lambda_value=0.0,
        cost_bounds=None,
        emission_bounds=None,
        emission_factors=EMISSION_FACTORS,
    )
    assert_improvement(
        result,
        instance,
        "swap_inter_crowd_crowd",
    )

    if (
        result.details["first_driver"]
        == result.details["second_driver"]
    ):
        raise AssertionError(
            "Crowd-crowd swap used the same OD twice."
        )

    print("[PASS] Crowd-crowd swaps Type-1 customers across ODs")
    print("[PASS] Crowd-crowd updates driver and pickup ownership")
    print("[PASS] Crowd-crowd accepted state is valid")

    report[result.operator_name] = {
        "base_objective": result.base_objective,
        "final_objective": result.final_objective,
        "details": result.details,
    }

    # ---------------------------------------------------------
    # LS-2E — Non-improving contract
    # ---------------------------------------------------------
    no_change = swap_inter_crowd_crowd(
        second,
        instance,
        lambda_value=0.0,
        cost_bounds=None,
        emission_bounds=None,
        emission_factors=EMISSION_FACTORS,
    )

    if no_change.improved:
        raise AssertionError(
            "A non-improving OD-OD swap was accepted."
        )

    print("[PASS] Non-improving swap candidates are rejected")

    output_dir = (
        root / "outputs" / "alns_local_search_fidelity_tests"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path = (
        output_dir / "local_search_ls2_report.json"
    )
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nReport saved to: {output_path}")
    print(
        "\nLOCAL SEARCH FIDELITY LS-2 — "
        "SWAP OPERATORS PASSED"
    )


if __name__ == "__main__":
    main()
