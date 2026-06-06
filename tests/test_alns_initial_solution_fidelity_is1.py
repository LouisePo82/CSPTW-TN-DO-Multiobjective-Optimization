from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import random

from core.instance_loader import load_instance
from alns_solver.paper_initial_solution import (
    InitialSolutionTrace,
    initialize_empty_state,
    phase1_assign_type1_to_ods,
    phase2_stabilize_active_tns,
    phase3_insert_remaining_customers,
)


EMISSION_FACTORS = (3.0, 1.0)


def manual_tn_phase1_state(
    instance: dict,
) -> tuple:
    state = initialize_empty_state(
        instance
    )

    state.od_routes["OD1"] = [
        instance["vehicles"]["OD1"]["origin"],
        "TN1",
        "C1",
        instance["vehicles"]["OD1"]["destination"],
    ]

    state.assign_customer(
        "C1",
        {
            "mode": "OD_HOME",
            "driver": "OD1",
            "pickup": "TN1",
        },
    )

    trace = InitialSolutionTrace(
        seed=0,
    )

    return state, trace


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
    # IS-1A — Phase 1
    # =========================================================
    phase1_orders = []

    for seed in range(10):
        trace = InitialSolutionTrace(
            seed=seed,
        )

        state = phase1_assign_type1_to_ods(
            initialize_empty_state(instance),
            instance,
            rng=random.Random(seed),
            trace=trace,
        )

        phase1_orders.append(
            tuple(trace.phase1_customer_order)
        )

        if set(state.assignments) - set(
            instance["type1"]
        ):
            raise AssertionError(
                "Phase 1 assigned a non-Type-1 customer."
            )

        if any(
            assignment.get("mode") != "OD_HOME"
            for assignment in state.assignments.values()
        ):
            raise AssertionError(
                "Phase 1 created a non-OD assignment."
            )

        if any(
            state.dv_routes[vehicle]
            for vehicle in instance["dvs"]
        ):
            raise AssertionError(
                "Phase 1 inserted a TN or customer into a DV route."
            )

    if len(set(phase1_orders)) < 2:
        raise AssertionError(
            "Phase 1 customer order did not vary across seeds."
        )

    print("[PASS] Phase 1 processes Type 1 customers only")
    print("[PASS] Phase 1 creates OD assignments only")
    print("[PASS] Phase 1 leaves all DV routes empty")
    print("[PASS] Phase 1 customer order varies by seed")

    report["phase1"] = {
        "unique_orders_10_seeds": len(
            set(phase1_orders)
        ),
        "orders": [
            list(order)
            for order in phase1_orders
        ],
    }

    # =========================================================
    # IS-1B — Phase 2 success
    # =========================================================
    state, trace = manual_tn_phase1_state(
        instance
    )

    phase2_success = phase2_stabilize_active_tns(
        state,
        instance,
        trace=trace,
    )

    if "TN1" not in trace.fixed_tn_positions:
        raise AssertionError(
            "Phase 2 did not fix active TN1."
        )

    fixed = trace.fixed_tn_positions["TN1"]
    vehicle = fixed["vehicle"]
    position = fixed["position"]

    if (
        phase2_success.dv_routes[vehicle][position]
        != "TN1"
    ):
        raise AssertionError(
            "TN1 was not inserted at the recorded fixed position."
        )

    print("[PASS] Phase 2 detects and inserts active TN")
    print("[PASS] Phase 2 records fixed TN position")

    report["phase2_success"] = {
        "fixed_tn_positions": (
            trace.fixed_tn_positions
        ),
        "dv_routes": phase2_success.dv_routes,
    }

    # =========================================================
    # IS-1C — Depot fallback
    # =========================================================
    fallback_instance = deepcopy(instance)

    for vehicle in fallback_instance["dvs"]:
        fallback_instance["vehicles"][vehicle][
            "capacity"
        ] = 0

    state, trace = manual_tn_phase1_state(
        fallback_instance
    )

    depot_fallback = phase2_stabilize_active_tns(
        state,
        fallback_instance,
        trace=trace,
    )

    if (
        depot_fallback.assignments["C1"]["pickup"]
        != fallback_instance["start_depot"]
    ):
        raise AssertionError(
            "Phase 2 did not fall back from TN to depot."
        )

    if not trace.depot_fallbacks:
        raise AssertionError(
            "Depot fallback was not recorded."
        )

    print("[PASS] Phase 2 falls back from TN to depot")

    report["phase2_depot_fallback"] = {
        "fallbacks": trace.depot_fallbacks,
        "od_route": depot_fallback.od_routes["OD1"],
    }

    # =========================================================
    # IS-1D — Unassigned fallback
    # =========================================================
    unassigned_instance = deepcopy(
        fallback_instance
    )

    unassigned_instance["vehicles"]["OD1"][
        "latest"
    ] = 1.0

    state, trace = manual_tn_phase1_state(
        unassigned_instance
    )

    unassigned_fallback = (
        phase2_stabilize_active_tns(
            state,
            unassigned_instance,
            trace=trace,
        )
    )

    if "C1" not in (
        unassigned_fallback.unassigned_customers
    ):
        raise AssertionError(
            "Phase 2 did not return C1 to unassigned."
        )

    if unassigned_fallback.od_routes["OD1"] != []:
        raise AssertionError(
            "Phase 2 did not deactivate OD1."
        )

    if not trace.unassigned_fallbacks:
        raise AssertionError(
            "Unassigned fallback was not recorded."
        )

    print("[PASS] Phase 2 returns customers to unassigned")
    print("[PASS] Phase 2 deactivates infeasible OD route")

    report["phase2_unassigned_fallback"] = {
        "fallbacks": trace.unassigned_fallbacks,
        "unassigned": sorted(
            unassigned_fallback.unassigned_customers
        ),
    }

    # =========================================================
    # IS-1E — Phase 3 completion and TN preservation
    # =========================================================
    state, trace = manual_tn_phase1_state(
        instance
    )

    phase2_state = phase2_stabilize_active_tns(
        state,
        instance,
        trace=trace,
    )

    fixed_before = deepcopy(
        trace.fixed_tn_positions
    )

    completed = phase3_insert_remaining_customers(
        phase2_state,
        instance,
        trace=trace,
        lambda_value=0.0,
        cost_bounds=None,
        emission_bounds=None,
        emission_factors=EMISSION_FACTORS,
        strategy_2_mode="paper_random_dv",
        strategy_2_seed=0,
    )

    for tn, info in fixed_before.items():
        route = completed.dv_routes[
            info["vehicle"]
        ]

        if route[info["position"]] != tn:
            raise AssertionError(
                "Phase 3 moved a fixed TN."
            )

    solution = completed.to_core_solution(
        instance=instance,
        lambda_value=0.0,
        objective_mode="cost",
        emission_factors=EMISSION_FACTORS,
        require_complete=True,
    )

    if not solution.validator_pass:
        raise AssertionError(
            "Phase 3 final solution is invalid: "
            f"{solution.validation_errors}"
        )

    print("[PASS] Phase 3 inserts all remaining customers")
    print("[PASS] Phase 3 preserves fixed TN positions")
    print("[PASS] Phase 3 final solution passes shared validator")

    report["phase3"] = {
        "initial_unassigned": (
            trace.phase3_initial_unassigned
        ),
        "insertion_order": (
            trace.phase3_insertion_order
        ),
        "fixed_tn_positions": (
            trace.fixed_tn_positions
        ),
        "validator_pass": solution.validator_pass,
        "cost": solution.cost,
        "emission": solution.emission,
    }

    output_dir = (
        root
        / "outputs"
        / "alns_initial_solution_fidelity_tests"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "initial_solution_is1_report.json"
    )

    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nReport saved to: {output_path}")
    print(
        "\nINITIAL SOLUTION FIDELITY IS-1 — "
        "THREE-PHASE CORRECTNESS PASSED"
    )


if __name__ == "__main__":
    main()
