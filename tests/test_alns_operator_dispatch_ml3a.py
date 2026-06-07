from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json

from core.instance_loader import load_instance
from alns_solver.paper_alns_main import (
    build_ml1_paper_initial_state,
)
from alns_solver.paper_operator_dispatch import (
    PAPER_DESTROY_OPERATOR_NAMES,
    PAPER_PERTURBATION_FACTOR,
    PAPER_RANDOMNESS_FACTOR,
    PAPER_RELATEDNESS_WEIGHTS,
    PAPER_REPAIR_OPERATOR_NAMES,
    PAPER_STRATEGY_2_MODE,
    PaperOperatorDispatchContext,
    dispatch_paper_destroy,
    dispatch_paper_repair,
    paper_destroy_operator_names,
    paper_repair_operator_names,
)


EXPECTED_DESTROY = (
    "random_customer_removal",
    "worst_customer_removal_deterministic",
    "worst_customer_removal_probabilistic",
    "route_removal",
    "random_adp_removal",
    "worst_adp_removal",
    "random_tn_removal",
    "related_removal_deterministic",
    "related_removal_probabilistic",
    "historical_node_removal",
    "neighborhood_removal",
    "node_neighborhood_removal",
)

EXPECTED_REPAIR = (
    "best_insertion",
    "regret_2",
    "perturbed_regret_2",
    "regret_3",
    "perturbed_best_insertion",
    "perturbed_regret_3",
)


def signature(state):
    return {
        "dv_routes": deepcopy(state.dv_routes),
        "od_routes": deepcopy(state.od_routes),
        "assignments": deepcopy(state.assignments),
        "unassigned": sorted(state.unassigned_customers),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    instance = load_instance(
        root / "data" / "small" / "instance_001"
    )

    lambda_value = 0.5
    cost_bounds = (0.0, 100.0)
    emission_bounds = (0.0, 300.0)
    emission_factors = (3.0, 1.0)

    state = build_ml1_paper_initial_state(
        instance,
        seed=42,
        lambda_value=lambda_value,
        cost_bounds=cost_bounds,
        emission_bounds=emission_bounds,
        emission_factors=emission_factors,
    )

    context = PaperOperatorDispatchContext(
        removal_count=1,
        seed=7,
        lambda_value=lambda_value,
        cost_bounds=cost_bounds,
        emission_bounds=emission_bounds,
        emission_factors=emission_factors,
    )

    # Registry contract
    if PAPER_DESTROY_OPERATOR_NAMES != EXPECTED_DESTROY:
        raise AssertionError("Paper destroy pool is incorrect.")
    if PAPER_REPAIR_OPERATOR_NAMES != EXPECTED_REPAIR:
        raise AssertionError("Paper repair pool is incorrect.")
    if paper_destroy_operator_names() != EXPECTED_DESTROY:
        raise AssertionError("Destroy name API is incorrect.")
    if paper_repair_operator_names() != EXPECTED_REPAIR:
        raise AssertionError("Repair name API is incorrect.")

    print("[PASS] Dispatcher exposes exactly 12 paper destroy operators")
    print("[PASS] Dispatcher exposes exactly 6 paper repair operators")

    if PAPER_RANDOMNESS_FACTOR != 5.0:
        raise AssertionError("Paper randomness factor is not 5.")
    if PAPER_RELATEDNESS_WEIGHTS != (5.0, 9.0, 1.0):
        raise AssertionError("Paper relatedness weights are incorrect.")
    if PAPER_PERTURBATION_FACTOR != 0.275:
        raise AssertionError("Paper perturbation factor is incorrect.")
    if PAPER_STRATEGY_2_MODE != "paper_random_dv":
        raise AssertionError("Paper Strategy II mode is incorrect.")

    print("[PASS] Paper destroy and repair constants are fixed")

    # Unsupported names
    for function, name in (
        (dispatch_paper_destroy, "unknown_destroy"),
        (dispatch_paper_repair, "unknown_repair"),
    ):
        try:
            function(
                name,
                state,
                instance,
                context=context,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"Unsupported operator was accepted: {name}"
            )

    print("[PASS] Unsupported operator names are rejected")

    # Historical context must be explicit.
    try:
        dispatch_paper_destroy(
            "historical_node_removal",
            state,
            instance,
            context=context,
        )
    except ValueError as error:
        if "best_historical_position_costs" not in str(error):
            raise
    else:
        raise AssertionError(
            "Historical removal accepted missing history context."
        )

    print("[PASS] Historical removal requires explicit history context")

    # Real controlled dispatch: route removal -> best insertion.
    before = signature(state)
    destroy_result = dispatch_paper_destroy(
        "route_removal",
        state,
        instance,
        context=context,
    )

    if signature(state) != before:
        raise AssertionError("Destroy dispatcher mutated input state.")
    if destroy_result.operator_name != "paper_route_removal":
        raise AssertionError("Route dispatch called wrong operator.")
    if not destroy_result.removed_customers:
        raise AssertionError("Route removal removed no customers.")

    print("[PASS] Route-removal dispatch calls paper implementation")
    print("[PASS] Destroy dispatch preserves input state")

    repair_result = dispatch_paper_repair(
        "best_insertion",
        destroy_result.state,
        instance,
        context=context,
    )

    if repair_result.operator_name != "best_insertion":
        raise AssertionError("Repair dispatch called wrong operator.")
    if not repair_result.validator_pass:
        raise AssertionError(
            f"Dispatched repair invalid: {repair_result.validation_errors}"
        )
    if repair_result.state.unassigned_customers:
        raise AssertionError("Dispatched repair is incomplete.")

    print("[PASS] Best-insertion dispatch completes partial state")
    print("[PASS] Repaired state passes shared validator")

    # Fixed seed reproducibility.
    repeated_destroy = dispatch_paper_destroy(
        "route_removal",
        state,
        instance,
        context=context,
    )
    repeated_repair = dispatch_paper_repair(
        "best_insertion",
        repeated_destroy.state,
        instance,
        context=context,
    )

    if signature(destroy_result.state) != signature(
        repeated_destroy.state
    ):
        raise AssertionError(
            "Destroy dispatcher is not reproducible."
        )
    if signature(repair_result.state) != signature(
        repeated_repair.state
    ):
        raise AssertionError(
            "Repair dispatcher is not reproducible."
        )

    print("[PASS] Fixed seed reproduces dispatch results")

    report = {
        "destroy_pool": list(PAPER_DESTROY_OPERATOR_NAMES),
        "repair_pool": list(PAPER_REPAIR_OPERATOR_NAMES),
        "paper_constants": {
            "randomness_factor": PAPER_RANDOMNESS_FACTOR,
            "relatedness_weights": list(
                PAPER_RELATEDNESS_WEIGHTS
            ),
            "perturbation_factor": PAPER_PERTURBATION_FACTOR,
            "strategy_2_mode": PAPER_STRATEGY_2_MODE,
        },
        "controlled_dispatch": {
            "destroy_operator": destroy_result.operator_name,
            "removed_customers": (
                destroy_result.removed_customers
            ),
            "repair_operator": repair_result.operator_name,
            "repair_validator_pass": (
                repair_result.validator_pass
            ),
        },
        "historical_contract": {
            "explicit_context_required": True,
            "metric": "two_arc_position_cost",
            "multiobjective_override": False,
        },
        "paper_faithful": True,
        "enhanced": False,
    }

    output_dir = (
        root
        / "outputs"
        / "alns_main_loop_fidelity_tests"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_dir
        / "alns_main_loop_ml3a_report.json"
    )
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nReport saved to: {output_path}")
    print(
        "\nALNS MAIN LOOP FIDELITY ML-3A — "
        "PAPER OPERATOR DISPATCH PASSED"
    )


if __name__ == "__main__":
    main()
