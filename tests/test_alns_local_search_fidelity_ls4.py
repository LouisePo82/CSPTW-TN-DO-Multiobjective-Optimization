from __future__ import annotations

from pathlib import Path
import json

from core.instance_loader import load_instance
from alns_solver.paper_local_search import (
    LocalSearchMoveResult,
    PAPER_LOCAL_SEARCH_OPERATOR_ORDER,
    run_paper_local_search,
)
from alns_solver.local_search_factory import (
    PAPER_LOCAL_SEARCH_MODE,
    ENHANCED_LOCAL_SEARCH_MODE,
    SUPPORTED_LOCAL_SEARCH_MODES,
    build_local_search,
)
from tests.test_alns_local_search_moves_ls1 import (
    controlled_intra_pair,
)

EMISSION_FACTORS = (3.0, 1.0)
EPSILON = 1e-10


def evaluate(state, instance) -> float:
    return float(
        state.evaluate(
            instance=instance,
            lambda_value=0.0,
            cost_bounds=None,
            emission_bounds=None,
            emission_factors=EMISSION_FACTORS,
        )["objective"]
    )


def state_signature(state) -> tuple:
    return (
        tuple(
            (name, tuple(route))
            for name, route
            in sorted(state.dv_routes.items())
        ),
        tuple(
            (name, tuple(route))
            for name, route
            in sorted(state.od_routes.items())
        ),
        tuple(
            (
                customer,
                tuple(sorted(assignment.items())),
            )
            for customer, assignment
            in sorted(state.assignments.items())
        ),
    )


def make_separation_registry(
    state_a,
    state_b,
    state_c,
    instance,
):
    """
    Controlled dependency:

    - operator 1 cannot improve A;
    - operator 2 changes A -> B;
    - only after that, operator 1 can change B -> C.

    Paper one-pass ends at B because operator 1 is not revisited.
    Enhanced full-sequence restart revisits operator 1 and reaches C.
    """
    signatures = {
        "A": state_signature(state_a),
        "B": state_signature(state_b),
        "C": state_signature(state_c),
    }

    def no_change(name):
        def operator(
            state,
            instance,
            *,
            lambda_value,
            cost_bounds,
            emission_bounds,
            emission_factors,
        ):
            base = evaluate(state, instance)
            return LocalSearchMoveResult(
                name,
                state.copy(),
                False,
                base,
                base,
                {},
            )
        return operator

    first_name = PAPER_LOCAL_SEARCH_OPERATOR_ORDER[0]
    second_name = PAPER_LOCAL_SEARCH_OPERATOR_ORDER[1]

    def operator_1(
        state,
        instance,
        *,
        lambda_value,
        cost_bounds,
        emission_bounds,
        emission_factors,
    ):
        base = evaluate(state, instance)

        if (
            state_signature(state) == signatures["B"]
            and evaluate(state_c, instance)
            < base - EPSILON
        ):
            return LocalSearchMoveResult(
                first_name,
                state_c.copy(),
                True,
                base,
                evaluate(state_c, instance),
                {"controlled_transition": "B_to_C"},
            )

        return LocalSearchMoveResult(
            first_name,
            state.copy(),
            False,
            base,
            base,
            {},
        )

    def operator_2(
        state,
        instance,
        *,
        lambda_value,
        cost_bounds,
        emission_bounds,
        emission_factors,
    ):
        base = evaluate(state, instance)

        if (
            state_signature(state) == signatures["A"]
            and evaluate(state_b, instance)
            < base - EPSILON
        ):
            return LocalSearchMoveResult(
                second_name,
                state_b.copy(),
                True,
                base,
                evaluate(state_b, instance),
                {"controlled_transition": "A_to_B"},
            )

        return LocalSearchMoveResult(
            second_name,
            state.copy(),
            False,
            base,
            base,
            {},
        )

    return (
        (first_name, operator_1),
        (second_name, operator_2),
        *tuple(
            (name, no_change(name))
            for name
            in PAPER_LOCAL_SEARCH_OPERATOR_ORDER[2:]
        ),
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    instance = load_instance(
        root / "data" / "small" / "instance_001"
    )

    report = {}

    if SUPPORTED_LOCAL_SEARCH_MODES != {
        PAPER_LOCAL_SEARCH_MODE,
        ENHANCED_LOCAL_SEARCH_MODE,
    }:
        raise AssertionError(
            "Local-search mode registry is incorrect."
        )

    try:
        build_local_search(
            controlled_intra_pair(instance)[0],
            instance,
            mode="unknown_mode",
            best_objective=1.0,
            lambda_value=0.0,
            cost_bounds=None,
            emission_bounds=None,
            emission_factors=EMISSION_FACTORS,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Unknown local-search mode was not rejected."
        )

    print("[PASS] Local-search factory exposes exactly two modes")
    print("[PASS] Unsupported local-search mode is rejected")

    high_state, intermediate_state = (
        controlled_intra_pair(instance)
    )

    real_final = run_paper_local_search(
        high_state,
        instance,
        best_objective=evaluate(high_state, instance),
        lambda_value=0.0,
        cost_bounds=None,
        emission_bounds=None,
        emission_factors=EMISSION_FACTORS,
    ).state

    if not (
        evaluate(real_final, instance)
        < evaluate(intermediate_state, instance)
        - EPSILON
    ):
        raise AssertionError(
            "Controlled LS-4 fixture requires a third, lower state."
        )

    registry = make_separation_registry(
        high_state,
        intermediate_state,
        real_final,
        instance,
    )

    paper = build_local_search(
        high_state,
        instance,
        mode=PAPER_LOCAL_SEARCH_MODE,
        best_objective=evaluate(high_state, instance),
        lambda_value=0.0,
        cost_bounds=None,
        emission_bounds=None,
        emission_factors=EMISSION_FACTORS,
        operator_registry=registry,
    )

    enhanced = build_local_search(
        high_state,
        instance,
        mode=ENHANCED_LOCAL_SEARCH_MODE,
        best_objective=evaluate(high_state, instance),
        lambda_value=0.0,
        cost_bounds=None,
        emission_bounds=None,
        emission_factors=EMISSION_FACTORS,
        operator_registry=registry,
        max_cycles=10,
    )

    if paper.metadata["paper_faithful"] is not True:
        raise AssertionError(
            "Paper mode lost paper_faithful=True."
        )

    if paper.metadata["restart_full_operator_sequence"]:
        raise AssertionError(
            "Paper mode incorrectly restarts full sequence."
        )

    if paper.cycles != 1:
        raise AssertionError(
            "Paper mode must perform one ordered pass."
        )

    if enhanced.metadata["paper_faithful"]:
        raise AssertionError(
            "Enhanced mode is incorrectly labeled paper-faithful."
        )

    if not enhanced.metadata[
        "restart_full_operator_sequence"
    ]:
        raise AssertionError(
            "Enhanced mode did not enable full-sequence restart."
        )

    if enhanced.cycles < 2:
        raise AssertionError(
            "Enhanced controlled fixture did not restart sequence."
        )

    if (
        state_signature(paper.state)
        != state_signature(intermediate_state)
    ):
        raise AssertionError(
            "Paper one-pass controller did not stop at state B."
        )

    if (
        state_signature(enhanced.state)
        != state_signature(real_final)
    ):
        raise AssertionError(
            "Enhanced controller did not revisit operator 1 "
            "and reach state C."
        )

    if not (
        enhanced.final_objective
        < paper.final_objective - EPSILON
    ):
        raise AssertionError(
            "Controlled enhanced mode did not improve beyond paper mode."
        )

    print("[PASS] Paper mode keeps delta_ls=0.1 and one ordered pass")
    print("[PASS] Paper mode is labeled paper-faithful")
    print("[PASS] Enhanced mode is explicitly non-paper")
    print("[PASS] Enhanced mode restarts the full operator sequence")
    print("[PASS] Controlled fixture separates paper and enhanced results")

    try:
        build_local_search(
            high_state,
            instance,
            mode=PAPER_LOCAL_SEARCH_MODE,
            best_objective=evaluate(high_state, instance),
            lambda_value=0.0,
            cost_bounds=None,
            emission_bounds=None,
            emission_factors=EMISSION_FACTORS,
            delta_ls=0.2,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Paper mode allowed delta_ls override."
        )

    print("[PASS] Paper delta_ls cannot be overridden")

    paper_repeat = build_local_search(
        high_state,
        instance,
        mode=PAPER_LOCAL_SEARCH_MODE,
        best_objective=evaluate(high_state, instance),
        lambda_value=0.0,
        cost_bounds=None,
        emission_bounds=None,
        emission_factors=EMISSION_FACTORS,
        operator_registry=registry,
    )

    enhanced_repeat = build_local_search(
        high_state,
        instance,
        mode=ENHANCED_LOCAL_SEARCH_MODE,
        best_objective=evaluate(high_state, instance),
        lambda_value=0.0,
        cost_bounds=None,
        emission_bounds=None,
        emission_factors=EMISSION_FACTORS,
        operator_registry=registry,
        max_cycles=10,
    )

    if (
        state_signature(paper_repeat.state)
        != state_signature(paper.state)
    ):
        raise AssertionError(
            "Paper factory is not reproducible."
        )

    if (
        state_signature(enhanced_repeat.state)
        != state_signature(enhanced.state)
    ):
        raise AssertionError(
            "Enhanced factory is not reproducible."
        )

    print("[PASS] Both modes are reproducible for the same input")

    paper_solution = paper.state.to_core_solution(
        instance=instance,
        lambda_value=0.0,
        objective_mode="cost",
        emission_factors=EMISSION_FACTORS,
        require_complete=True,
    )
    enhanced_solution = enhanced.state.to_core_solution(
        instance=instance,
        lambda_value=0.0,
        objective_mode="cost",
        emission_factors=EMISSION_FACTORS,
        require_complete=True,
    )

    if not paper_solution.validator_pass:
        raise AssertionError(
            f"Paper final state invalid: "
            f"{paper_solution.validation_errors}"
        )

    if not enhanced_solution.validator_pass:
        raise AssertionError(
            f"Enhanced final state invalid: "
            f"{enhanced_solution.validation_errors}"
        )

    print("[PASS] Both mode results pass the shared validator")

    report["paper_mode"] = {
        "metadata": paper.metadata,
        "cycles": paper.cycles,
        "base_objective": paper.base_objective,
        "final_objective": paper.final_objective,
        "accepted_moves": paper.accepted_moves,
        "operator_calls": paper.operator_calls,
        "validator_pass": paper_solution.validator_pass,
    }
    report["enhanced_mode"] = {
        "metadata": enhanced.metadata,
        "cycles": enhanced.cycles,
        "base_objective": enhanced.base_objective,
        "final_objective": enhanced.final_objective,
        "accepted_moves": enhanced.accepted_moves,
        "operator_calls": enhanced.operator_calls,
        "validator_pass": enhanced_solution.validator_pass,
    }
    report["separation"] = {
        "same_final_structure": (
            state_signature(paper.state)
            == state_signature(enhanced.state)
        ),
        "paper_minus_enhanced_objective": (
            paper.final_objective
            - enhanced.final_objective
        ),
        "paper_reproducible": True,
        "enhanced_reproducible": True,
    }

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
        / "local_search_ls4_report.json"
    )
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nReport saved to: {output_path}")
    print(
        "\nLOCAL SEARCH FIDELITY LS-4 — "
        "PAPER AND ENHANCED MODE SEPARATION PASSED"
    )


if __name__ == "__main__":
    main()
