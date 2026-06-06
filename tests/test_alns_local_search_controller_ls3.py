from __future__ import annotations

from pathlib import Path
import json

from core.instance_loader import load_instance
from alns_solver.solution_state import ALNSSolutionState
from alns_solver.paper_local_search import (
    LocalSearchMoveResult,
    PAPER_LOCAL_SEARCH_DELTA,
    PAPER_LOCAL_SEARCH_OPERATOR_ORDER,
    paper_local_search_eligible,
    run_paper_local_search,
)
from tests.test_alns_local_search_moves_ls1 import (
    controlled_intra_pair,
)

EMISSION_FACTORS = (3.0, 1.0)
EPSILON = 1e-10


def evaluate(
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


def make_stub_registry(
    call_log: list[str],
    state_counter: dict[str, int],
):
    """
    Stub registry with exactly the paper operator names/order.

    Operator 1 improves twice, then reports exhaustion.
    Operator 2 improves once, then reports exhaustion.
    Operators 3-8 report exhaustion immediately.
    """
    registry = []

    for index, name in enumerate(
        PAPER_LOCAL_SEARCH_OPERATOR_ORDER
    ):
        def operator(
            state,
            instance,
            *,
            lambda_value,
            cost_bounds,
            emission_bounds,
            emission_factors,
            _name=name,
            _index=index,
        ):
            call_log.append(_name)

            base = float(
                state.metadata.get(
                    "stub_objective",
                    100.0,
                )
            )

            limits = {
                PAPER_LOCAL_SEARCH_OPERATOR_ORDER[0]: 2,
                PAPER_LOCAL_SEARCH_OPERATOR_ORDER[1]: 1,
            }
            limit = limits.get(_name, 0)
            used = state_counter.get(_name, 0)

            if used >= limit:
                return LocalSearchMoveResult(
                    operator_name=_name,
                    state=state.copy(),
                    improved=False,
                    base_objective=base,
                    final_objective=base,
                    details={},
                )

            state_counter[_name] = used + 1
            candidate = state.copy()
            candidate.metadata["stub_objective"] = (
                base - 1.0
            )

            # The controller validates real state feasibility, while this
            # injected metadata drives only the stub result contract.
            return LocalSearchMoveResult(
                operator_name=_name,
                state=candidate,
                improved=True,
                base_objective=base,
                final_objective=base - 1.0,
                details={
                    "stub_acceptance": used + 1,
                },
            )

        registry.append((name, operator))

    return tuple(registry)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    instance = load_instance(
        root / "data" / "small" / "instance_001"
    )

    report = {}

    # ---------------------------------------------------------
    # LS-3A — Threshold and inclusive boundary
    # ---------------------------------------------------------
    if PAPER_LOCAL_SEARCH_DELTA != 0.1:
        raise AssertionError(
            "Paper delta_ls must equal 0.1."
        )

    if not paper_local_search_eligible(
        110.0,
        100.0,
        delta_ls=0.1,
    ):
        raise AssertionError(
            "Inclusive 1.1 threshold boundary was rejected."
        )

    if paper_local_search_eligible(
        110.0001,
        100.0,
        delta_ls=0.1,
        tolerance=0.0,
    ):
        raise AssertionError(
            "Solution above the 1.1 threshold was accepted."
        )

    print("[PASS] Paper local-search threshold uses delta_ls=0.1")
    print("[PASS] Threshold boundary is inclusive")
    print("[PASS] Working solution above threshold is rejected")

    # ---------------------------------------------------------
    # LS-3B — Skip contract with real state
    # ---------------------------------------------------------
    working, _ = controlled_intra_pair(
        instance
    )
    working_objective = evaluate(
        working,
        instance,
    )

    skipped = run_paper_local_search(
        working,
        instance,
        best_objective=(
            working_objective / 1.1001
        ),
        lambda_value=0.0,
        cost_bounds=None,
        emission_bounds=None,
        emission_factors=EMISSION_FACTORS,
    )

    if skipped.eligible:
        raise AssertionError(
            "Controller did not skip an out-of-threshold solution."
        )

    if skipped.accepted_moves:
        raise AssertionError(
            "Skipped local search executed accepted moves."
        )

    if any(
        skipped.operator_calls.values()
    ):
        raise AssertionError(
            "Skipped local search called an operator."
        )

    if (
        skipped.final_objective
        != skipped.base_objective
    ):
        raise AssertionError(
            "Skipped local search changed the objective."
        )

    print("[PASS] Out-of-threshold solution skips all operators")

    # ---------------------------------------------------------
    # LS-3C — Restart/exhaustion/order using controlled stubs
    # ---------------------------------------------------------
    # The real controller objective is derived from routes. To test exact
    # restart sequencing without changing production objective logic, use
    # wrappers that return valid real states from a monotonic chain.
    low_state = controlled_intra_pair(
        instance
    )[1]

    chain_states = [
        working,
        low_state,
    ]

    call_log: list[str] = []
    progress = {
        "operator_1_accepts": 0,
        "operator_2_accepts": 0,
    }

    def op1(
        state,
        instance,
        *,
        lambda_value,
        cost_bounds,
        emission_bounds,
        emission_factors,
    ):
        name = PAPER_LOCAL_SEARCH_OPERATOR_ORDER[0]
        call_log.append(name)
        base = evaluate(state, instance)

        if (
            progress["operator_1_accepts"] == 0
            and base > evaluate(low_state, instance) + EPSILON
        ):
            progress["operator_1_accepts"] += 1
            return LocalSearchMoveResult(
                name,
                low_state.copy(),
                True,
                base,
                evaluate(low_state, instance),
                {"controlled": 1},
            )

        return LocalSearchMoveResult(
            name,
            state.copy(),
            False,
            base,
            base,
            {},
        )

    def op2(
        state,
        instance,
        *,
        lambda_value,
        cost_bounds,
        emission_bounds,
        emission_factors,
    ):
        name = PAPER_LOCAL_SEARCH_OPERATOR_ORDER[1]
        call_log.append(name)
        base = evaluate(state, instance)

        # Deliberately exhausted: order test still proves operator 2 starts
        # only after operator 1 is re-run and exhausted.
        return LocalSearchMoveResult(
            name,
            state.copy(),
            False,
            base,
            base,
            {},
        )

    def exhausted_operator(name):
        def operator(
            state,
            instance,
            *,
            lambda_value,
            cost_bounds,
            emission_bounds,
            emission_factors,
        ):
            call_log.append(name)
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

    registry = (
        (
            PAPER_LOCAL_SEARCH_OPERATOR_ORDER[0],
            op1,
        ),
        (
            PAPER_LOCAL_SEARCH_OPERATOR_ORDER[1],
            op2,
        ),
        *tuple(
            (
                name,
                exhausted_operator(name),
            )
            for name in PAPER_LOCAL_SEARCH_OPERATOR_ORDER[2:]
        ),
    )

    controlled = run_paper_local_search(
        working,
        instance,
        best_objective=working_objective,
        lambda_value=0.0,
        cost_bounds=None,
        emission_bounds=None,
        emission_factors=EMISSION_FACTORS,
        operator_registry=registry,
    )

    expected_prefix = [
        PAPER_LOCAL_SEARCH_OPERATOR_ORDER[0],
        PAPER_LOCAL_SEARCH_OPERATOR_ORDER[0],
        PAPER_LOCAL_SEARCH_OPERATOR_ORDER[1],
    ]

    if call_log[:3] != expected_prefix:
        raise AssertionError(
            "Same-operator restart was not followed by exhaustion "
            f"before advancing. Calls={call_log}"
        )

    if tuple(
        call_log[2:]
    ) != PAPER_LOCAL_SEARCH_OPERATOR_ORDER[1:]:
        raise AssertionError(
            "Operators were not exhausted in paper-listed order. "
            f"Calls={call_log}"
        )

    if controlled.exhausted_operators != list(
        PAPER_LOCAL_SEARCH_OPERATOR_ORDER
    ):
        raise AssertionError(
            "Not all operators were recorded as exhausted."
        )

    if controlled.operator_calls[
        PAPER_LOCAL_SEARCH_OPERATOR_ORDER[0]
    ] != 2:
        raise AssertionError(
            "Improving operator was not restarted exactly once "
            "before exhaustion."
        )

    if len(controlled.accepted_moves) != 1:
        raise AssertionError(
            "Controlled restart test expected one accepted move."
        )

    if not (
        controlled.final_objective
        < controlled.base_objective - EPSILON
    ):
        raise AssertionError(
            "Controlled local search did not improve monotonically."
        )

    print("[PASS] Improvement restarts the same operator")
    print("[PASS] Exhausted operator advances to the next operator")
    print("[PASS] Eight operators follow the paper-listed order")
    print("[PASS] Controller terminates after final operator exhaustion")

    # ---------------------------------------------------------
    # LS-3D — Real eight-operator integration
    # ---------------------------------------------------------
    real_result = run_paper_local_search(
        working,
        instance,
        best_objective=working_objective,
        lambda_value=0.0,
        cost_bounds=None,
        emission_bounds=None,
        emission_factors=EMISSION_FACTORS,
    )

    final_solution = real_result.state.to_core_solution(
        instance=instance,
        lambda_value=0.0,
        objective_mode="cost",
        emission_factors=EMISSION_FACTORS,
        require_complete=True,
    )

    if not final_solution.validator_pass:
        raise AssertionError(
            "Real LS-3 final state failed validator: "
            f"{final_solution.validation_errors}"
        )

    objectives = [
        move["final_objective"]
        for move in real_result.accepted_moves
    ]

    if any(
        later >= earlier - EPSILON
        for earlier, later
        in zip(
            [real_result.base_objective] + objectives[:-1],
            objectives,
        )
    ):
        raise AssertionError(
            "Accepted local-search objectives are not strictly decreasing."
        )

    repeat_result = run_paper_local_search(
        working,
        instance,
        best_objective=working_objective,
        lambda_value=0.0,
        cost_bounds=None,
        emission_bounds=None,
        emission_factors=EMISSION_FACTORS,
    )

    if (
        repeat_result.final_objective
        != real_result.final_objective
    ):
        raise AssertionError(
            "Paper controller is not deterministic for the same input."
        )

    if (
        repeat_result.state.dv_routes
        != real_result.state.dv_routes
        or repeat_result.state.od_routes
        != real_result.state.od_routes
        or repeat_result.state.assignments
        != real_result.state.assignments
    ):
        raise AssertionError(
            "Paper controller returned different structures "
            "for the same input."
        )

    print("[PASS] Real eight-operator controller preserves feasibility")
    print("[PASS] Accepted objectives decrease monotonically")
    print("[PASS] Same input produces deterministic local-search result")

    report["threshold"] = {
        "delta_ls": PAPER_LOCAL_SEARCH_DELTA,
        "boundary_eligible": True,
        "above_boundary_eligible": False,
    }
    report["skip_contract"] = {
        "working_objective": working_objective,
        "best_objective": skipped.best_objective,
        "eligible": skipped.eligible,
        "operator_calls": skipped.operator_calls,
    }
    report["controlled_controller"] = {
        "call_log": call_log,
        "operator_calls": controlled.operator_calls,
        "accepted_moves": controlled.accepted_moves,
        "exhausted_operators": (
            controlled.exhausted_operators
        ),
        "base_objective": controlled.base_objective,
        "final_objective": controlled.final_objective,
    }
    report["real_controller"] = {
        "operator_order": list(
            real_result.operator_order
        ),
        "operator_calls": real_result.operator_calls,
        "accepted_moves": real_result.accepted_moves,
        "exhausted_operators": (
            real_result.exhausted_operators
        ),
        "base_objective": real_result.base_objective,
        "final_objective": real_result.final_objective,
        "validator_pass": (
            final_solution.validator_pass
        ),
        "deterministic_repeat": True,
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
        / "local_search_ls3_report.json"
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
        "\nLOCAL SEARCH FIDELITY LS-3 — "
        "PAPER SEARCH CONTROLLER PASSED"
    )


if __name__ == "__main__":
    main()
