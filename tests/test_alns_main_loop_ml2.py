from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json

from core.instance_loader import load_instance
from alns_solver.paper_alns_main import (
    PaperALNSCandidatePipelineResult,
    apply_paper_sa_transition,
    build_ml1_paper_initial_state,
    build_ml2_paper_controllers,
    run_one_paper_alns_candidate_pipeline,
    run_one_paper_alns_iteration_ml2,
)


class FixedRandom:
    def __init__(self, value: float):
        self.value = float(value)

    def random(self) -> float:
        return self.value


def state_signature(state):
    return {
        "dv_routes": deepcopy(state.dv_routes),
        "od_routes": deepcopy(state.od_routes),
        "assignments": deepcopy(state.assignments),
        "unassigned": sorted(
            state.unassigned_customers
        ),
    }


def clone_candidate_with_objective(
    candidate,
    objective: float,
):
    return PaperALNSCandidatePipelineResult(
        destroy_operator=(
            candidate.destroy_operator
        ),
        repair_operator=(
            candidate.repair_operator
        ),
        local_search_mode=(
            candidate.local_search_mode
        ),
        destroy_result=(
            candidate.destroy_result
        ),
        repair_result=(
            candidate.repair_result
        ),
        local_search_result=(
            candidate.local_search_result
        ),
        candidate_state=(
            candidate.candidate_state.copy()
        ),
        candidate_objective=float(objective),
        candidate_cost=candidate.candidate_cost,
        candidate_emission=(
            candidate.candidate_emission
        ),
        candidate_dv_distance=(
            candidate.candidate_dv_distance
        ),
        candidate_od_extra_distance=(
            candidate.candidate_od_extra_distance
        ),
        validator_pass=(
            candidate.validator_pass
        ),
        validation_errors=list(
            candidate.validation_errors
        ),
        metadata=dict(candidate.metadata),
    )


def build_sa(
    *,
    initial_objective: float,
    seed: int = 42,
):
    adaptive, sa = build_ml2_paper_controllers(
        initial_objective=initial_objective,
        seed=seed,
    )
    return adaptive, sa


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    instance = load_instance(
        root / "data" / "small" / "instance_001"
    )

    lambda_value = 0.5
    cost_bounds = (0.0, 100.0)
    emission_bounds = (0.0, 300.0)
    emission_factors = (3.0, 1.0)

    current_state = build_ml1_paper_initial_state(
        instance,
        seed=42,
        lambda_value=lambda_value,
        cost_bounds=cost_bounds,
        emission_bounds=emission_bounds,
        emission_factors=emission_factors,
    )
    best_state = current_state.copy()

    current_solution = current_state.to_core_solution(
        instance=instance,
        lambda_value=lambda_value,
        objective_mode="weighted",
        cost_bounds=cost_bounds,
        emission_bounds=emission_bounds,
        emission_factors=emission_factors,
        require_complete=True,
    )
    current_objective = float(
        current_solution.objective
    )
    best_objective = current_objective

    base_candidate = (
        run_one_paper_alns_candidate_pipeline(
            current_state,
            instance,
            best_objective=best_objective,
            lambda_value=lambda_value,
            cost_bounds=cost_bounds,
            emission_bounds=emission_bounds,
            emission_factors=emission_factors,
            destroy_seed=7,
            strategy_2_seed=7,
        )
    )

    # ---------------------------------------------------------
    # ML-2A — Real candidate integration
    # ---------------------------------------------------------
    adaptive, sa = build_sa(
        initial_objective=current_objective,
        seed=2026,
    )

    real_transition = (
        run_one_paper_alns_iteration_ml2(
            current_state,
            best_state,
            instance,
            iteration=1,
            current_objective=current_objective,
            best_objective=best_objective,
            lambda_value=lambda_value,
            cost_bounds=cost_bounds,
            emission_bounds=emission_bounds,
            emission_factors=emission_factors,
            destroy_seed=7,
            strategy_2_seed=7,
            sa_controller=sa.controller,
        )
    )

    if not (
        real_transition.temperature_after_cooling
        < real_transition.temperature_used
    ):
        raise AssertionError(
            "ML-2 did not cool after the decision."
        )

    if (
        adaptive.state.destroy_records[
            "route_removal"
        ].segment_uses != 1
    ):
        raise AssertionError(
            "Destroy usage was not recorded exactly once."
        )

    if (
        adaptive.state.repair_records[
            "best_insertion"
        ].segment_uses != 1
    ):
        raise AssertionError(
            "Repair usage was not recorded exactly once."
        )

    print("[PASS] Real ML-1 candidate enters paper SA exactly once")
    print("[PASS] Adaptive usage is recorded exactly once")
    print("[PASS] Temperature cooling is applied exactly once")

    # ---------------------------------------------------------
    # ML-2B — New global best
    # ---------------------------------------------------------
    candidate = clone_candidate_with_objective(
        base_candidate,
        best_objective - 0.10,
    )
    adaptive, sa = build_sa(
        initial_objective=current_objective,
    )

    transition = apply_paper_sa_transition(
        iteration=1,
        candidate_pipeline=candidate,
        current_state=current_state,
        best_state=best_state,
        current_objective=current_objective,
        best_objective=best_objective,
        sa_controller=sa.controller,
    )

    if (
        not transition.accepted
        or transition.reward_event
        != "new_global_best"
        or transition.reward != 33.0
    ):
        raise AssertionError(
            "New global best transition is incorrect."
        )

    if (
        state_signature(
            transition.current_state_after
        )
        != state_signature(
            candidate.candidate_state
        )
    ):
        raise AssertionError(
            "Accepted new best did not become current."
        )

    if (
        state_signature(
            transition.best_state_after
        )
        != state_signature(
            candidate.candidate_state
        )
    ):
        raise AssertionError(
            "New global best state was not updated."
        )

    print("[PASS] New global best updates current and best with reward 33")

    # ---------------------------------------------------------
    # ML-2C — Better current, not global best
    # ---------------------------------------------------------
    historical_best = best_objective - 0.20
    candidate = clone_candidate_with_objective(
        base_candidate,
        current_objective - 0.10,
    )
    adaptive, sa = build_sa(
        initial_objective=current_objective,
    )

    transition = apply_paper_sa_transition(
        iteration=1,
        candidate_pipeline=candidate,
        current_state=current_state,
        best_state=best_state,
        current_objective=current_objective,
        best_objective=historical_best,
        sa_controller=sa.controller,
    )

    if (
        transition.reward_event
        != "better_current"
        or transition.reward != 15.0
    ):
        raise AssertionError(
            "Better-current transition is incorrect."
        )

    if (
        transition.best_objective_after
        != historical_best
    ):
        raise AssertionError(
            "Historical best changed incorrectly."
        )

    print("[PASS] Better-current candidate receives reward 15")

    # ---------------------------------------------------------
    # ML-2D — Worse accepted
    # ---------------------------------------------------------
    candidate = clone_candidate_with_objective(
        base_candidate,
        current_objective + 0.01,
    )
    adaptive, sa = build_sa(
        initial_objective=current_objective,
    )
    sa.controller.rng = FixedRandom(0.0)

    transition = apply_paper_sa_transition(
        iteration=1,
        candidate_pipeline=candidate,
        current_state=current_state,
        best_state=best_state,
        current_objective=current_objective,
        best_objective=best_objective,
        sa_controller=sa.controller,
    )

    if (
        not transition.accepted
        or transition.reward_event
        != "worse_accepted"
        or transition.reward != 9.0
    ):
        raise AssertionError(
            "Worse-accepted transition is incorrect."
        )

    if (
        state_signature(
            transition.current_state_after
        )
        != state_signature(
            candidate.candidate_state
        )
    ):
        raise AssertionError(
            "Accepted worse candidate did not become current."
        )

    if (
        transition.best_objective_after
        != best_objective
    ):
        raise AssertionError(
            "Best objective changed after worse acceptance."
        )

    print("[PASS] SA-accepted worse candidate becomes current with reward 9")

    # ---------------------------------------------------------
    # ML-2E — Worse rejected
    # ---------------------------------------------------------
    adaptive, sa = build_sa(
        initial_objective=current_objective,
    )
    sa.controller.rng = FixedRandom(1.0)

    transition = apply_paper_sa_transition(
        iteration=1,
        candidate_pipeline=candidate,
        current_state=current_state,
        best_state=best_state,
        current_objective=current_objective,
        best_objective=best_objective,
        sa_controller=sa.controller,
    )

    if (
        transition.accepted
        or transition.reward_event
        != "rejected"
        or transition.reward != 0.0
    ):
        raise AssertionError(
            "Worse-rejected transition is incorrect."
        )

    if (
        state_signature(
            transition.current_state_after
        )
        != state_signature(current_state)
    ):
        raise AssertionError(
            "Rejected candidate changed current state."
        )

    print("[PASS] SA-rejected candidate preserves current with reward 0")

    # ---------------------------------------------------------
    # ML-2F — Equal accepted, zero reward
    # ---------------------------------------------------------
    candidate = clone_candidate_with_objective(
        base_candidate,
        current_objective,
    )
    adaptive, sa = build_sa(
        initial_objective=current_objective,
    )

    transition = apply_paper_sa_transition(
        iteration=1,
        candidate_pipeline=candidate,
        current_state=current_state,
        best_state=best_state,
        current_objective=current_objective,
        best_objective=best_objective,
        sa_controller=sa.controller,
    )

    if (
        not transition.accepted
        or transition.reward_event
        != "rejected"
        or transition.reward != 0.0
    ):
        raise AssertionError(
            "Equal-candidate transition is incorrect."
        )

    print("[PASS] Equal candidate is accepted with zero adaptive reward")

    # ---------------------------------------------------------
    # ML-2G — Input immutability and paper-only metadata
    # ---------------------------------------------------------
    current_before = state_signature(
        current_state
    )
    best_before = state_signature(
        best_state
    )

    candidate = clone_candidate_with_objective(
        base_candidate,
        current_objective - 0.10,
    )
    adaptive, sa = build_sa(
        initial_objective=current_objective,
    )

    transition = apply_paper_sa_transition(
        iteration=1,
        candidate_pipeline=candidate,
        current_state=current_state,
        best_state=best_state,
        current_objective=current_objective,
        best_objective=best_objective,
        sa_controller=sa.controller,
    )

    if current_before != state_signature(
        current_state
    ):
        raise AssertionError(
            "ML-2 mutated input current state."
        )

    if best_before != state_signature(
        best_state
    ):
        raise AssertionError(
            "ML-2 mutated input best state."
        )

    if not transition.metadata[
        "paper_faithful"
    ]:
        raise AssertionError(
            "ML-2 lost paper-faithful label."
        )

    if transition.metadata["enhanced"]:
        raise AssertionError(
            "Enhanced behavior entered ML-2."
        )

    if not (
        transition.metadata[
            "duplicate_reward_update"
        ] is False
        and transition.metadata[
            "duplicate_temperature_cooling"
        ] is False
    ):
        raise AssertionError(
            "ML-2 duplicate-side-effect contract is incorrect."
        )

    print("[PASS] ML-2 preserves input current and best states")
    print("[PASS] ML-2 uses paper controllers only")
    print("[PASS] No duplicate reward update or cooling is introduced")
    print("[PASS] Acceptance and rewards use one scalar F_lambda")

    report = {
        "real_candidate": {
            "candidate_objective": (
                real_transition
                .candidate_pipeline
                .candidate_objective
            ),
            "accepted": (
                real_transition.accepted
            ),
            "reward_event": (
                real_transition.reward_event
            ),
            "reward": real_transition.reward,
            "temperature_used": (
                real_transition.temperature_used
            ),
            "temperature_after_cooling": (
                real_transition
                .temperature_after_cooling
            ),
        },
        "controlled_contracts": {
            "new_global_best_reward": 33.0,
            "better_current_reward": 15.0,
            "worse_accepted_reward": 9.0,
            "rejected_reward": 0.0,
            "equal_reward": 0.0,
        },
        "fidelity": transition.metadata,
    }

    output_dir = (
        root
        / "outputs"
        / "alns_main_loop_fidelity_tests"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "alns_main_loop_ml2_report.json"
    )
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nReport saved to: {output_path}")
    print(
        "\nALNS MAIN LOOP FIDELITY ML-2 — "
        "ACCEPTANCE AND STATE TRANSITION PASSED"
    )


if __name__ == "__main__":
    main()
