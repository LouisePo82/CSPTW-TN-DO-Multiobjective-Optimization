from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from alns_solver.initial_solution_factory import (
    PAPER_MODE,
    build_initial_solution,
)
from alns_solver.local_search_factory import (
    PAPER_LOCAL_SEARCH_MODE,
    LocalSearchFactoryResult,
    build_local_search,
)
from alns_solver.paper_destroy_operators import (
    paper_route_removal,
)
from alns_solver.repair_operators import (
    best_insertion_repair,
)
from alns_solver.solution_state import (
    ALNSSolutionState,
)


@dataclass
class PaperALNSCandidatePipelineResult:
    """
    One paper-faithful destroy-repair-local-search candidate.

    This result does not contain an acceptance decision. Simulated annealing,
    adaptive rewards, operator roulette, and temperature cooling belong to
    later main-loop gates.
    """

    destroy_operator: str
    repair_operator: str
    local_search_mode: str

    destroy_result: Any
    repair_result: Any
    local_search_result: LocalSearchFactoryResult

    candidate_state: ALNSSolutionState
    candidate_objective: float
    candidate_cost: float
    candidate_emission: float
    candidate_dv_distance: float
    candidate_od_extra_distance: float

    validator_pass: bool
    validation_errors: list[str]

    metadata: dict[str, Any]


def run_one_paper_alns_candidate_pipeline(
    current_state: ALNSSolutionState,
    instance: dict,
    *,
    best_objective: float,
    lambda_value: float,
    cost_bounds: tuple[float, float] | None,
    emission_bounds: tuple[float, float] | None,
    emission_factors: tuple[float, float] = (3.0, 1.0),
    destroy_seed: int = 42,
    strategy_2_seed: int | None = None,
) -> PaperALNSCandidatePipelineResult:
    """
    Run one controlled paper-faithful candidate-generation pipeline:

        paper Route Removal
        -> Best Insertion repair
        -> paper Local Search
        -> shared objective and validator

    The only extension is the scalar objective F_lambda passed consistently
    through repair, local search, evaluation, and validation.
    """
    if not 0.0 <= float(lambda_value) <= 1.0:
        raise ValueError(
            "lambda_value must be between 0 and 1."
        )

    if strategy_2_seed is None:
        strategy_2_seed = destroy_seed

    # Snapshot is used only to enforce the no-mutation contract.
    original_snapshot = current_state.copy()

    current_solution = current_state.to_core_solution(
        instance=instance,
        lambda_value=lambda_value,
        objective_mode="weighted",
        cost_bounds=cost_bounds,
        emission_bounds=emission_bounds,
        emission_factors=emission_factors,
        require_complete=True,
        metadata={
            "validation_scope": "ml1_current_state",
        },
    )

    if not current_solution.validator_pass:
        raise ValueError(
            "ML-1 requires a complete valid current state. "
            f"Errors: {current_solution.validation_errors}"
        )

    destroy_result = paper_route_removal(
        current_state,
        instance,
        seed=destroy_seed,
    )

    partial_state = destroy_result.state

    if not destroy_result.removed_customers:
        raise RuntimeError(
            "Controlled route removal removed no customers."
        )

    if not partial_state.unassigned_customers:
        raise RuntimeError(
            "Controlled route removal did not create a partial state."
        )

    repair_result = best_insertion_repair(
        partial_state,
        instance,
        lambda_value=lambda_value,
        cost_bounds=cost_bounds,
        emission_bounds=emission_bounds,
        emission_factors=emission_factors,
        strategy_2_mode="paper_random_dv",
        strategy_2_seed=strategy_2_seed,
    )

    if not repair_result.validator_pass:
        raise RuntimeError(
            "Best-insertion repair failed shared validation. "
            f"Errors: {repair_result.validation_errors}"
        )

    if repair_result.state.unassigned_customers:
        raise RuntimeError(
            "Best-insertion repair left unassigned customers."
        )

    local_search_result = build_local_search(
        repair_result.state,
        instance,
        mode=PAPER_LOCAL_SEARCH_MODE,
        best_objective=best_objective,
        lambda_value=lambda_value,
        cost_bounds=cost_bounds,
        emission_bounds=emission_bounds,
        emission_factors=emission_factors,
    )

    candidate_state = local_search_result.state

    candidate_solution = candidate_state.to_core_solution(
        instance=instance,
        lambda_value=lambda_value,
        objective_mode="weighted",
        cost_bounds=cost_bounds,
        emission_bounds=emission_bounds,
        emission_factors=emission_factors,
        require_complete=True,
        metadata={
            "validation_scope": "ml1_candidate",
            "destroy_operator": "route_removal",
            "repair_operator": "best_insertion",
            "local_search_mode": PAPER_LOCAL_SEARCH_MODE,
        },
    )

    if not candidate_solution.validator_pass:
        raise RuntimeError(
            "ML-1 final candidate failed shared validation. "
            f"Errors: {candidate_solution.validation_errors}"
        )

    # Recompute once more through the shared ALNS state objective path.
    shared_metrics = candidate_state.evaluate(
        instance=instance,
        lambda_value=lambda_value,
        cost_bounds=cost_bounds,
        emission_bounds=emission_bounds,
        emission_factors=emission_factors,
    )

    if abs(
        float(candidate_solution.objective)
        - float(shared_metrics["objective"])
    ) > 1e-10:
        raise RuntimeError(
            "Candidate objective does not match shared evaluation."
        )

    # Enforce no mutation of the caller's current state.
    if (
        current_state.dv_routes
        != original_snapshot.dv_routes
        or current_state.od_routes
        != original_snapshot.od_routes
        or current_state.assignments
        != original_snapshot.assignments
        or current_state.unassigned_customers
        != original_snapshot.unassigned_customers
    ):
        raise RuntimeError(
            "ML-1 pipeline mutated the input current state."
        )

    return PaperALNSCandidatePipelineResult(
        destroy_operator="route_removal",
        repair_operator="best_insertion",
        local_search_mode=PAPER_LOCAL_SEARCH_MODE,
        destroy_result=destroy_result,
        repair_result=repair_result,
        local_search_result=local_search_result,
        candidate_state=candidate_state,
        candidate_objective=float(
            candidate_solution.objective
        ),
        candidate_cost=float(
            candidate_solution.cost
        ),
        candidate_emission=float(
            candidate_solution.emission
        ),
        candidate_dv_distance=float(
            candidate_solution.dv_distance
        ),
        candidate_od_extra_distance=float(
            candidate_solution.od_extra_distance
        ),
        validator_pass=bool(
            candidate_solution.validator_pass
        ),
        validation_errors=list(
            candidate_solution.validation_errors
        ),
        metadata={
            "paper_faithful": True,
            "enhanced": False,
            "pipeline_scope": (
                "destroy_repair_local_search_candidate_only"
            ),
            "destroy_seed": destroy_seed,
            "strategy_2_mode": "paper_random_dv",
            "strategy_2_seed": strategy_2_seed,
            "lambda_value": float(lambda_value),
            "objective_input": "scalar_F_lambda",
            "acceptance_applied": False,
            "adaptive_weights_applied": False,
            "temperature_cooling_applied": False,
            "local_search_metadata": dict(
                local_search_result.metadata
            ),
        },
    )


def build_ml1_paper_initial_state(
    instance: dict,
    *,
    seed: int,
    lambda_value: float,
    cost_bounds: tuple[float, float] | None,
    emission_bounds: tuple[float, float] | None,
    emission_factors: tuple[float, float] = (3.0, 1.0),
) -> ALNSSolutionState:
    """
    Test/support helper that obtains the current state from paper Algorithm 1.
    """
    result = build_initial_solution(
        instance,
        mode=PAPER_MODE,
        seed=seed,
        lambda_value=lambda_value,
        cost_bounds=cost_bounds,
        emission_bounds=emission_bounds,
        emission_factors=emission_factors,
        strategy_2_mode="paper_random_dv",
    )
    return result.state

# =============================================================
# ALNS Main Loop Fidelity ML-2 — Acceptance and State Transition
# =============================================================

import random

from alns_solver.acceptance_factory import (
    PAPER_SIMULATED_ANNEALING_MODE,
    build_simulated_annealing,
)
from alns_solver.adaptive_weights_factory import (
    PAPER_ADAPTIVE_WEIGHTS_MODE,
    build_adaptive_weights,
)


@dataclass
class PaperALNSIterationTransitionResult:
    iteration: int
    candidate_pipeline: PaperALNSCandidatePipelineResult
    acceptance_result: Any

    current_state_before: ALNSSolutionState
    current_state_after: ALNSSolutionState
    best_state_before: ALNSSolutionState
    best_state_after: ALNSSolutionState

    current_objective_before: float
    current_objective_after: float
    best_objective_before: float
    best_objective_after: float

    accepted: bool
    reward_event: str
    reward: float

    temperature_used: float
    temperature_after_cooling: float

    metadata: dict[str, Any]


def apply_paper_sa_transition(
    *,
    iteration: int,
    candidate_pipeline: PaperALNSCandidatePipelineResult,
    current_state: ALNSSolutionState,
    best_state: ALNSSolutionState,
    current_objective: float,
    best_objective: float,
    sa_controller: Any,
) -> PaperALNSIterationTransitionResult:
    """
    Apply the validated paper SA/adaptive controller to one ML-1 candidate.

    This function synchronizes the objective transition returned by the SA
    controller with the corresponding ALNSSolutionState objects.
    """
    current_before = current_state.copy()
    best_before = best_state.copy()

    acceptance_result = sa_controller.process_iteration(
        iteration=iteration,
        destroy_operator=(
            candidate_pipeline.destroy_operator
        ),
        repair_operator=(
            candidate_pipeline.repair_operator
        ),
        candidate_objective=(
            candidate_pipeline.candidate_objective
        ),
        current_objective=current_objective,
        best_objective=best_objective,
    )

    accepted = bool(
        acceptance_result.acceptance_decision.accepted
    )

    if accepted:
        current_after = (
            candidate_pipeline.candidate_state.copy()
        )
    else:
        current_after = current_state.copy()

    candidate_is_new_best = (
        candidate_pipeline.candidate_objective
        < float(best_objective)
    )

    if candidate_is_new_best:
        best_after = (
            candidate_pipeline.candidate_state.copy()
        )
    else:
        best_after = best_state.copy()

    expected_current_after = (
        candidate_pipeline.candidate_objective
        if accepted
        else float(current_objective)
    )
    expected_best_after = min(
        float(best_objective),
        candidate_pipeline.candidate_objective,
    )

    if abs(
        acceptance_result.current_objective_after
        - expected_current_after
    ) > 1e-12:
        raise RuntimeError(
            "SA objective transition disagrees with main-loop "
            "current-state transition."
        )

    if abs(
        acceptance_result.best_objective_after
        - expected_best_after
    ) > 1e-12:
        raise RuntimeError(
            "SA objective transition disagrees with main-loop "
            "best-state transition."
        )

    return PaperALNSIterationTransitionResult(
        iteration=iteration,
        candidate_pipeline=candidate_pipeline,
        acceptance_result=acceptance_result,
        current_state_before=current_before,
        current_state_after=current_after,
        best_state_before=best_before,
        best_state_after=best_after,
        current_objective_before=float(
            current_objective
        ),
        current_objective_after=float(
            acceptance_result.current_objective_after
        ),
        best_objective_before=float(
            best_objective
        ),
        best_objective_after=float(
            acceptance_result.best_objective_after
        ),
        accepted=accepted,
        reward_event=(
            acceptance_result.adaptive_result.event
        ),
        reward=float(
            acceptance_result.adaptive_result.reward
        ),
        temperature_used=float(
            acceptance_result.temperature_used
        ),
        temperature_after_cooling=float(
            acceptance_result.temperature_after_cooling
        ),
        metadata={
            "paper_faithful": True,
            "enhanced": False,
            "objective_input": "scalar_F_lambda",
            "acceptance_formula": (
                "exp(-(candidate-current)/temperature)"
            ),
            "adaptive_reward_source": (
                "paper_sa_controller"
            ),
            "cooling_source": (
                "paper_sa_controller"
            ),
            "duplicate_reward_update": False,
            "duplicate_temperature_cooling": False,
        },
    )


def run_one_paper_alns_iteration_ml2(
    current_state: ALNSSolutionState,
    best_state: ALNSSolutionState,
    instance: dict,
    *,
    iteration: int,
    current_objective: float,
    best_objective: float,
    lambda_value: float,
    cost_bounds: tuple[float, float] | None,
    emission_bounds: tuple[float, float] | None,
    emission_factors: tuple[float, float] = (3.0, 1.0),
    destroy_seed: int = 42,
    strategy_2_seed: int | None = None,
    sa_controller: Any,
) -> PaperALNSIterationTransitionResult:
    """
    Run one controlled ML-2 iteration.

    Candidate generation remains the ML-1 paper pipeline:
    Route Removal -> Best Insertion -> paper Local Search.

    Acceptance, adaptive reward, and cooling are delegated exactly once to
    the validated paper SA controller.
    """
    candidate = run_one_paper_alns_candidate_pipeline(
        current_state,
        instance,
        best_objective=best_objective,
        lambda_value=lambda_value,
        cost_bounds=cost_bounds,
        emission_bounds=emission_bounds,
        emission_factors=emission_factors,
        destroy_seed=destroy_seed,
        strategy_2_seed=strategy_2_seed,
    )

    return apply_paper_sa_transition(
        iteration=iteration,
        candidate_pipeline=candidate,
        current_state=current_state,
        best_state=best_state,
        current_objective=current_objective,
        best_objective=best_objective,
        sa_controller=sa_controller,
    )


def build_ml2_paper_controllers(
    *,
    initial_objective: float,
    destroy_operator_names: tuple[str, ...] = (
        "route_removal",
    ),
    repair_operator_names: tuple[str, ...] = (
        "best_insertion",
    ),
    seed: int = 42,
):
    """
    Construct paper adaptive weights first, then paper SA.

    Each call returns fresh independent state for one lambda run.
    """
    adaptive = build_adaptive_weights(
        mode=PAPER_ADAPTIVE_WEIGHTS_MODE,
        destroy_operator_names=(
            destroy_operator_names
        ),
        repair_operator_names=(
            repair_operator_names
        ),
    )

    simulated_annealing = build_simulated_annealing(
        mode=PAPER_SIMULATED_ANNEALING_MODE,
        initial_objective=initial_objective,
        adaptive_controller=adaptive.controller,
        rng=random.Random(seed),
    )

    return adaptive, simulated_annealing

# =============================================================
# ALNS Main Loop Fidelity ML-3B — Roulette and Segment Boundary
# =============================================================

from alns_solver.paper_adaptive_weights import (
    select_destroy_operator,
    select_repair_operator,
)
from alns_solver.paper_operator_dispatch import (
    PAPER_DESTROY_OPERATOR_NAMES,
    PAPER_REPAIR_OPERATOR_NAMES,
)


@dataclass
class PaperRouletteSelectionResult:
    iteration: int
    destroy_operator: str
    repair_operator: str
    destroy_probabilities: dict[str, float]
    repair_probabilities: dict[str, float]
    destroy_weights: dict[str, float]
    repair_weights: dict[str, float]


def _record_probabilities(records: dict[str, Any]) -> dict[str, float]:
    total = sum(
        float(record.weight)
        for record in records.values()
    )

    if total <= 0.0:
        raise ValueError(
            "Operator-weight total must be positive."
        )

    return {
        name: float(record.weight) / total
        for name, record in records.items()
    }


def select_paper_operator_pair(
    *,
    iteration: int,
    adaptive_state: Any,
    rng: random.Random,
) -> PaperRouletteSelectionResult:
    """
    Select one destroy and one repair operator by the validated paper
    roulette-wheel mechanism.

    F_lambda does not enter roulette normalization directly. It affects later
    rewards, which update weights at the paper segment boundary.
    """
    destroy_probabilities = _record_probabilities(
        adaptive_state.destroy_records
    )
    repair_probabilities = _record_probabilities(
        adaptive_state.repair_records
    )

    destroy_operator = select_destroy_operator(
        adaptive_state,
        rng=rng,
    )
    repair_operator = select_repair_operator(
        adaptive_state,
        rng=rng,
    )

    if destroy_operator not in PAPER_DESTROY_OPERATOR_NAMES:
        raise RuntimeError(
            "Roulette selected an operator outside the paper destroy pool."
        )

    if repair_operator not in PAPER_REPAIR_OPERATOR_NAMES:
        raise RuntimeError(
            "Roulette selected an operator outside the paper repair pool."
        )

    return PaperRouletteSelectionResult(
        iteration=iteration,
        destroy_operator=destroy_operator,
        repair_operator=repair_operator,
        destroy_probabilities=destroy_probabilities,
        repair_probabilities=repair_probabilities,
        destroy_weights={
            name: float(record.weight)
            for name, record
            in adaptive_state.destroy_records.items()
        },
        repair_weights={
            name: float(record.weight)
            for name, record
            in adaptive_state.repair_records.items()
        },
    )
