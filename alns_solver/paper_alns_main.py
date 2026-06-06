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
