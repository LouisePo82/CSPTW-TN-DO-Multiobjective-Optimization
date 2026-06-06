from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from alns_solver.solution_state import ALNSSolutionState
from alns_solver.paper_local_search import (
    EPSILON,
    PAPER_LOCAL_SEARCH_DELTA,
    PAPER_LOCAL_SEARCH_OPERATOR_ORDER,
    PaperLocalSearchResult,
    run_paper_local_search,
)

PAPER_LOCAL_SEARCH_MODE = "paper_local_search"
ENHANCED_LOCAL_SEARCH_MODE = "enhanced_local_search"

SUPPORTED_LOCAL_SEARCH_MODES = {
    PAPER_LOCAL_SEARCH_MODE,
    ENHANCED_LOCAL_SEARCH_MODE,
}


@dataclass
class LocalSearchFactoryResult:
    mode: str
    state: ALNSSolutionState
    base_objective: float
    final_objective: float
    improved: bool
    eligible: bool
    metadata: dict[str, Any]
    accepted_moves: list[dict[str, Any]]
    operator_calls: dict[str, int]
    cycles: int


def _merge_counts(
    target: dict[str, int],
    source: dict[str, int],
) -> None:
    for name, count in source.items():
        target[name] = target.get(name, 0) + int(count)


def _paper_result_to_factory_result(
    result: PaperLocalSearchResult,
) -> LocalSearchFactoryResult:
    return LocalSearchFactoryResult(
        mode=PAPER_LOCAL_SEARCH_MODE,
        state=result.state,
        base_objective=result.base_objective,
        final_objective=result.final_objective,
        improved=result.improved,
        eligible=result.eligible,
        metadata={
            "local_search_mode": PAPER_LOCAL_SEARCH_MODE,
            "paper_faithful": True,
            "enhanced": False,
            "delta_ls": PAPER_LOCAL_SEARCH_DELTA,
            "threshold_boundary": "inclusive",
            "selection": "first_improvement",
            "restart_same_operator": True,
            "restart_full_operator_sequence": False,
            "operator_order": list(
                PAPER_LOCAL_SEARCH_OPERATOR_ORDER
            ),
        },
        accepted_moves=list(result.accepted_moves),
        operator_calls=dict(result.operator_calls),
        cycles=1 if result.eligible else 0,
    )


def _run_enhanced_full_sequence_restart(
    working_state: ALNSSolutionState,
    instance: dict,
    *,
    best_objective: float,
    lambda_value: float,
    cost_bounds: tuple[float, float] | None,
    emission_bounds: tuple[float, float] | None,
    emission_factors: tuple[float, float],
    delta_ls: float,
    operator_registry,
    max_cycles: int,
    max_restarts_per_operator: int,
) -> LocalSearchFactoryResult:
    if max_cycles <= 0:
        raise ValueError("max_cycles must be positive.")

    current = working_state.copy()
    all_moves: list[dict[str, Any]] = []
    total_calls = {
        name: 0
        for name in PAPER_LOCAL_SEARCH_OPERATOR_ORDER
    }

    first_result = run_paper_local_search(
        current,
        instance,
        best_objective=best_objective,
        lambda_value=lambda_value,
        cost_bounds=cost_bounds,
        emission_bounds=emission_bounds,
        emission_factors=emission_factors,
        delta_ls=delta_ls,
        operator_registry=operator_registry,
        max_restarts_per_operator=max_restarts_per_operator,
    )

    base_objective = first_result.base_objective

    if not first_result.eligible:
        return LocalSearchFactoryResult(
            mode=ENHANCED_LOCAL_SEARCH_MODE,
            state=first_result.state,
            base_objective=base_objective,
            final_objective=base_objective,
            improved=False,
            eligible=False,
            metadata={
                "local_search_mode": ENHANCED_LOCAL_SEARCH_MODE,
                "paper_faithful": False,
                "enhanced": True,
                "delta_ls": float(delta_ls),
                "selection": "first_improvement",
                "restart_same_operator": True,
                "restart_full_operator_sequence": True,
                "operator_order": list(
                    PAPER_LOCAL_SEARCH_OPERATOR_ORDER
                ),
            },
            accepted_moves=[],
            operator_calls=total_calls,
            cycles=0,
        )

    cycle_result = first_result
    cycles = 0

    while True:
        cycles += 1
        _merge_counts(
            total_calls,
            cycle_result.operator_calls,
        )

        for move in cycle_result.accepted_moves:
            all_moves.append(
                {
                    **move,
                    "cycle": cycles,
                }
            )

        current = cycle_result.state

        if not cycle_result.improved:
            break

        if cycles >= max_cycles:
            raise RuntimeError(
                "Enhanced local search exceeded max_cycles."
            )

        cycle_result = run_paper_local_search(
            current,
            instance,
            best_objective=best_objective,
            lambda_value=lambda_value,
            cost_bounds=cost_bounds,
            emission_bounds=emission_bounds,
            emission_factors=emission_factors,
            delta_ls=delta_ls,
            operator_registry=operator_registry,
            max_restarts_per_operator=max_restarts_per_operator,
        )

    final_objective = cycle_result.final_objective

    return LocalSearchFactoryResult(
        mode=ENHANCED_LOCAL_SEARCH_MODE,
        state=current,
        base_objective=base_objective,
        final_objective=final_objective,
        improved=(
            final_objective
            < base_objective - EPSILON
        ),
        eligible=True,
        metadata={
            "local_search_mode": ENHANCED_LOCAL_SEARCH_MODE,
            "paper_faithful": False,
            "enhanced": True,
            "delta_ls": float(delta_ls),
            "selection": "first_improvement",
            "restart_same_operator": True,
            "restart_full_operator_sequence": True,
            "operator_order": list(
                PAPER_LOCAL_SEARCH_OPERATOR_ORDER
            ),
        },
        accepted_moves=all_moves,
        operator_calls=total_calls,
        cycles=cycles,
    )


def build_local_search(
    working_state: ALNSSolutionState,
    instance: dict,
    *,
    mode: str,
    best_objective: float,
    lambda_value: float,
    cost_bounds: tuple[float, float] | None,
    emission_bounds: tuple[float, float] | None,
    emission_factors: tuple[float, float] = (1.0, 1.0),
    delta_ls: float | None = None,
    operator_registry=None,
    max_cycles: int = 100,
    max_restarts_per_operator: int = 10_000,
) -> LocalSearchFactoryResult:
    """
    Central local-search factory.

    paper_local_search
        Exact LS-3 paper controller. delta_ls is fixed at 0.1.

    enhanced_local_search
        Explicit extension that restarts the complete eight-operator sequence
        after any improving pass. It is never labeled paper-faithful.
    """
    if mode not in SUPPORTED_LOCAL_SEARCH_MODES:
        raise ValueError(
            f"Unsupported local-search mode: {mode}. "
            f"Supported modes: {sorted(SUPPORTED_LOCAL_SEARCH_MODES)}"
        )

    if mode == PAPER_LOCAL_SEARCH_MODE:
        if (
            delta_ls is not None
            and abs(
                float(delta_ls)
                - PAPER_LOCAL_SEARCH_DELTA
            )
            > EPSILON
        ):
            raise ValueError(
                "paper_local_search fixes delta_ls at 0.1."
            )

        result = run_paper_local_search(
            working_state,
            instance,
            best_objective=best_objective,
            lambda_value=lambda_value,
            cost_bounds=cost_bounds,
            emission_bounds=emission_bounds,
            emission_factors=emission_factors,
            delta_ls=PAPER_LOCAL_SEARCH_DELTA,
            operator_registry=operator_registry,
            max_restarts_per_operator=max_restarts_per_operator,
        )
        return _paper_result_to_factory_result(result)

    enhanced_delta = (
        PAPER_LOCAL_SEARCH_DELTA
        if delta_ls is None
        else float(delta_ls)
    )

    return _run_enhanced_full_sequence_restart(
        working_state,
        instance,
        best_objective=best_objective,
        lambda_value=lambda_value,
        cost_bounds=cost_bounds,
        emission_bounds=emission_bounds,
        emission_factors=emission_factors,
        delta_ls=enhanced_delta,
        operator_registry=operator_registry,
        max_cycles=max_cycles,
        max_restarts_per_operator=max_restarts_per_operator,
    )
