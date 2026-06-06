from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from alns_solver.initial_solution import construct_initial_solution
from alns_solver.paper_initial_solution import (
    PaperInitialSolutionResult,
    construct_paper_initial_solution,
)
from alns_solver.solution_state import ALNSSolutionState

PAPER_MODE = "paper_algorithm_1"
ENHANCED_MODE = "enhanced_initial_solution"
SUPPORTED_INITIAL_SOLUTION_MODES = {PAPER_MODE, ENHANCED_MODE}


@dataclass
class InitialSolutionBuildResult:
    mode: str
    state: ALNSSolutionState
    trace: Any | None
    metadata: dict[str, Any]


def build_initial_solution(
    instance: dict,
    *,
    mode: str,
    seed: int = 42,
    lambda_value: float = 0.0,
    cost_bounds: tuple[float, float] | None = None,
    emission_bounds: tuple[float, float] | None = None,
    emission_factors: tuple[float, float] = (3.0, 1.0),
    strategy_2_mode: str = "paper_random_dv",
    max_attempts: int = 100,
) -> InitialSolutionBuildResult:
    if mode not in SUPPORTED_INITIAL_SOLUTION_MODES:
        raise ValueError(
            f"Unsupported initial-solution mode: {mode}. "
            f"Supported modes: {sorted(SUPPORTED_INITIAL_SOLUTION_MODES)}"
        )

    if mode == PAPER_MODE:
        result: PaperInitialSolutionResult = construct_paper_initial_solution(
            instance,
            seed=seed,
            lambda_value=lambda_value,
            cost_bounds=cost_bounds,
            emission_bounds=emission_bounds,
            emission_factors=emission_factors,
            strategy_2_mode=strategy_2_mode,
        )
        return InitialSolutionBuildResult(
            mode=PAPER_MODE,
            state=result.state,
            trace=result.trace,
            metadata={
                "construction_mode": PAPER_MODE,
                "paper_faithful": True,
                "enhanced": False,
                "seed": seed,
                "strategy_2_mode": strategy_2_mode,
            },
        )

    state = construct_initial_solution(
        instance,
        seed=seed,
        max_attempts=max_attempts,
    )
    state.register_operator_event(
        operator_type="construction",
        operator_name=ENHANCED_MODE,
        details={
            "seed": seed,
            "max_attempts": max_attempts,
            "paper_faithful": False,
        },
    )
    return InitialSolutionBuildResult(
        mode=ENHANCED_MODE,
        state=state,
        trace=None,
        metadata={
            "construction_mode": ENHANCED_MODE,
            "paper_faithful": False,
            "enhanced": True,
            "seed": seed,
            "max_attempts": max_attempts,
        },
    )
