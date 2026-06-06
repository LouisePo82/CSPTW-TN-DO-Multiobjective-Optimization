from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import random

from alns_solver.paper_acceptance import (
    PAPER_COOLING_RATE,
    PAPER_INITIAL_ACCEPTANCE_PROBABILITY,
    PAPER_INITIAL_RELATIVE_WORSENING,
    PaperSimulatedAnnealingController,
    PaperTemperatureSchedule,
    paper_initial_temperature,
)
from alns_solver.paper_adaptive_weights import (
    PaperAdaptiveWeightController,
)


PAPER_SIMULATED_ANNEALING_MODE = "paper_simulated_annealing"
ENHANCED_SIMULATED_ANNEALING_MODE = "enhanced_simulated_annealing"

SUPPORTED_SIMULATED_ANNEALING_MODES = {
    PAPER_SIMULATED_ANNEALING_MODE,
    ENHANCED_SIMULATED_ANNEALING_MODE,
}


@dataclass
class SimulatedAnnealingFactoryResult:
    mode: str
    controller: PaperSimulatedAnnealingController
    metadata: dict[str, Any]


class EnhancedTemperatureSchedule(PaperTemperatureSchedule):
    """
    Explicitly non-paper sensitivity-analysis schedule.

    It preserves the validated scheduling mechanics but allows controlled
    overrides for initial calibration and cooling.
    """

    def __init__(
        self,
        *,
        initial_objective: float,
        relative_worsening: float,
        target_acceptance_probability: float,
        cooling_rate: float,
    ) -> None:
        self.initial_objective = float(
            initial_objective
        )
        self.cooling_rate = float(
            cooling_rate
        )
        self.initial_temperature = (
            paper_initial_temperature(
                initial_objective=(
                    self.initial_objective
                ),
                relative_worsening=(
                    relative_worsening
                ),
                target_acceptance_probability=(
                    target_acceptance_probability
                ),
            )
        )
        self.current_temperature = (
            self.initial_temperature
        )
        self.completed_iterations = 0
        self.history = []

        if not 0.0 < self.cooling_rate < 1.0:
            raise ValueError(
                "Enhanced cooling_rate must be strictly "
                "between 0 and 1."
            )


def build_simulated_annealing(
    *,
    mode: str,
    initial_objective: float,
    adaptive_controller: PaperAdaptiveWeightController,
    rng: random.Random,
    relative_worsening: float | None = None,
    target_acceptance_probability: float | None = None,
    cooling_rate: float | None = None,
) -> SimulatedAnnealingFactoryResult:
    """
    Factory separating paper-faithful and enhanced SA modes.

    Paper mode:
    - relative worsening 0.5;
    - target acceptance probability 0.5;
    - cooling rate 0.9994;
    - inclusive acceptance boundary;
    - cooling after every iteration;
    - one scalar F_lambda objective.

    Enhanced mode:
    - explicit sensitivity-analysis variant;
    - calibration/cooling overrides allowed;
    - always labelled non-paper.
    """
    if mode not in SUPPORTED_SIMULATED_ANNEALING_MODES:
        raise ValueError(
            f"Unsupported simulated-annealing mode: {mode}. "
            f"Supported modes: "
            f"{sorted(SUPPORTED_SIMULATED_ANNEALING_MODES)}"
        )

    if mode == PAPER_SIMULATED_ANNEALING_MODE:
        if (
            relative_worsening is not None
            and abs(
                float(relative_worsening)
                - PAPER_INITIAL_RELATIVE_WORSENING
            ) > 1e-12
        ):
            raise ValueError(
                "paper_simulated_annealing fixes "
                "relative_worsening at 0.5."
            )

        if (
            target_acceptance_probability is not None
            and abs(
                float(target_acceptance_probability)
                - PAPER_INITIAL_ACCEPTANCE_PROBABILITY
            ) > 1e-12
        ):
            raise ValueError(
                "paper_simulated_annealing fixes target "
                "acceptance probability at 0.5."
            )

        if (
            cooling_rate is not None
            and abs(
                float(cooling_rate)
                - PAPER_COOLING_RATE
            ) > 1e-12
        ):
            raise ValueError(
                "paper_simulated_annealing fixes "
                "cooling_rate at 0.9994."
            )

        schedule = PaperTemperatureSchedule(
            initial_objective=initial_objective
        )

        controller = PaperSimulatedAnnealingController(
            temperature_schedule=schedule,
            adaptive_controller=adaptive_controller,
            rng=rng,
        )

        return SimulatedAnnealingFactoryResult(
            mode=PAPER_SIMULATED_ANNEALING_MODE,
            controller=controller,
            metadata={
                "simulated_annealing_mode": (
                    PAPER_SIMULATED_ANNEALING_MODE
                ),
                "paper_faithful": True,
                "enhanced": False,
                "relative_worsening": (
                    PAPER_INITIAL_RELATIVE_WORSENING
                ),
                "target_acceptance_probability": (
                    PAPER_INITIAL_ACCEPTANCE_PROBABILITY
                ),
                "cooling_rate": PAPER_COOLING_RATE,
                "acceptance_formula": (
                    "exp(-(candidate-current)/temperature)"
                ),
                "random_boundary": "inclusive",
                "cool_after_each_iteration": True,
                "objective_input": "scalar_F_lambda",
                "separate_cost_emission_acceptance": False,
                "independent_state_per_lambda_run": True,
            },
        )

    enhanced_relative_worsening = (
        PAPER_INITIAL_RELATIVE_WORSENING
        if relative_worsening is None
        else float(relative_worsening)
    )
    enhanced_target_probability = (
        PAPER_INITIAL_ACCEPTANCE_PROBABILITY
        if target_acceptance_probability is None
        else float(target_acceptance_probability)
    )
    enhanced_cooling_rate = (
        PAPER_COOLING_RATE
        if cooling_rate is None
        else float(cooling_rate)
    )

    schedule = EnhancedTemperatureSchedule(
        initial_objective=initial_objective,
        relative_worsening=(
            enhanced_relative_worsening
        ),
        target_acceptance_probability=(
            enhanced_target_probability
        ),
        cooling_rate=enhanced_cooling_rate,
    )

    controller = PaperSimulatedAnnealingController(
        temperature_schedule=schedule,
        adaptive_controller=adaptive_controller,
        rng=rng,
    )

    return SimulatedAnnealingFactoryResult(
        mode=ENHANCED_SIMULATED_ANNEALING_MODE,
        controller=controller,
        metadata={
            "simulated_annealing_mode": (
                ENHANCED_SIMULATED_ANNEALING_MODE
            ),
            "paper_faithful": False,
            "enhanced": True,
            "relative_worsening": (
                enhanced_relative_worsening
            ),
            "target_acceptance_probability": (
                enhanced_target_probability
            ),
            "cooling_rate": (
                enhanced_cooling_rate
            ),
            "acceptance_formula": (
                "exp(-(candidate-current)/temperature)"
            ),
            "random_boundary": "inclusive",
            "cool_after_each_iteration": True,
            "objective_input": "scalar_F_lambda",
            "separate_cost_emission_acceptance": False,
            "independent_state_per_lambda_run": True,
            "purpose": "sensitivity_analysis_only",
        },
    )
