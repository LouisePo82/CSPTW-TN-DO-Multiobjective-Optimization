from __future__ import annotations

from dataclasses import dataclass
import math
import random


EPSILON = 1e-12


@dataclass(frozen=True)
class SimulatedAnnealingDecision:
    candidate_objective: float
    current_objective: float
    temperature: float
    delta_objective: float
    acceptance_probability: float
    random_value: float | None
    accepted: bool
    reason: str


def _validate_finite(
    value: float,
    name: str,
) -> float:
    converted = float(value)

    if not math.isfinite(converted):
        raise ValueError(
            f"{name} must be finite."
        )

    return converted


def worse_candidate_acceptance_probability(
    *,
    candidate_objective: float,
    current_objective: float,
    temperature: float,
) -> float:
    """
    Return the simulated-annealing acceptance probability.

    For minimization:

        delta = candidate - current

    Better and equal candidates have probability 1. Worse candidates use:

        exp(-delta / temperature)

    The caller supplies one scalar objective. In the multi-objective
    extension this scalar is F_lambda.
    """
    candidate = _validate_finite(
        candidate_objective,
        "candidate_objective",
    )
    current = _validate_finite(
        current_objective,
        "current_objective",
    )
    validated_temperature = _validate_finite(
        temperature,
        "temperature",
    )

    if validated_temperature <= 0.0:
        raise ValueError(
            "temperature must be strictly positive."
        )

    delta = candidate - current

    if delta <= 0.0:
        return 1.0

    exponent = -delta / validated_temperature

    # math.exp safely underflows to 0.0 for sufficiently negative values.
    probability = math.exp(exponent)

    if not 0.0 <= probability <= 1.0:
        raise RuntimeError(
            "Acceptance probability is outside [0, 1]."
        )

    return probability


def accept_with_simulated_annealing(
    *,
    candidate_objective: float,
    current_objective: float,
    temperature: float,
    rng: random.Random,
) -> SimulatedAnnealingDecision:
    """
    Paper-mode simulated-annealing acceptance decision.

    - Better candidate: always accept.
    - Equal candidate: always accept.
    - Worse candidate: accept when u <= exp(-delta / T).

    The boundary is inclusive.
    """
    candidate = _validate_finite(
        candidate_objective,
        "candidate_objective",
    )
    current = _validate_finite(
        current_objective,
        "current_objective",
    )
    validated_temperature = _validate_finite(
        temperature,
        "temperature",
    )

    if validated_temperature <= 0.0:
        raise ValueError(
            "temperature must be strictly positive."
        )

    delta = candidate - current

    if delta < -EPSILON:
        return SimulatedAnnealingDecision(
            candidate_objective=candidate,
            current_objective=current,
            temperature=validated_temperature,
            delta_objective=delta,
            acceptance_probability=1.0,
            random_value=None,
            accepted=True,
            reason="better_candidate",
        )

    if abs(delta) <= EPSILON:
        return SimulatedAnnealingDecision(
            candidate_objective=candidate,
            current_objective=current,
            temperature=validated_temperature,
            delta_objective=delta,
            acceptance_probability=1.0,
            random_value=None,
            accepted=True,
            reason="equal_candidate",
        )

    probability = worse_candidate_acceptance_probability(
        candidate_objective=candidate,
        current_objective=current,
        temperature=validated_temperature,
    )
    random_value = float(rng.random())
    accepted = random_value <= probability

    return SimulatedAnnealingDecision(
        candidate_objective=candidate,
        current_objective=current,
        temperature=validated_temperature,
        delta_objective=delta,
        acceptance_probability=probability,
        random_value=random_value,
        accepted=accepted,
        reason=(
            "worse_candidate_accepted"
            if accepted
            else "worse_candidate_rejected"
        ),
    )

# =============================================================
# Simulated Annealing Fidelity SA-2 — Temperature Schedule
# =============================================================

from dataclasses import field
from typing import Any


PAPER_INITIAL_RELATIVE_WORSENING = 0.5
PAPER_INITIAL_ACCEPTANCE_PROBABILITY = 0.5
PAPER_COOLING_RATE = 0.9994


def paper_initial_temperature(
    *,
    initial_objective: float,
    relative_worsening: float = PAPER_INITIAL_RELATIVE_WORSENING,
    target_acceptance_probability: float = (
        PAPER_INITIAL_ACCEPTANCE_PROBABILITY
    ),
) -> float:
    """
    Calibrate T0 so that a solution worse than the initial solution by
    `relative_worsening` is accepted with the target probability.

        target_probability
        = exp(-(relative_worsening * initial_objective) / T0)

    Therefore:

        T0
        = -(relative_worsening * initial_objective)
          / ln(target_probability)

    Paper mode fixes relative_worsening=0.5 and target probability=0.5.
    """
    objective = _validate_finite(
        initial_objective,
        "initial_objective",
    )
    worsening = _validate_finite(
        relative_worsening,
        "relative_worsening",
    )
    probability = _validate_finite(
        target_acceptance_probability,
        "target_acceptance_probability",
    )

    if objective <= 0.0:
        raise ValueError(
            "initial_objective must be strictly positive for "
            "paper temperature calibration."
        )

    if worsening <= 0.0:
        raise ValueError(
            "relative_worsening must be strictly positive."
        )

    if not 0.0 < probability < 1.0:
        raise ValueError(
            "target_acceptance_probability must be strictly "
            "between 0 and 1."
        )

    temperature = -(
        worsening * objective
    ) / math.log(probability)

    if (
        not math.isfinite(temperature)
        or temperature <= 0.0
    ):
        raise RuntimeError(
            "Calibrated initial temperature is invalid."
        )

    return temperature


def cool_temperature(
    *,
    temperature: float,
    cooling_rate: float = PAPER_COOLING_RATE,
) -> float:
    """
    Apply geometric cooling:

        T_next = cooling_rate * T_current
    """
    current = _validate_finite(
        temperature,
        "temperature",
    )
    rate = _validate_finite(
        cooling_rate,
        "cooling_rate",
    )

    if current <= 0.0:
        raise ValueError(
            "temperature must be strictly positive."
        )

    if not 0.0 < rate < 1.0:
        raise ValueError(
            "cooling_rate must be strictly between 0 and 1."
        )

    next_temperature = current * rate

    if (
        not math.isfinite(next_temperature)
        or next_temperature <= 0.0
    ):
        raise RuntimeError(
            "Cooled temperature is invalid."
        )

    return next_temperature


@dataclass
class PaperTemperatureSchedule:
    """
    Paper geometric temperature schedule.

    Iteration k uses the current temperature. Cooling occurs after the
    iteration decision, so iteration 1 uses T0.
    """
    initial_objective: float
    cooling_rate: float = PAPER_COOLING_RATE
    initial_temperature: float = field(
        init=False
    )
    current_temperature: float = field(
        init=False
    )
    completed_iterations: int = 0
    history: list[dict[str, Any]] = field(
        default_factory=list
    )

    def __post_init__(self) -> None:
        if abs(
            float(self.cooling_rate)
            - PAPER_COOLING_RATE
        ) > EPSILON:
            raise ValueError(
                "Paper mode fixes cooling_rate at 0.9994."
            )

        self.initial_temperature = (
            paper_initial_temperature(
                initial_objective=(
                    self.initial_objective
                )
            )
        )
        self.current_temperature = (
            self.initial_temperature
        )

    def temperature_for_iteration(
        self,
        iteration: int,
    ) -> float:
        expected_iteration = (
            self.completed_iterations + 1
        )

        if iteration != expected_iteration:
            raise ValueError(
                "Temperature iterations must be consecutive. "
                f"Expected {expected_iteration}, received {iteration}."
            )

        return float(self.current_temperature)

    def cool_after_iteration(
        self,
        iteration: int,
    ) -> dict[str, float | int]:
        expected_iteration = (
            self.completed_iterations + 1
        )

        if iteration != expected_iteration:
            raise ValueError(
                "Cooling iterations must be consecutive. "
                f"Expected {expected_iteration}, received {iteration}."
            )

        used_temperature = float(
            self.current_temperature
        )
        next_temperature = cool_temperature(
            temperature=used_temperature,
            cooling_rate=self.cooling_rate,
        )

        event = {
            "iteration": iteration,
            "temperature_used": used_temperature,
            "temperature_after_cooling": (
                next_temperature
            ),
            "cooling_rate": self.cooling_rate,
        }

        self.current_temperature = (
            next_temperature
        )
        self.completed_iterations = iteration
        self.history.append(event)

        return event

    def snapshot(self) -> dict[str, Any]:
        return {
            "initial_objective": (
                float(self.initial_objective)
            ),
            "initial_temperature": (
                self.initial_temperature
            ),
            "current_temperature": (
                self.current_temperature
            ),
            "cooling_rate": self.cooling_rate,
            "completed_iterations": (
                self.completed_iterations
            ),
            "history": list(self.history),
        }

# =============================================================
# Simulated Annealing Fidelity SA-3 — Adaptive Reward Integration
# =============================================================

from alns_solver.paper_adaptive_weights import (
    AdaptiveIterationResult,
    PaperAdaptiveWeightController,
)


@dataclass
class SimulatedAnnealingIterationResult:
    iteration: int
    temperature_used: float
    temperature_after_cooling: float
    acceptance_decision: SimulatedAnnealingDecision
    adaptive_result: AdaptiveIterationResult
    current_objective_before: float
    current_objective_after: float
    best_objective_before: float
    best_objective_after: float


@dataclass
class PaperSimulatedAnnealingController:
    """
    Integration boundary between paper SA and paper adaptive rewards.

    The same scalar objective is used by:
    - simulated-annealing acceptance;
    - current/best solution updates;
    - adaptive-weight reward classification.

    In the multi-objective extension this scalar is F_lambda.
    """
    temperature_schedule: PaperTemperatureSchedule
    adaptive_controller: PaperAdaptiveWeightController
    rng: random.Random

    def process_iteration(
        self,
        *,
        iteration: int,
        destroy_operator: str,
        repair_operator: str,
        candidate_objective: float,
        current_objective: float,
        best_objective: float,
    ) -> SimulatedAnnealingIterationResult:
        temperature = (
            self.temperature_schedule
            .temperature_for_iteration(iteration)
        )

        decision = accept_with_simulated_annealing(
            candidate_objective=candidate_objective,
            current_objective=current_objective,
            temperature=temperature,
            rng=self.rng,
        )

        adaptive_result = (
            self.adaptive_controller.process_iteration(
                iteration=iteration,
                destroy_operator=destroy_operator,
                repair_operator=repair_operator,
                candidate_objective=candidate_objective,
                current_objective=current_objective,
                best_objective=best_objective,
                accepted=decision.accepted,
            )
        )

        current_after = (
            float(candidate_objective)
            if decision.accepted
            else float(current_objective)
        )

        best_after = min(
            float(best_objective),
            float(candidate_objective),
        )

        cooling_event = (
            self.temperature_schedule
            .cool_after_iteration(iteration)
        )

        return SimulatedAnnealingIterationResult(
            iteration=iteration,
            temperature_used=temperature,
            temperature_after_cooling=float(
                cooling_event[
                    "temperature_after_cooling"
                ]
            ),
            acceptance_decision=decision,
            adaptive_result=adaptive_result,
            current_objective_before=float(
                current_objective
            ),
            current_objective_after=current_after,
            best_objective_before=float(
                best_objective
            ),
            best_objective_after=best_after,
        )
