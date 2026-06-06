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
