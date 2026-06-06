from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Iterable

PAPER_SEGMENT_LENGTH = 300
PAPER_REACTION_FACTOR = 0.1
PAPER_REWARD_NEW_GLOBAL_BEST = 33.0
PAPER_REWARD_BETTER_CURRENT = 15.0
PAPER_REWARD_WORSE_ACCEPTED = 9.0
PAPER_REWARD_REJECTED = 0.0
PAPER_REWARDS = {
    "new_global_best": PAPER_REWARD_NEW_GLOBAL_BEST,
    "better_current": PAPER_REWARD_BETTER_CURRENT,
    "worse_accepted": PAPER_REWARD_WORSE_ACCEPTED,
    "rejected": PAPER_REWARD_REJECTED,
}

@dataclass
class OperatorWeightRecord:
    name: str
    weight: float = 1.0
    segment_score: float = 0.0
    segment_uses: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Operator name must not be empty.")
        if self.weight <= 0:
            raise ValueError(f"Initial weight for {self.name} must be positive.")
        if self.segment_score < 0:
            raise ValueError(f"Segment score for {self.name} must be non-negative.")
        if self.segment_uses < 0:
            raise ValueError(f"Segment uses for {self.name} must be non-negative.")

    def record(self, reward: float) -> None:
        if reward < 0:
            raise ValueError("Reward must be non-negative.")
        self.segment_uses += 1
        self.segment_score += float(reward)

    def updated_weight(self, reaction_factor: float) -> float:
        if not 0.0 <= reaction_factor <= 1.0:
            raise ValueError("reaction_factor must be between 0 and 1.")
        if self.segment_uses == 0:
            return float(self.weight)
        average_score = self.segment_score / self.segment_uses
        return ((1.0 - reaction_factor) * self.weight
                + reaction_factor * average_score)

    def reset_segment(self) -> None:
        self.segment_score = 0.0
        self.segment_uses = 0

@dataclass
class PaperAdaptiveWeightState:
    destroy_records: dict[str, OperatorWeightRecord]
    repair_records: dict[str, OperatorWeightRecord]
    segment_length: int = PAPER_SEGMENT_LENGTH
    reaction_factor: float = PAPER_REACTION_FACTOR
    completed_updates: int = 0
    update_history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.segment_length != PAPER_SEGMENT_LENGTH:
            raise ValueError("Paper mode fixes segment_length at 300.")
        if abs(self.reaction_factor - PAPER_REACTION_FACTOR) > 1e-12:
            raise ValueError("Paper mode fixes reaction_factor at 0.1.")
        if not self.destroy_records:
            raise ValueError("Destroy operator pool must not be empty.")
        if not self.repair_records:
            raise ValueError("Repair operator pool must not be empty.")
        overlap = set(self.destroy_records) & set(self.repair_records)
        if overlap:
            raise ValueError(
                "Destroy and repair operator names must be distinct. "
                f"Overlap: {sorted(overlap)}"
            )

    @classmethod
    def create(
        cls,
        destroy_operator_names: Iterable[str],
        repair_operator_names: Iterable[str],
        *,
        initial_weight: float = 1.0,
    ) -> "PaperAdaptiveWeightState":
        destroy_names = tuple(destroy_operator_names)
        repair_names = tuple(repair_operator_names)
        if len(set(destroy_names)) != len(destroy_names):
            raise ValueError("Destroy operator names must be unique.")
        if len(set(repair_names)) != len(repair_names):
            raise ValueError("Repair operator names must be unique.")
        return cls(
            destroy_records={
                name: OperatorWeightRecord(name=name, weight=initial_weight)
                for name in destroy_names
            },
            repair_records={
                name: OperatorWeightRecord(name=name, weight=initial_weight)
                for name in repair_names
            },
        )

    def _pool(self, pool: str) -> dict[str, OperatorWeightRecord]:
        if pool == "destroy":
            return self.destroy_records
        if pool == "repair":
            return self.repair_records
        raise ValueError("pool must be 'destroy' or 'repair'.")

    def record_operator_result(
        self,
        *,
        destroy_operator: str,
        repair_operator: str,
        reward: float,
    ) -> None:
        if destroy_operator not in self.destroy_records:
            raise KeyError(f"Unknown destroy operator: {destroy_operator}")
        if repair_operator not in self.repair_records:
            raise KeyError(f"Unknown repair operator: {repair_operator}")
        self.destroy_records[destroy_operator].record(reward)
        self.repair_records[repair_operator].record(reward)

    def record_event(
        self,
        *,
        destroy_operator: str,
        repair_operator: str,
        event: str,
    ) -> None:
        if event not in PAPER_REWARDS:
            raise ValueError(f"Unknown paper reward event: {event}")
        self.record_operator_result(
            destroy_operator=destroy_operator,
            repair_operator=repair_operator,
            reward=PAPER_REWARDS[event],
        )

    def should_update(self, iteration: int) -> bool:
        if iteration <= 0:
            raise ValueError("iteration must be positive.")
        return iteration % self.segment_length == 0

    def update_weights(self, *, iteration: int) -> dict[str, Any]:
        if not self.should_update(iteration):
            raise ValueError(f"Iteration {iteration} is not a segment boundary.")
        before = self.snapshot()
        updates = {"destroy": {}, "repair": {}}
        for pool_name in ("destroy", "repair"):
            records = self._pool(pool_name)
            for name, record in records.items():
                new_weight = record.updated_weight(self.reaction_factor)
                if new_weight <= 0:
                    raise RuntimeError(f"Updated weight for {name} is not positive.")
                updates[pool_name][name] = new_weight
            for name, new_weight in updates[pool_name].items():
                records[name].weight = new_weight
                records[name].reset_segment()
        self.completed_updates += 1
        event = {
            "iteration": iteration,
            "segment_length": self.segment_length,
            "reaction_factor": self.reaction_factor,
            "before": before,
            "updated_weights": updates,
            "after": self.snapshot(),
        }
        self.update_history.append(event)
        return event

    def snapshot(self) -> dict[str, Any]:
        def serialize(records):
            return {
                name: {
                    "weight": record.weight,
                    "segment_score": record.segment_score,
                    "segment_uses": record.segment_uses,
                }
                for name, record in records.items()
            }
        return {
            "destroy": serialize(self.destroy_records),
            "repair": serialize(self.repair_records),
            "segment_length": self.segment_length,
            "reaction_factor": self.reaction_factor,
            "completed_updates": self.completed_updates,
        }

def classify_paper_reward(
    *,
    candidate_objective: float,
    current_objective: float,
    best_objective: float,
    accepted: bool,
    tolerance: float = 1e-12,
) -> tuple[str, float]:
    candidate = float(candidate_objective)
    current = float(current_objective)
    best = float(best_objective)
    if candidate < best - tolerance:
        return "new_global_best", PAPER_REWARD_NEW_GLOBAL_BEST
    if candidate < current - tolerance:
        return "better_current", PAPER_REWARD_BETTER_CURRENT
    if accepted and candidate > current + tolerance:
        return "worse_accepted", PAPER_REWARD_WORSE_ACCEPTED
    return "rejected", PAPER_REWARD_REJECTED

# =============================================================
# Adaptive Weights Fidelity AW-2 — Roulette-Wheel Selection
# =============================================================

import math
import random


def operator_probabilities(
    records: dict[str, OperatorWeightRecord],
) -> dict[str, float]:
    """
    Normalize one operator pool independently:

        p_i = weight_i / sum_j(weight_j)

    Destroy and repair pools must be passed separately.
    """
    if not records:
        raise ValueError(
            "Operator pool must not be empty."
        )

    total_weight = 0.0

    for name, record in records.items():
        weight = float(record.weight)

        if not math.isfinite(weight):
            raise ValueError(
                f"Weight for {name} must be finite."
            )

        if weight <= 0.0:
            raise ValueError(
                f"Weight for {name} must be positive."
            )

        total_weight += weight

    if not math.isfinite(total_weight):
        raise ValueError(
            "Total operator weight must be finite."
        )

    if total_weight <= 0.0:
        raise ValueError(
            "Total operator weight must be positive."
        )

    probabilities = {
        name: float(record.weight) / total_weight
        for name, record in records.items()
    }

    probability_sum = sum(
        probabilities.values()
    )

    if abs(probability_sum - 1.0) > 1e-12:
        raise RuntimeError(
            "Operator probabilities do not sum to 1."
        )

    return probabilities


def select_operator_roulette(
    records: dict[str, OperatorWeightRecord],
    *,
    rng: random.Random,
) -> str:
    """
    Select one operator from one independently normalized pool.

    Iteration order is deterministic because the state preserves insertion
    order. With the same seed and the same weights, the selected sequence is
    reproducible.
    """
    probabilities = operator_probabilities(
        records
    )

    threshold = rng.random()
    cumulative = 0.0
    names = list(probabilities)

    for name in names:
        cumulative += probabilities[name]

        if threshold < cumulative:
            return name

    # Numerical safety for values extremely close to 1.
    return names[-1]


def select_destroy_operator(
    state: PaperAdaptiveWeightState,
    *,
    rng: random.Random,
) -> str:
    return select_operator_roulette(
        state.destroy_records,
        rng=rng,
    )


def select_repair_operator(
    state: PaperAdaptiveWeightState,
    *,
    rng: random.Random,
) -> str:
    return select_operator_roulette(
        state.repair_records,
        rng=rng,
    )
