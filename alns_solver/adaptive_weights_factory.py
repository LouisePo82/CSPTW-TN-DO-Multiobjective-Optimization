from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from alns_solver.paper_adaptive_weights import (
    PAPER_REACTION_FACTOR,
    PAPER_REWARD_BETTER_CURRENT,
    PAPER_REWARD_NEW_GLOBAL_BEST,
    PAPER_REWARD_REJECTED,
    PAPER_REWARD_WORSE_ACCEPTED,
    PAPER_SEGMENT_LENGTH,
    OperatorWeightRecord,
    PaperAdaptiveWeightController,
    PaperAdaptiveWeightState,
)


PAPER_ADAPTIVE_WEIGHTS_MODE = "paper_adaptive_weights"
ENHANCED_ADAPTIVE_WEIGHTS_MODE = "enhanced_adaptive_weights"

SUPPORTED_ADAPTIVE_WEIGHT_MODES = {
    PAPER_ADAPTIVE_WEIGHTS_MODE,
    ENHANCED_ADAPTIVE_WEIGHTS_MODE,
}


@dataclass
class AdaptiveWeightsFactoryResult:
    mode: str
    state: PaperAdaptiveWeightState
    controller: PaperAdaptiveWeightController
    metadata: dict[str, Any]


class EnhancedAdaptiveWeightState(PaperAdaptiveWeightState):
    """
    Explicitly non-paper sensitivity-analysis variant.

    It reuses the validated paper state mechanics but allows controlled
    parameter overrides. It must never be labelled paper-faithful.
    """

    def __post_init__(self) -> None:
        if self.segment_length <= 0:
            raise ValueError(
                "Enhanced segment_length must be positive."
            )

        if not 0.0 <= self.reaction_factor <= 1.0:
            raise ValueError(
                "Enhanced reaction_factor must be between 0 and 1."
            )

        if not self.destroy_records:
            raise ValueError(
                "Destroy operator pool must not be empty."
            )

        if not self.repair_records:
            raise ValueError(
                "Repair operator pool must not be empty."
            )

        overlap = (
            set(self.destroy_records)
            & set(self.repair_records)
        )
        if overlap:
            raise ValueError(
                "Destroy and repair operator names must be distinct."
            )

    @classmethod
    def create_enhanced(
        cls,
        destroy_operator_names: Iterable[str],
        repair_operator_names: Iterable[str],
        *,
        initial_weight: float,
        segment_length: int,
        reaction_factor: float,
    ) -> "EnhancedAdaptiveWeightState":
        destroy_names = tuple(destroy_operator_names)
        repair_names = tuple(repair_operator_names)

        if len(set(destroy_names)) != len(destroy_names):
            raise ValueError(
                "Destroy operator names must be unique."
            )

        if len(set(repair_names)) != len(repair_names):
            raise ValueError(
                "Repair operator names must be unique."
            )

        return cls(
            destroy_records={
                name: OperatorWeightRecord(
                    name=name,
                    weight=initial_weight,
                )
                for name in destroy_names
            },
            repair_records={
                name: OperatorWeightRecord(
                    name=name,
                    weight=initial_weight,
                )
                for name in repair_names
            },
            segment_length=segment_length,
            reaction_factor=reaction_factor,
        )


def build_adaptive_weights(
    *,
    mode: str,
    destroy_operator_names: Iterable[str],
    repair_operator_names: Iterable[str],
    initial_weight: float = 1.0,
    segment_length: int | None = None,
    reaction_factor: float | None = None,
    rewards: dict[str, float] | None = None,
) -> AdaptiveWeightsFactoryResult:
    """
    Factory separating paper-faithful and enhanced adaptive-weight modes.

    Paper mode:
    - segment length 300;
    - reaction factor 0.1;
    - rewards 33, 15, 9, 0;
    - equal initial weights;
    - unused operators preserve old weight.

    Enhanced mode:
    - explicit sensitivity-analysis variant;
    - custom segment length and reaction factor allowed;
    - always labelled non-paper.
    """
    if mode not in SUPPORTED_ADAPTIVE_WEIGHT_MODES:
        raise ValueError(
            f"Unsupported adaptive-weight mode: {mode}. "
            f"Supported modes: {sorted(SUPPORTED_ADAPTIVE_WEIGHT_MODES)}"
        )

    paper_rewards = {
        "new_global_best": PAPER_REWARD_NEW_GLOBAL_BEST,
        "better_current": PAPER_REWARD_BETTER_CURRENT,
        "worse_accepted": PAPER_REWARD_WORSE_ACCEPTED,
        "rejected": PAPER_REWARD_REJECTED,
    }

    if mode == PAPER_ADAPTIVE_WEIGHTS_MODE:
        if (
            segment_length is not None
            and segment_length != PAPER_SEGMENT_LENGTH
        ):
            raise ValueError(
                "paper_adaptive_weights fixes segment_length at 300."
            )

        if (
            reaction_factor is not None
            and abs(
                float(reaction_factor)
                - PAPER_REACTION_FACTOR
            )
            > 1e-12
        ):
            raise ValueError(
                "paper_adaptive_weights fixes reaction_factor at 0.1."
            )

        if rewards is not None and rewards != paper_rewards:
            raise ValueError(
                "paper_adaptive_weights fixes rewards at 33, 15, 9, and 0."
            )

        if initial_weight <= 0:
            raise ValueError(
                "initial_weight must be positive."
            )

        state = PaperAdaptiveWeightState.create(
            destroy_operator_names,
            repair_operator_names,
            initial_weight=initial_weight,
        )
        controller = PaperAdaptiveWeightController(
            state=state
        )

        return AdaptiveWeightsFactoryResult(
            mode=PAPER_ADAPTIVE_WEIGHTS_MODE,
            state=state,
            controller=controller,
            metadata={
                "adaptive_weights_mode": (
                    PAPER_ADAPTIVE_WEIGHTS_MODE
                ),
                "paper_faithful": True,
                "enhanced": False,
                "segment_length": PAPER_SEGMENT_LENGTH,
                "reaction_factor": PAPER_REACTION_FACTOR,
                "rewards": paper_rewards,
                "roulette_selection": True,
                "separate_destroy_repair_pools": True,
                "unused_operator_policy": (
                    "preserve_old_weight"
                ),
                "objective_input": "scalar_F_lambda",
                "separate_cost_emission_rewards": False,
                "independent_state_per_lambda_run": True,
            },
        )

    enhanced_segment_length = (
        PAPER_SEGMENT_LENGTH
        if segment_length is None
        else int(segment_length)
    )
    enhanced_reaction_factor = (
        PAPER_REACTION_FACTOR
        if reaction_factor is None
        else float(reaction_factor)
    )

    if rewards is not None and any(
        value < 0
        for value in rewards.values()
    ):
        raise ValueError(
            "Enhanced rewards must be non-negative."
        )

    state = EnhancedAdaptiveWeightState.create_enhanced(
        destroy_operator_names,
        repair_operator_names,
        initial_weight=initial_weight,
        segment_length=enhanced_segment_length,
        reaction_factor=enhanced_reaction_factor,
    )
    controller = PaperAdaptiveWeightController(
        state=state
    )

    return AdaptiveWeightsFactoryResult(
        mode=ENHANCED_ADAPTIVE_WEIGHTS_MODE,
        state=state,
        controller=controller,
        metadata={
            "adaptive_weights_mode": (
                ENHANCED_ADAPTIVE_WEIGHTS_MODE
            ),
            "paper_faithful": False,
            "enhanced": True,
            "segment_length": enhanced_segment_length,
            "reaction_factor": enhanced_reaction_factor,
            "rewards": (
                paper_rewards
                if rewards is None
                else dict(rewards)
            ),
            "roulette_selection": True,
            "separate_destroy_repair_pools": True,
            "unused_operator_policy": (
                "preserve_old_weight"
            ),
            "objective_input": "scalar_F_lambda",
            "separate_cost_emission_rewards": False,
            "independent_state_per_lambda_run": True,
        },
    )
