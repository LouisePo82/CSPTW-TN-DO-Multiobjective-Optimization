from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from alns_solver.destroy_operators import (
    random_adp_removal,
    random_customer_removal,
    random_tn_removal,
)
from alns_solver.paper_destroy_operators import (
    paper_historical_node_removal,
    paper_neighborhood_removal,
    paper_node_neighborhood_removal,
    paper_probabilistic_related_removal,
    paper_probabilistic_worst_customer_removal,
    paper_related_removal,
    paper_route_removal,
    paper_worst_adp_removal,
    paper_worst_customer_removal,
)
from alns_solver.repair_operators import (
    best_insertion_repair,
    perturbed_best_insertion_repair,
    perturbed_regret_3_repair,
    perturbed_regret_repair,
    regret_2_repair,
    regret_3_repair,
)
from alns_solver.solution_state import ALNSSolutionState


PAPER_DESTROY_OPERATOR_NAMES = (
    "random_customer_removal",
    "worst_customer_removal_deterministic",
    "worst_customer_removal_probabilistic",
    "route_removal",
    "random_adp_removal",
    "worst_adp_removal",
    "random_tn_removal",
    "related_removal_deterministic",
    "related_removal_probabilistic",
    "historical_node_removal",
    "neighborhood_removal",
    "node_neighborhood_removal",
)

PAPER_REPAIR_OPERATOR_NAMES = (
    "best_insertion",
    "regret_2",
    "perturbed_regret_2",
    "regret_3",
    "perturbed_best_insertion",
    "perturbed_regret_3",
)

PAPER_RANDOMNESS_FACTOR = 5.0
PAPER_RELATEDNESS_WEIGHTS = (5.0, 9.0, 1.0)
PAPER_PERTURBATION_FACTOR = 0.275
PAPER_STRATEGY_2_MODE = "paper_random_dv"


@dataclass(frozen=True)
class PaperOperatorDispatchContext:
    removal_count: int
    seed: int
    lambda_value: float
    cost_bounds: tuple[float, float] | None
    emission_bounds: tuple[float, float] | None
    emission_factors: tuple[float, float] = (3.0, 1.0)
    best_historical_position_costs: dict[str, float] | None = None

    def validate(self) -> None:
        if self.removal_count <= 0:
            raise ValueError("removal_count must be positive.")
        if not 0.0 <= float(self.lambda_value) <= 1.0:
            raise ValueError("lambda_value must be between 0 and 1.")


def paper_destroy_operator_names() -> tuple[str, ...]:
    return PAPER_DESTROY_OPERATOR_NAMES


def paper_repair_operator_names() -> tuple[str, ...]:
    return PAPER_REPAIR_OPERATOR_NAMES


def dispatch_paper_destroy(
    operator_name: str,
    state: ALNSSolutionState,
    instance: dict,
    *,
    context: PaperOperatorDispatchContext,
):
    """
    Dispatch one paper destroy operator with its exact validated signature.

    Historical and neighborhood operators keep their paper route-distance
    logic. Only objective-based destroy operators consume scalar F_lambda.
    """
    context.validate()

    if operator_name not in PAPER_DESTROY_OPERATOR_NAMES:
        raise ValueError(
            f"Unsupported paper destroy operator: {operator_name}"
        )

    if operator_name == "random_customer_removal":
        return random_customer_removal(
            state,
            instance,
            context.removal_count,
            seed=context.seed,
        )

    if operator_name == "worst_customer_removal_deterministic":
        return paper_worst_customer_removal(
            state,
            instance,
            context.removal_count,
            lambda_value=context.lambda_value,
            cost_bounds=context.cost_bounds,
            emission_bounds=context.emission_bounds,
            emission_factors=context.emission_factors,
        )

    if operator_name == "worst_customer_removal_probabilistic":
        return paper_probabilistic_worst_customer_removal(
            state,
            instance,
            context.removal_count,
            seed=context.seed,
            randomness_factor=PAPER_RANDOMNESS_FACTOR,
            lambda_value=context.lambda_value,
            cost_bounds=context.cost_bounds,
            emission_bounds=context.emission_bounds,
            emission_factors=context.emission_factors,
        )

    if operator_name == "route_removal":
        return paper_route_removal(
            state,
            instance,
            seed=context.seed,
        )

    if operator_name == "random_adp_removal":
        return random_adp_removal(
            state,
            instance,
            seed=context.seed,
        )

    if operator_name == "worst_adp_removal":
        return paper_worst_adp_removal(
            state,
            instance,
            lambda_value=context.lambda_value,
            cost_bounds=context.cost_bounds,
            emission_bounds=context.emission_bounds,
            emission_factors=context.emission_factors,
        )

    if operator_name == "random_tn_removal":
        return random_tn_removal(
            state,
            instance,
            seed=context.seed,
        )

    if operator_name == "related_removal_deterministic":
        phi_1, phi_2, phi_3 = PAPER_RELATEDNESS_WEIGHTS
        return paper_related_removal(
            state,
            instance,
            context.removal_count,
            seed=context.seed,
            phi_1=phi_1,
            phi_2=phi_2,
            phi_3=phi_3,
        )

    if operator_name == "related_removal_probabilistic":
        phi_1, phi_2, phi_3 = PAPER_RELATEDNESS_WEIGHTS
        return paper_probabilistic_related_removal(
            state,
            instance,
            context.removal_count,
            seed=context.seed,
            randomness_factor=PAPER_RANDOMNESS_FACTOR,
            phi_1=phi_1,
            phi_2=phi_2,
            phi_3=phi_3,
        )

    if operator_name == "historical_node_removal":
        if context.best_historical_position_costs is None:
            raise ValueError(
                "historical_node_removal requires "
                "best_historical_position_costs."
            )
        return paper_historical_node_removal(
            state,
            instance,
            context.removal_count,
            best_historical_position_costs=(
                context.best_historical_position_costs
            ),
        )

    if operator_name == "neighborhood_removal":
        return paper_neighborhood_removal(
            state,
            instance,
            context.removal_count,
        )

    if operator_name == "node_neighborhood_removal":
        return paper_node_neighborhood_removal(
            state,
            instance,
            context.removal_count,
            seed=context.seed,
        )

    raise RuntimeError("Unreachable paper destroy dispatch branch.")


def dispatch_paper_repair(
    operator_name: str,
    state: ALNSSolutionState,
    instance: dict,
    *,
    context: PaperOperatorDispatchContext,
):
    """
    Dispatch one paper repair operator.

    Strategy II, perturbation strength, and scalar objective context are fixed
    to the validated paper-mode contracts.
    """
    context.validate()

    if operator_name not in PAPER_REPAIR_OPERATOR_NAMES:
        raise ValueError(
            f"Unsupported paper repair operator: {operator_name}"
        )

    common: dict[str, Any] = {
        "lambda_value": context.lambda_value,
        "cost_bounds": context.cost_bounds,
        "emission_bounds": context.emission_bounds,
        "emission_factors": context.emission_factors,
        "strategy_2_mode": PAPER_STRATEGY_2_MODE,
        "strategy_2_seed": context.seed,
    }

    if operator_name == "best_insertion":
        return best_insertion_repair(
            state,
            instance,
            **common,
        )

    if operator_name == "regret_2":
        return regret_2_repair(
            state,
            instance,
            **common,
        )

    if operator_name == "perturbed_regret_2":
        return perturbed_regret_repair(
            state,
            instance,
            k=2,
            seed=context.seed,
            noise_strength=PAPER_PERTURBATION_FACTOR,
            **common,
        )

    if operator_name == "regret_3":
        return regret_3_repair(
            state,
            instance,
            **common,
        )

    if operator_name == "perturbed_best_insertion":
        return perturbed_best_insertion_repair(
            state,
            instance,
            seed=context.seed,
            noise_strength=PAPER_PERTURBATION_FACTOR,
            **common,
        )

    if operator_name == "perturbed_regret_3":
        return perturbed_regret_3_repair(
            state,
            instance,
            seed=context.seed,
            noise_strength=PAPER_PERTURBATION_FACTOR,
            **common,
        )

    raise RuntimeError("Unreachable paper repair dispatch branch.")
