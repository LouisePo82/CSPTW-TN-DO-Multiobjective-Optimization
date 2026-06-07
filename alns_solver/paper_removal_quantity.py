from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any


PAPER_REMOVAL_LOWER_FRACTION = 0.10
PAPER_REMOVAL_UPPER_FRACTION = 0.40

PAPER_COUNT_BASED_DESTROY_OPERATORS = (
    "random_customer_removal",
    "worst_customer_removal_deterministic",
    "worst_customer_removal_probabilistic",
    "related_removal_deterministic",
    "related_removal_probabilistic",
    "historical_node_removal",
    "neighborhood_removal",
    "node_neighborhood_removal",
)

PAPER_STRUCTURAL_DESTROY_OPERATORS = (
    "route_removal",
    "random_adp_removal",
    "worst_adp_removal",
    "random_tn_removal",
)


@dataclass(frozen=True)
class PaperRemovalQuantityBounds:
    customer_count: int
    lower_fraction: float
    upper_fraction: float
    minimum: int
    maximum: int
    rounding_rule: str


@dataclass(frozen=True)
class PaperRemovalQuantitySample:
    quantity: int
    bounds: PaperRemovalQuantityBounds
    seed: int | None
    metadata: dict[str, Any]


def paper_removal_quantity_bounds(
    customer_count: int,
) -> PaperRemovalQuantityBounds:
    """
    Integer realization of the paper removal interval:

        N_min = 0.1 |N|
        N_max = 0.4 |N|

    The paper reports fractional bounds but q must be an integer. Paper mode
    uses ceil for the lower bound and floor for the upper bound, with q >= 1.
    """
    if customer_count <= 0:
        raise ValueError(
            "customer_count must be positive."
        )

    minimum = max(
        1,
        math.ceil(
            PAPER_REMOVAL_LOWER_FRACTION
            * customer_count
        ),
    )
    maximum = max(
        minimum,
        math.floor(
            PAPER_REMOVAL_UPPER_FRACTION
            * customer_count
        ),
    )

    # q cannot exceed the total number of customers.
    maximum = min(maximum, customer_count)

    return PaperRemovalQuantityBounds(
        customer_count=customer_count,
        lower_fraction=(
            PAPER_REMOVAL_LOWER_FRACTION
        ),
        upper_fraction=(
            PAPER_REMOVAL_UPPER_FRACTION
        ),
        minimum=minimum,
        maximum=maximum,
        rounding_rule="ceil_lower_floor_upper",
    )


def sample_paper_removal_quantity(
    customer_count: int,
    *,
    rng: random.Random,
    seed: int | None = None,
) -> PaperRemovalQuantitySample:
    """
    Uniformly sample integer q from the paper interval.
    """
    bounds = paper_removal_quantity_bounds(
        customer_count
    )
    quantity = rng.randint(
        bounds.minimum,
        bounds.maximum,
    )

    return PaperRemovalQuantitySample(
        quantity=quantity,
        bounds=bounds,
        seed=seed,
        metadata={
            "paper_faithful": True,
            "enhanced": False,
            "distribution": (
                "discrete_uniform_inclusive"
            ),
            "formula": (
                "q in [ceil(0.1|N|), "
                "floor(0.4|N|)]"
            ),
            "lambda_dependent": False,
        },
    )


def destroy_operator_uses_removal_quantity(
    operator_name: str,
) -> bool:
    if operator_name in (
        PAPER_COUNT_BASED_DESTROY_OPERATORS
    ):
        return True

    if operator_name in (
        PAPER_STRUCTURAL_DESTROY_OPERATORS
    ):
        return False

    raise ValueError(
        f"Unknown paper destroy operator: {operator_name}"
    )
