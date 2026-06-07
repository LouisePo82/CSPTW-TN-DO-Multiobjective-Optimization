from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from alns_solver.paper_destroy_operators import current_position_cost
from alns_solver.solution_state import ALNSSolutionState


def _observable_historical_customers(
    state: ALNSSolutionState,
    instance: dict,
) -> list[str]:
    """
    Return Type-1/3 customers currently served by home delivery.

    Historical position costs are observable only when the customer appears
    as a customer node in a DV_HOME or OD_HOME route.
    """
    observable: list[str] = []

    for customer, assignment in state.assignments.items():
        customer_type = int(
            instance["nodes"][customer]["customer_type"]
        )
        mode = assignment.get("mode")

        if (
            customer_type in {1, 3}
            and mode in {"DV_HOME", "OD_HOME"}
        ):
            observable.append(customer)

    return sorted(observable)


@dataclass
class PaperHistoricalPositionState:
    """
    Historical-position lifecycle for paper Historical Node Removal.

    The stored value for customer i is:

        min position cost observed before the next iteration

    where the paper position cost is:

        c(prev,i) + c(i,next)

    ADP observations are skipped. Existing historical values are retained.
    """

    best_position_costs: dict[str, float] = field(
        default_factory=dict
    )
    observation_counts: dict[str, int] = field(
        default_factory=dict
    )
    completed_observations: int = 0

    @classmethod
    def initialize_from_state(
        cls,
        state: ALNSSolutionState,
        instance: dict,
    ) -> "PaperHistoricalPositionState":
        history = cls()
        history.observe_current_state(
            state,
            instance,
        )
        return history

    def snapshot(self) -> dict[str, float]:
        """
        Return an independent map for destroy-dispatch context.
        """
        return dict(self.best_position_costs)

    def observe_current_state(
        self,
        state: ALNSSolutionState,
        instance: dict,
    ) -> dict[str, float]:
        """
        Observe the accepted/current state after a state transition.

        Returns the current observations made in this call. A customer at ADP
        produces no observation and does not erase its previous best.
        """
        observations: dict[str, float] = {}

        for customer in _observable_historical_customers(
            state,
            instance,
        ):
            value = float(
                current_position_cost(
                    state,
                    instance,
                    customer,
                )
            )
            observations[customer] = value

            if customer not in self.best_position_costs:
                self.best_position_costs[customer] = value
            else:
                self.best_position_costs[customer] = min(
                    self.best_position_costs[customer],
                    value,
                )

            self.observation_counts[customer] = (
                self.observation_counts.get(customer, 0)
                + 1
            )

        self.completed_observations += 1
        return observations

    def best_cost(self, customer: str) -> float:
        if customer not in self.best_position_costs:
            raise KeyError(
                f"No historical position cost for {customer}."
            )
        return float(self.best_position_costs[customer])

    def metadata(self) -> dict[str, Any]:
        return {
            "paper_faithful": True,
            "enhanced": False,
            "metric": "two_arc_position_cost",
            "formula": "c(prev,i)+c(i,next)",
            "objective_extension_applied": False,
            "completed_observations": (
                self.completed_observations
            ),
            "tracked_customers": sorted(
                self.best_position_costs
            ),
        }
