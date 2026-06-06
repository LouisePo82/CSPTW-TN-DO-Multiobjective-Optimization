from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable

from alns_solver.solution_state import ALNSSolutionState
from alns_solver.destroy_operators import (
    DestroyResult,
    remove_customer,
)


@dataclass
class ADPRemovalScore:
    vehicle: str
    adp: str
    removed_customers: list[str]
    objective_before: float
    objective_after: float
    total_saving: float
    average_saving: float


def _active_dv_routes(
    state: ALNSSolutionState,
    instance: dict,
) -> list[str]:
    """
    Return only dedicated vehicles with a non-empty active route.

    Paper-faithful route removal must select a DV route, not an OD route.
    """
    return sorted(
        vehicle
        for vehicle in instance["dvs"]
        if state.dv_routes.get(vehicle, [])
    )


def _active_adp_pairs(
    state: ALNSSolutionState,
) -> list[tuple[str, str]]:
    return sorted(
        {
            (
                assignment["vehicle"],
                assignment["adp"],
            )
            for assignment in state.assignments.values()
            if assignment.get("mode") == "ADP"
        }
    )


def _customers_at_adp_pair(
    state: ALNSSolutionState,
    vehicle: str,
    adp: str,
) -> list[str]:
    return sorted(
        customer
        for customer, assignment in state.assignments.items()
        if assignment.get("mode") == "ADP"
        and assignment.get("vehicle") == vehicle
        and assignment.get("adp") == adp
    )


def _objective_value(
    state: ALNSSolutionState,
    instance: dict,
    *,
    lambda_value: float,
    cost_bounds: tuple[float, float] | None,
    emission_bounds: tuple[float, float] | None,
    emission_factors: tuple[float, float],
) -> float:
    metrics = state.evaluate(
        instance=instance,
        lambda_value=lambda_value,
        cost_bounds=cost_bounds,
        emission_bounds=emission_bounds,
        emission_factors=emission_factors,
    )

    return float(metrics["objective"])


def paper_route_removal(
    state: ALNSSolutionState,
    instance: dict,
    *,
    seed: int | None = None,
    vehicle: str | None = None,
) -> DestroyResult:
    """
    Paper-faithful Route Removal.

    Rules:
    1. Select one active dedicated-vehicle route only.
    2. Remove all DV_HOME and ADP customers assigned to that DV.
    3. If the selected DV route visits a TN, also remove every OD_HOME
       customer using that TN.
    4. Deactivate OD routes that lose all their customers.
    5. Set the selected DV route to [].
    """
    candidates = _active_dv_routes(
        state=state,
        instance=instance,
    )

    if vehicle is not None:
        candidates = [
            candidate
            for candidate in candidates
            if candidate == vehicle
        ]

    if not candidates:
        raise ValueError(
            "No active dedicated-vehicle route matches the request."
        )

    rng = random.Random(seed)
    selected_vehicle = rng.choice(candidates)

    destroyed = state.copy()
    selected_route = list(
        destroyed.dv_routes.get(selected_vehicle, [])
    )

    affected_tns = {
        tn
        for tn in instance["tns"]
        if tn in selected_route
    }

    direct_customers = {
        customer
        for customer, assignment in destroyed.assignments.items()
        if assignment.get("vehicle") == selected_vehicle
        and assignment.get("mode") in {"DV_HOME", "ADP"}
    }

    tn_customers = {
        customer
        for customer, assignment in destroyed.assignments.items()
        if assignment.get("mode") == "OD_HOME"
        and assignment.get("pickup") in affected_tns
    }

    removed_customers = sorted(
        direct_customers | tn_customers
    )

    for customer in list(removed_customers):
        if customer in destroyed.assignments:
            remove_customer(
                state=destroyed,
                instance=instance,
                customer_id=customer,
            )

    destroyed.dv_routes[selected_vehicle] = []
    destroyed.invalidate_cache()

    destroyed.register_operator_event(
        operator_type="destroy",
        operator_name="paper_route_removal",
        details={
            "seed": seed,
            "vehicle": selected_vehicle,
            "affected_tns": sorted(affected_tns),
            "removed_customers": list(removed_customers),
        },
    )

    return DestroyResult(
        operator_name="paper_route_removal",
        state=destroyed,
        removed_customers=removed_customers,
        removed_route={
            "route_type": "DV",
            "route_id": selected_vehicle,
            "affected_tns": sorted(affected_tns),
        },
    )


def score_adp_pair_eq46(
    state: ALNSSolutionState,
    instance: dict,
    *,
    vehicle: str,
    adp: str,
    lambda_value: float,
    cost_bounds: tuple[float, float] | None,
    emission_bounds: tuple[float, float] | None,
    emission_factors: tuple[float, float],
) -> ADPRemovalScore:
    """
    Compute the paper Eq. (46) average ADP removal score:

        c_d = [f(S) - f(S_d^-)] / N_d

    where N_d is the number of customers assigned to the selected DV-ADP pair.
    """
    removed_customers = _customers_at_adp_pair(
        state,
        vehicle,
        adp,
    )

    if not removed_customers:
        raise ValueError(
            f"No customers are assigned to ({vehicle}, {adp})."
        )

    objective_before = _objective_value(
        state,
        instance,
        lambda_value=lambda_value,
        cost_bounds=cost_bounds,
        emission_bounds=emission_bounds,
        emission_factors=emission_factors,
    )

    candidate_state = state.copy()

    for customer in list(removed_customers):
        remove_customer(
            state=candidate_state,
            instance=instance,
            customer_id=customer,
        )

    objective_after = _objective_value(
        candidate_state,
        instance,
        lambda_value=lambda_value,
        cost_bounds=cost_bounds,
        emission_bounds=emission_bounds,
        emission_factors=emission_factors,
    )

    total_saving = objective_before - objective_after
    average_saving = total_saving / len(removed_customers)

    return ADPRemovalScore(
        vehicle=vehicle,
        adp=adp,
        removed_customers=removed_customers,
        objective_before=objective_before,
        objective_after=objective_after,
        total_saving=total_saving,
        average_saving=average_saving,
    )


def paper_worst_adp_removal(
    state: ALNSSolutionState,
    instance: dict,
    *,
    lambda_value: float,
    cost_bounds: tuple[float, float] | None,
    emission_bounds: tuple[float, float] | None,
    emission_factors: tuple[float, float] = (1.0, 1.0),
    candidate_pairs: Iterable[tuple[str, str]] | None = None,
) -> DestroyResult:
    """
    Paper-faithful Worst ADP Removal using Eq. (46).

    Select the active DV-ADP pair with the highest average objective saving
    per removed customer, not the highest total saving.
    """
    active_pairs = _active_adp_pairs(state)

    if candidate_pairs is not None:
        allowed = set(candidate_pairs)
        active_pairs = [
            pair
            for pair in active_pairs
            if pair in allowed
        ]

    if not active_pairs:
        raise ValueError(
            "No active DV-ADP pair is available."
        )

    scores = [
        score_adp_pair_eq46(
            state,
            instance,
            vehicle=vehicle,
            adp=adp,
            lambda_value=lambda_value,
            cost_bounds=cost_bounds,
            emission_bounds=emission_bounds,
            emission_factors=emission_factors,
        )
        for vehicle, adp in active_pairs
    ]

    selected = max(
        scores,
        key=lambda score: (
            score.average_saving,
            score.total_saving,
            score.vehicle,
            score.adp,
        ),
    )

    destroyed = state.copy()

    for customer in list(selected.removed_customers):
        remove_customer(
            state=destroyed,
            instance=instance,
            customer_id=customer,
        )

    destroyed.invalidate_cache()

    destroyed.register_operator_event(
        operator_type="destroy",
        operator_name="paper_worst_adp_removal",
        details={
            "vehicle": selected.vehicle,
            "adp": selected.adp,
            "removed_customers": list(
                selected.removed_customers
            ),
            "objective_before": selected.objective_before,
            "objective_after": selected.objective_after,
            "total_saving": selected.total_saving,
            "average_saving_eq46": selected.average_saving,
            "lambda_value": lambda_value,
        },
    )

    return DestroyResult(
        operator_name="paper_worst_adp_removal",
        state=destroyed,
        removed_customers=list(
            selected.removed_customers
        ),
        removed_route={
            "facility_type": "ADP",
            "vehicle": selected.vehicle,
            "facility_id": selected.adp,
            "total_saving": selected.total_saving,
            "average_saving_eq46": selected.average_saving,
        },
    )
