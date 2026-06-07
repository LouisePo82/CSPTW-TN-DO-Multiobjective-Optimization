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

# =============================================================
# Destroy Fidelity Gate 2
# Related / Historical / Neighborhood / Node-Neighborhood
# =============================================================

@dataclass
class RelatednessScore:
    seed_customer: str
    customer: str
    normalized_distance: float
    normalized_demand_difference: float
    type_similarity: float
    total_score: float


@dataclass
class HistoricalRemovalScore:
    customer: str
    current_position_cost: float
    best_historical_position_cost: float
    score: float


@dataclass
class NeighborhoodRemovalScore:
    customer: str
    route_type: str
    route_id: str
    route_cost_before: float
    route_cost_after: float
    contribution: float


def _eligible_type_13_customers(
    state: ALNSSolutionState,
    instance: dict,
) -> list[str]:
    """
    Customers eligible for the paper's historical and neighborhood operators:
    Type 1 and Type 3 only.
    """
    return sorted(
        customer
        for customer in state.assignments
        if int(
            instance["nodes"][customer]["customer_type"]
        ) in {1, 3}
    )


def _customer_demand(
    instance: dict,
    customer: str,
) -> float:
    return float(
        instance["nodes"][customer]["demand"]
    )


def _max_customer_distance(
    instance: dict,
    customers: list[str],
) -> float:
    maximum = 0.0

    for i in customers:
        for j in customers:
            maximum = max(
                maximum,
                float(instance["distance"][i][j]),
            )

    return maximum


def _demand_range(
    instance: dict,
    customers: list[str],
) -> float:
    demands = [
        _customer_demand(instance, customer)
        for customer in customers
    ]

    if not demands:
        return 0.0

    return max(demands) - min(demands)


def _type_similarity_term(
    state: ALNSSolutionState,
    instance: dict,
    customer_i: str,
    customer_j: str,
) -> float:
    """
    Paper Eq. (47) TS_ij term.

    - two Type-2 customers assigned to the same ADP: 0.0
    - two Type-3 customers assigned to the same ADP: 0.5
    - otherwise: 1.0
    """
    type_i = int(
        instance["nodes"][customer_i]["customer_type"]
    )
    type_j = int(
        instance["nodes"][customer_j]["customer_type"]
    )

    assignment_i = state.assignments[customer_i]
    assignment_j = state.assignments[customer_j]

    same_adp = (
        assignment_i.get("mode") == "ADP"
        and assignment_j.get("mode") == "ADP"
        and assignment_i.get("adp")
        == assignment_j.get("adp")
    )

    if type_i == 2 and type_j == 2 and same_adp:
        return 0.0

    if type_i == 3 and type_j == 3 and same_adp:
        return 0.5

    return 1.0


def relatedness_eq47(
    state: ALNSSolutionState,
    instance: dict,
    seed_customer: str,
    customer: str,
    *,
    phi_1: float = 5.0,
    phi_2: float = 9.0,
    phi_3: float = 1.0,
) -> RelatednessScore:
    """
    Paper Eq. (47):

        S(i,j)
        = phi_1 * c_ij / max(c)
        + phi_2 * |d_i-d_j| / (max(d)-min(d))
        + phi_3 * TS_ij

    Lower score means more related.
    """
    active_customers = sorted(state.assignments)

    if seed_customer not in state.assignments:
        raise ValueError(
            f"Seed customer {seed_customer} is not active."
        )

    if customer not in state.assignments:
        raise ValueError(
            f"Customer {customer} is not active."
        )

    max_distance = _max_customer_distance(
        instance,
        active_customers,
    )

    demand_range = _demand_range(
        instance,
        active_customers,
    )

    raw_distance = float(
        instance["distance"][seed_customer][customer]
    )

    normalized_distance = (
        raw_distance / max_distance
        if max_distance > 1e-12
        else 0.0
    )

    demand_difference = abs(
        _customer_demand(instance, seed_customer)
        - _customer_demand(instance, customer)
    )

    normalized_demand_difference = (
        demand_difference / demand_range
        if demand_range > 1e-12
        else 0.0
    )

    type_similarity = _type_similarity_term(
        state,
        instance,
        seed_customer,
        customer,
    )

    total_score = (
        phi_1 * normalized_distance
        + phi_2 * normalized_demand_difference
        + phi_3 * type_similarity
    )

    return RelatednessScore(
        seed_customer=seed_customer,
        customer=customer,
        normalized_distance=normalized_distance,
        normalized_demand_difference=(
            normalized_demand_difference
        ),
        type_similarity=type_similarity,
        total_score=total_score,
    )


def paper_related_removal(
    state: ALNSSolutionState,
    instance: dict,
    removal_count: int,
    *,
    seed: int | None = None,
    seed_customer: str | None = None,
    candidate_customers: Iterable[str] | None = None,
    phi_1: float = 5.0,
    phi_2: float = 9.0,
    phi_3: float = 1.0,
) -> DestroyResult:
    """
    Deterministic paper Related Removal using Eq. (47).
    """
    if removal_count <= 0:
        raise ValueError(
            "removal_count must be positive."
        )

    active = set(state.assignments)

    candidates = (
        sorted(active)
        if candidate_customers is None
        else sorted(
            active & set(candidate_customers)
        )
    )

    if removal_count > len(candidates):
        raise ValueError(
            "removal_count exceeds candidate count."
        )

    rng = random.Random(seed)

    if seed_customer is None:
        seed_customer = rng.choice(candidates)

    if seed_customer not in candidates:
        raise ValueError(
            "seed_customer is not an active candidate."
        )

    scores = [
        relatedness_eq47(
            state,
            instance,
            seed_customer,
            customer,
            phi_1=phi_1,
            phi_2=phi_2,
            phi_3=phi_3,
        )
        for customer in candidates
    ]

    selected = [
        score.customer
        for score in sorted(
            scores,
            key=lambda score: (
                score.total_score,
                score.customer,
            ),
        )[:removal_count]
    ]

    destroyed = state.copy()

    for customer in selected:
        remove_customer(
            state=destroyed,
            instance=instance,
            customer_id=customer,
        )

    destroyed.register_operator_event(
        operator_type="destroy",
        operator_name="paper_related_removal",
        details={
            "seed_customer": seed_customer,
            "removed_customers": list(selected),
            "phi_1": phi_1,
            "phi_2": phi_2,
            "phi_3": phi_3,
            "scores": {
                score.customer: score.total_score
                for score in scores
            },
        },
    )

    return DestroyResult(
        operator_name="paper_related_removal",
        state=destroyed,
        removed_customers=selected,
    )


def _route_distance_value(
    route: list[str],
    instance: dict,
) -> float:
    if len(route) < 2:
        return 0.0

    return sum(
        float(instance["distance"][route[index]][route[index + 1]])
        for index in range(len(route) - 1)
    )


def _find_customer_route(
    state: ALNSSolutionState,
    customer: str,
) -> tuple[str, str, list[str]]:
    assignment = state.assignments[customer]
    mode = assignment.get("mode")

    if mode == "DV_HOME":
        vehicle = assignment["vehicle"]
        return (
            "DV",
            vehicle,
            list(state.dv_routes[vehicle]),
        )

    if mode == "OD_HOME":
        driver = assignment["driver"]
        return (
            "OD",
            driver,
            list(state.od_routes[driver]),
        )

    raise ValueError(
        "Historical and neighborhood operators only "
        "support Type 1/3 home-delivery customers."
    )


def _remove_customer_from_route_copy(
    route: list[str],
    customer: str,
) -> list[str]:
    return [
        node
        for node in route
        if node != customer
    ]


def current_position_cost(
    state: ALNSSolutionState,
    instance: dict,
    customer: str,
) -> float:
    """
    Paper Historical Node Removal position cost:

        f_i = c(prev,i) + c(i,next)

    This deliberately differs from the marginal route contribution used by
    Neighborhood Removal.
    """
    _, _, route = _find_customer_route(
        state,
        customer,
    )

    if customer not in route:
        raise ValueError(
            f"{customer} is not present in its assigned route."
        )

    position = route.index(customer)

    if position == 0 or position == len(route) - 1:
        raise ValueError(
            "Customer cannot be a route endpoint."
        )

    prev_node = route[position - 1]
    next_node = route[position + 1]

    return (
        float(instance["distance"][prev_node][customer])
        + float(instance["distance"][customer][next_node])
    )


def paper_historical_node_removal(
    state: ALNSSolutionState,
    instance: dict,
    removal_count: int,
    *,
    best_historical_position_costs: dict[str, float],
) -> DestroyResult:
    """
    Paper Historical Node Removal.

    Score:
        current position cost - best historical position cost

    Only Type 1 and Type 3 customers are eligible.
    """
    if removal_count <= 0:
        raise ValueError(
            "removal_count must be positive."
        )

    candidates = _eligible_type_13_customers(
        state,
        instance,
    )

    if removal_count > len(candidates):
        raise ValueError(
            "removal_count exceeds eligible customers."
        )

    missing = [
        customer
        for customer in candidates
        if customer not in best_historical_position_costs
    ]

    if missing:
        raise ValueError(
            "Missing historical costs for: "
            f"{missing}"
        )

    scores = []

    for customer in candidates:
        current = current_position_cost(
            state,
            instance,
            customer,
        )

        best_historical = float(
            best_historical_position_costs[customer]
        )

        scores.append(
            HistoricalRemovalScore(
                customer=customer,
                current_position_cost=current,
                best_historical_position_cost=(
                    best_historical
                ),
                score=current - best_historical,
            )
        )

    selected = [
        score.customer
        for score in sorted(
            scores,
            key=lambda score: (
                -score.score,
                score.customer,
            ),
        )[:removal_count]
    ]

    destroyed = state.copy()

    for customer in selected:
        remove_customer(
            state=destroyed,
            instance=instance,
            customer_id=customer,
        )

    destroyed.register_operator_event(
        operator_type="destroy",
        operator_name="paper_historical_node_removal",
        details={
            "removed_customers": list(selected),
            "scores": {
                score.customer: {
                    "current": score.current_position_cost,
                    "best_historical": (
                        score.best_historical_position_cost
                    ),
                    "difference": score.score,
                }
                for score in scores
            },
        },
    )

    return DestroyResult(
        operator_name="paper_historical_node_removal",
        state=destroyed,
        removed_customers=selected,
    )


def neighborhood_contribution(
    state: ALNSSolutionState,
    instance: dict,
    customer: str,
) -> NeighborhoodRemovalScore:
    """
    Paper Neighborhood Removal contribution:

        f_R - f_{R \\ {j}}

    evaluated on the customer's current route.
    """
    route_type, route_id, route = _find_customer_route(
        state,
        customer,
    )

    route_after = _remove_customer_from_route_copy(
        route,
        customer,
    )

    route_cost_before = _route_distance_value(
        route,
        instance,
    )

    route_cost_after = _route_distance_value(
        route_after,
        instance,
    )

    return NeighborhoodRemovalScore(
        customer=customer,
        route_type=route_type,
        route_id=route_id,
        route_cost_before=route_cost_before,
        route_cost_after=route_cost_after,
        contribution=(
            route_cost_before - route_cost_after
        ),
    )


def paper_neighborhood_removal(
    state: ALNSSolutionState,
    instance: dict,
    removal_count: int,
) -> DestroyResult:
    """
    Paper Neighborhood Removal.

    Remove Type 1/3 customers with the largest route-cost contribution.
    """
    if removal_count <= 0:
        raise ValueError(
            "removal_count must be positive."
        )

    candidates = _eligible_type_13_customers(
        state,
        instance,
    )

    if removal_count > len(candidates):
        raise ValueError(
            "removal_count exceeds eligible customers."
        )

    scores = [
        neighborhood_contribution(
            state,
            instance,
            customer,
        )
        for customer in candidates
    ]

    selected = [
        score.customer
        for score in sorted(
            scores,
            key=lambda score: (
                -score.contribution,
                score.customer,
            ),
        )[:removal_count]
    ]

    destroyed = state.copy()

    for customer in selected:
        remove_customer(
            state=destroyed,
            instance=instance,
            customer_id=customer,
        )

    destroyed.register_operator_event(
        operator_type="destroy",
        operator_name="paper_neighborhood_removal",
        details={
            "removed_customers": list(selected),
            "contributions": {
                score.customer: score.contribution
                for score in scores
            },
        },
    )

    return DestroyResult(
        operator_name="paper_neighborhood_removal",
        state=destroyed,
        removed_customers=selected,
    )


def paper_node_neighborhood_removal(
    state: ALNSSolutionState,
    instance: dict,
    removal_count: int,
    *,
    seed: int | None = None,
    seed_customer: str | None = None,
) -> DestroyResult:
    """
    Paper Node-Neighborhood Removal.

    1. Randomly select one Type 1/3 customer as seed.
    2. Remove the seed.
    3. Remove the q-1 geographically nearest eligible Type 1/3 customers.
    """
    if removal_count <= 0:
        raise ValueError(
            "removal_count must be positive."
        )

    candidates = _eligible_type_13_customers(
        state,
        instance,
    )

    if removal_count > len(candidates):
        raise ValueError(
            "removal_count exceeds eligible customers."
        )

    rng = random.Random(seed)

    if seed_customer is None:
        seed_customer = rng.choice(candidates)

    if seed_customer not in candidates:
        raise ValueError(
            "seed_customer must be an active Type 1/3 customer."
        )

    ordered = sorted(
        candidates,
        key=lambda customer: (
            float(
                instance["distance"][seed_customer][customer]
            ),
            customer,
        ),
    )

    selected = ordered[:removal_count]

    destroyed = state.copy()

    for customer in selected:
        remove_customer(
            state=destroyed,
            instance=instance,
            customer_id=customer,
        )

    destroyed.register_operator_event(
        operator_type="destroy",
        operator_name="paper_node_neighborhood_removal",
        details={
            "seed_customer": seed_customer,
            "removed_customers": list(selected),
        },
    )

    return DestroyResult(
        operator_name="paper_node_neighborhood_removal",
        state=destroyed,
        removed_customers=selected,
    )

# =============================================================
# Destroy Fidelity Gate 3
# Deterministic / probabilistic ranked operators and registry
# =============================================================

@dataclass
class WorstCustomerScore:
    customer: str
    objective_before: float
    objective_after: float
    saving: float


def _rank_biased_index(
    candidate_count: int,
    rng: random.Random,
    *,
    randomness_factor: float = 5.0,
) -> int:
    """
    Paper rank-based probabilistic selection:

        floor(U(0,1)^p * |L|)

    where p is the randomness factor. With p > 1, lower ranks are favored.
    """
    if candidate_count <= 0:
        raise ValueError(
            "candidate_count must be positive."
        )

    if randomness_factor <= 0:
        raise ValueError(
            "randomness_factor must be positive."
        )

    raw_index = int(
        (rng.random() ** randomness_factor)
        * candidate_count
    )

    return min(
        raw_index,
        candidate_count - 1,
    )


def worst_customer_scores(
    state: ALNSSolutionState,
    instance: dict,
    *,
    lambda_value: float,
    cost_bounds: tuple[float, float] | None,
    emission_bounds: tuple[float, float] | None,
    emission_factors: tuple[float, float],
    candidate_customers: Iterable[str] | None = None,
) -> list[WorstCustomerScore]:
    """
    Compute objective saving for removing each currently assigned customer.
    """
    active_customers = set(state.assignments)

    candidates = (
        sorted(active_customers)
        if candidate_customers is None
        else sorted(
            active_customers
            & set(candidate_customers)
        )
    )

    if not candidates:
        raise ValueError(
            "No active customer is available for scoring."
        )

    objective_before = _objective_value(
        state,
        instance,
        lambda_value=lambda_value,
        cost_bounds=cost_bounds,
        emission_bounds=emission_bounds,
        emission_factors=emission_factors,
    )

    scores = []

    for customer in candidates:
        candidate_state = state.copy()

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

        scores.append(
            WorstCustomerScore(
                customer=customer,
                objective_before=objective_before,
                objective_after=objective_after,
                saving=(
                    objective_before - objective_after
                ),
            )
        )

    return sorted(
        scores,
        key=lambda score: (
            -score.saving,
            score.customer,
        ),
    )


def paper_worst_customer_removal(
    state: ALNSSolutionState,
    instance: dict,
    removal_count: int,
    *,
    lambda_value: float,
    cost_bounds: tuple[float, float] | None,
    emission_bounds: tuple[float, float] | None,
    emission_factors: tuple[float, float] = (1.0, 1.0),
    candidate_customers: Iterable[str] | None = None,
) -> DestroyResult:
    """
    Deterministic paper Worst Customer Removal.

    Recompute the ranked list after every removal, then remove the current
    highest-saving customer.
    """
    if removal_count <= 0:
        raise ValueError(
            "removal_count must be positive."
        )

    destroyed = state.copy()
    removed_customers = []
    score_history = []

    for _ in range(removal_count):
        scores = worst_customer_scores(
            destroyed,
            instance,
            lambda_value=lambda_value,
            cost_bounds=cost_bounds,
            emission_bounds=emission_bounds,
            emission_factors=emission_factors,
            candidate_customers=candidate_customers,
        )

        selected = scores[0]

        remove_customer(
            state=destroyed,
            instance=instance,
            customer_id=selected.customer,
        )

        removed_customers.append(
            selected.customer
        )

        score_history.append(
            {
                "selected_customer": selected.customer,
                "saving": selected.saving,
                "ranked_scores": {
                    score.customer: score.saving
                    for score in scores
                },
            }
        )

    destroyed.register_operator_event(
        operator_type="destroy",
        operator_name="paper_worst_customer_removal",
        details={
            "removed_customers": list(
                removed_customers
            ),
            "selection_mode": "deterministic",
            "score_history": score_history,
        },
    )

    return DestroyResult(
        operator_name="paper_worst_customer_removal",
        state=destroyed,
        removed_customers=removed_customers,
    )


def paper_probabilistic_worst_customer_removal(
    state: ALNSSolutionState,
    instance: dict,
    removal_count: int,
    *,
    seed: int,
    randomness_factor: float = 5.0,
    lambda_value: float,
    cost_bounds: tuple[float, float] | None,
    emission_bounds: tuple[float, float] | None,
    emission_factors: tuple[float, float] = (1.0, 1.0),
    candidate_customers: Iterable[str] | None = None,
) -> DestroyResult:
    """
    Probabilistic paper Worst Customer Removal.

    Customers remain ranked from worst to best, but the selected rank follows:

        floor(U^p * |L|)

    with paper randomness factor p = 5.
    """
    if removal_count <= 0:
        raise ValueError(
            "removal_count must be positive."
        )

    rng = random.Random(seed)
    destroyed = state.copy()
    removed_customers = []
    selection_history = []

    for _ in range(removal_count):
        scores = worst_customer_scores(
            destroyed,
            instance,
            lambda_value=lambda_value,
            cost_bounds=cost_bounds,
            emission_bounds=emission_bounds,
            emission_factors=emission_factors,
            candidate_customers=candidate_customers,
        )

        selected_rank = _rank_biased_index(
            len(scores),
            rng,
            randomness_factor=randomness_factor,
        )

        selected = scores[selected_rank]

        remove_customer(
            state=destroyed,
            instance=instance,
            customer_id=selected.customer,
        )

        removed_customers.append(
            selected.customer
        )

        selection_history.append(
            {
                "selected_rank": selected_rank,
                "selected_customer": selected.customer,
                "saving": selected.saving,
                "ranked_customers": [
                    score.customer
                    for score in scores
                ],
            }
        )

    destroyed.register_operator_event(
        operator_type="destroy",
        operator_name=(
            "paper_probabilistic_worst_customer_removal"
        ),
        details={
            "removed_customers": list(
                removed_customers
            ),
            "selection_mode": "probabilistic",
            "randomness_factor": randomness_factor,
            "seed": seed,
            "selection_history": selection_history,
        },
    )

    return DestroyResult(
        operator_name=(
            "paper_probabilistic_worst_customer_removal"
        ),
        state=destroyed,
        removed_customers=removed_customers,
    )


def paper_probabilistic_related_removal(
    state: ALNSSolutionState,
    instance: dict,
    removal_count: int,
    *,
    seed: int,
    seed_customer: str | None = None,
    randomness_factor: float = 5.0,
    candidate_customers: Iterable[str] | None = None,
    phi_1: float = 5.0,
    phi_2: float = 9.0,
    phi_3: float = 1.0,
) -> DestroyResult:
    """
    Probabilistic paper Related Removal.

    Relatedness follows Eq. (47), then rank selection follows floor(U^p |L|).
    The seed is removed first. Remaining removals are chosen probabilistically
    from the relatedness-ranked list.
    """
    if removal_count <= 0:
        raise ValueError(
            "removal_count must be positive."
        )

    active = set(state.assignments)

    candidates = (
        sorted(active)
        if candidate_customers is None
        else sorted(
            active & set(candidate_customers)
        )
    )

    if removal_count > len(candidates):
        raise ValueError(
            "removal_count exceeds candidate count."
        )

    rng = random.Random(seed)

    if seed_customer is None:
        seed_customer = rng.choice(candidates)

    if seed_customer not in candidates:
        raise ValueError(
            "seed_customer is not an active candidate."
        )

    destroyed = state.copy()
    removed_customers = []

    remove_customer(
        state=destroyed,
        instance=instance,
        customer_id=seed_customer,
    )

    removed_customers.append(
        seed_customer
    )

    selection_history = [
        {
            "selected_rank": 0,
            "selected_customer": seed_customer,
            "reason": "seed",
        }
    ]

    while len(removed_customers) < removal_count:
        remaining_candidates = [
            customer
            for customer in candidates
            if customer in destroyed.assignments
        ]

        if not remaining_candidates:
            raise RuntimeError(
                "No related-removal candidates remain."
            )

        scores = [
            relatedness_eq47(
                state,
                instance,
                seed_customer,
                customer,
                phi_1=phi_1,
                phi_2=phi_2,
                phi_3=phi_3,
            )
            for customer in remaining_candidates
        ]

        ranked_scores = sorted(
            scores,
            key=lambda score: (
                score.total_score,
                score.customer,
            ),
        )

        selected_rank = _rank_biased_index(
            len(ranked_scores),
            rng,
            randomness_factor=randomness_factor,
        )

        selected = ranked_scores[selected_rank]

        remove_customer(
            state=destroyed,
            instance=instance,
            customer_id=selected.customer,
        )

        removed_customers.append(
            selected.customer
        )

        selection_history.append(
            {
                "selected_rank": selected_rank,
                "selected_customer": selected.customer,
                "relatedness": selected.total_score,
                "ranked_customers": [
                    score.customer
                    for score in ranked_scores
                ],
            }
        )

    destroyed.register_operator_event(
        operator_type="destroy",
        operator_name=(
            "paper_probabilistic_related_removal"
        ),
        details={
            "seed_customer": seed_customer,
            "removed_customers": list(
                removed_customers
            ),
            "randomness_factor": randomness_factor,
            "seed": seed,
            "phi_1": phi_1,
            "phi_2": phi_2,
            "phi_3": phi_3,
            "selection_history": selection_history,
        },
    )

    return DestroyResult(
        operator_name=(
            "paper_probabilistic_related_removal"
        ),
        state=destroyed,
        removed_customers=removed_customers,
    )


PAPER_DESTROY_OPERATOR_REGISTRY = {
    "random_customer_removal": (
        "generic_random_customer_removal"
    ),
    "worst_customer_removal_deterministic": (
        "paper_worst_customer_removal"
    ),
    "worst_customer_removal_probabilistic": (
        "paper_probabilistic_worst_customer_removal"
    ),
    "route_removal": (
        "paper_route_removal"
    ),
    "random_adp_removal": (
        "generic_random_adp_removal"
    ),
    "worst_adp_removal": (
        "paper_worst_adp_removal"
    ),
    "random_tn_removal": (
        "generic_random_tn_removal"
    ),
    "related_removal_deterministic": (
        "paper_related_removal"
    ),
    "related_removal_probabilistic": (
        "paper_probabilistic_related_removal"
    ),
    "historical_node_removal": (
        "paper_historical_node_removal"
    ),
    "neighborhood_removal": (
        "paper_neighborhood_removal"
    ),
    "node_neighborhood_removal": (
        "paper_node_neighborhood_removal"
    ),
}


def paper_destroy_operator_names() -> list[str]:
    """
    Return the selectable destroy operator concepts expected by Table 10.

    Worst-customer and related removal each have deterministic and
    probabilistic variants, resulting in 12 selectable registry entries.
    """
    return list(
        PAPER_DESTROY_OPERATOR_REGISTRY.keys()
    )
