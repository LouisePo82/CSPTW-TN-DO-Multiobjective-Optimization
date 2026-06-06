from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable

from alns_solver.solution_state import ALNSSolutionState


@dataclass
class DestroyResult:
    operator_name: str
    state: ALNSSolutionState
    removed_customers: list[str]
    removed_route: dict | None = None


def _active_customers(state: ALNSSolutionState) -> list[str]:
    return sorted(state.assignments.keys())


def _remove_node_once(route: list[str], node: str) -> list[str]:
    removed = False
    cleaned = []

    for current in route:
        if current == node and not removed:
            removed = True
            continue
        cleaned.append(current)

    return cleaned


def _od_customer_count(
    state: ALNSSolutionState,
    driver: str,
) -> int:
    return sum(
        1
        for assignment in state.assignments.values()
        if assignment.get("mode") == "OD_HOME"
        and assignment.get("driver") == driver
    )


def _remaining_adp_assignments(
    state: ALNSSolutionState,
    vehicle: str,
    adp: str,
) -> int:
    return sum(
        1
        for assignment in state.assignments.values()
        if assignment.get("mode") == "ADP"
        and assignment.get("vehicle") == vehicle
        and assignment.get("adp") == adp
    )


def _remaining_tn_assignments(
    state: ALNSSolutionState,
    tn: str,
) -> int:
    return sum(
        1
        for assignment in state.assignments.values()
        if assignment.get("mode") == "OD_HOME"
        and assignment.get("pickup") == tn
    )


def _cleanup_unused_adp(
    state: ALNSSolutionState,
    vehicle: str,
    adp: str,
) -> None:
    if _remaining_adp_assignments(state, vehicle, adp) > 0:
        return

    state.dv_routes[vehicle] = _remove_node_once(
        state.dv_routes.get(vehicle, []),
        adp,
    )


def _cleanup_unused_tn(
    state: ALNSSolutionState,
    instance: dict,
    tn: str,
) -> None:
    if _remaining_tn_assignments(state, tn) > 0:
        return

    for vehicle in instance["dvs"]:
        state.dv_routes[vehicle] = _remove_node_once(
            state.dv_routes.get(vehicle, []),
            tn,
        )


def remove_customer(
    state: ALNSSolutionState,
    instance: dict,
    customer_id: str,
) -> None:
    """
    Remove one served customer and clean route/facility artifacts.

    Rules:
    - DV_HOME: remove customer node from its DV route.
    - OD_HOME: remove customer node from its OD route; deactivate the OD if
      no customers remain; remove an orphan TN from DV routes.
    - ADP: remove assignment; remove ADP node only when no remaining parcel
      is assigned to that vehicle-ADP pair.
    """
    if customer_id not in state.assignments:
        raise ValueError(
            f"Customer {customer_id} is not assigned and cannot be removed."
        )

    assignment = dict(state.assignments[customer_id])
    mode = assignment.get("mode")

    if mode == "DV_HOME":
        vehicle = assignment["vehicle"]
        state.dv_routes[vehicle] = _remove_node_once(
            state.dv_routes.get(vehicle, []),
            customer_id,
        )

    elif mode == "OD_HOME":
        driver = assignment["driver"]
        pickup = assignment["pickup"]

        state.od_routes[driver] = _remove_node_once(
            state.od_routes.get(driver, []),
            customer_id,
        )

        state.mark_customer_unassigned(customer_id)

        if _od_customer_count(state, driver) == 0:
            state.od_routes[driver] = []

        if pickup in instance["tns"]:
            _cleanup_unused_tn(
                state=state,
                instance=instance,
                tn=pickup,
            )

        state.invalidate_cache()
        return

    elif mode == "ADP":
        vehicle = assignment["vehicle"]
        adp = assignment["adp"]

        state.mark_customer_unassigned(customer_id)
        _cleanup_unused_adp(
            state=state,
            vehicle=vehicle,
            adp=adp,
        )
        state.invalidate_cache()
        return

    else:
        raise ValueError(
            f"Unsupported assignment mode for {customer_id}: {mode}"
        )

    state.mark_customer_unassigned(customer_id)
    state.invalidate_cache()


def random_customer_removal(
    state: ALNSSolutionState,
    instance: dict,
    removal_count: int,
    *,
    seed: int | None = None,
    candidate_customers: Iterable[str] | None = None,
) -> DestroyResult:
    """
    Paper destroy operator: Random Customer Removal.

    Select `removal_count` currently assigned customers uniformly without
    replacement, then remove them using the shared cleanup rules.
    """
    if removal_count <= 0:
        raise ValueError("removal_count must be positive.")

    candidates = (
        sorted(set(candidate_customers) & set(state.assignments))
        if candidate_customers is not None
        else _active_customers(state)
    )

    if removal_count > len(candidates):
        raise ValueError(
            f"Cannot remove {removal_count} customers from "
            f"{len(candidates)} candidates."
        )

    rng = random.Random(seed)
    selected = rng.sample(candidates, removal_count)

    destroyed = state.copy()

    for customer_id in selected:
        remove_customer(
            state=destroyed,
            instance=instance,
            customer_id=customer_id,
        )

    destroyed.register_operator_event(
        operator_type="destroy",
        operator_name="random_customer_removal",
        details={
            "seed": seed,
            "removal_count": removal_count,
            "removed_customers": list(selected),
        },
    )

    return DestroyResult(
        operator_name="random_customer_removal",
        state=destroyed,
        removed_customers=list(selected),
    )


def _state_objective(
    state: ALNSSolutionState,
    instance: dict,
    *,
    lambda_value: float,
    cost_bounds: tuple[float, float] | None,
    emission_bounds: tuple[float, float] | None,
    emission_factors: tuple[float, float],
) -> float:
    return state.evaluate(
        instance=instance,
        lambda_value=lambda_value,
        cost_bounds=cost_bounds,
        emission_bounds=emission_bounds,
        emission_factors=emission_factors,
    )["objective"]


def worst_customer_removal(
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
    Paper destroy operator: Worst Customer Removal.

    At each removal step, remove the customer producing the greatest decrease
    in the current scalarized objective. Scores are recomputed after every
    removal because route and facility cleanup changes marginal savings.
    """
    if removal_count <= 0:
        raise ValueError("removal_count must be positive.")

    destroyed = state.copy()
    removed: list[str] = []

    allowed = (
        set(candidate_customers)
        if candidate_customers is not None
        else set(state.assignments)
    )

    for _ in range(removal_count):
        candidates = sorted(
            set(destroyed.assignments) & allowed
        )

        if not candidates:
            raise ValueError(
                "No assigned candidate customers remain for worst removal."
            )

        current_objective = _state_objective(
            destroyed,
            instance,
            lambda_value=lambda_value,
            cost_bounds=cost_bounds,
            emission_bounds=emission_bounds,
            emission_factors=emission_factors,
        )

        scored: list[tuple[float, str, ALNSSolutionState]] = []

        for customer_id in candidates:
            candidate_state = destroyed.copy()
            remove_customer(
                state=candidate_state,
                instance=instance,
                customer_id=customer_id,
            )

            candidate_objective = _state_objective(
                candidate_state,
                instance,
                lambda_value=lambda_value,
                cost_bounds=cost_bounds,
                emission_bounds=emission_bounds,
                emission_factors=emission_factors,
            )

            saving = current_objective - candidate_objective
            scored.append(
                (
                    saving,
                    customer_id,
                    candidate_state,
                )
            )

        # Largest saving is "worst"; customer id gives deterministic tie-break.
        saving, customer_id, destroyed = max(
            scored,
            key=lambda item: (item[0], item[1]),
        )
        removed.append(customer_id)

    destroyed.register_operator_event(
        operator_type="destroy",
        operator_name="worst_customer_removal",
        details={
            "removal_count": removal_count,
            "removed_customers": list(removed),
            "lambda_value": lambda_value,
        },
    )

    return DestroyResult(
        operator_name="worst_customer_removal",
        state=destroyed,
        removed_customers=removed,
    )


def _active_route_candidates(
    state: ALNSSolutionState,
    *,
    route_type: str | None,
) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []

    if route_type in {None, "DV"}:
        candidates.extend(
            ("DV", vehicle)
            for vehicle, route in state.dv_routes.items()
            if route
        )

    if route_type in {None, "OD"}:
        candidates.extend(
            ("OD", driver)
            for driver, route in state.od_routes.items()
            if route
        )

    return sorted(candidates)


def route_removal(
    state: ALNSSolutionState,
    instance: dict,
    *,
    seed: int | None = None,
    route_type: str | None = None,
    route_id: str | None = None,
) -> DestroyResult:
    """
    Paper destroy operator: Route Removal.

    OD route:
      remove every OD_HOME customer assigned to the selected driver.

    DV route:
      remove DV_HOME and ADP customers assigned to that DV. If the DV is the
      unique carrier to a TN, all OD customers using that TN are also removed
      and their affected OD routes are deactivated.
    """
    if route_type not in {None, "DV", "OD"}:
        raise ValueError("route_type must be None, 'DV', or 'OD'.")

    candidates = _active_route_candidates(
        state,
        route_type=route_type,
    )

    if route_id is not None:
        candidates = [
            item
            for item in candidates
            if item[1] == route_id
        ]

    if not candidates:
        raise ValueError("No active route matches the removal request.")

    rng = random.Random(seed)
    selected_type, selected_id = rng.choice(candidates)

    destroyed = state.copy()
    removed_customers: list[str] = []

    if selected_type == "OD":
        removed_customers = sorted(
            customer
            for customer, assignment
            in destroyed.assignments.items()
            if assignment.get("mode") == "OD_HOME"
            and assignment.get("driver") == selected_id
        )

        for customer in list(removed_customers):
            remove_customer(
                state=destroyed,
                instance=instance,
                customer_id=customer,
            )

        destroyed.od_routes[selected_id] = []

    else:
        selected_route = list(
            destroyed.dv_routes.get(selected_id, [])
        )

        direct_customers = [
            customer
            for customer, assignment
            in destroyed.assignments.items()
            if assignment.get("vehicle") == selected_id
            and assignment.get("mode") in {"DV_HOME", "ADP"}
        ]

        affected_tns = [
            tn
            for tn in instance["tns"]
            if tn in selected_route
        ]

        tn_customers = [
            customer
            for customer, assignment
            in destroyed.assignments.items()
            if assignment.get("mode") == "OD_HOME"
            and assignment.get("pickup") in affected_tns
        ]

        removed_customers = sorted(
            set(direct_customers + tn_customers)
        )

        for customer in list(removed_customers):
            if customer in destroyed.assignments:
                remove_customer(
                    state=destroyed,
                    instance=instance,
                    customer_id=customer,
                )

        destroyed.dv_routes[selected_id] = []

    destroyed.register_operator_event(
        operator_type="destroy",
        operator_name="route_removal",
        details={
            "route_type": selected_type,
            "route_id": selected_id,
            "removed_customers": list(removed_customers),
            "seed": seed,
        },
    )

    return DestroyResult(
        operator_name="route_removal",
        state=destroyed,
        removed_customers=removed_customers,
        removed_route={
            "route_type": selected_type,
            "route_id": selected_id,
        },
    )

# =============================================================
# Gate 4B — Facility-level destroy operators
# =============================================================

def _active_adp_pairs(
    state: ALNSSolutionState,
) -> list[tuple[str, str]]:
    """
    Return active (DV, ADP) pairs that currently carry at least one parcel.
    """
    pairs = {
        (
            assignment["vehicle"],
            assignment["adp"],
        )
        for assignment in state.assignments.values()
        if assignment.get("mode") == "ADP"
    }
    return sorted(pairs)


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


def random_adp_removal(
    state: ALNSSolutionState,
    instance: dict,
    *,
    seed: int | None = None,
    vehicle: str | None = None,
    adp: str | None = None,
) -> DestroyResult:
    """
    Paper destroy operator: Random ADP Removal.

    Select one active DV-ADP pair and remove every customer parcel assigned
    through that pair. The ADP node is removed from the selected DV route
    once its final parcel is removed.
    """
    candidates = _active_adp_pairs(state)

    if vehicle is not None:
        candidates = [
            pair for pair in candidates
            if pair[0] == vehicle
        ]

    if adp is not None:
        candidates = [
            pair for pair in candidates
            if pair[1] == adp
        ]

    if not candidates:
        raise ValueError(
            "No active DV-ADP pair matches the removal request."
        )

    rng = random.Random(seed)
    selected_vehicle, selected_adp = rng.choice(candidates)

    destroyed = state.copy()

    removed_customers = _customers_at_adp_pair(
        destroyed,
        selected_vehicle,
        selected_adp,
    )

    for customer_id in list(removed_customers):
        remove_customer(
            state=destroyed,
            instance=instance,
            customer_id=customer_id,
        )

    destroyed.register_operator_event(
        operator_type="destroy",
        operator_name="random_adp_removal",
        details={
            "seed": seed,
            "vehicle": selected_vehicle,
            "adp": selected_adp,
            "removed_customers": list(removed_customers),
        },
    )

    return DestroyResult(
        operator_name="random_adp_removal",
        state=destroyed,
        removed_customers=removed_customers,
        removed_route={
            "facility_type": "ADP",
            "vehicle": selected_vehicle,
            "facility_id": selected_adp,
        },
    )


def worst_adp_removal(
    state: ALNSSolutionState,
    instance: dict,
    *,
    lambda_value: float,
    cost_bounds: tuple[float, float] | None,
    emission_bounds: tuple[float, float] | None,
    emission_factors: tuple[float, float] = (1.0, 1.0),
) -> DestroyResult:
    """
    Paper destroy operator: Worst ADP Removal.

    Evaluate each active DV-ADP pair by removing all parcels assigned to it.
    Select the pair producing the greatest scalarized-objective saving.
    """
    candidates = _active_adp_pairs(state)

    if not candidates:
        raise ValueError(
            "No active ADP pair is available for worst removal."
        )

    current_objective = _state_objective(
        state,
        instance,
        lambda_value=lambda_value,
        cost_bounds=cost_bounds,
        emission_bounds=emission_bounds,
        emission_factors=emission_factors,
    )

    scored = []

    for vehicle, adp in candidates:
        candidate_state = state.copy()

        removed_customers = _customers_at_adp_pair(
            candidate_state,
            vehicle,
            adp,
        )

        for customer_id in list(removed_customers):
            remove_customer(
                state=candidate_state,
                instance=instance,
                customer_id=customer_id,
            )

        candidate_objective = _state_objective(
            candidate_state,
            instance,
            lambda_value=lambda_value,
            cost_bounds=cost_bounds,
            emission_bounds=emission_bounds,
            emission_factors=emission_factors,
        )

        saving = current_objective - candidate_objective

        scored.append(
            (
                saving,
                vehicle,
                adp,
                removed_customers,
                candidate_state,
            )
        )

    (
        saving,
        selected_vehicle,
        selected_adp,
        removed_customers,
        destroyed,
    ) = max(
        scored,
        key=lambda item: (
            item[0],
            item[1],
            item[2],
        ),
    )

    destroyed.register_operator_event(
        operator_type="destroy",
        operator_name="worst_adp_removal",
        details={
            "vehicle": selected_vehicle,
            "adp": selected_adp,
            "objective_saving": saving,
            "removed_customers": list(removed_customers),
            "lambda_value": lambda_value,
        },
    )

    return DestroyResult(
        operator_name="worst_adp_removal",
        state=destroyed,
        removed_customers=list(removed_customers),
        removed_route={
            "facility_type": "ADP",
            "vehicle": selected_vehicle,
            "facility_id": selected_adp,
            "objective_saving": saving,
        },
    )


def _active_tns(
    state: ALNSSolutionState,
    instance: dict,
) -> list[str]:
    return sorted(
        {
            assignment["pickup"]
            for assignment in state.assignments.values()
            if assignment.get("mode") == "OD_HOME"
            and assignment.get("pickup") in instance["tns"]
        }
    )


def random_tn_removal(
    state: ALNSSolutionState,
    instance: dict,
    *,
    seed: int | None = None,
    tn: str | None = None,
) -> DestroyResult:
    """
    Paper destroy operator: Random TN Removal.

    Select one active TN and remove every OD-served customer whose parcel is
    picked up at that TN. Affected OD routes are deactivated when they no
    longer serve customers, and the orphan TN is removed from all DV routes.
    """
    candidates = _active_tns(
        state=state,
        instance=instance,
    )

    if tn is not None:
        candidates = [
            candidate_tn
            for candidate_tn in candidates
            if candidate_tn == tn
        ]

    if not candidates:
        raise ValueError(
            "No active TN matches the removal request."
        )

    rng = random.Random(seed)
    selected_tn = rng.choice(candidates)

    destroyed = state.copy()

    removed_customers = sorted(
        customer
        for customer, assignment in destroyed.assignments.items()
        if assignment.get("mode") == "OD_HOME"
        and assignment.get("pickup") == selected_tn
    )

    affected_drivers = sorted(
        {
            destroyed.assignments[customer]["driver"]
            for customer in removed_customers
        }
    )

    for customer_id in list(removed_customers):
        remove_customer(
            state=destroyed,
            instance=instance,
            customer_id=customer_id,
        )

    # Defensive cleanup in case multiple affected routes shared the TN.
    for vehicle in instance["dvs"]:
        destroyed.dv_routes[vehicle] = _remove_node_once(
            destroyed.dv_routes.get(vehicle, []),
            selected_tn,
        )

    destroyed.register_operator_event(
        operator_type="destroy",
        operator_name="random_tn_removal",
        details={
            "seed": seed,
            "tn": selected_tn,
            "affected_drivers": affected_drivers,
            "removed_customers": list(removed_customers),
        },
    )

    return DestroyResult(
        operator_name="random_tn_removal",
        state=destroyed,
        removed_customers=removed_customers,
        removed_route={
            "facility_type": "TN",
            "facility_id": selected_tn,
            "affected_drivers": affected_drivers,
        },
    )

# =============================================================
# Gate 4C — Related/history/neighborhood destroy operators
# =============================================================

def _assignment_mode(
    state: ALNSSolutionState,
    customer_id: str,
) -> str:
    return str(
        state.assignments[customer_id].get("mode", "")
    )


def _customer_distance(
    instance: dict,
    customer_a: str,
    customer_b: str,
) -> float:
    return float(
        instance["distance"][customer_a][customer_b]
    )


def related_removal(
    state: ALNSSolutionState,
    instance: dict,
    removal_count: int,
    *,
    seed: int | None = None,
    seed_customer: str | None = None,
    candidate_customers: Iterable[str] | None = None,
    distance_weight: float = 1.0,
    type_weight: float = 1.0,
    mode_weight: float = 1.0,
) -> DestroyResult:
    """
    Related Removal.

    Relatedness combines:
    - spatial distance;
    - customer-type mismatch;
    - delivery-mode mismatch.

    Lower score means more related to the seed customer.
    """
    if removal_count <= 0:
        raise ValueError("removal_count must be positive.")

    candidates = (
        sorted(set(candidate_customers) & set(state.assignments))
        if candidate_customers is not None
        else _active_customers(state)
    )

    if removal_count > len(candidates):
        raise ValueError(
            f"Cannot remove {removal_count} customers from "
            f"{len(candidates)} candidates."
        )

    rng = random.Random(seed)

    if seed_customer is None:
        seed_customer = rng.choice(candidates)

    if seed_customer not in candidates:
        raise ValueError(
            f"Seed customer {seed_customer} is not an active candidate."
        )

    seed_type = instance["nodes"][seed_customer]["customer_type"]
    seed_mode = _assignment_mode(state, seed_customer)

    scored = []

    for customer_id in candidates:
        customer_type = instance["nodes"][customer_id]["customer_type"]
        customer_mode = _assignment_mode(state, customer_id)

        score = (
            distance_weight
            * _customer_distance(instance, seed_customer, customer_id)
            + type_weight
            * (0.0 if customer_type == seed_type else 1.0)
            + mode_weight
            * (0.0 if customer_mode == seed_mode else 1.0)
        )

        scored.append((score, customer_id))

    selected = [
        customer_id
        for _, customer_id in sorted(
            scored,
            key=lambda item: (item[0], item[1]),
        )[:removal_count]
    ]

    destroyed = state.copy()

    for customer_id in selected:
        remove_customer(
            state=destroyed,
            instance=instance,
            customer_id=customer_id,
        )

    destroyed.register_operator_event(
        operator_type="destroy",
        operator_name="related_removal",
        details={
            "seed": seed,
            "seed_customer": seed_customer,
            "removal_count": removal_count,
            "removed_customers": list(selected),
            "distance_weight": distance_weight,
            "type_weight": type_weight,
            "mode_weight": mode_weight,
        },
    )

    return DestroyResult(
        operator_name="related_removal",
        state=destroyed,
        removed_customers=selected,
    )


def historical_node_removal(
    state: ALNSSolutionState,
    instance: dict,
    removal_count: int,
    *,
    historical_scores: dict[str, float],
    candidate_customers: Iterable[str] | None = None,
) -> DestroyResult:
    """
    Historical Node Removal.

    Customers with the highest supplied historical score are removed first.
    The ALNS main loop can later update these scores from past destroy/repair
    performance or historical marginal costs.
    """
    if removal_count <= 0:
        raise ValueError("removal_count must be positive.")

    candidates = (
        sorted(set(candidate_customers) & set(state.assignments))
        if candidate_customers is not None
        else _active_customers(state)
    )

    if removal_count > len(candidates):
        raise ValueError(
            f"Cannot remove {removal_count} customers from "
            f"{len(candidates)} candidates."
        )

    missing_scores = [
        customer_id
        for customer_id in candidates
        if customer_id not in historical_scores
    ]

    if missing_scores:
        raise ValueError(
            "Missing historical scores for customers: "
            f"{missing_scores}"
        )

    selected = [
        customer_id
        for customer_id in sorted(
            candidates,
            key=lambda customer_id: (
                -float(historical_scores[customer_id]),
                customer_id,
            ),
        )[:removal_count]
    ]

    destroyed = state.copy()

    for customer_id in selected:
        remove_customer(
            state=destroyed,
            instance=instance,
            customer_id=customer_id,
        )

    destroyed.register_operator_event(
        operator_type="destroy",
        operator_name="historical_node_removal",
        details={
            "removal_count": removal_count,
            "removed_customers": list(selected),
            "selected_scores": {
                customer_id: float(historical_scores[customer_id])
                for customer_id in selected
            },
        },
    )

    return DestroyResult(
        operator_name="historical_node_removal",
        state=destroyed,
        removed_customers=selected,
    )


def neighborhood_removal(
    state: ALNSSolutionState,
    instance: dict,
    removal_count: int,
    *,
    seed: int | None = None,
    seed_customer: str | None = None,
    candidate_customers: Iterable[str] | None = None,
) -> DestroyResult:
    """
    Neighborhood Removal.

    Select one seed customer and remove the geographically nearest active
    customers, including the seed itself.
    """
    if removal_count <= 0:
        raise ValueError("removal_count must be positive.")

    candidates = (
        sorted(set(candidate_customers) & set(state.assignments))
        if candidate_customers is not None
        else _active_customers(state)
    )

    if removal_count > len(candidates):
        raise ValueError(
            f"Cannot remove {removal_count} customers from "
            f"{len(candidates)} candidates."
        )

    rng = random.Random(seed)

    if seed_customer is None:
        seed_customer = rng.choice(candidates)

    if seed_customer not in candidates:
        raise ValueError(
            f"Seed customer {seed_customer} is not an active candidate."
        )

    selected = [
        customer_id
        for customer_id in sorted(
            candidates,
            key=lambda customer_id: (
                _customer_distance(
                    instance,
                    seed_customer,
                    customer_id,
                ),
                customer_id,
            ),
        )[:removal_count]
    ]

    destroyed = state.copy()

    for customer_id in selected:
        remove_customer(
            state=destroyed,
            instance=instance,
            customer_id=customer_id,
        )

    destroyed.register_operator_event(
        operator_type="destroy",
        operator_name="neighborhood_removal",
        details={
            "seed": seed,
            "seed_customer": seed_customer,
            "removal_count": removal_count,
            "removed_customers": list(selected),
        },
    )

    return DestroyResult(
        operator_name="neighborhood_removal",
        state=destroyed,
        removed_customers=selected,
    )


def _customer_service_anchor(
    state: ALNSSolutionState,
    customer_id: str,
) -> str:
    """
    Return the logistics node representing the customer's current service.

    DV_HOME -> customer node
    ADP     -> assigned ADP
    OD_HOME -> assigned pickup point
    """
    assignment = state.assignments[customer_id]
    mode = assignment.get("mode")

    if mode == "DV_HOME":
        return customer_id

    if mode == "ADP":
        return assignment["adp"]

    if mode == "OD_HOME":
        return assignment["pickup"]

    raise ValueError(
        f"Unsupported assignment mode for {customer_id}: {mode}"
    )


def node_neighborhood_removal(
    state: ALNSSolutionState,
    instance: dict,
    *,
    center_node: str,
    radius: float | None = None,
    removal_count: int | None = None,
    candidate_customers: Iterable[str] | None = None,
) -> DestroyResult:
    """
    Node-Neighborhood Removal.

    Rank customers by the distance between their current logistics service
    anchor and the selected center node.

    The operator supports either:
    - a radius; or
    - a fixed removal count; or
    - both, in which case the radius filter is applied first.
    """
    if center_node not in instance["nodes"]:
        raise ValueError(f"Unknown center node: {center_node}")

    if radius is None and removal_count is None:
        raise ValueError(
            "Provide radius, removal_count, or both."
        )

    if radius is not None and radius < 0:
        raise ValueError("radius must be nonnegative.")

    if removal_count is not None and removal_count <= 0:
        raise ValueError("removal_count must be positive.")

    candidates = (
        sorted(set(candidate_customers) & set(state.assignments))
        if candidate_customers is not None
        else _active_customers(state)
    )

    scored = []

    for customer_id in candidates:
        anchor = _customer_service_anchor(
            state,
            customer_id,
        )
        distance = float(
            instance["distance"][center_node][anchor]
        )

        if radius is not None and distance > radius + 1e-9:
            continue

        scored.append(
            (
                distance,
                customer_id,
                anchor,
            )
        )

    scored.sort(
        key=lambda item: (
            item[0],
            item[1],
        )
    )

    if removal_count is not None:
        scored = scored[:removal_count]

    if not scored:
        raise ValueError(
            "No customers fall inside the requested node neighborhood."
        )

    selected = [
        customer_id
        for _, customer_id, _ in scored
    ]

    destroyed = state.copy()

    for customer_id in selected:
        remove_customer(
            state=destroyed,
            instance=instance,
            customer_id=customer_id,
        )

    destroyed.register_operator_event(
        operator_type="destroy",
        operator_name="node_neighborhood_removal",
        details={
            "center_node": center_node,
            "radius": radius,
            "removal_count": removal_count,
            "removed_customers": list(selected),
            "service_anchors": {
                customer_id: anchor
                for _, customer_id, anchor in scored
            },
        },
    )

    return DestroyResult(
        operator_name="node_neighborhood_removal",
        state=destroyed,
        removed_customers=selected,
        removed_route={
            "center_node": center_node,
            "radius": radius,
        },
    )
