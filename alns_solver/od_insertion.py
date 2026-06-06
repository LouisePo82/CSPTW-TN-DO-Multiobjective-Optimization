from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from alns_solver.solution_state import ALNSSolutionState


@dataclass
class ODInsertionResult:
    """
    Result returned by an OD insertion strategy.

    `objective_delta` is measured against the supplied base state whenever
    the base state is evaluable. For partial base states it may be None.
    """
    strategy: str
    state: ALNSSolutionState
    driver: str
    pickup: str
    customer: str
    insertion_position: int
    dv_vehicle: str | None
    dv_insertion_position: int | None
    cost: float
    emission: float
    objective: float
    objective_delta: float | None
    validator_pass: bool
    validation_errors: list[str]


def _route_insert_delta(
    route: list[str],
    node: str,
    position: int,
    distance: dict,
) -> float:
    prev_node = route[position - 1]
    next_node = route[position]

    return (
        distance[prev_node][node]
        + distance[node][next_node]
        - distance[prev_node][next_node]
    )


def _valid_od_insert_positions(route: list[str]) -> range:
    """
    OD route structure:
        origin -> pickup point -> customers -> destination

    A newly inserted customer must therefore be placed after the pickup point
    and before the destination.
    """
    if len(route) < 3:
        return range(0)

    return range(2, len(route))


def _driver_customer_count(
    state: ALNSSolutionState,
    driver: str,
) -> int:
    return sum(
        1
        for assignment in state.assignments.values()
        if assignment.get("mode") == "OD_HOME"
        and assignment.get("driver") == driver
    )


def _evaluate_complete_candidate(
    state: ALNSSolutionState,
    instance: dict,
    *,
    lambda_value: float,
    cost_bounds: tuple[float, float] | None,
    emission_bounds: tuple[float, float] | None,
    emission_factors: tuple[float, float],
    strategy: str,
    customer: str,
    driver: str,
    pickup: str,
    insertion_position: int,
    dv_vehicle: str | None = None,
    dv_insertion_position: int | None = None,
    base_objective: float | None = None,
) -> ODInsertionResult | None:
    solution = state.to_core_solution(
        instance=instance,
        lambda_value=lambda_value,
        objective_mode="weighted",
        cost_bounds=cost_bounds,
        emission_bounds=emission_bounds,
        emission_factors=emission_factors,
        require_complete=True,
        metadata={
            "od_insertion_strategy": strategy,
            "inserted_customer": customer,
            "driver": driver,
            "pickup": pickup,
        },
    )

    if not solution.validator_pass:
        return None

    objective_delta = (
        None
        if base_objective is None
        else solution.objective - base_objective
    )

    return ODInsertionResult(
        strategy=strategy,
        state=state,
        driver=driver,
        pickup=pickup,
        customer=customer,
        insertion_position=insertion_position,
        dv_vehicle=dv_vehicle,
        dv_insertion_position=dv_insertion_position,
        cost=solution.cost,
        emission=solution.emission,
        objective=solution.objective,
        objective_delta=objective_delta,
        validator_pass=solution.validator_pass,
        validation_errors=list(solution.validation_errors),
    )


def _try_base_objective(
    state: ALNSSolutionState,
    instance: dict,
    *,
    lambda_value: float,
    cost_bounds: tuple[float, float] | None,
    emission_bounds: tuple[float, float] | None,
    emission_factors: tuple[float, float],
) -> float | None:
    """
    A destroy-state may be incomplete and therefore invalid. In that case,
    objective delta is not defined and selection uses the candidate objective.
    """
    try:
        solution = state.to_core_solution(
            instance=instance,
            lambda_value=lambda_value,
            objective_mode="weighted",
            cost_bounds=cost_bounds,
            emission_bounds=emission_bounds,
            emission_factors=emission_factors,
            require_complete=True,
        )
    except Exception:
        return None

    return solution.objective if solution.validator_pass else None


def od_insertion_strategy_1(
    state: ALNSSolutionState,
    instance: dict,
    customer_id: str,
    *,
    lambda_value: float,
    cost_bounds: tuple[float, float] | None,
    emission_bounds: tuple[float, float] | None,
    emission_factors: tuple[float, float] = (1.0, 1.0),
    candidate_drivers: Iterable[str] | None = None,
) -> ODInsertionResult | None:
    """
    Paper Algorithm 3 frame: insert a customer through pickup points already
    used by active occasional drivers.

    For each eligible OD:
    1. Keep its existing pickup point unchanged.
    2. Test every customer insertion position after the pickup point.
    3. Convert the complete candidate through the shared core validator.
    4. Return the feasible candidate with the lowest weighted objective.
    """
    if customer_id not in instance["customers"]:
        raise ValueError(f"Unknown customer: {customer_id}")

    if customer_id in state.assignments:
        raise ValueError(
            f"Customer {customer_id} is already assigned. "
            "Destroy it before repair insertion."
        )

    drivers = (
        list(candidate_drivers)
        if candidate_drivers is not None
        else list(instance["ods"])
    )

    base_objective = _try_base_objective(
        state,
        instance,
        lambda_value=lambda_value,
        cost_bounds=cost_bounds,
        emission_bounds=emission_bounds,
        emission_factors=emission_factors,
    )

    candidates: list[ODInsertionResult] = []

    for driver in drivers:
        route = state.od_routes.get(driver, [])

        # Strategy I only uses an already active pickup point.
        if len(route) < 3:
            continue

        pickup = route[1]

        if pickup not in instance["pickup_points"]:
            continue

        capacity = int(instance["vehicles"][driver]["capacity"])
        if _driver_customer_count(state, driver) >= capacity:
            continue

        for position in _valid_od_insert_positions(route):
            candidate_state = state.copy()
            candidate_state.od_routes[driver].insert(
                position,
                customer_id,
            )
            candidate_state.assign_customer(
                customer_id,
                {
                    "mode": "OD_HOME",
                    "driver": driver,
                    "pickup": pickup,
                },
            )
            candidate_state.register_operator_event(
                operator_type="repair",
                operator_name="od_insertion_strategy_1",
                details={
                    "customer": customer_id,
                    "driver": driver,
                    "pickup": pickup,
                    "position": position,
                },
            )

            result = _evaluate_complete_candidate(
                candidate_state,
                instance,
                lambda_value=lambda_value,
                cost_bounds=cost_bounds,
                emission_bounds=emission_bounds,
                emission_factors=emission_factors,
                strategy="strategy_1_existing_pickup",
                customer=customer_id,
                driver=driver,
                pickup=pickup,
                insertion_position=position,
                base_objective=base_objective,
            )

            if result is not None:
                candidates.append(result)

    if not candidates:
        return None

    return min(
        candidates,
        key=lambda result: (
            result.objective,
            result.cost,
            result.emission,
            result.driver,
            result.insertion_position,
        ),
    )


def _dv_load_before_new_tn_assignment(
    state: ALNSSolutionState,
    instance: dict,
    vehicle: str,
) -> float:
    return float(
        state._compute_vehicle_loads(instance).get(vehicle, 0.0)
    )


def _candidate_dv_tn_insertions(
    state: ALNSSolutionState,
    instance: dict,
    tn: str,
    customer_id: str,
) -> list[tuple[str, int, float]]:
    """
    Return feasible (vehicle, position, distance_delta) combinations for a
    newly used TN. Capacity includes the customer parcel transferred at TN.
    """
    demand = float(instance["nodes"][customer_id]["demand"])
    candidates: list[tuple[str, int, float]] = []

    for vehicle in instance["dvs"]:
        capacity = float(instance["vehicles"][vehicle]["capacity"])
        current_load = _dv_load_before_new_tn_assignment(
            state,
            instance,
            vehicle,
        )

        if current_load + demand > capacity + 1e-9:
            continue

        route = list(state.dv_routes.get(vehicle, []))

        if not route:
            route = [
                instance["start_depot"],
                instance["end_depot"],
            ]

        if tn in route:
            # The TN is already physically visited by this DV. No insertion.
            candidates.append((vehicle, route.index(tn), 0.0))
            continue

        for position in range(1, len(route)):
            delta = _route_insert_delta(
                route=route,
                node=tn,
                position=position,
                distance=instance["distance"],
            )
            candidates.append((vehicle, position, delta))

    return candidates


def od_insertion_strategy_2(
    state: ALNSSolutionState,
    instance: dict,
    customer_id: str,
    *,
    lambda_value: float,
    cost_bounds: tuple[float, float] | None,
    emission_bounds: tuple[float, float] | None,
    emission_factors: tuple[float, float] = (1.0, 1.0),
    candidate_drivers: Iterable[str] | None = None,
    candidate_tns: Iterable[str] | None = None,
) -> ODInsertionResult | None:
    """
    Paper Algorithm 4 frame: evaluate a newly used TN pickup point.

    Gate-3 scope:
    - considers currently inactive ODs;
    - creates origin -> TN -> destination;
    - inserts the customer after TN;
    - inserts TN into every capacity-feasible DV route/position;
    - validates full routing, time windows, loads and synchronization;
    - selects the lowest weighted-objective candidate.

    Replacing the pickup point of an already active OD is intentionally
    deferred until the later neighborhood/local-search gate.
    """
    if customer_id not in instance["customers"]:
        raise ValueError(f"Unknown customer: {customer_id}")

    if customer_id in state.assignments:
        raise ValueError(
            f"Customer {customer_id} is already assigned. "
            "Destroy it before repair insertion."
        )

    drivers = (
        list(candidate_drivers)
        if candidate_drivers is not None
        else list(instance["ods"])
    )
    tns = (
        list(candidate_tns)
        if candidate_tns is not None
        else list(instance["tns"])
    )

    base_objective = _try_base_objective(
        state,
        instance,
        lambda_value=lambda_value,
        cost_bounds=cost_bounds,
        emission_bounds=emission_bounds,
        emission_factors=emission_factors,
    )

    candidates: list[ODInsertionResult] = []

    for driver in drivers:
        existing_route = state.od_routes.get(driver, [])

        # Gate 3 only creates a new TN pickup for an inactive OD.
        if existing_route:
            continue

        capacity = int(instance["vehicles"][driver]["capacity"])
        if _driver_customer_count(state, driver) >= capacity:
            continue

        info = instance["vehicles"][driver]

        for tn in tns:
            if tn not in instance["tns"]:
                continue

            dv_candidates = _candidate_dv_tn_insertions(
                state=state,
                instance=instance,
                tn=tn,
                customer_id=customer_id,
            )

            for vehicle, dv_position, _ in dv_candidates:
                candidate_state = state.copy()

                dv_route = list(
                    candidate_state.dv_routes.get(vehicle, [])
                )
                if not dv_route:
                    dv_route = [
                        instance["start_depot"],
                        instance["end_depot"],
                    ]
                    candidate_state.dv_routes[vehicle] = dv_route

                if tn not in dv_route:
                    candidate_state.dv_routes[vehicle].insert(
                        dv_position,
                        tn,
                    )

                od_route = [
                    info["origin"],
                    tn,
                    customer_id,
                    info["destination"],
                ]
                candidate_state.od_routes[driver] = od_route
                candidate_state.assign_customer(
                    customer_id,
                    {
                        "mode": "OD_HOME",
                        "driver": driver,
                        "pickup": tn,
                    },
                )
                candidate_state.register_operator_event(
                    operator_type="repair",
                    operator_name="od_insertion_strategy_2",
                    details={
                        "customer": customer_id,
                        "driver": driver,
                        "pickup": tn,
                        "od_position": 2,
                        "dv_vehicle": vehicle,
                        "dv_position": dv_position,
                    },
                )

                result = _evaluate_complete_candidate(
                    candidate_state,
                    instance,
                    lambda_value=lambda_value,
                    cost_bounds=cost_bounds,
                    emission_bounds=emission_bounds,
                    emission_factors=emission_factors,
                    strategy="strategy_2_new_tn",
                    customer=customer_id,
                    driver=driver,
                    pickup=tn,
                    insertion_position=2,
                    dv_vehicle=vehicle,
                    dv_insertion_position=dv_position,
                    base_objective=base_objective,
                )

                if result is not None:
                    candidates.append(result)

    if not candidates:
        return None

    return min(
        candidates,
        key=lambda result: (
            result.objective,
            result.cost,
            result.emission,
            result.driver,
            result.pickup,
            result.dv_vehicle or "",
            result.dv_insertion_position or 0,
        ),
    )
