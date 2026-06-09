from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

from alns_solver.solution_state import ALNSSolutionState


@dataclass
class InsertionCandidate:
    customer_id: str
    mode: str
    state: ALNSSolutionState
    objective: float
    insertion_cost: float
    cost: float
    emission: float
    details: dict[str, Any]


@dataclass
class RepairResult:
    operator_name: str
    state: ALNSSolutionState
    insertion_order: list[str]
    final_objective: float
    final_cost: float
    final_emission: float
    validator_pass: bool
    validation_errors: list[str]


def _canonicalize_inactive_routes(
    state: ALNSSolutionState,
    instance: dict,
) -> None:
    """
    Convert degenerate inactive routes to the repository's canonical form.

    Repository convention:
    - unused DV route: []
    - unused OD route: []

    Destroy operators may temporarily leave:
    - DV: [S, T]
    - OD: [origin, pickup, destination]

    Those routes contain no service activity and must not survive into a
    complete solution, because the shared validator correctly treats them as
    degenerate active routes.
    """
    start = instance["start_depot"]
    end = instance["end_depot"]

    # Canonicalize DV routes that contain no service node.
    for vehicle in instance["dvs"]:
        route = list(state.dv_routes.get(vehicle, []))

        if not route:
            state.dv_routes[vehicle] = []
            continue

        assigned_to_vehicle = any(
            assignment.get("vehicle") == vehicle
            and assignment.get("mode") in {"DV_HOME", "ADP"}
            for assignment in state.assignments.values()
        )

        active_tns = {
            assignment.get("pickup")
            for assignment in state.assignments.values()
            if assignment.get("mode") == "OD_HOME"
            and assignment.get("pickup") in instance["tns"]
        }

        route_serves_active_tn = any(
            tn in route
            for tn in active_tns
        )

        service_nodes = [
            node
            for node in route
            if node not in {start, end}
        ]

        if (
            route == [start, end]
            or (
                not service_nodes
                and not assigned_to_vehicle
                and not route_serves_active_tn
            )
        ):
            state.dv_routes[vehicle] = []

    # Canonicalize OD routes with no assigned customer.
    for driver in instance["ods"]:
        route = list(state.od_routes.get(driver, []))

        if not route:
            state.od_routes[driver] = []
            continue

        assigned_customers = [
            customer
            for customer, assignment in state.assignments.items()
            if assignment.get("mode") == "OD_HOME"
            and assignment.get("driver") == driver
        ]

        if not assigned_customers:
            state.od_routes[driver] = []

    state.invalidate_cache()


def _weighted_metrics(
    state: ALNSSolutionState,
    instance: dict,
    *,
    lambda_value: float,
    cost_bounds: tuple[float, float] | None,
    emission_bounds: tuple[float, float] | None,
    emission_factors: tuple[float, float],
) -> dict[str, float]:
    return state.evaluate(
        instance=instance,
        lambda_value=lambda_value,
        cost_bounds=cost_bounds,
        emission_bounds=emission_bounds,
        emission_factors=emission_factors,
    )


def _partial_solution_is_valid(
    state: ALNSSolutionState,
    instance: dict,
    *,
    lambda_value: float,
    cost_bounds: tuple[float, float] | None,
    emission_bounds: tuple[float, float] | None,
    emission_factors: tuple[float, float],
) -> bool:
    """
    Validate an intermediate ALNS state without requiring full customer coverage.

    During destroy/repair, customers may legitimately remain in
    `unassigned_customers`. Therefore, this function checks only structural,
    route, capacity and compatibility conditions that must already hold.

    Full feasibility is checked after repair completion through the shared
    core validator in `_finalize_repair`.
    """
    structure_errors = state.basic_structure_errors(instance)

    if structure_errors:
        return False

    valid_customers = set(instance["customers"])
    assigned_customers = set(state.assignments)
    unassigned_customers = set(state.unassigned_customers)

    if assigned_customers & unassigned_customers:
        return False

    if not assigned_customers.issubset(valid_customers):
        return False

    if not unassigned_customers.issubset(valid_customers):
        return False

    # Every customer not assigned should remain explicitly tracked as
    # unassigned during the repair process.
    missing_customers = (
        valid_customers
        - assigned_customers
        - unassigned_customers
    )

    if missing_customers:
        return False

    # DV route endpoint and node checks.
    for vehicle in instance["dvs"]:
        route = state.dv_routes.get(vehicle, [])

        if not route:
            continue

        if route[0] != instance["start_depot"]:
            return False

        if route[-1] != instance["end_depot"]:
            return False

        if any(node not in instance["nodes"] for node in route):
            return False

    # OD route endpoint, pickup and node checks.
    for driver in instance["ods"]:
        route = state.od_routes.get(driver, [])

        if not route:
            continue

        info = instance["vehicles"][driver]

        if len(route) < 3:
            return False

        if route[0] != info["origin"]:
            return False

        if route[-1] != info["destination"]:
            return False

        if route[1] not in instance["pickup_points"]:
            return False

        if any(node not in instance["nodes"] for node in route):
            return False

    # Assignment-to-route consistency.
    for customer, assignment in state.assignments.items():
        mode = assignment.get("mode")

        if mode == "DV_HOME":
            vehicle = assignment.get("vehicle")

            if vehicle not in instance["dvs"]:
                return False

            if customer not in state.dv_routes.get(vehicle, []):
                return False

        elif mode == "ADP":
            vehicle = assignment.get("vehicle")
            adp = assignment.get("adp")

            if vehicle not in instance["dvs"]:
                return False

            if adp not in instance["adps"]:
                return False

            if instance["gamma"].get((customer, adp), 0) != 1:
                return False

            if adp not in state.dv_routes.get(vehicle, []):
                return False

        elif mode == "OD_HOME":
            driver = assignment.get("driver")
            pickup = assignment.get("pickup")

            if driver not in instance["ods"]:
                return False

            route = state.od_routes.get(driver, [])

            if not route:
                return False

            if pickup not in instance["pickup_points"]:
                return False

            if route[1] != pickup:
                return False

            if customer not in route:
                return False

            # If pickup is a TN, some DV must physically visit that TN.
            if pickup in instance["tns"]:
                if not any(
                    pickup in state.dv_routes.get(vehicle, [])
                    for vehicle in instance["dvs"]
                ):
                    return False

        else:
            return False

    # DV capacity.
    # Use the same load semantics as the shared final
    # validator, including OD handoff demand at every TN
    # visited by a dedicated vehicle.
    dv_loads = _validator_contract_dv_loads(
        state,
        instance,
    )

    for vehicle, load in dv_loads.items():
        capacity = float(
            instance["vehicles"][vehicle]["capacity"]
        )

        if float(load) > capacity + 1e-9:
            return False

    # OD capacity.
    for driver in instance["ods"]:
        load = sum(
            float(instance["nodes"][customer]["demand"])
            for customer, assignment in state.assignments.items()
            if assignment.get("mode") == "OD_HOME"
            and assignment.get("driver") == driver
        )

        capacity = float(
            instance["vehicles"][driver]["capacity"]
        )

        if load > capacity + 1e-9:
            return False

    return True


def _candidate_from_state(
    customer_id: str,
    mode: str,
    state: ALNSSolutionState,
    details: dict[str, Any],
    instance: dict,
    *,
    lambda_value: float,
    cost_bounds: tuple[float, float] | None,
    emission_bounds: tuple[float, float] | None,
    emission_factors: tuple[float, float],
) -> InsertionCandidate | None:
    _canonicalize_inactive_routes(
        state,
        instance,
    )

    if not _partial_solution_is_valid(
        state,
        instance,
        lambda_value=lambda_value,
        cost_bounds=cost_bounds,
        emission_bounds=emission_bounds,
        emission_factors=emission_factors,
    ):
        return None

    metrics = _weighted_metrics(
        state,
        instance,
        lambda_value=lambda_value,
        cost_bounds=cost_bounds,
        emission_bounds=emission_bounds,
        emission_factors=emission_factors,
    )

    # All candidates enumerated during one repair iteration share the same
    # parent state. Therefore ranking by candidate objective is mathematically
    # equivalent to ranking by objective increase:
    #
    #   argmin [f(S + i) - f(S)] = argmin f(S + i)
    #
    # Reconstructing the parent by only removing the assignment is unsafe,
    # because it does not undo inserted route nodes such as a customer, ADP,
    # or TN. Use the candidate objective as the stable insertion score.
    insertion_cost = float(metrics["objective"])

    return InsertionCandidate(
        customer_id=customer_id,
        mode=mode,
        state=state,
        objective=float(metrics["objective"]),
        insertion_cost=insertion_cost,
        cost=float(metrics["cost"]),
        emission=float(metrics["emission"]),
        details=details,
    )


def _route_insert_positions(route: list[str]) -> range:
    return range(1, len(route))


def _od_insert_positions(route: list[str]) -> range:
    """
    OD structure:
        origin -> pickup -> customers -> destination

    New customers may only be inserted after the pickup point and before the
    destination.
    """
    return range(2, len(route))


def _validator_contract_dv_loads(
    state: ALNSSolutionState,
    instance: dict,
) -> dict[str, float]:
    """
    Reproduce the shared validator's DV-capacity semantics.

    A DV carries its direct-home and ADP assignments.
    It also carries the full handoff demand of every active
    OD whose TN pickup is visited by that DV route.
    """
    loads = {
        vehicle: 0.0
        for vehicle in instance["dvs"]
    }

    for customer, assignment in state.assignments.items():
        mode = assignment.get("mode")
        vehicle = assignment.get("vehicle")

        if (
            mode in {"DV_HOME", "ADP"}
            and vehicle in loads
        ):
            loads[vehicle] += float(
                instance["nodes"][customer][
                    "demand"
                ]
            )

    for vehicle in instance["dvs"]:
        dv_route = state.dv_routes.get(
            vehicle,
            [],
        )

        for driver in instance["ods"]:
            od_route = state.od_routes.get(
                driver,
                [],
            )

            if not od_route:
                continue

            pickup = next(
                (
                    tn
                    for tn in instance["tns"]
                    if tn in od_route
                ),
                None,
            )

            if (
                pickup is None
                or pickup not in dv_route
            ):
                continue

            loads[vehicle] += sum(
                float(
                    instance["nodes"][customer][
                        "demand"
                    ]
                )
                for customer, assignment
                in state.assignments.items()
                if (
                    assignment.get("mode")
                    == "OD_HOME"
                    and assignment.get("driver")
                    == driver
                    and assignment.get("pickup")
                    == pickup
                )
            )

    return loads


def _current_dv_load(
    state: ALNSSolutionState,
    instance: dict,
    vehicle: str,
) -> float:
    """
    Return the DV load using the same capacity semantics
    as the shared final validator.

    This includes DV-home demand, ADP demand, and the full
    handoff demand of every OD using a TN visited by the DV.
    """
    return float(
        _validator_contract_dv_loads(
            state,
            instance,
        ).get(
            vehicle,
            0.0,
        )
    )


def _current_od_load(
    state: ALNSSolutionState,
    instance: dict,
    driver: str,
) -> float:
    return sum(
        float(instance["nodes"][customer]["demand"])
        for customer, assignment in state.assignments.items()
        if assignment.get("mode") == "OD_HOME"
        and assignment.get("driver") == driver
    )


def _default_dv_route(instance: dict) -> list[str]:
    return [
        instance["start_depot"],
        instance["end_depot"],
    ]


def _enumerate_dv_home_candidates(
    state: ALNSSolutionState,
    instance: dict,
    customer_id: str,
    *,
    lambda_value: float,
    cost_bounds: tuple[float, float] | None,
    emission_bounds: tuple[float, float] | None,
    emission_factors: tuple[float, float],
) -> list[InsertionCandidate]:
    candidates: list[InsertionCandidate] = []
    demand = float(instance["nodes"][customer_id]["demand"])

    for vehicle in instance["dvs"]:
        capacity = float(instance["vehicles"][vehicle]["capacity"])

        if (
            _current_dv_load(state, instance, vehicle) + demand
            > capacity + 1e-9
        ):
            continue

        base_route = list(state.dv_routes.get(vehicle, []))

        if not base_route:
            base_route = _default_dv_route(instance)

        for position in _route_insert_positions(base_route):
            candidate_state = state.copy()
            candidate_state.dv_routes[vehicle] = list(base_route)
            candidate_state.dv_routes[vehicle].insert(
                position,
                customer_id,
            )
            candidate_state.assign_customer(
                customer_id,
                {
                    "mode": "DV_HOME",
                    "vehicle": vehicle,
                },
            )

            candidate = _candidate_from_state(
                customer_id=customer_id,
                mode="DV_HOME",
                state=candidate_state,
                details={
                    "vehicle": vehicle,
                    "position": position,
                },
                instance=instance,
                lambda_value=lambda_value,
                cost_bounds=cost_bounds,
                emission_bounds=emission_bounds,
                emission_factors=emission_factors,
            )

            if candidate is not None:
                candidates.append(candidate)

    return candidates


def _enumerate_adp_candidates(
    state: ALNSSolutionState,
    instance: dict,
    customer_id: str,
    *,
    lambda_value: float,
    cost_bounds: tuple[float, float] | None,
    emission_bounds: tuple[float, float] | None,
    emission_factors: tuple[float, float],
) -> list[InsertionCandidate]:
    candidates: list[InsertionCandidate] = []
    demand = float(instance["nodes"][customer_id]["demand"])

    compatible_adps = [
        adp
        for adp in instance["adps"]
        if instance["gamma"].get((customer_id, adp), 0) == 1
    ]

    for vehicle in instance["dvs"]:
        capacity = float(instance["vehicles"][vehicle]["capacity"])

        if (
            _current_dv_load(state, instance, vehicle) + demand
            > capacity + 1e-9
        ):
            continue

        base_route = list(state.dv_routes.get(vehicle, []))

        if not base_route:
            base_route = _default_dv_route(instance)

        for adp in compatible_adps:
            if adp in base_route:
                route_options = [
                    (
                        list(base_route),
                        base_route.index(adp),
                    )
                ]
            else:
                route_options = []

                for position in _route_insert_positions(base_route):
                    route = list(base_route)
                    route.insert(position, adp)
                    route_options.append(
                        (
                            route,
                            position,
                        )
                    )

            for route, position in route_options:
                candidate_state = state.copy()
                candidate_state.dv_routes[vehicle] = route
                candidate_state.assign_customer(
                    customer_id,
                    {
                        "mode": "ADP",
                        "vehicle": vehicle,
                        "adp": adp,
                    },
                )

                candidate = _candidate_from_state(
                    customer_id=customer_id,
                    mode="ADP",
                    state=candidate_state,
                    details={
                        "vehicle": vehicle,
                        "adp": adp,
                        "position": position,
                    },
                    instance=instance,
                    lambda_value=lambda_value,
                    cost_bounds=cost_bounds,
                    emission_bounds=emission_bounds,
                    emission_factors=emission_factors,
                )

                if candidate is not None:
                    candidates.append(candidate)

    return candidates


def _enumerate_existing_od_candidates(
    state: ALNSSolutionState,
    instance: dict,
    customer_id: str,
    *,
    lambda_value: float,
    cost_bounds: tuple[float, float] | None,
    emission_bounds: tuple[float, float] | None,
    emission_factors: tuple[float, float],
) -> list[InsertionCandidate]:
    candidates: list[InsertionCandidate] = []
    demand = float(instance["nodes"][customer_id]["demand"])


    for driver in instance["ods"]:
        route = list(state.od_routes.get(driver, []))

        if len(route) < 3:
            continue

        capacity = float(instance["vehicles"][driver]["capacity"])

        if (
            _current_od_load(state, instance, driver) + demand
            > capacity + 1e-9
        ):
            continue

        pickup = route[1]

        for position in _od_insert_positions(route):
            candidate_state = state.copy()
            candidate_state.od_routes[driver] = list(route)
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

            # Adding a customer to an existing OD route
            # increases the handoff demand at its TN pickup.
            # Reject the insertion if it would violate the
            # shared validator's DV-capacity contract.
            if pickup in instance["tns"]:
                dv_loads = (
                    _validator_contract_dv_loads(
                        candidate_state,
                        instance,
                    )
                )

                capacity_violated = any(
                    float(
                        dv_loads.get(
                            vehicle,
                            0.0,
                        )
                    )
                    > float(
                        instance["vehicles"][
                            vehicle
                        ]["capacity"]
                    )
                    + 1e-9
                    for vehicle in instance["dvs"]
                )

                if capacity_violated:
                    continue

            candidate = _candidate_from_state(
                customer_id=customer_id,
                mode="OD_HOME",
                state=candidate_state,
                details={
                    "driver": driver,
                    "pickup": pickup,
                    "position": position,
                    "pickup_status": "existing",
                },
                instance=instance,
                lambda_value=lambda_value,
                cost_bounds=cost_bounds,
                emission_bounds=emission_bounds,
                emission_factors=emission_factors,
            )

            if candidate is not None:
                candidates.append(candidate)

    return candidates


def _enumerate_inactive_od_depot_candidates(
    state: ALNSSolutionState,
    instance: dict,
    customer_id: str,
    *,
    lambda_value: float,
    cost_bounds: tuple[float, float] | None,
    emission_bounds: tuple[float, float] | None,
    emission_factors: tuple[float, float],
) -> list[InsertionCandidate]:
    candidates: list[InsertionCandidate] = []
    demand = float(instance["nodes"][customer_id]["demand"])
    depot = instance["start_depot"]

    for driver in instance["ods"]:
        if state.od_routes.get(driver, []):
            continue

        capacity = float(instance["vehicles"][driver]["capacity"])

        if demand > capacity + 1e-9:
            continue

        info = instance["vehicles"][driver]
        candidate_state = state.copy()
        candidate_state.od_routes[driver] = [
            info["origin"],
            depot,
            customer_id,
            info["destination"],
        ]
        candidate_state.assign_customer(
            customer_id,
            {
                "mode": "OD_HOME",
                "driver": driver,
                "pickup": depot,
            },
        )

        candidate = _candidate_from_state(
            customer_id=customer_id,
            mode="OD_HOME",
            state=candidate_state,
            details={
                "driver": driver,
                "pickup": depot,
                "position": 2,
                "pickup_status": "new_depot",
            },
            instance=instance,
            lambda_value=lambda_value,
            cost_bounds=cost_bounds,
            emission_bounds=emission_bounds,
            emission_factors=emission_factors,
        )

        if candidate is not None:
            candidates.append(candidate)

    return candidates



def _active_pickup_points(
    state: ALNSSolutionState,
    instance: dict,
) -> list[str]:
    """
    Pickup points currently available in the partial solution.

    The depot is always available. A TN is active only when at least one
    dedicated-vehicle route currently visits it.
    """
    active = [instance["start_depot"]]

    for tn in instance["tns"]:
        if any(
            tn in state.dv_routes.get(vehicle, [])
            for vehicle in instance["dvs"]
        ):
            active.append(tn)

    return active


def _enumerate_inactive_od_active_pickup_candidates(
    state: ALNSSolutionState,
    instance: dict,
    customer_id: str,
    *,
    lambda_value: float,
    cost_bounds: tuple[float, float] | None,
    emission_bounds: tuple[float, float] | None,
    emission_factors: tuple[float, float],
) -> list[InsertionCandidate]:
    """
    Paper CS Insertion Strategy I for an inactive occasional driver.

    The driver may use any pickup point already available in the current
    solution: the depot or an already visited TN. No new TN insertion is
    performed here.
    """
    candidates: list[InsertionCandidate] = []
    demand = float(instance["nodes"][customer_id]["demand"])
    pickup_points = _active_pickup_points(state, instance)

    for driver in instance["ods"]:
        if state.od_routes.get(driver, []):
            continue

        capacity = float(instance["vehicles"][driver]["capacity"])
        if demand > capacity + 1e-9:
            continue

        info = instance["vehicles"][driver]

        for pickup in pickup_points:
            candidate_state = state.copy()
            candidate_state.od_routes[driver] = [
                info["origin"],
                pickup,
                customer_id,
                info["destination"],
            ]
            candidate_state.assign_customer(
                customer_id,
                {
                    "mode": "OD_HOME",
                    "driver": driver,
                    "pickup": pickup,
                },
            )

            if pickup in instance["tns"]:
                dv_loads = (
                    _validator_contract_dv_loads(
                        candidate_state,
                        instance,
                    )
                )

                capacity_violated = any(
                    float(
                        dv_loads.get(
                            vehicle,
                            0.0,
                        )
                    )
                    > float(
                        instance["vehicles"][
                            vehicle
                        ]["capacity"]
                    )
                    + 1e-9
                    for vehicle in instance["dvs"]
                )

                if capacity_violated:
                    continue

            candidate = _candidate_from_state(
                customer_id=customer_id,
                mode="OD_HOME",
                state=candidate_state,
                details={
                    "driver": driver,
                    "pickup": pickup,
                    "position": 2,
                    "pickup_status": "active_strategy_1",
                },
                instance=instance,
                lambda_value=lambda_value,
                cost_bounds=cost_bounds,
                emission_bounds=emission_bounds,
                emission_factors=emission_factors,
            )

            if candidate is not None:
                candidates.append(candidate)

    return candidates


def _enumerate_inactive_od_tn_candidates(
    state: ALNSSolutionState,
    instance: dict,
    customer_id: str,
    *,
    lambda_value: float,
    cost_bounds: tuple[float, float] | None,
    emission_bounds: tuple[float, float] | None,
    emission_factors: tuple[float, float],
    strategy_2_mode: str = "paper_random_dv",
    strategy_2_seed: int | None = None,
) -> list[InsertionCandidate]:
    candidates: list[InsertionCandidate] = []
    demand = float(instance["nodes"][customer_id]["demand"])

    if strategy_2_mode not in {
        "paper_random_dv",
        "exhaustive_best_dv",
    }:
        raise ValueError(
            "strategy_2_mode must be 'paper_random_dv' or "
            "'exhaustive_best_dv'."
        )

    available_dvs = list(instance["dvs"])

    if not available_dvs:
        return []

    if strategy_2_mode == "paper_random_dv":
        rng = random.Random(strategy_2_seed)
        selected_dvs = [rng.choice(available_dvs)]
    else:
        selected_dvs = available_dvs

    for driver in instance["ods"]:
        if state.od_routes.get(driver, []):
            continue

        od_capacity = float(instance["vehicles"][driver]["capacity"])

        if demand > od_capacity + 1e-9:
            continue

        od_info = instance["vehicles"][driver]

        for tn in instance["tns"]:
            for vehicle in selected_dvs:
                dv_capacity = float(
                    instance["vehicles"][vehicle]["capacity"]
                )

                if (
                    _current_dv_load(state, instance, vehicle)
                    + demand
                    > dv_capacity + 1e-9
                ):
                    continue

                base_route = list(
                    state.dv_routes.get(vehicle, [])
                )

                if not base_route:
                    base_route = _default_dv_route(instance)

                if tn in base_route:
                    route_options = [
                        (
                            list(base_route),
                            base_route.index(tn),
                        )
                    ]
                else:
                    route_options = []

                    for position in _route_insert_positions(
                        base_route
                    ):
                        route = list(base_route)
                        route.insert(position, tn)
                        route_options.append(
                            (
                                route,
                                position,
                            )
                        )

                for route, dv_position in route_options:
                    candidate_state = state.copy()
                    candidate_state.dv_routes[vehicle] = route
                    candidate_state.od_routes[driver] = [
                        od_info["origin"],
                        tn,
                        customer_id,
                        od_info["destination"],
                    ]
                    candidate_state.assign_customer(
                        customer_id,
                        {
                            "mode": "OD_HOME",
                            "driver": driver,
                            "pickup": tn,
                        },
                    )

                    candidate = _candidate_from_state(
                        customer_id=customer_id,
                        mode="OD_HOME",
                        state=candidate_state,
                        details={
                            "driver": driver,
                            "pickup": tn,
                            "position": 2,
                            "pickup_status": "new_tn",
                            "dv_vehicle": vehicle,
                            "dv_position": dv_position,
                        },
                        instance=instance,
                        lambda_value=lambda_value,
                        cost_bounds=cost_bounds,
                        emission_bounds=emission_bounds,
                        emission_factors=emission_factors,
                    )

                    if candidate is not None:
                        candidates.append(candidate)

    return candidates


def enumerate_insertion_candidates(
    state: ALNSSolutionState,
    instance: dict,
    customer_id: str,
    *,
    lambda_value: float,
    cost_bounds: tuple[float, float] | None,
    emission_bounds: tuple[float, float] | None,
    emission_factors: tuple[float, float] = (1.0, 1.0),
    strategy_2_mode: str = "paper_random_dv",
    strategy_2_seed: int | None = None,
) -> list[InsertionCandidate]:
    """
    Enumerate all currently feasible insertions for one unassigned customer.
    """
    if customer_id in state.assignments:
        raise ValueError(
            f"Customer {customer_id} is already assigned."
        )

    if customer_id not in instance["customers"]:
        raise ValueError(
            f"Unknown customer: {customer_id}"
        )

    customer_type = int(
        instance["nodes"][customer_id]["customer_type"]
    )

    candidates: list[InsertionCandidate] = []

    if customer_type in {1, 3}:
        candidates.extend(
            _enumerate_dv_home_candidates(
                state,
                instance,
                customer_id,
                lambda_value=lambda_value,
                cost_bounds=cost_bounds,
                emission_bounds=emission_bounds,
                emission_factors=emission_factors,
            )
        )

        candidates.extend(
            _enumerate_existing_od_candidates(
                state,
                instance,
                customer_id,
                lambda_value=lambda_value,
                cost_bounds=cost_bounds,
                emission_bounds=emission_bounds,
                emission_factors=emission_factors,
            )
        )

        candidates.extend(
            _enumerate_inactive_od_active_pickup_candidates(
                state,
                instance,
                customer_id,
                lambda_value=lambda_value,
                cost_bounds=cost_bounds,
                emission_bounds=emission_bounds,
                emission_factors=emission_factors,
            )
        )

        candidates.extend(
            _enumerate_inactive_od_tn_candidates(
                state,
                instance,
                customer_id,
                lambda_value=lambda_value,
                cost_bounds=cost_bounds,
                emission_bounds=emission_bounds,
                emission_factors=emission_factors,
                strategy_2_mode=strategy_2_mode,
                strategy_2_seed=strategy_2_seed,
            )
        )

    if customer_type in {2, 3}:
        candidates.extend(
            _enumerate_adp_candidates(
                state,
                instance,
                customer_id,
                lambda_value=lambda_value,
                cost_bounds=cost_bounds,
                emission_bounds=emission_bounds,
                emission_factors=emission_factors,
            )
        )

    candidates.sort(
        key=lambda candidate: (
            candidate.insertion_cost,
            candidate.objective,
            candidate.cost,
            candidate.emission,
            candidate.mode,
            str(candidate.details),
        )
    )

    return candidates


def _finalize_repair(
    operator_name: str,
    state: ALNSSolutionState,
    insertion_order: list[str],
    instance: dict,
    *,
    lambda_value: float,
    cost_bounds: tuple[float, float] | None,
    emission_bounds: tuple[float, float] | None,
    emission_factors: tuple[float, float],
    metadata: dict[str, Any] | None = None,
) -> RepairResult:
    _canonicalize_inactive_routes(
        state,
        instance,
    )

    state.register_operator_event(
        operator_type="repair",
        operator_name=operator_name,
        details={
            "insertion_order": list(insertion_order),
            **(metadata or {}),
        },
    )

    solution = state.to_core_solution(
        instance=instance,
        lambda_value=lambda_value,
        objective_mode="weighted",
        cost_bounds=cost_bounds,
        emission_bounds=emission_bounds,
        emission_factors=emission_factors,
        require_complete=True,
    )

    if not solution.validator_pass:
        raise RuntimeError(
            f"{operator_name} produced an invalid complete solution. "
            f"Errors: {solution.validation_errors}. "
            f"DV routes: {state.dv_routes}. "
            f"OD routes: {state.od_routes}. "
            f"Assignments: {state.assignments}."
        )

    return RepairResult(
        operator_name=operator_name,
        state=state,
        insertion_order=insertion_order,
        final_objective=float(solution.objective),
        final_cost=float(solution.cost),
        final_emission=float(solution.emission),
        validator_pass=bool(solution.validator_pass),
        validation_errors=list(solution.validation_errors),
    )


def best_insertion_repair(
    state: ALNSSolutionState,
    instance: dict,
    *,
    lambda_value: float,
    cost_bounds: tuple[float, float] | None,
    emission_bounds: tuple[float, float] | None,
    emission_factors: tuple[float, float] = (1.0, 1.0),
    strategy_2_mode: str = "paper_random_dv",
    strategy_2_seed: int | None = None,
) -> RepairResult:
    """
    Repeatedly select the globally cheapest feasible insertion among all
    remaining unassigned customers.
    """
    repaired = state.copy()
    _canonicalize_inactive_routes(
        repaired,
        instance,
    )
    insertion_order: list[str] = []

    while repaired.unassigned_customers:
        all_candidates: list[InsertionCandidate] = []
        candidate_counts: dict[str, int] = {}

        for customer_id in sorted(
            repaired.unassigned_customers
        ):
            customer_candidates = enumerate_insertion_candidates(
                repaired,
                instance,
                customer_id,
                lambda_value=lambda_value,
                cost_bounds=cost_bounds,
                emission_bounds=emission_bounds,
                emission_factors=emission_factors,
                strategy_2_mode=strategy_2_mode,
                strategy_2_seed=strategy_2_seed,
            )

            candidate_counts[customer_id] = len(
                customer_candidates
            )
            all_candidates.extend(customer_candidates)

        if not all_candidates:
            raise RuntimeError(
                "Best insertion could not find any feasible candidate. "
                f"Remaining unassigned customers: "
                f"{sorted(repaired.unassigned_customers)}. "
                f"Candidate counts: {candidate_counts}. "
                f"Current DV routes: {repaired.dv_routes}. "
                f"Current OD routes: {repaired.od_routes}. "
                f"Current assignments: "
                f"{sorted(repaired.assignments)}."
            )

        selected = min(
            all_candidates,
            key=lambda candidate: (
                candidate.insertion_cost,
                candidate.objective,
                candidate.cost,
                candidate.emission,
                candidate.customer_id,
                candidate.mode,
                str(candidate.details),
            ),
        )

        repaired = selected.state
        insertion_order.append(selected.customer_id)

    return _finalize_repair(
        operator_name="best_insertion",
        state=repaired,
        insertion_order=insertion_order,
        instance=instance,
        lambda_value=lambda_value,
        cost_bounds=cost_bounds,
        emission_bounds=emission_bounds,
        emission_factors=emission_factors,
    )


def regret_k_repair(
    state: ALNSSolutionState,
    instance: dict,
    *,
    k: int,
    lambda_value: float,
    cost_bounds: tuple[float, float] | None,
    emission_bounds: tuple[float, float] | None,
    emission_factors: tuple[float, float] = (1.0, 1.0),
    strategy_2_mode: str = "paper_random_dv",
    strategy_2_seed: int | None = None,
) -> RepairResult:
    """
    Regret-k insertion.

    A customer with fewer than k feasible alternatives receives a large
    urgency bonus.
    """
    if k < 2:
        raise ValueError("Regret-k requires k >= 2.")

    repaired = state.copy()
    _canonicalize_inactive_routes(
        repaired,
        instance,
    )
    insertion_order: list[str] = []

    while repaired.unassigned_customers:
        customer_options: list[
            tuple[float, float, str, InsertionCandidate]
        ] = []

        candidate_counts: dict[str, int] = {}

        for customer_id in sorted(
            repaired.unassigned_customers
        ):
            candidates = enumerate_insertion_candidates(
                repaired,
                instance,
                customer_id,
                lambda_value=lambda_value,
                cost_bounds=cost_bounds,
                emission_bounds=emission_bounds,
                emission_factors=emission_factors,
                strategy_2_mode=strategy_2_mode,
                strategy_2_seed=strategy_2_seed,
            )

            candidate_counts[customer_id] = len(candidates)

            if not candidates:
                continue

            best = candidates[0]

            if len(candidates) >= k:
                regret = sum(
                    candidates[index].insertion_cost
                    - candidates[index - 1].insertion_cost
                    for index in range(1, k)
                )
            else:
                regret = 1_000_000.0 + (
                    candidates[-1].insertion_cost
                    - best.insertion_cost
                )

            customer_options.append(
                (
                    regret,
                    -best.objective,
                    customer_id,
                    best,
                )
            )

        if not customer_options:
            raise RuntimeError(
                f"Regret-{k} repair could not find a feasible "
                f"candidate. Remaining customers: "
                f"{sorted(repaired.unassigned_customers)}. "
                f"Candidate counts: {candidate_counts}."
            )

        _, _, customer_id, selected = max(
            customer_options,
            key=lambda item: (
                item[0],
                item[1],
                item[2],
            ),
        )

        repaired = selected.state
        insertion_order.append(customer_id)

    return _finalize_repair(
        operator_name=f"regret_{k}",
        state=repaired,
        insertion_order=insertion_order,
        instance=instance,
        lambda_value=lambda_value,
        cost_bounds=cost_bounds,
        emission_bounds=emission_bounds,
        emission_factors=emission_factors,
        metadata={"k": k},
    )


def regret_2_repair(
    state: ALNSSolutionState,
    instance: dict,
    **kwargs,
) -> RepairResult:
    return regret_k_repair(
        state,
        instance,
        k=2,
        **kwargs,
    )


def regret_3_repair(
    state: ALNSSolutionState,
    instance: dict,
    **kwargs,
) -> RepairResult:
    return regret_k_repair(
        state,
        instance,
        k=3,
        **kwargs,
    )


def perturbed_best_insertion_repair(
    state: ALNSSolutionState,
    instance: dict,
    *,
    seed: int,
    noise_strength: float = 0.275,
    lambda_value: float,
    cost_bounds: tuple[float, float] | None,
    emission_bounds: tuple[float, float] | None,
    emission_factors: tuple[float, float] = (1.0, 1.0),
    strategy_2_mode: str = "paper_random_dv",
    strategy_2_seed: int | None = None,
) -> RepairResult:
    """
    Best insertion with multiplicative random perturbation.
    """
    if noise_strength < 0:
        raise ValueError(
            "noise_strength must be nonnegative."
        )

    rng = random.Random(seed)
    repaired = state.copy()
    _canonicalize_inactive_routes(
        repaired,
        instance,
    )
    insertion_order: list[str] = []

    while repaired.unassigned_customers:
        scored: list[
            tuple[float, InsertionCandidate]
        ] = []

        candidate_counts: dict[str, int] = {}

        for customer_id in sorted(
            repaired.unassigned_customers
        ):
            candidates = enumerate_insertion_candidates(
                repaired,
                instance,
                customer_id,
                lambda_value=lambda_value,
                cost_bounds=cost_bounds,
                emission_bounds=emission_bounds,
                emission_factors=emission_factors,
                strategy_2_mode=strategy_2_mode,
                strategy_2_seed=strategy_2_seed,
            )

            candidate_counts[customer_id] = len(candidates)

            for candidate in candidates:
                noise = rng.uniform(
                    -noise_strength,
                    noise_strength,
                )
                score = candidate.insertion_cost * (
                    1.0 + noise
                )
                scored.append(
                    (
                        score,
                        candidate,
                    )
                )

        if not scored:
            raise RuntimeError(
                "Perturbed best insertion found no feasible "
                f"candidate. Remaining customers: "
                f"{sorted(repaired.unassigned_customers)}. "
                f"Candidate counts: {candidate_counts}."
            )

        _, selected = min(
            scored,
            key=lambda item: (
                item[0],
                item[1].customer_id,
                item[1].mode,
                str(item[1].details),
            ),
        )

        repaired = selected.state
        insertion_order.append(selected.customer_id)

    return _finalize_repair(
        operator_name="perturbed_best_insertion",
        state=repaired,
        insertion_order=insertion_order,
        instance=instance,
        lambda_value=lambda_value,
        cost_bounds=cost_bounds,
        emission_bounds=emission_bounds,
        emission_factors=emission_factors,
        metadata={
            "seed": seed,
            "noise_strength": noise_strength,
        },
    )


def perturbed_regret_repair(
    state: ALNSSolutionState,
    instance: dict,
    *,
    k: int,
    seed: int,
    noise_strength: float = 0.275,
    lambda_value: float,
    cost_bounds: tuple[float, float] | None,
    emission_bounds: tuple[float, float] | None,
    emission_factors: tuple[float, float] = (1.0, 1.0),
    strategy_2_mode: str = "paper_random_dv",
    strategy_2_seed: int | None = None,
) -> RepairResult:
    """
    Regret-k insertion with random perturbation of the regret score.
    """
    if k < 2:
        raise ValueError(
            "Perturbed regret requires k >= 2."
        )

    if noise_strength < 0:
        raise ValueError(
            "noise_strength must be nonnegative."
        )

    rng = random.Random(seed)
    repaired = state.copy()
    _canonicalize_inactive_routes(
        repaired,
        instance,
    )
    insertion_order: list[str] = []

    while repaired.unassigned_customers:
        customer_options: list[
            tuple[
                float,
                float,
                str,
                InsertionCandidate,
            ]
        ] = []

        candidate_counts: dict[str, int] = {}

        for customer_id in sorted(
            repaired.unassigned_customers
        ):
            candidates = enumerate_insertion_candidates(
                repaired,
                instance,
                customer_id,
                lambda_value=lambda_value,
                cost_bounds=cost_bounds,
                emission_bounds=emission_bounds,
                emission_factors=emission_factors,
                strategy_2_mode=strategy_2_mode,
                strategy_2_seed=strategy_2_seed,
            )

            candidate_counts[customer_id] = len(candidates)

            if not candidates:
                continue

            best = candidates[0]

            if len(candidates) >= k:
                regret = sum(
                    candidates[index].insertion_cost
                    - candidates[index - 1].insertion_cost
                    for index in range(1, k)
                )
            else:
                regret = 1_000_000.0 + (
                    candidates[-1].insertion_cost
                    - best.insertion_cost
                )

            perturbation = rng.uniform(
                -noise_strength,
                noise_strength,
            )
            perturbed_regret = regret * (
                1.0 + perturbation
            )

            customer_options.append(
                (
                    perturbed_regret,
                    -best.objective,
                    customer_id,
                    best,
                )
            )

        if not customer_options:
            raise RuntimeError(
                f"Perturbed regret-{k} found no feasible "
                f"candidate. Remaining customers: "
                f"{sorted(repaired.unassigned_customers)}. "
                f"Candidate counts: {candidate_counts}."
            )

        _, _, customer_id, selected = max(
            customer_options,
            key=lambda item: (
                item[0],
                item[1],
                item[2],
            ),
        )

        repaired = selected.state
        insertion_order.append(customer_id)

    return _finalize_repair(
        operator_name=f"perturbed_regret_{k}",
        state=repaired,
        insertion_order=insertion_order,
        instance=instance,
        lambda_value=lambda_value,
        cost_bounds=cost_bounds,
        emission_bounds=emission_bounds,
        emission_factors=emission_factors,
        metadata={
            "k": k,
            "seed": seed,
            "noise_strength": noise_strength,
        },
    )


def perturbed_regret_3_repair(
    state: ALNSSolutionState,
    instance: dict,
    **kwargs,
) -> RepairResult:
    return perturbed_regret_repair(
        state,
        instance,
        k=3,
        **kwargs,
    )
