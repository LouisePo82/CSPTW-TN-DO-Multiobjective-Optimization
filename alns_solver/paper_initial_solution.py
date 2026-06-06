from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
import random
from typing import Any

from alns_solver.solution_state import ALNSSolutionState
from alns_solver.repair_operators import (
    enumerate_insertion_candidates,
)


@dataclass
class InitialSolutionTrace:
    seed: int
    phase1_customer_order: list[str] = field(default_factory=list)
    phase1_assignments: list[dict[str, Any]] = field(default_factory=list)
    phase1_unassigned: list[str] = field(default_factory=list)

    active_tns_before_phase2: list[str] = field(default_factory=list)
    phase2_tn_results: list[dict[str, Any]] = field(default_factory=list)
    depot_fallbacks: list[dict[str, Any]] = field(default_factory=list)
    unassigned_fallbacks: list[dict[str, Any]] = field(default_factory=list)
    fixed_tn_positions: dict[str, dict[str, Any]] = field(default_factory=dict)

    phase3_initial_unassigned: list[str] = field(default_factory=list)
    phase3_insertion_order: list[str] = field(default_factory=list)

    final_validator_pass: bool = False
    final_validation_errors: list[str] = field(default_factory=list)


@dataclass
class PaperInitialSolutionResult:
    state: ALNSSolutionState
    trace: InitialSolutionTrace


def initialize_empty_state(
    instance: dict,
) -> ALNSSolutionState:
    return ALNSSolutionState(
        dv_routes={
            vehicle: []
            for vehicle in instance["dvs"]
        },
        od_routes={
            driver: []
            for driver in instance["ods"]
        },
        assignments={},
        unassigned_customers=set(
            instance["customers"]
        ),
    )


def _route_distance(
    route: list[str],
    distance: dict,
) -> float:
    if len(route) < 2:
        return 0.0

    return sum(
        float(
            distance[route[index]][route[index + 1]]
        )
        for index in range(len(route) - 1)
    )


def _best_customer_position(
    route: list[str],
    customer: str,
    distance: dict,
) -> tuple[int, float]:
    best_position = 2
    best_delta = float("inf")

    for position in range(2, len(route)):
        previous = route[position - 1]
        following = route[position]

        delta = (
            float(distance[previous][customer])
            + float(distance[customer][following])
            - float(distance[previous][following])
        )

        if delta < best_delta:
            best_position = position
            best_delta = delta

    return best_position, best_delta


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


def _od_route_time_feasible(
    state: ALNSSolutionState,
    instance: dict,
    driver: str,
    *,
    synchronized_tn_completion: dict[str, float] | None = None,
) -> bool:
    route = state.od_routes.get(driver, [])

    if not route:
        return True

    if len(route) < 3:
        return False

    vehicle = instance["vehicles"][driver]
    current_time = float(vehicle["earliest"])
    service_rate = float(
        instance["service_time_per_weight"]
    )

    for index in range(1, len(route)):
        previous = route[index - 1]
        node = route[index]

        current_time += float(
            instance["travel_time"][previous][node]
        )

        if index == 1:
            pickup = node

            if (
                synchronized_tn_completion is not None
                and pickup in synchronized_tn_completion
            ):
                current_time = max(
                    current_time,
                    synchronized_tn_completion[pickup],
                )

            pickup_demand = sum(
                float(
                    instance["nodes"][customer]["demand"]
                )
                for customer, assignment
                in state.assignments.items()
                if assignment.get("mode") == "OD_HOME"
                and assignment.get("driver") == driver
            )
            current_time += service_rate * pickup_demand

        elif node in instance["customers"]:
            tw_start = float(
                instance["nodes"][node]["tw_start"]
            )
            tw_end = float(
                instance["nodes"][node]["tw_end"]
            )

            current_time = max(
                current_time,
                tw_start,
            )

            if current_time > tw_end + 1e-9:
                return False

            current_time += (
                service_rate
                * float(
                    instance["nodes"][node]["demand"]
                )
            )

    return (
        current_time
        <= float(vehicle["latest"]) + 1e-9
    )


def _phase1_candidate_score(
    state: ALNSSolutionState,
    instance: dict,
    customer: str,
    driver: str,
    pickup: str,
) -> tuple[float, int]:
    info = instance["vehicles"][driver]
    current_route = list(
        state.od_routes.get(driver, [])
    )

    if current_route:
        if current_route[1] != pickup:
            return float("inf"), -1

        base_route = current_route
    else:
        base_route = [
            info["origin"],
            pickup,
            info["destination"],
        ]

    position, delta = _best_customer_position(
        base_route,
        customer,
        instance["distance"],
    )

    return (
        float(instance["rho"]) * delta,
        position,
    )


def phase1_assign_type1_to_ods(
    state: ALNSSolutionState,
    instance: dict,
    *,
    rng: random.Random,
    trace: InitialSolutionTrace,
) -> ALNSSolutionState:
    """
    Paper Algorithm 1 — Phase 1.

    Only Type-1 customers are processed.
    ODs may temporarily use depot or TN pickup points.
    No TN is inserted into a DV route in this phase.
    """
    result = state.copy()

    customer_order = list(instance["type1"])
    rng.shuffle(customer_order)

    trace.phase1_customer_order = list(
        customer_order
    )

    for customer in customer_order:
        candidates = []

        for driver in instance["ods"]:
            capacity = int(
                instance["vehicles"][driver]["capacity"]
            )

            if (
                _od_customer_count(result, driver)
                >= capacity
            ):
                continue

            existing_route = result.od_routes.get(
                driver,
                [],
            )

            pickups = (
                [existing_route[1]]
                if existing_route
                else list(instance["pickup_points"])
            )

            for pickup in pickups:
                score, position = _phase1_candidate_score(
                    result,
                    instance,
                    customer,
                    driver,
                    pickup,
                )

                if position < 0:
                    continue

                candidate_state = result.copy()
                info = instance["vehicles"][driver]

                if not existing_route:
                    candidate_state.od_routes[driver] = [
                        info["origin"],
                        pickup,
                        info["destination"],
                    ]

                candidate_state.od_routes[driver].insert(
                    position,
                    customer,
                )
                candidate_state.assign_customer(
                    customer,
                    {
                        "mode": "OD_HOME",
                        "driver": driver,
                        "pickup": pickup,
                    },
                )

                # Phase 1 checks the OD route itself only.
                # TN availability is deliberately deferred to Phase 2.
                if not _od_route_time_feasible(
                    candidate_state,
                    instance,
                    driver,
                ):
                    continue

                candidates.append(
                    (
                        score,
                        rng.random(),
                        driver,
                        pickup,
                        position,
                        candidate_state,
                    )
                )

        if not candidates:
            trace.phase1_unassigned.append(
                customer
            )
            continue

        selected = min(
            candidates,
            key=lambda item: (
                item[0],
                item[1],
                item[2],
                item[3],
                item[4],
            ),
        )

        (
            score,
            _,
            driver,
            pickup,
            position,
            result,
        ) = selected

        trace.phase1_assignments.append(
            {
                "customer": customer,
                "driver": driver,
                "pickup": pickup,
                "position": position,
                "score": score,
            }
        )

    result.register_operator_event(
        operator_type="construction",
        operator_name="paper_algorithm_1_phase_1",
        details={
            "customer_order": list(
                trace.phase1_customer_order
            ),
            "assignments": deepcopy(
                trace.phase1_assignments
            ),
            "unassigned": list(
                trace.phase1_unassigned
            ),
        },
    )

    return result


def _active_tns_from_assignments(
    state: ALNSSolutionState,
    instance: dict,
) -> list[str]:
    return sorted(
        {
            assignment["pickup"]
            for assignment in state.assignments.values()
            if assignment.get("mode") == "OD_HOME"
            and assignment.get("pickup")
            in instance["tns"]
        }
    )


def _tn_demand(
    state: ALNSSolutionState,
    instance: dict,
    tn: str,
) -> float:
    return sum(
        float(instance["nodes"][customer]["demand"])
        for customer, assignment in state.assignments.items()
        if assignment.get("mode") == "OD_HOME"
        and assignment.get("pickup") == tn
    )


def _empty_dv_candidates(
    state: ALNSSolutionState,
    instance: dict,
    tn: str,
) -> list[tuple[float, str, list[str]]]:
    demand = _tn_demand(
        state,
        instance,
        tn,
    )

    candidates = []

    for vehicle in instance["dvs"]:
        if state.dv_routes.get(vehicle, []):
            continue

        capacity = float(
            instance["vehicles"][vehicle]["capacity"]
        )

        if demand > capacity + 1e-9:
            continue

        route = [
            instance["start_depot"],
            tn,
            instance["end_depot"],
        ]

        distance = _route_distance(
            route,
            instance["distance"],
        )

        latest = float(
            instance["vehicles"][vehicle]["latest"]
        )

        route_time = (
            distance
            + float(
                instance["service_time_per_weight"]
            )
            * demand
        )

        if route_time > latest + 1e-9:
            continue

        candidates.append(
            (
                distance,
                vehicle,
                route,
            )
        )

    return sorted(candidates)


def _tn_completion_time(
    state: ALNSSolutionState,
    instance: dict,
    vehicle: str,
    tn: str,
) -> float:
    route = state.dv_routes[vehicle]
    current_time = float(
        instance["vehicles"][vehicle]["earliest"]
    )
    service_rate = float(
        instance["service_time_per_weight"]
    )

    for index in range(1, len(route)):
        previous = route[index - 1]
        node = route[index]

        current_time += float(
            instance["travel_time"][previous][node]
        )

        if node == tn:
            current_time += (
                service_rate
                * _tn_demand(
                    state,
                    instance,
                    tn,
                )
            )
            return current_time

    raise ValueError(
        f"{tn} not found in {vehicle} route."
    )


def _drivers_using_tn(
    state: ALNSSolutionState,
    tn: str,
) -> list[str]:
    return sorted(
        {
            assignment["driver"]
            for assignment in state.assignments.values()
            if assignment.get("mode") == "OD_HOME"
            and assignment.get("pickup") == tn
        }
    )


def _replace_driver_pickup(
    state: ALNSSolutionState,
    driver: str,
    new_pickup: str,
) -> None:
    route = list(state.od_routes[driver])

    if len(route) < 3:
        raise ValueError(
            f"Invalid OD route for {driver}: {route}"
        )

    route[1] = new_pickup
    state.od_routes[driver] = route

    for assignment in state.assignments.values():
        if (
            assignment.get("mode") == "OD_HOME"
            and assignment.get("driver") == driver
        ):
            assignment["pickup"] = new_pickup

    state.invalidate_cache()


def _unassign_driver_customers(
    state: ALNSSolutionState,
    driver: str,
) -> list[str]:
    customers = sorted(
        customer
        for customer, assignment in state.assignments.items()
        if assignment.get("mode") == "OD_HOME"
        and assignment.get("driver") == driver
    )

    for customer in customers:
        state.mark_customer_unassigned(
            customer
        )

    state.od_routes[driver] = []
    state.invalidate_cache()

    return customers


def phase2_stabilize_active_tns(
    state: ALNSSolutionState,
    instance: dict,
    *,
    trace: InitialSolutionTrace,
) -> ALNSSolutionState:
    """
    Paper Algorithm 1 — Phase 2.

    Each active TN is inserted into a separate empty DV route where feasible.
    If stabilization fails, affected ODs fall back to depot; if depot is also
    infeasible, their customers return to the unassigned pool.
    """
    result = state.copy()

    active_tns = _active_tns_from_assignments(
        result,
        instance,
    )

    trace.active_tns_before_phase2 = list(
        active_tns
    )

    for tn in active_tns:
        dv_candidates = _empty_dv_candidates(
            result,
            instance,
            tn,
        )

        stabilized = False

        for _, vehicle, route in dv_candidates:
            candidate_state = result.copy()
            candidate_state.dv_routes[vehicle] = list(
                route
            )

            completion = _tn_completion_time(
                candidate_state,
                instance,
                vehicle,
                tn,
            )

            synchronized = {
                tn: completion,
            }

            affected_drivers = _drivers_using_tn(
                candidate_state,
                tn,
            )

            if all(
                _od_route_time_feasible(
                    candidate_state,
                    instance,
                    driver,
                    synchronized_tn_completion=(
                        synchronized
                    ),
                )
                for driver in affected_drivers
            ):
                result = candidate_state
                fixed_position = (
                    result.dv_routes[vehicle].index(tn)
                )

                trace.fixed_tn_positions[tn] = {
                    "vehicle": vehicle,
                    "position": fixed_position,
                }

                trace.phase2_tn_results.append(
                    {
                        "tn": tn,
                        "status": "inserted",
                        "vehicle": vehicle,
                        "position": fixed_position,
                        "completion_time": completion,
                    }
                )

                stabilized = True
                break

        if stabilized:
            continue

        for driver in _drivers_using_tn(
            result,
            tn,
        ):
            depot_candidate = result.copy()
            _replace_driver_pickup(
                depot_candidate,
                driver,
                instance["start_depot"],
            )

            if _od_route_time_feasible(
                depot_candidate,
                instance,
                driver,
            ):
                result = depot_candidate

                trace.depot_fallbacks.append(
                    {
                        "tn": tn,
                        "driver": driver,
                        "new_pickup": (
                            instance["start_depot"]
                        ),
                    }
                )
            else:
                removed = (
                    _unassign_driver_customers(
                        result,
                        driver,
                    )
                )

                trace.unassigned_fallbacks.append(
                    {
                        "tn": tn,
                        "driver": driver,
                        "customers": removed,
                    }
                )

        trace.phase2_tn_results.append(
            {
                "tn": tn,
                "status": "fallback",
            }
        )

    result.register_operator_event(
        operator_type="construction",
        operator_name="paper_algorithm_1_phase_2",
        details={
            "active_tns": list(
                trace.active_tns_before_phase2
            ),
            "tn_results": deepcopy(
                trace.phase2_tn_results
            ),
            "depot_fallbacks": deepcopy(
                trace.depot_fallbacks
            ),
            "unassigned_fallbacks": deepcopy(
                trace.unassigned_fallbacks
            ),
            "fixed_tn_positions": deepcopy(
                trace.fixed_tn_positions
            ),
        },
    )

    return result


def _fixed_tns_preserved(
    candidate_state: ALNSSolutionState,
    fixed_tn_positions: dict[str, dict[str, Any]],
) -> bool:
    for tn, info in fixed_tn_positions.items():
        vehicle = info["vehicle"]
        position = int(info["position"])
        route = candidate_state.dv_routes.get(
            vehicle,
            [],
        )

        if (
            len(route) <= position
            or route[position] != tn
        ):
            return False

    return True


def phase3_insert_remaining_customers(
    state: ALNSSolutionState,
    instance: dict,
    *,
    trace: InitialSolutionTrace,
    lambda_value: float = 0.0,
    cost_bounds: tuple[float, float] | None = None,
    emission_bounds: tuple[float, float] | None = None,
    emission_factors: tuple[float, float] = (3.0, 1.0),
    strategy_2_mode: str = "paper_random_dv",
    strategy_2_seed: int | None = None,
) -> ALNSSolutionState:
    """
    Paper Algorithm 1 — Phase 3.

    Best-insert all remaining customers while preserving the fixed TN
    positions established by Phase 2.
    """
    result = state.copy()

    trace.phase3_initial_unassigned = sorted(
        result.unassigned_customers
    )

    while result.unassigned_customers:
        all_candidates = []

        for customer in sorted(
            result.unassigned_customers
        ):
            candidates = enumerate_insertion_candidates(
                result,
                instance,
                customer,
                lambda_value=lambda_value,
                cost_bounds=cost_bounds,
                emission_bounds=emission_bounds,
                emission_factors=emission_factors,
                strategy_2_mode=strategy_2_mode,
                strategy_2_seed=strategy_2_seed,
            )

            candidates = [
                candidate
                for candidate in candidates
                if _fixed_tns_preserved(
                    candidate.state,
                    trace.fixed_tn_positions,
                )
            ]

            all_candidates.extend(
                candidates
            )

        if not all_candidates:
            raise RuntimeError(
                "Phase 3 could not find a feasible insertion "
                f"for remaining customers: "
                f"{sorted(result.unassigned_customers)}"
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

        result = selected.state

        trace.phase3_insertion_order.append(
            selected.customer_id
        )

    result.register_operator_event(
        operator_type="construction",
        operator_name="paper_algorithm_1_phase_3",
        details={
            "initial_unassigned": list(
                trace.phase3_initial_unassigned
            ),
            "insertion_order": list(
                trace.phase3_insertion_order
            ),
            "fixed_tn_positions": deepcopy(
                trace.fixed_tn_positions
            ),
        },
    )

    return result


def construct_paper_initial_solution(
    instance: dict,
    *,
    seed: int = 42,
    lambda_value: float = 0.0,
    cost_bounds: tuple[float, float] | None = None,
    emission_bounds: tuple[float, float] | None = None,
    emission_factors: tuple[float, float] = (3.0, 1.0),
    strategy_2_mode: str = "paper_random_dv",
) -> PaperInitialSolutionResult:
    rng = random.Random(seed)

    trace = InitialSolutionTrace(
        seed=seed,
    )

    state = initialize_empty_state(
        instance
    )

    state = phase1_assign_type1_to_ods(
        state,
        instance,
        rng=rng,
        trace=trace,
    )

    state = phase2_stabilize_active_tns(
        state,
        instance,
        trace=trace,
    )

    state = phase3_insert_remaining_customers(
        state,
        instance,
        trace=trace,
        lambda_value=lambda_value,
        cost_bounds=cost_bounds,
        emission_bounds=emission_bounds,
        emission_factors=emission_factors,
        strategy_2_mode=strategy_2_mode,
        strategy_2_seed=seed,
    )

    solution = state.to_core_solution(
        instance=instance,
        lambda_value=lambda_value,
        objective_mode="weighted",
        cost_bounds=cost_bounds,
        emission_bounds=emission_bounds,
        emission_factors=emission_factors,
        require_complete=True,
        metadata={
            "construction_mode": (
                "paper_algorithm_1"
            ),
            "seed": seed,
        },
    )

    trace.final_validator_pass = bool(
        solution.validator_pass
    )
    trace.final_validation_errors = list(
        solution.validation_errors
    )

    if not solution.validator_pass:
        raise RuntimeError(
            "Paper initial solution failed shared validator: "
            f"{solution.validation_errors}"
        )

    return PaperInitialSolutionResult(
        state=state,
        trace=trace,
    )
