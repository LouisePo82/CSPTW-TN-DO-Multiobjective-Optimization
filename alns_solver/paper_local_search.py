from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from alns_solver.solution_state import ALNSSolutionState
from alns_solver.repair_operators import enumerate_insertion_candidates

EPSILON = 1e-10


@dataclass
class LocalSearchMoveResult:
    operator_name: str
    state: ALNSSolutionState
    improved: bool
    base_objective: float
    final_objective: float
    details: dict[str, Any]


def _objective(state, instance, *, lambda_value, cost_bounds, emission_bounds, emission_factors):
    return float(state.evaluate(
        instance=instance,
        lambda_value=lambda_value,
        cost_bounds=cost_bounds,
        emission_bounds=emission_bounds,
        emission_factors=emission_factors,
    )["objective"])


def _validated_objective(state, instance, *, lambda_value, cost_bounds, emission_bounds, emission_factors):
    solution = state.to_core_solution(
        instance=instance,
        lambda_value=lambda_value,
        objective_mode="weighted",
        cost_bounds=cost_bounds,
        emission_bounds=emission_bounds,
        emission_factors=emission_factors,
        require_complete=True,
    )
    return float(solution.objective) if solution.validator_pass else None


def _canonicalize_empty_routes(state, instance):
    for vehicle in instance["dvs"]:
        if state.dv_routes.get(vehicle, []) == [instance["start_depot"], instance["end_depot"]]:
            state.dv_routes[vehicle] = []
    for driver in instance["ods"]:
        has_customers = any(
            a.get("mode") == "OD_HOME" and a.get("driver") == driver
            for a in state.assignments.values()
        )
        if not has_customers:
            state.od_routes[driver] = []
    state.invalidate_cache()


def _cleanup_orphan_tns(state, instance):
    active_tns = {
        a["pickup"] for a in state.assignments.values()
        if a.get("mode") == "OD_HOME" and a.get("pickup") in instance["tns"]
    }
    for vehicle in instance["dvs"]:
        state.dv_routes[vehicle] = [
            n for n in state.dv_routes.get(vehicle, [])
            if n not in instance["tns"] or n in active_tns
        ]
    _canonicalize_empty_routes(state, instance)


def _detach_customer(state, instance, customer):
    detached = state.copy()
    assignment = detached.assignments.get(customer)
    if assignment is None:
        raise ValueError(f"Customer {customer} is not assigned.")
    mode = assignment.get("mode")
    if mode == "DV_HOME":
        vehicle = assignment["vehicle"]
        detached.dv_routes[vehicle] = [n for n in detached.dv_routes.get(vehicle, []) if n != customer]
    elif mode == "OD_HOME":
        driver = assignment["driver"]
        detached.od_routes[driver] = [n for n in detached.od_routes.get(driver, []) if n != customer]
    else:
        raise ValueError("LS-1 cross-fleet moves only support home-delivery customers.")
    detached.assignments.pop(customer)
    detached.unassigned_customers.add(customer)
    detached.invalidate_cache()
    _cleanup_orphan_tns(detached, instance)
    return detached


def _no_improvement(name, state, base_objective):
    return LocalSearchMoveResult(name, state.copy(), False, base_objective, base_objective, {})


def move_intra_classic_classic(state, instance, *, lambda_value, cost_bounds, emission_bounds, emission_factors=(1.0, 1.0)):
    name = "move_intra_classic_classic"
    base = _objective(state, instance, lambda_value=lambda_value, cost_bounds=cost_bounds,
                      emission_bounds=emission_bounds, emission_factors=emission_factors)
    for vehicle in sorted(instance["dvs"]):
        route = list(state.dv_routes.get(vehicle, []))
        if len(route) < 4:
            continue
        for source in range(1, len(route) - 1):
            node = route[source]
            eligible = node in instance["tns"] or (
                node in instance["customers"]
                and state.assignments.get(node, {}).get("mode") == "DV_HOME"
                and state.assignments[node].get("vehicle") == vehicle
            )
            if not eligible:
                continue
            reduced = route[:source] + route[source + 1:]
            for target in range(1, len(reduced)):
                candidate_route = list(reduced)
                candidate_route.insert(target, node)
                if candidate_route == route:
                    continue
                candidate = state.copy()
                candidate.dv_routes[vehicle] = candidate_route
                candidate.invalidate_cache()
                obj = _validated_objective(candidate, instance, lambda_value=lambda_value,
                                           cost_bounds=cost_bounds, emission_bounds=emission_bounds,
                                           emission_factors=emission_factors)
                if obj is not None and obj < base - EPSILON:
                    details = {"vehicle": vehicle, "node": node, "source_position": source,
                               "target_position": target, "selection": "first_improvement"}
                    candidate.register_operator_event("local_search", name, details)
                    return LocalSearchMoveResult(name, candidate, True, base, obj, details)
    return _no_improvement(name, state, base)


def move_inter_classic_classic(state, instance, *, lambda_value, cost_bounds, emission_bounds, emission_factors=(1.0, 1.0)):
    name = "move_inter_classic_classic"
    base = _objective(state, instance, lambda_value=lambda_value, cost_bounds=cost_bounds,
                      emission_bounds=emission_bounds, emission_factors=emission_factors)
    for source_vehicle in sorted(instance["dvs"]):
        source_route = list(state.dv_routes.get(source_vehicle, []))
        if len(source_route) < 3:
            continue
        for source in range(1, len(source_route) - 1):
            node = source_route[source]
            is_customer = (
                node in instance["customers"]
                and state.assignments.get(node, {}).get("mode") == "DV_HOME"
                and state.assignments[node].get("vehicle") == source_vehicle
            )
            is_tn = node in instance["tns"]
            if not (is_customer or is_tn):
                continue
            for target_vehicle in sorted(instance["dvs"]):
                if target_vehicle == source_vehicle:
                    continue
                target_route = list(state.dv_routes.get(target_vehicle, [])) or [
                    instance["start_depot"], instance["end_depot"]
                ]
                for target in range(1, len(target_route)):
                    candidate = state.copy()
                    candidate.dv_routes[source_vehicle] = [
                        n for idx, n in enumerate(source_route) if idx != source
                    ]
                    candidate.dv_routes[target_vehicle] = list(target_route)
                    candidate.dv_routes[target_vehicle].insert(target, node)
                    if is_customer:
                        candidate.assignments[node] = {"mode": "DV_HOME", "vehicle": target_vehicle}
                    _canonicalize_empty_routes(candidate, instance)
                    candidate.invalidate_cache()
                    obj = _validated_objective(candidate, instance, lambda_value=lambda_value,
                                               cost_bounds=cost_bounds, emission_bounds=emission_bounds,
                                               emission_factors=emission_factors)
                    if obj is not None and obj < base - EPSILON:
                        details = {"node": node, "source_vehicle": source_vehicle,
                                   "target_vehicle": target_vehicle, "source_position": source,
                                   "target_position": target, "selection": "first_improvement"}
                        candidate.register_operator_event("local_search", name, details)
                        return LocalSearchMoveResult(name, candidate, True, base, obj, details)
    return _no_improvement(name, state, base)


def _cross_fleet_candidates(detached, instance, customer, *, required_mode, lambda_value,
                            cost_bounds, emission_bounds, emission_factors, strategy_2_seed):
    candidates = enumerate_insertion_candidates(
        detached, instance, customer,
        lambda_value=lambda_value,
        cost_bounds=cost_bounds,
        emission_bounds=emission_bounds,
        emission_factors=emission_factors,
        strategy_2_mode="paper_random_dv",
        strategy_2_seed=strategy_2_seed,
    )
    filtered = [c for c in candidates if c.mode == required_mode]
    return sorted(filtered, key=lambda c: (
        str(c.details.get("driver", "")), str(c.details.get("pickup", "")),
        str(c.details.get("vehicle", "")), int(c.details.get("position", 0)),
        int(c.details.get("dv_position", 0)), str(c.details)
    ))


def move_inter_classic_crowd(state, instance, *, lambda_value, cost_bounds, emission_bounds,
                             emission_factors=(1.0, 1.0), strategy_2_seed=None):
    name = "move_inter_classic_crowd"
    base = _objective(state, instance, lambda_value=lambda_value, cost_bounds=cost_bounds,
                      emission_bounds=emission_bounds, emission_factors=emission_factors)
    customers = sorted(c for c, a in state.assignments.items()
                       if int(instance["nodes"][c]["customer_type"]) == 1
                       and a.get("mode") == "DV_HOME")
    for customer in customers:
        detached = _detach_customer(state, instance, customer)
        candidates = _cross_fleet_candidates(
            detached, instance, customer, required_mode="OD_HOME",
            lambda_value=lambda_value, cost_bounds=cost_bounds,
            emission_bounds=emission_bounds, emission_factors=emission_factors,
            strategy_2_seed=strategy_2_seed,
        )
        for candidate in candidates:
            if candidate.objective < base - EPSILON:
                details = {"customer": customer, **candidate.details, "selection": "first_improvement"}
                candidate.state.register_operator_event("local_search", name, details)
                return LocalSearchMoveResult(name, candidate.state, True, base,
                                             float(candidate.objective), details)
    return _no_improvement(name, state, base)


def move_inter_crowd_classic(state, instance, *, lambda_value, cost_bounds, emission_bounds,
                             emission_factors=(1.0, 1.0)):
    name = "move_inter_crowd_classic"
    base = _objective(state, instance, lambda_value=lambda_value, cost_bounds=cost_bounds,
                      emission_bounds=emission_bounds, emission_factors=emission_factors)
    customers = sorted(c for c, a in state.assignments.items()
                       if int(instance["nodes"][c]["customer_type"]) == 1
                       and a.get("mode") == "OD_HOME")
    for customer in customers:
        detached = _detach_customer(state, instance, customer)
        candidates = _cross_fleet_candidates(
            detached, instance, customer, required_mode="DV_HOME",
            lambda_value=lambda_value, cost_bounds=cost_bounds,
            emission_bounds=emission_bounds, emission_factors=emission_factors,
            strategy_2_seed=None,
        )
        for candidate in candidates:
            if candidate.objective < base - EPSILON:
                improved = candidate.state
                _cleanup_orphan_tns(improved, instance)
                details = {"customer": customer, **candidate.details, "selection": "first_improvement"}
                improved.register_operator_event("local_search", name, details)
                return LocalSearchMoveResult(name, improved, True, base,
                                             float(candidate.objective), details)
    return _no_improvement(name, state, base)

# =============================================================
# Local Search Fidelity LS-2 — Swap Operators
# =============================================================

def _paper_dv_swap_node(
    state: ALNSSolutionState,
    instance: dict,
    vehicle: str,
    node: str,
) -> bool:
    if node in instance["tns"]:
        return True

    return (
        node in instance["customers"]
        and state.assignments.get(node, {}).get("mode")
        == "DV_HOME"
        and state.assignments[node].get("vehicle")
        == vehicle
    )


def _paper_type1_dv_customer(
    state: ALNSSolutionState,
    instance: dict,
    vehicle: str,
    customer: str,
) -> bool:
    return (
        customer in instance["customers"]
        and int(
            instance["nodes"][customer]["customer_type"]
        )
        == 1
        and state.assignments.get(customer, {}).get("mode")
        == "DV_HOME"
        and state.assignments[customer].get("vehicle")
        == vehicle
    )


def _paper_type1_od_customer(
    state: ALNSSolutionState,
    instance: dict,
    driver: str,
    customer: str,
) -> bool:
    return (
        customer in instance["customers"]
        and int(
            instance["nodes"][customer]["customer_type"]
        )
        == 1
        and state.assignments.get(customer, {}).get("mode")
        == "OD_HOME"
        and state.assignments[customer].get("driver")
        == driver
    )


def swap_intra_classic_classic(
    state: ALNSSolutionState,
    instance: dict,
    *,
    lambda_value: float,
    cost_bounds: tuple[float, float] | None,
    emission_bounds: tuple[float, float] | None,
    emission_factors: tuple[float, float] = (1.0, 1.0),
) -> LocalSearchMoveResult:
    """
    First-improvement swap of two paper-eligible nodes within one DV route.

    Paper-mode eligible nodes are DV-home customers and TNs. ADP nodes are
    deliberately excluded because grouped ADP reassignment semantics are not
    explicitly defined for this neighborhood.
    """
    name = "swap_intra_classic_classic"
    base = _objective(
        state,
        instance,
        lambda_value=lambda_value,
        cost_bounds=cost_bounds,
        emission_bounds=emission_bounds,
        emission_factors=emission_factors,
    )

    for vehicle in sorted(instance["dvs"]):
        route = list(state.dv_routes.get(vehicle, []))

        for first in range(1, len(route) - 1):
            first_node = route[first]

            if not _paper_dv_swap_node(
                state,
                instance,
                vehicle,
                first_node,
            ):
                continue

            for second in range(first + 1, len(route) - 1):
                second_node = route[second]

                if not _paper_dv_swap_node(
                    state,
                    instance,
                    vehicle,
                    second_node,
                ):
                    continue

                candidate = state.copy()
                candidate_route = list(route)
                candidate_route[first], candidate_route[second] = (
                    candidate_route[second],
                    candidate_route[first],
                )
                candidate.dv_routes[vehicle] = candidate_route
                candidate.invalidate_cache()

                objective_value = _validated_objective(
                    candidate,
                    instance,
                    lambda_value=lambda_value,
                    cost_bounds=cost_bounds,
                    emission_bounds=emission_bounds,
                    emission_factors=emission_factors,
                )

                if (
                    objective_value is not None
                    and objective_value < base - EPSILON
                ):
                    details = {
                        "vehicle": vehicle,
                        "first_node": first_node,
                        "second_node": second_node,
                        "first_position": first,
                        "second_position": second,
                        "selection": "first_improvement",
                    }
                    candidate.register_operator_event(
                        "local_search",
                        name,
                        details,
                    )
                    return LocalSearchMoveResult(
                        name,
                        candidate,
                        True,
                        base,
                        objective_value,
                        details,
                    )

    return _no_improvement(name, state, base)


def swap_inter_classic_classic(
    state: ALNSSolutionState,
    instance: dict,
    *,
    lambda_value: float,
    cost_bounds: tuple[float, float] | None,
    emission_bounds: tuple[float, float] | None,
    emission_factors: tuple[float, float] = (1.0, 1.0),
) -> LocalSearchMoveResult:
    """
    First-improvement swap between two different DV routes.

    Eligible nodes are DV-home customers and TNs. Customer vehicle ownership
    is updated when a customer crosses routes.
    """
    name = "swap_inter_classic_classic"
    base = _objective(
        state,
        instance,
        lambda_value=lambda_value,
        cost_bounds=cost_bounds,
        emission_bounds=emission_bounds,
        emission_factors=emission_factors,
    )

    vehicles = sorted(instance["dvs"])

    for source_index, first_vehicle in enumerate(vehicles):
        first_route = list(
            state.dv_routes.get(first_vehicle, [])
        )

        for second_vehicle in vehicles[source_index + 1 :]:
            second_route = list(
                state.dv_routes.get(second_vehicle, [])
            )

            for first_position in range(
                1,
                len(first_route) - 1,
            ):
                first_node = first_route[first_position]

                if not _paper_dv_swap_node(
                    state,
                    instance,
                    first_vehicle,
                    first_node,
                ):
                    continue

                for second_position in range(
                    1,
                    len(second_route) - 1,
                ):
                    second_node = second_route[second_position]

                    if not _paper_dv_swap_node(
                        state,
                        instance,
                        second_vehicle,
                        second_node,
                    ):
                        continue

                    candidate = state.copy()
                    candidate.dv_routes[first_vehicle][first_position] = (
                        second_node
                    )
                    candidate.dv_routes[second_vehicle][second_position] = (
                        first_node
                    )

                    if first_node in instance["customers"]:
                        candidate.assignments[first_node] = {
                            "mode": "DV_HOME",
                            "vehicle": second_vehicle,
                        }

                    if second_node in instance["customers"]:
                        candidate.assignments[second_node] = {
                            "mode": "DV_HOME",
                            "vehicle": first_vehicle,
                        }

                    candidate.invalidate_cache()

                    objective_value = _validated_objective(
                        candidate,
                        instance,
                        lambda_value=lambda_value,
                        cost_bounds=cost_bounds,
                        emission_bounds=emission_bounds,
                        emission_factors=emission_factors,
                    )

                    if (
                        objective_value is not None
                        and objective_value < base - EPSILON
                    ):
                        details = {
                            "first_vehicle": first_vehicle,
                            "second_vehicle": second_vehicle,
                            "first_node": first_node,
                            "second_node": second_node,
                            "first_position": first_position,
                            "second_position": second_position,
                            "selection": "first_improvement",
                        }
                        candidate.register_operator_event(
                            "local_search",
                            name,
                            details,
                        )
                        return LocalSearchMoveResult(
                            name,
                            candidate,
                            True,
                            base,
                            objective_value,
                            details,
                        )

    return _no_improvement(name, state, base)


def swap_inter_classic_crowd(
    state: ALNSSolutionState,
    instance: dict,
    *,
    lambda_value: float,
    cost_bounds: tuple[float, float] | None,
    emission_bounds: tuple[float, float] | None,
    emission_factors: tuple[float, float] = (1.0, 1.0),
) -> LocalSearchMoveResult:
    """
    First-improvement swap of one Type-1 DV-home customer and one Type-1
    OD-home customer.

    The customers exchange route positions and delivery modes. The OD pickup
    point remains the pickup point of the receiving OD route.
    """
    name = "swap_inter_classic_crowd"
    base = _objective(
        state,
        instance,
        lambda_value=lambda_value,
        cost_bounds=cost_bounds,
        emission_bounds=emission_bounds,
        emission_factors=emission_factors,
    )

    for vehicle in sorted(instance["dvs"]):
        dv_route = list(state.dv_routes.get(vehicle, []))

        for dv_position in range(1, len(dv_route) - 1):
            dv_customer = dv_route[dv_position]

            if not _paper_type1_dv_customer(
                state,
                instance,
                vehicle,
                dv_customer,
            ):
                continue

            for driver in sorted(instance["ods"]):
                od_route = list(state.od_routes.get(driver, []))

                if not od_route:
                    continue

                pickup = od_route[1]

                for od_position in range(2, len(od_route) - 1):
                    od_customer = od_route[od_position]

                    if not _paper_type1_od_customer(
                        state,
                        instance,
                        driver,
                        od_customer,
                    ):
                        continue

                    candidate = state.copy()
                    candidate.dv_routes[vehicle][dv_position] = (
                        od_customer
                    )
                    candidate.od_routes[driver][od_position] = (
                        dv_customer
                    )

                    candidate.assignments[dv_customer] = {
                        "mode": "OD_HOME",
                        "driver": driver,
                        "pickup": pickup,
                    }
                    candidate.assignments[od_customer] = {
                        "mode": "DV_HOME",
                        "vehicle": vehicle,
                    }
                    candidate.invalidate_cache()

                    objective_value = _validated_objective(
                        candidate,
                        instance,
                        lambda_value=lambda_value,
                        cost_bounds=cost_bounds,
                        emission_bounds=emission_bounds,
                        emission_factors=emission_factors,
                    )

                    if (
                        objective_value is not None
                        and objective_value < base - EPSILON
                    ):
                        details = {
                            "vehicle": vehicle,
                            "driver": driver,
                            "dv_customer": dv_customer,
                            "od_customer": od_customer,
                            "dv_position": dv_position,
                            "od_position": od_position,
                            "pickup": pickup,
                            "selection": "first_improvement",
                        }
                        candidate.register_operator_event(
                            "local_search",
                            name,
                            details,
                        )
                        return LocalSearchMoveResult(
                            name,
                            candidate,
                            True,
                            base,
                            objective_value,
                            details,
                        )

    return _no_improvement(name, state, base)


def swap_inter_crowd_crowd(
    state: ALNSSolutionState,
    instance: dict,
    *,
    lambda_value: float,
    cost_bounds: tuple[float, float] | None,
    emission_bounds: tuple[float, float] | None,
    emission_factors: tuple[float, float] = (1.0, 1.0),
) -> LocalSearchMoveResult:
    """
    First-improvement swap of two Type-1 customers on different OD routes.

    Each customer adopts the receiving driver's pickup point and driver.
    """
    name = "swap_inter_crowd_crowd"
    base = _objective(
        state,
        instance,
        lambda_value=lambda_value,
        cost_bounds=cost_bounds,
        emission_bounds=emission_bounds,
        emission_factors=emission_factors,
    )

    drivers = sorted(instance["ods"])

    for first_index, first_driver in enumerate(drivers):
        first_route = list(
            state.od_routes.get(first_driver, [])
        )

        if not first_route:
            continue

        first_pickup = first_route[1]

        for second_driver in drivers[first_index + 1 :]:
            second_route = list(
                state.od_routes.get(second_driver, [])
            )

            if not second_route:
                continue

            second_pickup = second_route[1]

            for first_position in range(
                2,
                len(first_route) - 1,
            ):
                first_customer = first_route[first_position]

                if not _paper_type1_od_customer(
                    state,
                    instance,
                    first_driver,
                    first_customer,
                ):
                    continue

                for second_position in range(
                    2,
                    len(second_route) - 1,
                ):
                    second_customer = second_route[second_position]

                    if not _paper_type1_od_customer(
                        state,
                        instance,
                        second_driver,
                        second_customer,
                    ):
                        continue

                    candidate = state.copy()
                    candidate.od_routes[first_driver][first_position] = (
                        second_customer
                    )
                    candidate.od_routes[second_driver][second_position] = (
                        first_customer
                    )

                    candidate.assignments[first_customer] = {
                        "mode": "OD_HOME",
                        "driver": second_driver,
                        "pickup": second_pickup,
                    }
                    candidate.assignments[second_customer] = {
                        "mode": "OD_HOME",
                        "driver": first_driver,
                        "pickup": first_pickup,
                    }
                    candidate.invalidate_cache()

                    objective_value = _validated_objective(
                        candidate,
                        instance,
                        lambda_value=lambda_value,
                        cost_bounds=cost_bounds,
                        emission_bounds=emission_bounds,
                        emission_factors=emission_factors,
                    )

                    if (
                        objective_value is not None
                        and objective_value < base - EPSILON
                    ):
                        details = {
                            "first_driver": first_driver,
                            "second_driver": second_driver,
                            "first_customer": first_customer,
                            "second_customer": second_customer,
                            "first_position": first_position,
                            "second_position": second_position,
                            "selection": "first_improvement",
                        }
                        candidate.register_operator_event(
                            "local_search",
                            name,
                            details,
                        )
                        return LocalSearchMoveResult(
                            name,
                            candidate,
                            True,
                            base,
                            objective_value,
                            details,
                        )

    return _no_improvement(name, state, base)
