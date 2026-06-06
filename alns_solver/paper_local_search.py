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
