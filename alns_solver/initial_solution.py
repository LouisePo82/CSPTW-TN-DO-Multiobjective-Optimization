from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from alns_solver.solution_state import ALNSSolutionState


@dataclass
class Candidate:
    score: float
    apply: Callable[[ALNSSolutionState], None]
    description: str


def _best_position(route: list[str], node: str, distance: dict) -> tuple[int, float]:
    best_pos, best_delta = 1, float("inf")
    for pos in range(1, len(route)):
        i, j = route[pos - 1], route[pos]
        delta = distance[i][node] + distance[node][j] - distance[i][j]
        if delta < best_delta:
            best_pos, best_delta = pos, delta
    return best_pos, best_delta


def _ensure_dv_route(
    state: ALNSSolutionState, instance: dict, vehicle: str
) -> list[str]:
    if not state.dv_routes.get(vehicle):
        state.dv_routes[vehicle] = [instance["start_depot"], instance["end_depot"]]
    return state.dv_routes[vehicle]


def _ensure_od_route(
    state: ALNSSolutionState, instance: dict, driver: str, pickup: str
) -> list[str]:
    route = state.od_routes.get(driver, [])
    if route:
        if route[1] != pickup:
            raise ValueError(f"{driver} already uses pickup {route[1]}")
        return route
    info = instance["vehicles"][driver]
    route = [info["origin"], pickup, info["destination"]]
    state.od_routes[driver] = route
    return route


def _dv_load(state: ALNSSolutionState, instance: dict, vehicle: str) -> float:
    return state._compute_vehicle_loads(instance).get(vehicle, 0.0)


def _od_count(state: ALNSSolutionState, driver: str) -> int:
    return sum(
        1
        for a in state.assignments.values()
        if a.get("mode") == "OD_HOME" and a.get("driver") == driver
    )


def _insert_node_once(route: list[str], node: str, distance: dict) -> float:
    if node in route:
        return 0.0
    pos, delta = _best_position(route, node, distance)
    route.insert(pos, node)
    return delta


def _build_candidates(
    state: ALNSSolutionState,
    instance: dict,
    customer: str,
    rng: random.Random,
) -> list[Candidate]:
    nodes = instance["nodes"]
    distance = instance["distance"]
    demand = nodes[customer]["demand"]
    ctype = nodes[customer]["customer_type"]
    candidates: list[Candidate] = []

    # Home delivery by DV: available to Type 1 and Type 3.
    if ctype in {1, 3}:
        for vehicle in instance["dvs"]:
            if (
                _dv_load(state, instance, vehicle) + demand
                > instance["vehicles"][vehicle]["capacity"] + 1e-9
            ):
                continue
            route = state.dv_routes.get(vehicle) or [
                instance["start_depot"],
                instance["end_depot"],
            ]
            pos, delta = _best_position(route, customer, distance)

            def apply_dv(s, v=vehicle, p=pos):
                r = _ensure_dv_route(s, instance, v)
                r.insert(p, customer)
                s.assign_customer(customer, {"mode": "DV_HOME", "vehicle": v})

            candidates.append(Candidate(delta, apply_dv, f"DV_HOME:{vehicle}"))

    # ADP delivery: mandatory for Type 2, optional for Type 3.
    if ctype in {2, 3}:
        for adp in instance["adps"]:
            if instance["gamma"].get((customer, adp), 0) != 1:
                continue
            for vehicle in instance["dvs"]:
                if (
                    _dv_load(state, instance, vehicle) + demand
                    > instance["vehicles"][vehicle]["capacity"] + 1e-9
                ):
                    continue
                route = state.dv_routes.get(vehicle) or [
                    instance["start_depot"],
                    instance["end_depot"],
                ]
                if adp in route:
                    delta = 0.0
                    pos = None
                else:
                    pos, delta = _best_position(route, adp, distance)

                def apply_adp(s, v=vehicle, a=adp, p=pos):
                    r = _ensure_dv_route(s, instance, v)
                    if a not in r:
                        r.insert(p, a)
                    s.assign_customer(customer, {"mode": "ADP", "vehicle": v, "adp": a})

                candidates.append(Candidate(delta, apply_adp, f"ADP:{vehicle}:{adp}"))

    # Home delivery by OD: available to Type 1 and Type 3.
    if ctype in {1, 3}:
        for driver in instance["ods"]:
            if _od_count(state, driver) >= instance["vehicles"][driver]["capacity"]:
                continue
            current = state.od_routes.get(driver, [])
            pickups = [current[1]] if current else list(instance["pickup_points"])
            for pickup in pickups:
                od_route = current or [
                    instance["vehicles"][driver]["origin"],
                    pickup,
                    instance["vehicles"][driver]["destination"],
                ]
                pos, od_delta = _best_position(od_route, customer, distance)
                extra_dv_delta = 0.0
                tn_vehicle = None
                tn_pos = None

                if pickup in instance["tns"]:
                    feasible_dvs = []
                    for vehicle in instance["dvs"]:
                        if (
                            _dv_load(state, instance, vehicle) + demand
                            > instance["vehicles"][vehicle]["capacity"] + 1e-9
                        ):
                            continue
                        dv_route = state.dv_routes.get(vehicle) or [
                            instance["start_depot"],
                            instance["end_depot"],
                        ]
                        if pickup in dv_route:
                            p, delta = None, 0.0
                        else:
                            p, delta = _best_position(dv_route, pickup, distance)
                        feasible_dvs.append((delta, vehicle, p))
                    if not feasible_dvs:
                        continue
                    extra_dv_delta, tn_vehicle, tn_pos = min(feasible_dvs)

                # Approximate original cost increment; rho discounts OD detour.
                score = extra_dv_delta + instance["rho"] * od_delta

                def apply_od(s, d=driver, pp=pickup, p=pos, tv=tn_vehicle, tp=tn_pos):
                    if pp in instance["tns"]:
                        r_dv = _ensure_dv_route(s, instance, tv)
                        if pp not in r_dv:
                            r_dv.insert(tp, pp)
                    r_od = _ensure_od_route(s, instance, d, pp)
                    r_od.insert(p, customer)
                    s.assign_customer(
                        customer, {"mode": "OD_HOME", "driver": d, "pickup": pp}
                    )

                candidates.append(
                    Candidate(score, apply_od, f"OD_HOME:{driver}:{pickup}")
                )

    # Random tie noise changes construction without overwhelming greedy logic.
    for candidate in candidates:
        candidate.score += rng.random() * 1e-6
    return candidates


def _construct_once(instance: dict, seed: int) -> ALNSSolutionState:
    rng = random.Random(seed)
    state = ALNSSolutionState(
        dv_routes={v: [] for v in instance["dvs"]},
        od_routes={o: [] for o in instance["ods"]},
        assignments={},
        unassigned_customers=set(instance["customers"]),
    )

    # Paper Algorithm 1 staging: Type 1 first; then mandatory ADP Type 2;
    # then flexible Type 3. Within each stage, order is randomized.
    stages = [list(instance["type1"]), list(instance["type2"]), list(instance["type3"])]
    for stage in stages:
        rng.shuffle(stage)
        for customer in stage:
            candidates = _build_candidates(state, instance, customer, rng)
            if not candidates:
                raise RuntimeError(f"No insertion candidate for {customer}")
            selected = min(candidates, key=lambda c: c.score)
            selected.apply(state)

    state.normalize_routes(instance)
    state.register_operator_event(
        "construction",
        "algorithm_1_gate2",
        {"seed": seed},
    )
    return state


def construct_initial_solution(
    instance: dict,
    seed: int = 42,
    max_attempts: int = 100,
) -> ALNSSolutionState:
    """
    Build a complete feasible initial state.

    Gate-2 implementation follows the paper's three-stage construction logic.
    Full OD insertion Strategies I and II are intentionally deferred to Gate 3.
    Multiple deterministic attempts are used only to recover from a poor
    randomized insertion order; every returned state must pass shared validator.
    """
    last_errors: list[str] = []
    for attempt in range(max_attempts):
        attempt_seed = seed * 1000 + attempt
        try:
            state = _construct_once(instance, attempt_seed)
            solution = state.to_core_solution(
                instance=instance,
                lambda_value=0.0,
                objective_mode="cost",
                emission_factors=(1.0, 1.0),
                require_complete=True,
                metadata={"construction_seed": attempt_seed},
            )
            if solution.validator_pass:
                return state
            last_errors = solution.validation_errors
        except Exception as exc:
            last_errors = [f"{type(exc).__name__}: {exc}"]

    raise RuntimeError(
        f"Unable to construct a valid initial solution after {max_attempts} attempts. "
        f"Last errors: {last_errors}"
    )
