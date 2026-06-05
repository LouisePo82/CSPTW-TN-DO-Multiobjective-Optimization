from __future__ import annotations

from collections import Counter
from math import isclose
from .schedule import recompute_earliest_schedule


def _route_distance(route, distance):
    return sum(distance[route[i]][route[i + 1]] for i in range(len(route) - 1))


def validate_solution(
    data: dict,
    result: dict,
    emission_factors: tuple[float, float] = (1.0, 1.0),
    tolerance: float = 1e-5,
) -> dict:
    errors = []
    warnings = []

    if result.get("status") not in {"OPTIMAL", "FEASIBLE"}:
        return {"valid": False, "errors": [f"Solver status is {result.get('status')}"], "warnings": []}

    nodes = data["nodes"]
    vehicles = data["vehicles"]
    c = data["customers"]
    nh, na, nha = data["type1"], data["type2"], data["type3"]
    nt, nl = data["tns"], data["adps"]
    gamma = data["gamma"]
    rho = data["rho"]
    st = data["service_time_per_weight"]
    dist = data["distance"]
    assignments = result["assignments"]
    dv_routes = result["dv_routes"]
    od_routes = result["od_routes"]
    times = result["times"]
    earliest_times = recompute_earliest_schedule(data, result)

    # 1. Every customer has one valid delivery option.
    if set(assignments) != set(c):
        errors.append(f"Assignment coverage mismatch: expected {set(c)}, got {set(assignments)}")

    for i in c:
        a = assignments.get(i, {})
        mode = a.get("mode")
        if i in nh and mode not in {"DV_HOME", "OD_HOME"}:
            errors.append(f"{i} is Type 1 but mode={mode}")
        if i in na and mode != "ADP":
            errors.append(f"{i} is Type 2 but mode={mode}")
        if i in nha and mode not in {"DV_HOME", "OD_HOME", "ADP"}:
            errors.append(f"{i} is Type 3 but mode={mode}")
        if mode == "ADP":
            adp = a.get("adp")
            if gamma.get((i, adp), 0) != 1:
                errors.append(f"{i} assigned to incompatible ADP {adp}")

    # 2. Route endpoints and visit consistency.
    for k in data["dvs"]:
        route = dv_routes.get(k, [])
        if route:
            if route[0] != data["start_depot"] or route[-1] != data["end_depot"]:
                errors.append(f"{k} route has invalid endpoints: {route}")
            if route == [data["start_depot"], data["end_depot"]]:
                errors.append(f"{k} has a degenerate empty active route S->T")

    for od in data["ods"]:
        route = od_routes.get(od, [])
        if route:
            if route[0] != vehicles[od]["origin"] or route[-1] != vehicles[od]["destination"]:
                errors.append(f"{od} route has invalid endpoints: {route}")
            pickup_visits = [p for p in data["pickup_points"] if p in route]
            if len(pickup_visits) != 1:
                errors.append(f"{od} must visit exactly one PP when active; route={route}")
            if any(a in route for a in nl):
                errors.append(f"{od} illegally visits an ADP: {route}")

    # Direct-home visits.
    for i, a in assignments.items():
        if a["mode"] == "DV_HOME":
            if i not in dv_routes.get(a["vehicle"], []):
                errors.append(f"{i} assigned DV_HOME but absent from {a['vehicle']} route")
        elif a["mode"] == "OD_HOME":
            if i not in od_routes.get(a["driver"], []):
                errors.append(f"{i} assigned OD_HOME but absent from {a['driver']} route")
        elif a["mode"] == "ADP":
            if a["adp"] not in dv_routes.get(a["vehicle"], []):
                errors.append(f"{i} assigned to {a['adp']} but DV does not visit it")

    # 3. Capacity.
    for k in data["dvs"]:
        load = 0.0
        for i, a in assignments.items():
            if a.get("vehicle") == k and a["mode"] in {"DV_HOME", "ADP"}:
                load += nodes[i]["demand"]
        for od in data["ods"]:
            route = od_routes.get(od, [])
            if route and any(p in route for p in nt):
                p = next(p for p in nt if p in route)
                if p in dv_routes.get(k, []):
                    load += sum(
                        nodes[i]["demand"]
                        for i, a in assignments.items()
                        if a["mode"] == "OD_HOME"
                        and a["driver"] == od
                        and a["pickup"] == p
                    )
        if load > vehicles[k]["capacity"] + tolerance:
            errors.append(f"{k} capacity exceeded: {load}>{vehicles[k]['capacity']}")

    for od in data["ods"]:
        count = sum(1 for a in assignments.values() if a["mode"] == "OD_HOME" and a["driver"] == od)
        if count > vehicles[od]["capacity"]:
            errors.append(f"{od} customer-count capacity exceeded: {count}>{vehicles[od]['capacity']}")

    # 4. Time windows and synchronization using solver-extracted times.
    for k, route in dv_routes.items():
        if not route:
            continue
        for i in route:
            if i in c:
                arr = times["dv"].get(k, {}).get(i)
                if arr is None or arr < nodes[i]["tw_start"] - tolerance or arr > nodes[i]["tw_end"] + tolerance:
                    errors.append(f"{k} violates TW at {i}: {arr}")

    for od, route in od_routes.items():
        if not route:
            continue
        for i in c:
            if i in route:
                arr = times["od_customer"].get(od, {}).get(i)
                if arr is None or arr < nodes[i]["tw_start"] - tolerance or arr > nodes[i]["tw_end"] + tolerance:
                    errors.append(f"{od} violates TW at {i}: {arr}")

        pp = next((p for p in data["pickup_points"] if p in route), None)
        if pp in nt:
            pickup_time = times["od_pickup"].get(od, {}).get(pp)
            dv_candidates = [k for k, r in dv_routes.items() if pp in r]
            if not dv_candidates:
                errors.append(f"{od} picks at {pp} but no DV visits the TN")
            else:
                tn_total = result["tn_demand"][pp]
                latest_drop = max(times["dv"][k][pp] + st * tn_total for k in dv_candidates)
                if pickup_time + tolerance < latest_drop:
                    errors.append(
                        f"TN synchronization violated at {pp}: OD pickup {pickup_time} < DV completion {latest_drop}"
                    )

    # 5. Independent objective recomputation.
    dv_distance = sum(_route_distance(route, dist) for route in dv_routes.values() if route)
    od_extra = 0.0
    for od, route in od_routes.items():
        if not route:
            continue
        route_d = _route_distance(route, dist)
        direct_d = dist[vehicles[od]["origin"]][vehicles[od]["destination"]]
        extra = route_d - direct_d
        if extra < -tolerance:
            errors.append(f"{od} has negative extra distance {extra}")
        od_extra += max(0.0, extra)

    cost = dv_distance + rho * od_extra
    e_dv, e_od = emission_factors
    emission = e_dv * dv_distance + e_od * od_extra

    if not isclose(cost, result["cost"], abs_tol=tolerance, rel_tol=1e-7):
        errors.append(f"Cost mismatch: validator={cost}, solver={result['cost']}")
    if not isclose(emission, result["emission"], abs_tol=tolerance, rel_tol=1e-7):
        errors.append(f"Emission mismatch: validator={emission}, solver={result['emission']}")

    if result.get("status") == "FEASIBLE":
        warnings.append("Solution is feasible but not proven optimal; do not call it ground truth.")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "earliest_schedule": earliest_times,
        "recomputed": {
            "dv_distance": dv_distance,
            "od_extra_distance": od_extra,
            "cost": cost,
            "emission": emission,
        },
    }
