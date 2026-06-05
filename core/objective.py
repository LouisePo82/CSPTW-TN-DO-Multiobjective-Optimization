from __future__ import annotations

def route_distance(route: list[str], distance: dict) -> float:
    return sum(distance[route[i]][route[i+1]] for i in range(len(route)-1))

def recompute_objectives(
    instance: dict,
    dv_routes: dict[str, list[str]],
    od_routes: dict[str, list[str]],
    emission_factors: tuple[float, float] = (1.0, 1.0),
) -> dict:
    dv_distance = sum(route_distance(r, instance["distance"]) for r in dv_routes.values() if r)
    od_extra = 0.0
    for od, route in od_routes.items():
        if not route:
            continue
        route_d = route_distance(route, instance["distance"])
        info = instance["vehicles"][od]
        direct = instance["distance"][info["origin"]][info["destination"]]
        od_extra += max(0.0, route_d - direct)
    cost = dv_distance + instance["rho"] * od_extra
    e_dv, e_od = emission_factors
    emission = e_dv * dv_distance + e_od * od_extra
    return {
        "dv_distance": dv_distance,
        "od_extra_distance": od_extra,
        "cost": cost,
        "emission": emission,
    }
