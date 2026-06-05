from __future__ import annotations

from itertools import permutations, product


def route_distance(route, dist):
    return sum(dist[route[i]][route[i + 1]] for i in range(len(route) - 1))


def enumerate_micro_reference(data: dict) -> dict:
    """
    Independent finite enumeration for the supplied 3-customer micro instance.

    C1 Type 1: DV home or OD home
    C2 Type 2: ADP A1
    C3 Type 3: A1 is incompatible, therefore home by DV or OD
    OD1 capacity is one customer.
    If OD is active, pickup can be S or TN1.
    """
    s, t = data["start_depot"], data["end_depot"]
    dist = data["distance"]
    nodes = data["nodes"]
    rho = data["rho"]
    dv = data["dvs"][0]
    od = data["ods"][0]
    o = data["vehicles"][od]["origin"]
    d = data["vehicles"][od]["destination"]

    best = None
    candidates = []

    # assignment choices for C1 and C3; C2 always ADP.
    for c1_mode, c3_mode in product(["DV", "OD"], repeat=2):
        od_customers = [i for i, mode in [("C1", c1_mode), ("C3", c3_mode)] if mode == "OD"]
        if len(od_customers) > data["vehicles"][od]["capacity"]:
            continue

        for pickup in ([None] if not od_customers else ["S", "TN1"]):
            dv_required = ["A1"]
            dv_required += [i for i, mode in [("C1", c1_mode), ("C3", c3_mode)] if mode == "DV"]
            if pickup == "TN1":
                dv_required.append("TN1")

            for dv_order in permutations(dv_required):
                dv_route = [s] + list(dv_order) + [t]
                dv_distance = route_distance(dv_route, dist)

                if od_customers:
                    for od_order in permutations(od_customers):
                        od_route = [o, pickup] + list(od_order) + [d]
                        od_distance = route_distance(od_route, dist)
                        od_extra = od_distance - dist[o][d]
                        total_cost = dv_distance + rho * od_extra
                        candidates.append({
                            "cost": total_cost,
                            "dv_distance": dv_distance,
                            "od_extra_distance": od_extra,
                            "dv_route": dv_route,
                            "od_route": od_route,
                            "assignments": {"C1": c1_mode, "C2": "ADP", "C3": c3_mode},
                            "pickup": pickup,
                        })
                else:
                    candidates.append({
                        "cost": dv_distance,
                        "dv_distance": dv_distance,
                        "od_extra_distance": 0.0,
                        "dv_route": dv_route,
                        "od_route": [],
                        "assignments": {"C1": c1_mode, "C2": "ADP", "C3": c3_mode},
                        "pickup": None,
                    })

    # The model includes time/service constraints. Filter candidates by a small simulator.
    feasible = []
    st = data["service_time_per_weight"]
    for cand in candidates:
        # DV cumulative time.
        time = 0.0
        arrival = {}
        ok = True
        for idx in range(len(cand["dv_route"]) - 1):
            i, j = cand["dv_route"][idx], cand["dv_route"][idx + 1]
            service = 0.0
            if i in data["customers"] and cand["assignments"].get(i) == "DV":
                service = st * nodes[i]["demand"]
            elif i == "A1":
                service = st * nodes["C2"]["demand"]
            elif i == "TN1" and cand["pickup"] == "TN1":
                service = st * sum(nodes[q]["demand"] for q in ["C1", "C3"] if cand["assignments"][q] == "OD")
            time += service + data["travel_time"][i][j]
            arrival[j] = time
            if j in data["customers"]:
                if not (nodes[j]["tw_start"] <= time <= nodes[j]["tw_end"]):
                    ok = False

        if not ok:
            continue

        if cand["od_route"]:
            time = data["vehicles"][od]["earliest"]
            # origin -> pickup
            pickup = cand["pickup"]
            time += data["travel_time"][o][pickup]
            if pickup == "TN1":
                dv_finish = arrival["TN1"] + st * sum(
                    nodes[q]["demand"] for q in ["C1", "C3"] if cand["assignments"][q] == "OD"
                )
                time = max(time, dv_finish)
            # pickup service
            time += st * sum(nodes[q]["demand"] for q in ["C1", "C3"] if cand["assignments"][q] == "OD")
            prev = pickup
            for customer in cand["od_route"][2:-1]:
                time += data["travel_time"][prev][customer]
                if not (nodes[customer]["tw_start"] <= time <= nodes[customer]["tw_end"]):
                    ok = False
                time += st * nodes[customer]["demand"]
                prev = customer
            time += data["travel_time"][prev][d]
            if time > data["vehicles"][od]["latest"]:
                ok = False

        if ok:
            feasible.append(cand)

    if not feasible:
        raise RuntimeError("Manual micro enumeration found no feasible solution.")

    best = min(feasible, key=lambda x: (x["cost"], x["od_extra_distance"], x["dv_distance"]))
    return {"best": best, "feasible_candidate_count": len(feasible)}
