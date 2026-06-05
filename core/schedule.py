from __future__ import annotations

def recompute_earliest_schedule(data: dict, result: dict) -> dict:
    """
    Recompute operationally meaningful earliest times from extracted routes.

    This does not alter feasibility or objective values. It replaces arbitrary
    solver slack times with earliest physically feasible times for reporting.
    """
    nodes = data["nodes"]
    vehicles = data["vehicles"]
    tt = data["travel_time"]
    st = data["service_time_per_weight"]
    assignments = result["assignments"]
    dv_routes = result["dv_routes"]
    od_routes = result["od_routes"]

    dv_times = {}
    tn_completion = {}

    # Helper demand delivered at nodes.
    def dv_service_demand(k, node):
        if node in data["customers"]:
            a = assignments[node]
            return nodes[node]["demand"] if a.get("mode") == "DV_HOME" and a.get("vehicle") == k else 0.0
        if node in data["adps"]:
            return sum(
                nodes[i]["demand"]
                for i, a in assignments.items()
                if a.get("mode") == "ADP" and a.get("vehicle") == k and a.get("adp") == node
            )
        if node in data["tns"]:
            return sum(
                nodes[i]["demand"]
                for i, a in assignments.items()
                if a.get("mode") == "OD_HOME" and a.get("pickup") == node
            )
        return 0.0

    for k, route in dv_routes.items():
        if not route:
            dv_times[k] = {}
            continue
        current_time = max(nodes[route[0]]["tw_start"], vehicles[k]["earliest"])
        times = {route[0]: current_time}
        for idx in range(len(route)-1):
            i, j = route[idx], route[idx+1]
            service = st * dv_service_demand(k, i)
            arrival = current_time + service + tt[i][j]
            arrival = max(arrival, nodes[j]["tw_start"])
            times[j] = arrival
            current_time = arrival
            if j in data["tns"]:
                tn_completion[j] = max(
                    tn_completion.get(j, 0.0),
                    arrival + st * dv_service_demand(k, j)
                )
        dv_times[k] = times

    od_pickup_times = {}
    od_customer_times = {}
    od_destination_times = {}

    for od, route in od_routes.items():
        if not route:
            od_pickup_times[od] = {}
            od_customer_times[od] = {}
            od_destination_times[od] = None
            continue

        o = vehicles[od]["origin"]
        d = vehicles[od]["destination"]
        current_time = vehicles[od]["earliest"]
        current = o
        pickup = next((p for p in data["pickup_points"] if p in route), None)

        # origin -> pickup
        current_time += tt[current][pickup]
        if pickup in data["tns"]:
            current_time = max(current_time, tn_completion.get(pickup, current_time))
        od_pickup_times[od] = {pickup: current_time}

        assigned = [
            i for i, a in assignments.items()
            if a.get("mode") == "OD_HOME" and a.get("driver") == od
        ]
        pickup_load = sum(nodes[i]["demand"] for i in assigned)
        current_time += st * pickup_load
        current = pickup

        customer_times = {}
        for node in route[2:-1]:
            current_time += tt[current][node]
            current_time = max(current_time, nodes[node]["tw_start"])
            customer_times[node] = current_time
            current_time += st * nodes[node]["demand"]
            current = node

        current_time += tt[current][d]
        od_customer_times[od] = customer_times
        od_destination_times[od] = current_time

    return {
        "dv": dv_times,
        "od_pickup": od_pickup_times,
        "od_customer": od_customer_times,
        "od_destination": od_destination_times,
        "tn_completion": tn_completion,
    }
