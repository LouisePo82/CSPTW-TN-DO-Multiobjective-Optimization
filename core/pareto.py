from __future__ import annotations

def is_close(a, b, tol=1e-7):
    return abs(a - b) <= tol

def remove_duplicates(rows, tol=1e-7):
    unique = []
    for row in rows:
        if not any(
            is_close(row["cost"], u["cost"], tol)
            and is_close(row["emission"], u["emission"], tol)
            for u in unique
        ):
            unique.append(row)
    return unique

def nondominated(rows, tol=1e-7):
    nd = []
    for i, row in enumerate(rows):
        dominated = False
        for j, other in enumerate(rows):
            if i == j:
                continue
            no_worse = (
                other["cost"] <= row["cost"] + tol
                and other["emission"] <= row["emission"] + tol
            )
            strictly_better = (
                other["cost"] < row["cost"] - tol
                or other["emission"] < row["emission"] - tol
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            nd.append(row)
    return sorted(nd, key=lambda x: (x["cost"], x["emission"]))

def coverage_report(data, result):
    assignments = result.get("assignments", {})
    dv_routes = result.get("dv_routes", {})
    od_routes = result.get("od_routes", {})

    used_dvs = [k for k, r in dv_routes.items() if r]
    used_ods = [k for k, r in od_routes.items() if r]
    pickups = {
        a.get("pickup")
        for a in assignments.values()
        if a.get("mode") == "OD_HOME"
    }

    return {
        "used_dv_count": len(used_dvs),
        "unused_dv_count": len(data["dvs"]) - len(used_dvs),
        "used_od_count": len(used_ods),
        "unused_od_count": len(data["ods"]) - len(used_ods),
        "uses_depot_pickup": data["start_depot"] in pickups,
        "uses_tn_pickup": any(p in data["tns"] for p in pickups),
        "od_serves_two_customers": any(
            sum(
                1 for a in assignments.values()
                if a.get("mode") == "OD_HOME" and a.get("driver") == od
            ) >= 2
            for od in data["ods"]
        ),
        "type3_home": any(
            i in data["type3"] and a.get("mode") in {"DV_HOME", "OD_HOME"}
            for i, a in assignments.items()
        ),
        "type3_adp": any(
            i in data["type3"] and a.get("mode") == "ADP"
            for i, a in assignments.items()
        ),
        "positive_od_extra_distance": result.get("od_extra_distance", 0.0) > 1e-7,
        "multiple_dvs_used": len(used_dvs) >= 2,
        "multiple_ods_used": len(used_ods) >= 2,
    }
