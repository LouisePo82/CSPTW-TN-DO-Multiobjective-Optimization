from __future__ import annotations

from collections import defaultdict
import time
from typing import Any

from ortools.linear_solver import pywraplp


TOL = 1e-7


def _create_solver():
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        solver = pywraplp.Solver.CreateSolver("CBC_MIXED_INTEGER_PROGRAMMING")
    if solver is None:
        raise RuntimeError("Neither SCIP nor CBC is available in OR-Tools.")
    return solver


def _binary(solver, name):
    return solver.IntVar(0.0, 1.0, name)


def _continuous(solver, lb, ub, name):
    return solver.NumVar(lb, ub, name)


def _extract_route(active_arcs, start, end):
    outgoing = defaultdict(list)
    for i, j in active_arcs:
        outgoing[i].append(j)

    route = [start]
    current = start
    used = set()
    while current != end:
        candidates = [j for j in outgoing.get(current, []) if (current, j) not in used]
        if not candidates:
            break
        nxt = candidates[0]
        used.add((current, nxt))
        route.append(nxt)
        current = nxt
        if len(route) > len(active_arcs) + 2:
            break
    return route


def solve_exact_milp(
    data: dict,
    lambda_val: float = 0.0,
    objective_mode: str = "cost",
    cost_bounds: tuple[float, float] | None = None,
    emission_bounds: tuple[float, float] | None = None,
    emission_factors: tuple[float, float] = (1.0, 1.0),
    epsilon_emission: float | None = None,
    time_limit_sec: int = 300,
    mip_gap: float | None = 0.0,
    enable_output: bool = False,
) -> dict[str, Any]:
    """
    Reduced-size paper-frame MILP for CSPTW-TN-DO.

    Mathematical correspondence:
      x[k,i,j]       paper x^k_ij (DV arcs)
      sigma[k,i,j]   paper sigma^k_ij (DV load on arcs)
      delta_d[k,i]   paper delta^D_ik
      z[k,i]         paper z^k_i
      y[l,i,j]       paper y^l_ij (OD arcs)
      beta[l,i,j]    paper beta^l_ij (OD load on arcs)
      r[l,i]         paper r^l_i
      omega[l,p]     paper omega^l_p
      f[l,p]         paper f^l_p
      delta_cs[l,i]  paper delta^CS_i, indexed by OD in implementation
      d_t[p]         paper dT_p
      h[i]           paper h_i
      epsilon[k,i,a] paper epsilon^k_ia
      d_l[k,a]       paper dL^k_a

    Auxiliary variables linearize nonlinear products:
      home_visit[k,i] = z[k,i] * h[i]
      w[l,i,p]        = r[l,i] * omega[l,p]
      tn_load[k,p]    = z[k,p] * d_t[p]

    The model is intended for tiny/small exact ground-truth instances.
    """
    if not 0.0 <= lambda_val <= 1.0:
        raise ValueError("lambda_val must be in [0,1].")

    solver = _create_solver()
    solver.SetTimeLimit(int(time_limit_sec * 1000))
    if enable_output:
        solver.EnableOutput()
    if mip_gap is not None and solver.SolverVersion().lower().find("scip") >= 0:
        solver.SetSolverSpecificParametersAsString(f"limits/gap = {mip_gap}")

    n = data["nodes"]
    vinfo = data["vehicles"]
    c = data["customers"]
    nh, na, nha = data["type1"], data["type2"], data["type3"]
    nt, nl = data["tns"], data["adps"]
    kd, ko = data["dvs"], data["ods"]
    s, t = data["start_depot"], data["end_depot"]
    pp = data["pickup_points"]
    dist, tt = data["distance"], data["travel_time"]
    gamma = data["gamma"]
    rho = data["rho"]
    st = data["service_time_per_weight"]
    big_m = data["big_m_time"]

    nv = c + nt + nl
    dv_nodes = [s] + nv + [t]

    # Valid DV arcs: s -> NV; NV -> NV/t. No arcs into s or out of t.
    a_dv = [(i, j) for i in [s] + nv for j in nv + [t] if i != j and not (i == s and j == t)]

    # Valid OD arcs depend on each OD:
    # o(k)->PP, PP->customer, customer->customer/d(k).
    a_od = {}
    for od in ko:
        o, d = vinfo[od]["origin"], vinfo[od]["destination"]
        arcs = []
        arcs += [(o, p) for p in pp]
        arcs += [(p, i) for p in pp for i in c]
        arcs += [(i, j) for i in c for j in c if i != j]
        arcs += [(i, d) for i in c]
        a_od[od] = arcs

    # Upper bounds for linearization/load variables.
    total_demand = sum(n[i]["demand"] for i in c)
    max_od_weight = {}
    sorted_demands = sorted((n[i]["demand"] for i in c), reverse=True)
    for od in ko:
        q_count = vinfo[od]["capacity"]
        max_od_weight[od] = sum(sorted_demands[:q_count])

    # ------------------------------------------------------------------
    # Decision variables
    # ------------------------------------------------------------------
    use_dv = {k: _binary(solver, f"use_dv[{k}]") for k in kd}
    use_od = {od: _binary(solver, f"use_od[{od}]") for od in ko}

    h = {i: _binary(solver, f"h[{i}]") for i in nh + nha}
    epsilon = {
        (k, i, a): _binary(solver, f"epsilon[{k},{i},{a}]")
        for k in kd for i in na + nha for a in nl
    }

    z = {(k, i): _binary(solver, f"z[{k},{i}]") for k in kd for i in nv}
    r = {(od, i): _binary(solver, f"r[{od},{i}]") for od in ko for i in c}
    omega = {(od, p): _binary(solver, f"omega[{od},{p}]") for od in ko for p in pp}

    x = {(k, i, j): _binary(solver, f"x[{k},{i},{j}]") for k in kd for i, j in a_dv}
    y = {
        (od, i, j): _binary(solver, f"y[{od},{i},{j}]")
        for od in ko for i, j in a_od[od]
    }

    sigma = {
        (k, i, j): _continuous(solver, 0.0, vinfo[k]["capacity"], f"sigma[{k},{i},{j}]")
        for k in kd for i, j in a_dv
    }
    beta = {
        (od, i, j): _continuous(solver, 0.0, max_od_weight[od], f"beta[{od},{i},{j}]")
        for od in ko
        for i, j in a_od[od]
        if i in pp + c
    }

    delta_d = {
        (k, i): _continuous(solver, 0.0, big_m, f"delta_d[{k},{i}]")
        for k in kd for i in dv_nodes
    }
    delta_cs = {
        (od, i): _continuous(solver, 0.0, big_m, f"delta_cs[{od},{i}]")
        for od in ko for i in c
    }
    f = {
        (od, p): _continuous(solver, 0.0, big_m, f"f[{od},{p}]")
        for od in ko for p in pp
    }

    d_t = {p: _continuous(solver, 0.0, total_demand, f"d_t[{p}]") for p in pp}
    d_l = {
        (k, a): _continuous(solver, 0.0, vinfo[k]["capacity"], f"d_l[{k},{a}]")
        for k in kd for a in nl
    }
    theta = {
        k: _continuous(solver, 0.0, vinfo[k]["capacity"], f"theta[{k}]")
        for k in kd
    }

    home_visit = {
        (k, i): _binary(solver, f"home_visit[{k},{i}]")
        for k in kd for i in nh + nha
    }
    w = {
        (od, i, p): _binary(solver, f"w[{od},{i},{p}]")
        for od in ko for i in c for p in pp
    }
    tn_load = {
        (k, p): _continuous(solver, 0.0, total_demand, f"tn_load[{k},{p}]")
        for k in kd for p in nt
    }

    # ------------------------------------------------------------------
    # Delivery-option and assignment constraints (paper 1-6)
    # ------------------------------------------------------------------
    for i in nh:
        solver.Add(h[i] == 1)

    for i in na:
        solver.Add(solver.Sum(epsilon[k, i, a] for k in kd for a in nl) == 1)

    for i in na + nha:
        for k in kd:
            for a in nl:
                solver.Add(epsilon[k, i, a] <= gamma.get((i, a), 0))

    for i in nha:
        solver.Add(h[i] + solver.Sum(epsilon[k, i, a] for k in kd for a in nl) == 1)

    for i in nh + nha:
        solver.Add(
            solver.Sum(z[k, i] for k in kd) + solver.Sum(r[od, i] for od in ko) == h[i]
        )

    for i in na:
        for k in kd:
            solver.Add(z[k, i] == 0)
        for od in ko:
            solver.Add(r[od, i] == 0)

    for k in kd:
        for a in nl:
            solver.Add(
                solver.Sum(epsilon[k, i, a] for i in na + nha)
                <= len(na + nha) * z[k, a]
            )

    # Linearize home_visit = z*h.
    for k in kd:
        for i in nh + nha:
            solver.Add(home_visit[k, i] <= z[k, i])
            solver.Add(home_visit[k, i] <= h[i])
            solver.Add(home_visit[k, i] >= z[k, i] + h[i] - 1)

    # ------------------------------------------------------------------
    # DV routing, load flow and time constraints (paper 7-18)
    # ------------------------------------------------------------------
    out_dv = defaultdict(list)
    in_dv = defaultdict(list)
    for k in kd:
        for i, j in a_dv:
            out_dv[k, i].append(x[k, i, j])
            in_dv[k, j].append(x[k, i, j])

    for k in kd:
        solver.Add(solver.Sum(out_dv[k, s]) == use_dv[k])
        solver.Add(solver.Sum(in_dv[k, t]) == use_dv[k])

        for i in nv:
            solver.Add(solver.Sum(out_dv[k, i]) - solver.Sum(in_dv[k, i]) == 0)
            solver.Add(solver.Sum(out_dv[k, i]) == z[k, i])

        # A used DV must visit at least one service node; an unused DV visits none.
        solver.Add(solver.Sum(z[k, i] for i in nv) >= use_dv[k])
        solver.Add(solver.Sum(z[k, i] for i in nv) <= len(nv) * use_dv[k])

        # ADP delivered load.
        for a in nl:
            solver.Add(
                d_l[k, a]
                == solver.Sum(n[i]["demand"] * epsilon[k, i, a] for i in na + nha)
            )

    # w = r * omega; dT at every PP.
    for od in ko:
        for i in c:
            for p in pp:
                solver.Add(w[od, i, p] <= r[od, i])
                solver.Add(w[od, i, p] <= omega[od, p])
                solver.Add(w[od, i, p] >= r[od, i] + omega[od, p] - 1)

    for p in pp:
        solver.Add(
            d_t[p]
            == solver.Sum(n[i]["demand"] * w[od, i, p] for od in ko for i in c)
        )

    # tn_load = z * dT, exact McCormick for binary-continuous.
    for k in kd:
        for p in nt:
            solver.Add(tn_load[k, p] <= d_t[p])
            solver.Add(tn_load[k, p] <= total_demand * z[k, p])
            solver.Add(tn_load[k, p] >= d_t[p] - total_demand * (1 - z[k, p]))

        solver.Add(
            theta[k]
            == solver.Sum(n[i]["demand"] * home_visit[k, i] for i in nh + nha)
            + solver.Sum(tn_load[k, p] for p in nt)
            + solver.Sum(d_l[k, a] for a in nl)
        )
        solver.Add(theta[k] <= vinfo[k]["capacity"] * use_dv[k])

        # DV arc load-flow balance.
        for node in dv_nodes:
            incoming = solver.Sum(
                sigma[k, i, j] for i, j in a_dv if j == node
            )
            outgoing = solver.Sum(
                sigma[k, i, j] for i, j in a_dv if i == node
            )

            if node == s:
                solver.Add(incoming - outgoing == -theta[k])
            elif node == t:
                solver.Add(incoming - outgoing == 0)
            elif node in c:
                delivered = (
                    n[node]["demand"] * home_visit[k, node]
                    if node in nh + nha else 0.0
                )
                solver.Add(incoming - outgoing == delivered)
            elif node in nt:
                solver.Add(incoming - outgoing == tn_load[k, node])
            elif node in nl:
                solver.Add(incoming - outgoing == d_l[k, node])

        for i, j in a_dv:
            solver.Add(sigma[k, i, j] <= vinfo[k]["capacity"] * x[k, i, j])

        # Time variables.
        solver.Add(delta_d[k, s] >= n[s]["tw_start"])
        solver.Add(delta_d[k, s] <= n[s]["tw_end"])
        solver.Add(delta_d[k, t] >= n[t]["tw_start"])
        solver.Add(delta_d[k, t] <= n[t]["tw_end"])

        for i, j in a_dv:
            if i == s:
                service_i = 0.0
            elif i in c:
                service_i = st * n[i]["demand"]
            elif i in nt:
                service_i = st * d_t[i]
            elif i in nl:
                service_i = st * d_l[k, i]
            else:
                service_i = 0.0

            solver.Add(
                delta_d[k, j]
                >= delta_d[k, i] + tt[i][j] + service_i - big_m * (1 - x[k, i, j])
            )

        for i in c:
            solver.Add(delta_d[k, i] >= n[i]["tw_start"] - big_m * (1 - z[k, i]))
            solver.Add(delta_d[k, i] <= n[i]["tw_end"] + big_m * (1 - z[k, i]))

    # ------------------------------------------------------------------
    # OD routing, load flow and time constraints (paper 19-32)
    # ------------------------------------------------------------------
    for od in ko:
        o, d = vinfo[od]["origin"], vinfo[od]["destination"]

        solver.Add(solver.Sum(omega[od, p] for p in pp) == use_od[od])
        solver.Add(solver.Sum(r[od, i] for i in c) <= vinfo[od]["capacity"])
        solver.Add(
            solver.Sum(r[od, i] for i in c)
            <= len(c) * solver.Sum(omega[od, p] for p in pp)
        )

        for p in pp:
            solver.Add(y[od, o, p] == omega[od, p])
            solver.Add(solver.Sum(y[od, p, i] for i in c) == y[od, o, p])

        for i in c:
            inbound = solver.Sum(
                y[od, a, b] for a, b in a_od[od] if b == i
            )
            outbound = solver.Sum(
                y[od, a, b] for a, b in a_od[od] if a == i
            )
            solver.Add(inbound == r[od, i])
            solver.Add(outbound == r[od, i])

        solver.Add(
            solver.Sum(y[od, p, i] for p in pp for i in c)
            == solver.Sum(y[od, i, d] for i in c)
        )

        # Pickup load by PP.
        pickup_load = {
            p: solver.Sum(n[i]["demand"] * w[od, i, p] for i in c)
            for p in pp
        }

        # OD load flow: beta exists only after pickup.
        for p in pp:
            outgoing_beta = solver.Sum(
                beta[od, a, b]
                for a, b in a_od[od]
                if a == p and (od, a, b) in beta
            )
            solver.Add(outgoing_beta == pickup_load[p])

        for i in c:
            incoming_beta = solver.Sum(
                beta[od, a, b]
                for a, b in a_od[od]
                if b == i and (od, a, b) in beta
            )
            outgoing_beta = solver.Sum(
                beta[od, a, b]
                for a, b in a_od[od]
                if a == i and (od, a, b) in beta
            )
            solver.Add(incoming_beta - outgoing_beta == n[i]["demand"] * r[od, i])

        for (a, b) in a_od[od]:
            if (od, a, b) in beta:
                solver.Add(beta[od, a, b] <= max_od_weight[od] * y[od, a, b])

        # Time at PP and customers.
        for p in pp:
            solver.Add(
                f[od, p]
                >= vinfo[od]["earliest"] + tt[o][p] - big_m * (1 - y[od, o, p])
            )
            solver.Add(f[od, p] <= big_m * omega[od, p])

            pickup_service = st * pickup_load[p]
            for i in c:
                solver.Add(
                    delta_cs[od, i]
                    >= f[od, p] + pickup_service + tt[p][i]
                    - big_m * (1 - y[od, p, i])
                )

        for i in c:
            for j in c:
                if i != j:
                    solver.Add(
                        delta_cs[od, j]
                        >= delta_cs[od, i] + st * n[i]["demand"] + tt[i][j]
                        - big_m * (1 - y[od, i, j])
                    )

            solver.Add(
                delta_cs[od, i] >= n[i]["tw_start"] - big_m * (1 - r[od, i])
            )
            solver.Add(
                delta_cs[od, i] <= n[i]["tw_end"] + big_m * (1 - r[od, i])
            )
            solver.Add(
                delta_cs[od, i] + st * n[i]["demand"] + tt[i][d]
                <= vinfo[od]["latest"] + big_m * (1 - y[od, i, d])
            )

    # ------------------------------------------------------------------
    # TN synchronization (paper 33-35)
    # ------------------------------------------------------------------
    for p in nt:
        solver.Add(
            solver.Sum(omega[od, p] for od in ko)
            <= len(ko) * solver.Sum(z[k, p] for k in kd)
        )
        # Avoid duplicate delivery of the same TN demand in the exact ground-truth model.
        solver.Add(solver.Sum(z[k, p] for k in kd) <= 1)

        for od in ko:
            for k in kd:
                solver.Add(
                    delta_d[k, p] + st * d_t[p]
                    <= f[od, p]
                    + big_m * (2 - y[od, vinfo[od]["origin"], p] - z[k, p])
                )

    # ------------------------------------------------------------------
    # Cost and emission objectives
    # ------------------------------------------------------------------
    dv_distance = solver.Sum(dist[i][j] * x[k, i, j] for k in kd for i, j in a_dv)

    od_route_distance = {}
    od_extra_distance = {}
    for od in ko:
        o, d = vinfo[od]["origin"], vinfo[od]["destination"]
        od_route_distance[od] = solver.Sum(
            dist[i][j] * y[od, i, j] for i, j in a_od[od]
        )
        od_extra_distance[od] = (
            od_route_distance[od] - dist[o][d] * use_od[od]
        )

    total_od_extra = solver.Sum(od_extra_distance[od] for od in ko)
    total_cost = dv_distance + rho * total_od_extra

    e_dv, e_od = emission_factors
    total_emission = e_dv * dv_distance + e_od * total_od_extra

    if objective_mode == "cost":
        objective_expr = total_cost
    elif objective_mode == "emission":
        objective_expr = total_emission
    elif objective_mode == "epsilon_cost":
        if epsilon_emission is None:
            raise ValueError("epsilon_cost mode requires epsilon_emission.")
        solver.Add(total_emission <= float(epsilon_emission))
        objective_expr = total_cost
    elif objective_mode == "weighted":
        if cost_bounds is None or emission_bounds is None:
            raise ValueError("weighted mode requires cost_bounds and emission_bounds.")
        z_min, z_max = cost_bounds
        e_min, e_max = emission_bounds
        if z_max - z_min <= TOL or e_max - e_min <= TOL:
            raise ValueError("Normalization ranges must be positive.")
        objective_expr = (
            (1.0 - lambda_val) * (total_cost - z_min) / (z_max - z_min)
            + lambda_val * (total_emission - e_min) / (e_max - e_min)
        )
    else:
        raise ValueError("objective_mode must be cost, emission, epsilon_cost, or weighted.")

    solver.Minimize(objective_expr)

    start = time.perf_counter()
    status = solver.Solve()
    elapsed = time.perf_counter() - start

    status_name = {
        pywraplp.Solver.OPTIMAL: "OPTIMAL",
        pywraplp.Solver.FEASIBLE: "FEASIBLE",
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
        pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
        pywraplp.Solver.ABNORMAL: "ABNORMAL",
        pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
    }.get(status, str(status))

    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return {
            "status": status_name,
            "comp_time_sec": elapsed,
            "routes": {},
            "assignments": {},
        }

    def sv(var):
        return var.solution_value()

    dv_routes = {}
    active_x = {}
    for k in kd:
        edges = [(i, j) for i, j in a_dv if sv(x[k, i, j]) > 0.5]
        active_x[k] = edges
        dv_routes[k] = _extract_route(edges, s, t) if sv(use_dv[k]) > 0.5 else []

    od_routes = {}
    active_y = {}
    for od in ko:
        o, d = vinfo[od]["origin"], vinfo[od]["destination"]
        edges = [(i, j) for i, j in a_od[od] if sv(y[od, i, j]) > 0.5]
        active_y[od] = edges
        od_routes[od] = _extract_route(edges, o, d) if sv(use_od[od]) > 0.5 else []

    assignments = {}
    for i in c:
        if i in nh + nha and sv(h[i]) > 0.5:
            dv = next((k for k in kd if sv(z[k, i]) > 0.5), None)
            od = next((q for q in ko if sv(r[q, i]) > 0.5), None)
            assignments[i] = (
                {"mode": "DV_HOME", "vehicle": dv}
                if dv is not None
                else {
                    "mode": "OD_HOME",
                    "driver": od,
                    "pickup": next((p for p in pp if sv(omega[od, p]) > 0.5), None),
                }
            )
        else:
            selected = next(
                (
                    (k, a)
                    for k in kd for a in nl
                    if (k, i, a) in epsilon and sv(epsilon[k, i, a]) > 0.5
                ),
                None,
            )
            assignments[i] = {
                "mode": "ADP",
                "vehicle": selected[0] if selected else None,
                "adp": selected[1] if selected else None,
            }

    times = {
        "dv": {
            k: {i: sv(delta_d[k, i]) for i in dv_nodes if i in dv_routes[k]}
            for k in kd
        },
        "od_customer": {
            od: {i: sv(delta_cs[od, i]) for i in c if sv(r[od, i]) > 0.5}
            for od in ko
        },
        "od_pickup": {
            od: {p: sv(f[od, p]) for p in pp if sv(omega[od, p]) > 0.5}
            for od in ko
        },
    }

    def clean_number(value, tol=1e-8):
        value = float(value)
        return 0.0 if abs(value) < tol else value

    objective_value = clean_number(
        solver.Objective().Value()
    )
    best_bound = clean_number(
        solver.Objective().BestBound()
    )
    mip_gap_abs = clean_number(
        abs(
            objective_value
            - best_bound
        )
    )
    mip_gap_rel = (
        mip_gap_abs
        / max(
            abs(objective_value),
            1e-12,
        )
    )

    return {
        "status": status_name,
        "objective_mode": objective_mode,
        "lambda": lambda_val,
        "cost": clean_number(total_cost.solution_value()),
        "emission": clean_number(total_emission.solution_value()),
        "objective": objective_value,
        "dv_distance": clean_number(dv_distance.solution_value()),
        "od_extra_distance": clean_number(total_od_extra.solution_value()),
        "routes": {**dv_routes, **od_routes},
        "dv_routes": dv_routes,
        "od_routes": od_routes,
        "assignments": assignments,
        "times": times,
        "tn_demand": {p: clean_number(sv(d_t[p])) for p in pp},
        "adp_load": {
            k: {a: clean_number(sv(d_l[k, a])) for a in nl}
            for k in kd
        },
        "vehicle_load": {k: clean_number(sv(theta[k])) for k in kd},
        "active_arcs": {"dv": active_x, "od": active_y},
        "comp_time_sec": elapsed,
        "solver_version": solver.SolverVersion(),
        "best_bound": best_bound,
        "mip_gap_abs": mip_gap_abs,
        "mip_gap_rel": mip_gap_rel,
        "mip_gap_percent": (
            100.0 * mip_gap_rel
        ),
    }
