from __future__ import annotations
from core.pareto import remove_duplicates, nondominated
from reporting import OutputManager
from visualization.pareto_plot import plot_tradeoff
from visualization.route_plot import plot_routes
from .solver_factory import create_solver

def run_exact_multiobjective(config, instance):
    solver = create_solver("exact")
    out = OutputManager(config.output.root_dir, config.experiment_name)
    out.save_config(config.raw)
    if config.output.save_instance_snapshot:
        out.snapshot_instance(config.instance_path)

    base_obj = {
        "emission_factor_dv": config.objective.emission_factor_dv,
        "emission_factor_od": config.objective.emission_factor_od,
    }
    exact_cfg = vars(config.exact)

    cost_anchor = solver.solve(instance, {**base_obj, "mode": "cost", "lambda_value": 0.0}, exact_cfg)
    emission_anchor = solver.solve(instance, {**base_obj, "mode": "emission", "lambda_value": 1.0}, exact_cfg)
    for label, sol in [("cost_anchor", cost_anchor), ("emission_anchor", emission_anchor)]:
        if sol.status != "OPTIMAL" or not sol.validator_pass:
            raise RuntimeError(f"{label} failed ground-truth requirements: {sol.status}, {sol.validation_errors}")
        out.save_solution(label, sol)

    z_min, z_max = cost_anchor.cost, emission_anchor.cost
    e_min, e_max = emission_anchor.emission, cost_anchor.emission
    if z_max <= z_min or e_max <= e_min:
        raise RuntimeError("No positive cost-emission payoff range.")

    lambda_rows, candidate_rows = [], []
    for lam in config.objective.lambda_values:
        sol = solver.solve(
            instance,
            {
                **base_obj, "mode": "weighted", "lambda_value": lam,
                "cost_bounds": (z_min, z_max), "emission_bounds": (e_min, e_max),
            },
            exact_cfg,
        )
        if sol.status != "OPTIMAL" or not sol.validator_pass:
            raise RuntimeError(f"lambda={lam} failed validation.")
        label = f"lambda_{lam:.2f}"
        out.save_solution(label, sol)
        row = {
            "method": "weighted_sum", "lambda": lam, "epsilon": None,
            "cost": sol.cost, "emission": sol.emission,
            "objective": sol.objective, "dv_distance": sol.dv_distance,
            "od_extra_distance": sol.od_extra_distance,
            "runtime_sec": sol.runtime_sec, "gap": sol.optimality_gap,
            "validator_pass": sol.validator_pass,
        }
        lambda_rows.append(row); candidate_rows.append({**row, "label": f"λ={lam:g}"})

    epsilon_rows = []
    levels = config.objective.epsilon_levels
    for idx in range(levels):
        eps = e_min + (e_max - e_min) * idx / (levels - 1)
        sol = solver.solve(
            instance,
            {**base_obj, "mode": "epsilon_cost", "epsilon_emission": eps},
            exact_cfg,
        )
        if sol.status == "INFEASIBLE":
            continue
        if sol.status != "OPTIMAL" or not sol.validator_pass:
            raise RuntimeError(f"epsilon={eps} failed validation.")
        row = {
            "method": "epsilon_constraint", "lambda": None, "epsilon": eps,
            "cost": sol.cost, "emission": sol.emission,
            "objective": sol.objective, "dv_distance": sol.dv_distance,
            "od_extra_distance": sol.od_extra_distance,
            "runtime_sec": sol.runtime_sec, "gap": sol.optimality_gap,
            "validator_pass": sol.validator_pass,
        }
        epsilon_rows.append(row); candidate_rows.append({**row, "label": "ε"})

    unique = remove_duplicates(candidate_rows)
    pareto = nondominated(unique)

    out.save_rows("summary/payoff_table.csv", [
        {"anchor": "cost", "cost": cost_anchor.cost, "emission": cost_anchor.emission},
        {"anchor": "emission", "cost": emission_anchor.cost, "emission": emission_anchor.emission},
    ])
    out.save_rows("summary/lambda_summary.csv", lambda_rows)
    out.save_rows("summary/epsilon_summary.csv", epsilon_rows)
    out.save_rows("summary/nondominated_solutions.csv", pareto)
    out.save_manifest({
        "experiment": config.experiment_name,
        "solver": "exact",
        "instance": instance.get("metadata", {}),
        "cost_anchor": cost_anchor.to_dict(),
        "emission_anchor": emission_anchor.to_dict(),
        "nondominated_count": len(pareto),
    })
    if config.output.save_charts:
        plot_tradeoff(pareto, out.run_dir / "charts" / "exact_cost_emission_tradeoff.png")
        plot_routes(instance, cost_anchor, out.run_dir / "charts" / "cost_anchor_route.png", "Cost-optimal route")
        plot_routes(instance, emission_anchor, out.run_dir / "charts" / "emission_anchor_route.png", "Emission-optimal route")
    return out.run_dir
