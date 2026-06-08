from __future__ import annotations
from core.solution import Solution
from core.validator import validate_solution
from core.schedule import recompute_earliest_schedule
from .model_exact_milp import solve_exact_milp

class ExactSolver:
    name = "exact"

    def solve(self, instance: dict, objective_config: dict, solver_config: dict) -> Solution:
        factors = (
            float(objective_config.get("emission_factor_dv", 1.0)),
            float(objective_config.get("emission_factor_od", 1.0)),
        )
        raw = solve_exact_milp(
            instance,
            lambda_val=float(objective_config.get("lambda_value", 0.0)),
            objective_mode=objective_config.get("mode", "cost"),
            cost_bounds=objective_config.get("cost_bounds"),
            emission_bounds=objective_config.get("emission_bounds"),
            emission_factors=factors,
            epsilon_emission=objective_config.get("epsilon_emission"),
            time_limit_sec=int(solver_config.get("time_limit_sec", 300)),
            mip_gap=solver_config.get("mip_gap", 0.0),
            enable_output=bool(solver_config.get("enable_output", False)),
        )
        if raw.get("status") not in {"OPTIMAL", "FEASIBLE"}:
            return Solution(
                status=raw.get("status", "UNKNOWN"),
                solver_name=self.name,
                objective_mode=objective_config.get("mode", "cost"),
                lambda_value=objective_config.get("lambda_value"),
                runtime_sec=raw.get("comp_time_sec", 0.0),
            )

        validation = validate_solution(instance, raw, emission_factors=factors)
        earliest = recompute_earliest_schedule(instance, raw)
        return Solution(
            status=raw["status"],
            solver_name=self.name,
            objective_mode=raw["objective_mode"],
            lambda_value=raw.get("lambda"),
            cost=raw["cost"],
            emission=raw["emission"],
            objective=raw["objective"],
            dv_distance=raw["dv_distance"],
            od_extra_distance=raw["od_extra_distance"],
            dv_routes=raw["dv_routes"],
            od_routes=raw["od_routes"],
            assignments=raw["assignments"],
            arrival_times=earliest,
            vehicle_loads=raw["vehicle_load"],
            tn_demands=raw["tn_demand"],
            adp_loads=raw["adp_load"],
            runtime_sec=raw["comp_time_sec"],
            optimality_gap=raw["mip_gap_abs"],
            validator_pass=validation["valid"],
            validation_errors=validation["errors"],
            metadata={
                "solver_version": raw.get(
                    "solver_version"
                ),
                "best_bound": raw.get(
                    "best_bound"
                ),
                "absolute_mip_gap": raw.get(
                    "mip_gap_abs"
                ),
                "relative_mip_gap": raw.get(
                    "mip_gap_rel"
                ),
                "relative_mip_gap_percent": (
                    raw.get(
                        "mip_gap_percent"
                    )
                ),
                "optimality_gap_semantics": (
                    "absolute_objective_gap"
                ),
                "configured_mip_gap_target": (
                    solver_config.get(
                        "mip_gap",
                        0.0,
                    )
                ),
                "raw_solver_times": raw.get(
                    "times"
                ),
            },
        )
