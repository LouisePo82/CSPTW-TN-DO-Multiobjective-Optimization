from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from time import perf_counter
from typing import Any
import csv, json, random

from alns_solver.paper_alns_main import (
    PaperALNSCandidatePipelineResult, apply_paper_sa_transition,
    build_ml2_paper_controllers, select_paper_operator_pair,
)
from alns_solver.paper_operator_dispatch import (
    PAPER_DESTROY_OPERATOR_NAMES, PAPER_REPAIR_OPERATOR_NAMES,
    PaperOperatorDispatchContext, dispatch_paper_destroy,
    dispatch_paper_repair,
)
from alns_solver.paper_removal_quantity import (
    destroy_operator_uses_removal_quantity,
    sample_paper_removal_quantity,
)
from alns_solver.paper_history import PaperHistoricalPositionState
from alns_solver.local_search_factory import PAPER_LOCAL_SEARCH_MODE, build_local_search

@dataclass(frozen=True)
class PaperALNSRunConfig:
    instance_id: str
    lambda_value: float
    run_seed: int
    iteration_limit: int
    cost_bounds: tuple[float, float]
    emission_bounds: tuple[float, float]
    emission_factors: tuple[float, float] = (1.0, 1.0)
    objective_mode: str = "weighted"

    def validate(self):
        if not 0 <= self.lambda_value <= 1:
            raise ValueError("lambda_value must be in [0,1]")
        if self.iteration_limit <= 0:
            raise ValueError("iteration_limit must be positive")

@dataclass
class PaperALNSProductionResult:
    config: PaperALNSRunConfig
    initial_state: Any
    best_state: Any
    final_current_state: Any
    initial_solution: Any
    best_solution: Any
    final_current_solution: Any
    iteration_history: list[dict[str, Any]]
    operator_statistics: list[dict[str, Any]]
    runtime_seconds: float
    termination_reason: str
    metadata: dict[str, Any]

def _solution(state, instance, cfg):
    return state.to_core_solution(
        instance=instance,
        lambda_value=cfg.lambda_value,
        objective_mode=cfg.objective_mode,
        cost_bounds=cfg.cost_bounds,
        emission_bounds=cfg.emission_bounds,
        emission_factors=cfg.emission_factors,
        require_complete=True,
    )

def _seed(run_seed, iteration, stream):
    return run_seed * 1_000_003 + iteration * 101 + stream

def _state_payload(state):
    return {
        "dv_routes": {k: list(v) for k, v in state.dv_routes.items()},
        "od_routes": {k: list(v) for k, v in state.od_routes.items()},
        "assignments": {k: dict(v) for k, v in state.assignments.items()},
        "unassigned_customers": sorted(state.unassigned_customers),
    }

def _operator_rows(adaptive_state):
    rows = []
    for pool, records in (
        ("destroy", adaptive_state.destroy_records),
        ("repair", adaptive_state.repair_records),
    ):
        for name, record in records.items():
            rows.append({
                "pool": pool, "operator": name,
                "weight": float(record.weight),
                "segment_score": float(record.segment_score),
                "segment_uses": int(record.segment_uses),
            })
    return rows

def _record_non_effective(
    adaptive_controller,
    sa_controller,
    iteration,
    destroy_name,
    repair_name,
    current_objective,
    best_objective,
):
    """
    Record one selected but non-effective paper operator pair.

    No candidate is generated, so:
    - current and best objectives remain unchanged;
    - the selected pair records one zero-reward use;
    - SA temperature cools exactly once;
    - no resampling, fallback, or substitution occurs.
    """
    temperature_used = float(
        sa_controller.temperature_schedule
        .temperature_for_iteration(iteration)
    )

    adaptive_controller.process_iteration(
        iteration=iteration,
        destroy_operator=destroy_name,
        repair_operator=repair_name,
        candidate_objective=current_objective,
        current_objective=current_objective,
        best_objective=best_objective,
        accepted=False,
    )

    cooling_event = (
        sa_controller.temperature_schedule
        .cool_after_iteration(iteration)
    )

    return {
        "accepted": False,
        "reward_event": "non_effective_operator",
        "reward": 0.0,
        "temperature_used": temperature_used,
        "temperature_after_cooling": float(
            cooling_event[
                "temperature_after_cooling"
            ]
        ),
    }

def run_paper_alns_production(*, instance, initial_state, config):
    config.validate()
    started = perf_counter()
    initial_state = initial_state.copy()
    initial_solution = _solution(initial_state, instance, config)
    if not initial_solution.validator_pass:
        raise ValueError(initial_solution.validation_errors)

    current_state = initial_state.copy()
    best_state = initial_state.copy()
    current_solution = initial_solution
    best_solution = initial_solution

    adaptive, sa = build_ml2_paper_controllers(
        initial_objective=float(initial_solution.objective),
        destroy_operator_names=PAPER_DESTROY_OPERATOR_NAMES,
        repair_operator_names=PAPER_REPAIR_OPERATOR_NAMES,
        seed=_seed(config.run_seed, 0, 1),
    )
    roulette_rng = random.Random(_seed(config.run_seed, 0, 2))
    quantity_rng = random.Random(_seed(config.run_seed, 0, 3))
    history = PaperHistoricalPositionState.initialize_from_state(
        current_state, instance
    )
    rows = []

    for iteration in range(1, config.iteration_limit + 1):
        current_before = current_state.copy()
        best_before = best_state.copy()
        current_before_obj = float(current_solution.objective)
        best_before_obj = float(best_solution.objective)

        selection = select_paper_operator_pair(
            iteration=iteration,
            adaptive_state=adaptive.state,
            rng=roulette_rng,
        )
        destroy_name = selection.destroy_operator
        repair_name = selection.repair_operator

        if destroy_operator_uses_removal_quantity(destroy_name):
            q = sample_paper_removal_quantity(
                len(instance["customers"]),
                rng=quantity_rng,
                seed=_seed(config.run_seed, iteration, 4),
            ).quantity
        else:
            q = 1

        context = PaperOperatorDispatchContext(
            removal_count=q,
            seed=_seed(config.run_seed, iteration, 5),
            lambda_value=config.lambda_value,
            cost_bounds=config.cost_bounds,
            emission_bounds=config.emission_bounds,
            emission_factors=config.emission_factors,
            best_historical_position_costs=history.snapshot(),
        )

        destroy_result = repair_result = ls_result = candidate_solution = None
        non_effective = False
        error = ""
        try:
            destroy_result = dispatch_paper_destroy(
                destroy_name, current_state, instance, context=context
            )
            non_effective = not bool(destroy_result.removed_customers)
        except (ValueError, RuntimeError, KeyError) as exc:
            non_effective = True
            error = f"{type(exc).__name__}: {exc}"

        if non_effective:
            transition = _record_non_effective(
                adaptive.controller, sa.controller, iteration,
                destroy_name, repair_name, current_before_obj, best_before_obj,
            )
            current_state = current_before
            best_state = best_before
            current_solution = _solution(current_state, instance, config)
            best_solution = _solution(best_state, instance, config)
        else:
            repair_result = dispatch_paper_repair(
                repair_name, destroy_result.state, instance, context=context
            )
            if not repair_result.validator_pass:
                raise RuntimeError(repair_result.validation_errors)

            ls_result = build_local_search(
                repair_result.state, instance,
                mode=PAPER_LOCAL_SEARCH_MODE,
                best_objective=best_before_obj,
                lambda_value=config.lambda_value,
                cost_bounds=config.cost_bounds,
                emission_bounds=config.emission_bounds,
                emission_factors=config.emission_factors,
            )
            if not ls_result.metadata["paper_faithful"] or ls_result.metadata["enhanced"]:
                raise RuntimeError("Non-paper local search entered production")

            candidate_solution = _solution(ls_result.state, instance, config)
            if not candidate_solution.validator_pass:
                raise RuntimeError(candidate_solution.validation_errors)

            pipeline = PaperALNSCandidatePipelineResult(
                destroy_operator=destroy_name,
                repair_operator=repair_name,
                local_search_mode=PAPER_LOCAL_SEARCH_MODE,
                destroy_result=destroy_result,
                repair_result=repair_result,
                local_search_result=ls_result,
                candidate_state=ls_result.state.copy(),
                candidate_objective=float(candidate_solution.objective),
                candidate_cost=float(candidate_solution.cost),
                candidate_emission=float(candidate_solution.emission),
                candidate_dv_distance=float(candidate_solution.dv_distance),
                candidate_od_extra_distance=float(candidate_solution.od_extra_distance),
                validator_pass=True,
                validation_errors=[],
                metadata={
                    "paper_faithful": True, "enhanced": False,
                    "fallback": False, "operator_substitution": False,
                    "objective_input": "scalar_F_lambda",
                },
            )
            t = apply_paper_sa_transition(
                iteration=iteration,
                candidate_pipeline=pipeline,
                current_state=current_before,
                best_state=best_before,
                current_objective=current_before_obj,
                best_objective=best_before_obj,
                sa_controller=sa.controller,
            )
            current_state = t.current_state_after.copy()
            best_state = t.best_state_after.copy()
            current_solution = _solution(current_state, instance, config)
            best_solution = _solution(best_state, instance, config)
            transition = {
                "accepted": bool(t.accepted),
                "reward_event": t.reward_event,
                "reward": float(t.reward),
                "temperature_used": float(t.temperature_used),
                "temperature_after_cooling": float(t.temperature_after_cooling),
            }

        history.observe_current_state(current_state, instance)
        rows.append({
            "iteration": iteration,
            "destroy_operator": destroy_name,
            "repair_operator": repair_name,
            "removal_count": int(q),
            "removed_count": 0 if destroy_result is None else len(destroy_result.removed_customers),
            "non_effective_operator": non_effective,
            "non_effective_error": error,
            "candidate_objective": None if candidate_solution is None else float(candidate_solution.objective),
            "candidate_cost": None if candidate_solution is None else float(candidate_solution.cost),
            "candidate_emission": None if candidate_solution is None else float(candidate_solution.emission),
            "current_objective_before": current_before_obj,
            "current_objective_after": float(current_solution.objective),
            "best_objective_before": best_before_obj,
            "best_objective_after": float(best_solution.objective),
            "accepted": bool(transition["accepted"]),
            "reward_event": transition["reward_event"],
            "reward": float(transition["reward"]),
            "temperature_used": float(transition["temperature_used"]),
            "temperature_after_cooling": float(transition["temperature_after_cooling"]),
            "validation_pass": bool(current_solution.validator_pass),
        })

    return PaperALNSProductionResult(
        config=config,
        initial_state=initial_state,
        best_state=best_state,
        final_current_state=current_state,
        initial_solution=initial_solution,
        best_solution=best_solution,
        final_current_solution=current_solution,
        iteration_history=rows,
        operator_statistics=_operator_rows(adaptive.state),
        runtime_seconds=perf_counter() - started,
        termination_reason="iteration_limit",
        metadata={
            "paper_faithful": True, "enhanced": False,
            "full_destroy_pool": 12, "full_repair_pool": 6,
            "local_search_mode": PAPER_LOCAL_SEARCH_MODE,
            "objective_input": "scalar_F_lambda",
            "fallback": False, "operator_substitution": False,
            "operator_resampling": False,
            "non_effective_policy": "zero_reward_unchanged_state_no_resampling",
        },
    )

def export_production_result(result, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "run_config": output_dir / "run_config.json",
        "run_results_json": output_dir / "run_results.json",
        "run_results_csv": output_dir / "run_results.csv",
        "iteration_history": output_dir / "iteration_history.csv",
        "operator_statistics": output_dir / "operator_statistics.csv",
        "best_solution": output_dir / "best_solution.json",
    }
    paths["run_config"].write_text(json.dumps(asdict(result.config), indent=2), encoding="utf-8")
    summary = {
        "instance_id": result.config.instance_id,
        "lambda": result.config.lambda_value,
        "cost_weight": 1.0 - result.config.lambda_value,
        "emission_weight": result.config.lambda_value,
        "seed": result.config.run_seed,
        "iteration_limit": result.config.iteration_limit,
        "runtime_seconds": result.runtime_seconds,
        "initial_cost": float(result.initial_solution.cost),
        "initial_emission": float(result.initial_solution.emission),
        "initial_F_lambda": float(result.initial_solution.objective),
        "best_cost": float(result.best_solution.cost),
        "best_emission": float(result.best_solution.emission),
        "best_F_lambda": float(result.best_solution.objective),
        "dv_distance": float(result.best_solution.dv_distance),
        "od_extra_distance": float(result.best_solution.od_extra_distance),
        "accepted_count": sum(r["accepted"] for r in result.iteration_history),
        "new_global_best_count": sum(r["reward_event"] == "new_global_best" for r in result.iteration_history),
        "better_current_count": sum(r["reward_event"] == "better_current" for r in result.iteration_history),
        "worse_accepted_count": sum(r["reward_event"] == "worse_accepted" for r in result.iteration_history),
        "rejected_count": sum(r["reward_event"] == "rejected" for r in result.iteration_history),
        "non_effective_count": sum(r["non_effective_operator"] for r in result.iteration_history),
        "validation_pass": bool(result.best_solution.validator_pass),
        "termination_reason": result.termination_reason,
        "paper_faithful": True,
        "enhanced": False,
    }
    paths["run_results_json"].write_text(json.dumps({"summary":summary,"metadata":result.metadata}, indent=2), encoding="utf-8")
    with paths["run_results_csv"].open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary)); w.writeheader(); w.writerow(summary)
    with paths["iteration_history"].open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(result.iteration_history[0])); w.writeheader(); w.writerows(result.iteration_history)
    with paths["operator_statistics"].open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(result.operator_statistics[0])); w.writeheader(); w.writerows(result.operator_statistics)
    paths["best_solution"].write_text(json.dumps({
        "state": _state_payload(result.best_state),
        "metrics": {
            "cost": float(result.best_solution.cost),
            "emission": float(result.best_solution.emission),
            "F_lambda": float(result.best_solution.objective),
            "dv_distance": float(result.best_solution.dv_distance),
            "od_extra_distance": float(result.best_solution.od_extra_distance),
        },
        "validator_pass": bool(result.best_solution.validator_pass),
    }, indent=2), encoding="utf-8")
    return paths
