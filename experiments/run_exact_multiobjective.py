from __future__ import annotations

from pathlib import Path
import csv

from core.pareto import (
    nondominated,
    remove_duplicates,
)
from reporting import OutputManager
from reporting.common_artifact_exporter import (
    export_exact_common_artifacts,
)
from reporting.exact_lambda_comparison import (
    plot_exact_lambda_route_comparison,
)
from reporting.route_visualization import (
    plot_solution_routes,
    plot_vehicle_routes,
)
from visualization.pareto_plot import (
    plot_tradeoff,
)

from .solver_factory import create_solver


def _assert_exact_ground_truth(
    *,
    label: str,
    solution,
    require_optimal: bool = True,
) -> None:
    """
    Validate whether an exact-solver result is usable.

    Production ground-truth runs require OPTIMAL status by
    default. Time-limited smoke or bounded benchmark runs may
    explicitly allow a valid FEASIBLE incumbent, while preserving
    its non-optimal status and reported MIP gap.
    """
    accepted_statuses = (
        {"OPTIMAL"}
        if require_optimal
        else {"OPTIMAL", "FEASIBLE"}
    )

    if (
        solution.status not in accepted_statuses
        or not solution.validator_pass
    ):
        policy = (
            "OPTIMAL"
            if require_optimal
            else "OPTIMAL or FEASIBLE"
        )

        raise RuntimeError(
            f"{label} failed exact-result "
            f"requirements: expected={policy}, "
            f"status={solution.status}, "
            f"gap={solution.optimality_gap}, "
            f"errors={solution.validation_errors}"
        )


def _resolved_config(
    config,
) -> dict:
    return {
        **config.raw,
        "instance_path": str(
            config.instance_path
        ),
        "objective": {
            **config.raw.get(
                "objective",
                {},
            ),
            "mode": (
                config.objective.mode
            ),
            "emission_factor_dv": (
                config.objective
                .emission_factor_dv
            ),
            "emission_factor_od": (
                config.objective
                .emission_factor_od
            ),
            "lambda_values": list(
                config.objective
                .lambda_values
            ),
            "epsilon_levels": (
                config.objective
                .epsilon_levels
            ),
        },
        "exact": {
            **config.raw.get(
                "exact",
                {},
            ),
            "time_limit_sec": (
                config.exact
                .time_limit_sec
            ),
            "mip_gap": (
                config.exact.mip_gap
            ),
            "enable_output": (
                config.exact
                .enable_output
            ),
            "require_optimal": bool(
                config.exact.require_optimal
            ),
        },
        "output": {
            **config.raw.get(
                "output",
                {},
            ),
            "root_dir": str(
                config.output
                .root_dir
            ),
            "save_solution_details": (
                config.output
                .save_solution_details
            ),
            "save_charts": (
                config.output
                .save_charts
            ),
            "save_instance_snapshot": (
                config.output
                .save_instance_snapshot
            ),
        },
    }


def _save_shared_route_artifacts(
    *,
    instance: dict,
    solution,
    output_dir: Path,
    instance_id: str,
    title: str,
    lambda_value: float | None,
) -> dict[str, object]:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined = (
        plot_solution_routes(
            instance=instance,
            state=solution,
            output_path=(
                output_dir
                / "best_route_map.png"
            ),
            instance_id=instance_id,
            lambda_value=lambda_value,
            seed=None,
            title=title,
        )
    )

    vehicle_maps = (
        plot_vehicle_routes(
            instance=instance,
            state=solution,
            output_dir=(
                output_dir
                / "vehicle_routes"
            ),
            instance_id=instance_id,
            lambda_value=lambda_value,
            seed=None,
        )
    )

    return {
        "combined_route_map": combined,
        "vehicle_route_maps": (
            vehicle_maps
        ),
    }


def _read_single_csv_row(
    path: Path,
) -> dict[str, str]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return next(
            csv.DictReader(file)
        )


def run_exact_multiobjective(
    config,
    instance,
):
    solver = create_solver("exact")

    out = OutputManager(
        config.output.root_dir,
        config.experiment_name,
    )

    out.save_config(
        _resolved_config(config)
    )

    if (
        config.output
        .save_instance_snapshot
    ):
        out.snapshot_instance(
            config.instance_path
        )

    metadata = instance.get(
        "metadata",
        {},
    )

    instance_id = str(
        metadata.get(
            "instance_id",
            Path(
                config.instance_path
            ).name,
        )
    )

    base_objective = {
        "emission_factor_dv": (
            config.objective
            .emission_factor_dv
        ),
        "emission_factor_od": (
            config.objective
            .emission_factor_od
        ),
    }

    emission_factors = (
        float(
            config.objective
            .emission_factor_dv
        ),
        float(
            config.objective
            .emission_factor_od
        ),
    )

    require_optimal = bool(
        config.exact.require_optimal
    )

    # Runner-only acceptance policy must not be
    # forwarded to the exact solver.
    exact_config = {
        key: value
        for key, value in vars(
            config.exact
        ).items()
        if key != "require_optimal"
    }

    cost_anchor = solver.solve(
        instance,
        {
            **base_objective,
            "mode": "cost",
            "lambda_value": 0.0,
        },
        exact_config,
    )

    emission_anchor = solver.solve(
        instance,
        {
            **base_objective,
            "mode": "emission",
            "lambda_value": 1.0,
        },
        exact_config,
    )

    for label, solution in (
        (
            "cost_anchor",
            cost_anchor,
        ),
        (
            "emission_anchor",
            emission_anchor,
        ),
    ):
        _assert_exact_ground_truth(
            label=label,
            solution=solution,
            require_optimal=require_optimal,
        )

        out.save_solution(
            label,
            solution,
        )

    cost_min = float(
        cost_anchor.cost
    )
    cost_max = float(
        emission_anchor.cost
    )
    emission_min = float(
        emission_anchor.emission
    )
    emission_max = float(
        cost_anchor.emission
    )

    if (
        cost_max <= cost_min
        or emission_max <= emission_min
    ):
        raise RuntimeError(
            "No positive cost-emission payoff range."
        )

    cost_bounds = (
        cost_min,
        cost_max,
    )
    emission_bounds = (
        emission_min,
        emission_max,
    )

    anchor_root = (
        out.run_dir
        / "anchor_artifacts"
    )

    cost_anchor_artifacts = (
        _save_shared_route_artifacts(
            instance=instance,
            solution=cost_anchor,
            output_dir=(
                anchor_root
                / "cost_anchor"
            ),
            instance_id=instance_id,
            lambda_value=0.0,
            title=(
                "Crowd-Shipping Route Map — "
                "Exact cost-optimal solution\n"
                f"{instance_id} | Exact MILP | "
                f"{cost_anchor.status}"
            ),
        )
    )

    emission_anchor_artifacts = (
        _save_shared_route_artifacts(
            instance=instance,
            solution=emission_anchor,
            output_dir=(
                anchor_root
                / "emission_anchor"
            ),
            instance_id=instance_id,
            lambda_value=1.0,
            title=(
                "Crowd-Shipping Route Map — "
                "Exact emission-optimal solution\n"
                f"{instance_id} | Exact MILP | "
                f"{emission_anchor.status}"
            ),
        )
    )

    lambda_rows: list[dict] = []
    candidate_rows: list[dict] = []
    common_result_rows: list[
        dict
    ] = []
    common_run_dirs: list[
        str
    ] = []
    comparison_panels: list[
        dict
    ] = []

    for lambda_value in (
        config.objective.lambda_values
    ):
        lambda_value = float(
            lambda_value
        )

        solution = solver.solve(
            instance,
            {
                **base_objective,
                "mode": "weighted",
                "lambda_value": (
                    lambda_value
                ),
                "cost_bounds": (
                    cost_bounds
                ),
                "emission_bounds": (
                    emission_bounds
                ),
            },
            exact_config,
        )

        _assert_exact_ground_truth(
            label=(
                f"lambda="
                f"{lambda_value:g}"
            ),
            solution=solution,
            require_optimal=require_optimal,
        )

        label = (
            f"lambda_"
            f"{lambda_value:.2f}"
        )

        out.save_solution(
            label,
            solution,
        )

        legacy_row = {
            "method": (
                "weighted_sum"
            ),
            "lambda": lambda_value,
            "epsilon": None,
            "cost": solution.cost,
            "emission": (
                solution.emission
            ),
            "objective": (
                solution.objective
            ),
            "dv_distance": (
                solution.dv_distance
            ),
            "od_extra_distance": (
                solution.od_extra_distance
            ),
            "runtime_sec": (
                solution.runtime_sec
            ),
            "gap": (
                solution.optimality_gap
            ),
            "validator_pass": (
                solution.validator_pass
            ),
        }

        lambda_rows.append(
            legacy_row
        )

        candidate_rows.append(
            {
                **legacy_row,
                "label": (
                    f"λ={lambda_value:g}"
                ),
            }
        )

        common_run_dir = (
            out.run_dir
            / "common_runs"
            / (
                f"lambda_"
                f"{lambda_value:g}"
            )
        )

        common_artifacts = (
            export_exact_common_artifacts(
                instance=instance,
                solution=solution,
                output_dir=(
                    common_run_dir
                ),
                instance_id=(
                    instance_id
                ),
                lambda_value=(
                    lambda_value
                ),
                cost_bounds=(
                    cost_bounds
                ),
                emission_bounds=(
                    emission_bounds
                ),
                emission_factors=(
                    emission_factors
                ),
                time_limit_seconds=(
                    config.exact
                    .time_limit_sec
                ),
                mip_gap_target=(
                    config.exact
                    .mip_gap
                ),
            )
        )

        common_run_dirs.append(
            str(
                common_run_dir
                .relative_to(
                    out.run_dir
                )
            )
        )

        common_result_rows.append(
            _read_single_csv_row(
                Path(
                    common_artifacts[
                        "run_results_csv"
                    ]
                )
            )
        )

        comparison_panels.append(
            {
                "lambda_value": (
                    lambda_value
                ),
                "image_path": Path(
                    common_artifacts[
                        "combined_route_map"
                    ]
                ),
                "cost": (
                    solution.cost
                ),
                "emission": (
                    solution.emission
                ),
                "status": (
                    solution.status
                ),
            }
        )

    comparison_map = (
        plot_exact_lambda_route_comparison(
            panels=(
                comparison_panels
            ),
            output_path=(
                out.run_dir
                / "charts"
                / (
                    "exact_lambda_"
                    "route_comparison.png"
                )
            ),
            title=(
                "Exact weighted-sum route "
                f"comparison — {instance_id}"
            ),
        )
    )

    epsilon_rows: list[
        dict
    ] = []

    levels = int(
        config.objective
        .epsilon_levels
    )

    if levels < 2:
        raise ValueError(
            "epsilon_levels must be at least 2."
        )

    for index in range(
        levels
    ):
        epsilon_value = (
            emission_min
            + (
                emission_max
                - emission_min
            )
            * index
            / (levels - 1)
        )

        solution = solver.solve(
            instance,
            {
                **base_objective,
                "mode": (
                    "epsilon_cost"
                ),
                "epsilon_emission": (
                    epsilon_value
                ),
            },
            exact_config,
        )

        if (
            solution.status
            == "INFEASIBLE"
        ):
            continue

        _assert_exact_ground_truth(
            label=(
                f"epsilon="
                f"{epsilon_value}"
            ),
            solution=solution,
            require_optimal=require_optimal,
        )

        row = {
            "method": (
                "epsilon_constraint"
            ),
            "lambda": None,
            "epsilon": (
                epsilon_value
            ),
            "cost": solution.cost,
            "emission": (
                solution.emission
            ),
            "objective": (
                solution.objective
            ),
            "dv_distance": (
                solution.dv_distance
            ),
            "od_extra_distance": (
                solution.od_extra_distance
            ),
            "runtime_sec": (
                solution.runtime_sec
            ),
            "gap": (
                solution.optimality_gap
            ),
            "validator_pass": (
                solution.validator_pass
            ),
        }

        epsilon_rows.append(
            row
        )

        candidate_rows.append(
            {
                **row,
                "label": "ε",
            }
        )

    unique_candidates = (
        remove_duplicates(
            candidate_rows
        )
    )

    pareto_rows = (
        nondominated(
            unique_candidates
        )
    )

    out.save_rows(
        "summary/payoff_table.csv",
        [
            {
                "anchor": "cost",
                "cost": (
                    cost_anchor.cost
                ),
                "emission": (
                    cost_anchor.emission
                ),
            },
            {
                "anchor": "emission",
                "cost": (
                    emission_anchor.cost
                ),
                "emission": (
                    emission_anchor
                    .emission
                ),
            },
        ],
    )

    out.save_rows(
        "summary/lambda_summary.csv",
        lambda_rows,
    )

    out.save_rows(
        "summary/epsilon_summary.csv",
        epsilon_rows,
    )

    out.save_rows(
        (
            "summary/"
            "nondominated_solutions.csv"
        ),
        pareto_rows,
    )

    out.save_rows(
        (
            "summary/"
            "benchmark_outputs_"
            "consolidate.csv"
        ),
        common_result_rows,
    )

    manifest = {
        "experiment": (
            config.experiment_name
        ),
        "solver": "exact",
        "instance": metadata,
        "cost_anchor": (
            cost_anchor.to_dict()
        ),
        "emission_anchor": (
            emission_anchor.to_dict()
        ),
        "normalization_anchors": {
            "cost_bounds": list(
                cost_bounds
            ),
            "emission_bounds": list(
                emission_bounds
            ),
            "source": (
                "exact_objective_extremes"
            ),
        },
        "nondominated_count": len(
            pareto_rows
        ),
        "common_artifact_contract": True,
        "common_run_directories": (
            common_run_dirs
        ),
        "shared_route_visualization": True,
        "lambda_route_comparison": str(
            comparison_map.relative_to(
                out.run_dir
            )
        ),
        "anchor_artifacts": {
            "cost_anchor": {
                "combined_route_map": str(
                    Path(
                        cost_anchor_artifacts[
                            "combined_route_map"
                        ]
                    ).relative_to(
                        out.run_dir
                    )
                ),
                "vehicle_route_maps": [
                    str(
                        Path(path)
                        .relative_to(
                            out.run_dir
                        )
                    )
                    for path in (
                        cost_anchor_artifacts[
                            "vehicle_route_maps"
                        ]
                    )
                ],
            },
            "emission_anchor": {
                "combined_route_map": str(
                    Path(
                        emission_anchor_artifacts[
                            "combined_route_map"
                        ]
                    ).relative_to(
                        out.run_dir
                    )
                ),
                "vehicle_route_maps": [
                    str(
                        Path(path)
                        .relative_to(
                            out.run_dir
                        )
                    )
                    for path in (
                        emission_anchor_artifacts[
                            "vehicle_route_maps"
                        ]
                    )
                ],
            },
        },
    }

    out.save_manifest(
        manifest
    )

    if (
        config.output
        .save_charts
    ):
        plot_tradeoff(
            pareto_rows,
            (
                out.run_dir
                / "charts"
                / (
                    "exact_cost_emission_"
                    "tradeoff.png"
                )
            ),
        )

    return out.run_dir
