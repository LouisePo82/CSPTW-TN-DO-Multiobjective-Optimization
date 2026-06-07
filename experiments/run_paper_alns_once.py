from __future__ import annotations

from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path
from typing import Any
import json

from core.instance_loader import load_instance
from alns_solver.paper_alns_main import (
    build_ml1_paper_initial_state,
)
from alns_solver.paper_production_runner import (
    PaperALNSRunConfig,
    export_production_result,
    run_paper_alns_production,
)
from reporting.route_visualization import (
    plot_solution_routes,
    plot_vehicle_routes,
)


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description=(
            "Run one paper-faithful ALNS production experiment "
            "and export all run artifacts."
        )
    )

    parser.add_argument("--instance", required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument(
        "--lambda-value",
        type=float,
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--iterations",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--cost-min",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--cost-max",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--emission-min",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--emission-max",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--dv-emission-factor",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--od-emission-factor",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--output-root",
        default="outputs/production_runs",
    )

    return parser.parse_args()


def _build_run_directory(
    *,
    output_root: Path,
    config: PaperALNSRunConfig,
    timestamp: str,
) -> Path:
    lambda_label = f"{config.lambda_value:g}"

    return (
        output_root
        / config.instance_id
        / f"lambda_{lambda_label}"
        / f"seed_{config.run_seed}"
        / timestamp
    )


def _relative_artifact_path(
    *,
    run_dir: Path,
    artifact_path: Path,
) -> str:
    try:
        return str(artifact_path.relative_to(run_dir))
    except ValueError:
        return str(artifact_path)


def _write_artifact_manifest(
    *,
    run_dir: Path,
    config: PaperALNSRunConfig,
    artifacts: dict[str, Any],
    vehicle_route_paths: list[Path],
) -> Path:
    manifest_path = run_dir / "artifact_manifest.json"

    manifest = {
        "run_identity": {
            "instance_id": config.instance_id,
            "lambda_value": float(config.lambda_value),
            "seed": int(config.run_seed),
            "iteration_limit": int(config.iteration_limit),
        },
        "scope_lock": {
            "paper_faithful_alns": True,
            "objective_extension": "scalar_F_lambda",
            "enhanced_alns": False,
        },
        "artifacts": {
            key: _relative_artifact_path(
                run_dir=run_dir,
                artifact_path=Path(value),
            )
            for key, value in artifacts.items()
        },
        "vehicle_route_maps": [
            _relative_artifact_path(
                run_dir=run_dir,
                artifact_path=path,
            )
            for path in vehicle_route_paths
        ],
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return manifest_path


def _json_ready_artifacts(
    artifacts: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}

    for key, value in artifacts.items():
        if isinstance(value, list):
            result[key] = [
                str(item)
                for item in value
            ]
        else:
            result[key] = str(value)

    return result


def main() -> None:
    args = parse_args()
    instance = load_instance(Path(args.instance))

    config = PaperALNSRunConfig(
        instance_id=args.instance_id,
        lambda_value=args.lambda_value,
        run_seed=args.seed,
        iteration_limit=args.iterations,
        cost_bounds=(
            args.cost_min,
            args.cost_max,
        ),
        emission_bounds=(
            args.emission_min,
            args.emission_max,
        ),
        emission_factors=(
            args.dv_emission_factor,
            args.od_emission_factor,
        ),
    )

    initial_state = build_ml1_paper_initial_state(
        instance,
        seed=config.run_seed,
        lambda_value=config.lambda_value,
        cost_bounds=config.cost_bounds,
        emission_bounds=config.emission_bounds,
        emission_factors=config.emission_factors,
    )

    result = run_paper_alns_production(
        instance=instance,
        initial_state=initial_state,
        config=config,
    )

    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    run_dir = _build_run_directory(
        output_root=Path(args.output_root),
        config=config,
        timestamp=timestamp,
    )

    artifacts = export_production_result(
        result,
        run_dir,
    )

    combined_route_map = plot_solution_routes(
        instance=instance,
        state=result.best_state,
        output_path=run_dir / "best_route_map.png",
        instance_id=config.instance_id,
        lambda_value=config.lambda_value,
        seed=config.run_seed,
    )

    vehicle_route_paths = plot_vehicle_routes(
        instance=instance,
        state=result.best_state,
        output_dir=run_dir / "vehicle_routes",
        instance_id=config.instance_id,
        lambda_value=config.lambda_value,
        seed=config.run_seed,
    )

    artifacts["combined_route_map"] = combined_route_map
    artifacts["vehicle_route_maps"] = vehicle_route_paths

    artifact_manifest = _write_artifact_manifest(
        run_dir=run_dir,
        config=config,
        artifacts={
            key: value
            for key, value in artifacts.items()
            if key != "vehicle_route_maps"
        },
        vehicle_route_paths=vehicle_route_paths,
    )

    artifacts["artifact_manifest"] = artifact_manifest

    print(
        json.dumps(
            _json_ready_artifacts(artifacts),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
