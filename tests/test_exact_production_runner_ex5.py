from __future__ import annotations

from pathlib import Path
import csv
import json

from core import (
    load_config,
    load_instance,
)
from experiments.run_exact_multiobjective import (
    run_exact_multiobjective,
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    config_path = (
        project_root
        / "configs"
        / "small_exact_multiobjective.yaml"
    )

    config = load_config(
        config_path
    )

    instance_path = Path(
        config.instance_path
    )
    if not instance_path.is_absolute():
        instance_path = (
            project_root / instance_path
        )

    config.instance_path = str(
        instance_path
    )
    config.output.root_dir = str(
        project_root
        / "outputs"
        / "exact_production_migration_tests"
    )

    # Keep the migration gate fast while exercising
    # both regimes and the tie point.
    config.objective.lambda_values = [
        0.0,
        0.5,
        1.0,
    ]
    config.objective.epsilon_levels = 3
    config.output.save_charts = True

    instance = load_instance(
        config.instance_path
    )

    run_dir = run_exact_multiobjective(
        config,
        instance,
    )

    required_top_level = {
        "manifest.json",
        "config_used.yaml",
    }

    actual_top_level = {
        path.name
        for path in run_dir.iterdir()
        if path.is_file()
    }

    missing_top_level = (
        required_top_level
        - actual_top_level
    )

    if missing_top_level:
        raise AssertionError(
            "Missing exact production files: "
            f"{sorted(missing_top_level)}"
        )

    manifest = json.loads(
        (
            run_dir / "manifest.json"
        ).read_text(encoding="utf-8")
    )

    if (
        manifest[
            "common_artifact_contract"
        ]
        is not True
    ):
        raise AssertionError(
            "Exact production manifest does not "
            "enable the common artifact contract."
        )

    if (
        manifest[
            "shared_route_visualization"
        ]
        is not True
    ):
        raise AssertionError(
            "Exact production runner is not using "
            "shared route visualization."
        )

    common_runs = sorted(
        (
            run_dir / "common_runs"
        ).glob("lambda_*")
    )

    if len(common_runs) != 3:
        raise AssertionError(
            "Expected 3 exact common runs, "
            f"found {len(common_runs)}."
        )

    required_common_files = {
        "run_config.json",
        "run_results.json",
        "run_results.csv",
        "best_solution.json",
        "best_route_map.png",
        "artifact_manifest.json",
    }

    for common_run in common_runs:
        actual = {
            path.name
            for path in common_run.iterdir()
            if path.is_file()
        }

        missing = (
            required_common_files - actual
        )

        if missing:
            raise AssertionError(
                f"{common_run} is missing: "
                f"{sorted(missing)}"
            )

        for filename in (
            required_common_files
        ):
            path = common_run / filename

            if path.stat().st_size <= 0:
                raise AssertionError(
                    f"Empty exact artifact: {path}"
                )

    for anchor_name in (
        "cost_anchor",
        "emission_anchor",
    ):
        anchor_dir = (
            run_dir
            / "anchor_artifacts"
            / anchor_name
        )

        map_path = (
            anchor_dir
            / "best_route_map.png"
        )

        if (
            not map_path.exists()
            or map_path.stat().st_size <= 0
        ):
            raise AssertionError(
                f"Missing anchor map: {map_path}"
            )

    consolidated_path = (
        run_dir
        / "summary"
        / "benchmark_outputs_consolidate.csv"
    )

    with consolidated_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(
            csv.DictReader(file)
        )

    if len(rows) != 3:
        raise AssertionError(
            "Exact common consolidation must "
            "contain one row per lambda."
        )

    if {
        float(row["lambda"])
        for row in rows
    } != {0.0, 0.5, 1.0}:
        raise AssertionError(
            "Unexpected lambda values in exact "
            "common consolidation."
        )

    legacy_summary = (
        run_dir
        / "summary"
        / "lambda_summary.csv"
    )

    if not legacy_summary.exists():
        raise AssertionError(
            "Backward-compatible exact summary "
            "was not retained."
        )

    legacy_route_imports = (
        project_root
        / "experiments"
        / "run_exact_multiobjective.py"
    ).read_text(encoding="utf-8")

    if (
        "visualization.route_plot"
        in legacy_route_imports
    ):
        raise AssertionError(
            "Legacy exact route_plot import "
            "is still present."
        )

    print(
        "[PASS] Exact production runner retains legacy summaries"
    )
    print(
        "[PASS] Exact weighted runs export the common artifact contract"
    )
    print(
        "[PASS] Exact cost and emission anchors use shared route visualization"
    )
    print(
        "[PASS] Exact common consolidation contains one row per lambda"
    )
    print(
        "[PASS] Legacy exact route_plot dependency is removed"
    )
    print(f"\nRun directory: {run_dir}")
    print(
        "\nEX-5 — EXACT PRODUCTION RUNNER "
        "MIGRATION PASSED"
    )


if __name__ == "__main__":
    main()
