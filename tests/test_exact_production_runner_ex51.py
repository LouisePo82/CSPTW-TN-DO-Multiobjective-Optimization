from __future__ import annotations

from pathlib import Path
import csv
import json
import yaml

from core import (
    load_config,
    load_instance,
)
from experiments.run_exact_multiobjective import (
    run_exact_multiobjective,
)


def main() -> None:
    project_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

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
            project_root
            / instance_path
        )

    config.instance_path = str(
        instance_path
    )

    config.output.root_dir = str(
        project_root
        / "outputs"
        / (
            "exact_production_"
            "migration_tests"
        )
    )

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

    run_dir = (
        run_exact_multiobjective(
            config,
            instance,
        )
    )

    resolved_config = yaml.safe_load(
        (
            run_dir
            / "config_used.yaml"
        ).read_text(
            encoding="utf-8"
        )
    )

    if (
        resolved_config[
            "objective"
        ]["lambda_values"]
        != [0.0, 0.5, 1.0]
    ):
        raise AssertionError(
            "config_used.yaml does not "
            "contain resolved lambdas."
        )

    if (
        resolved_config[
            "objective"
        ]["epsilon_levels"]
        != 3
    ):
        raise AssertionError(
            "config_used.yaml does not "
            "contain resolved epsilon levels."
        )

    comparison_path = (
        run_dir
        / "charts"
        / (
            "exact_lambda_"
            "route_comparison.png"
        )
    )

    if (
        not comparison_path.exists()
        or comparison_path.stat().st_size
        <= 0
    ):
        raise AssertionError(
            "Combined lambda route "
            "comparison was not created."
        )

    common_runs = sorted(
        (
            run_dir
            / "common_runs"
        ).glob("lambda_*")
    )

    if len(common_runs) != 3:
        raise AssertionError(
            "Expected 3 common runs."
        )

    for common_run in (
        common_runs
    ):
        run_results = json.loads(
            (
                common_run
                / "run_results.json"
            ).read_text(
                encoding="utf-8"
            )
        )

        summary = (
            run_results["summary"]
        )

        if (
            float(
                summary[
                    "time_limit_seconds"
                ]
            )
            != float(
                config.exact
                .time_limit_sec
            )
        ):
            raise AssertionError(
                "Exact time limit was not "
                "preserved."
            )

        if (
            float(
                summary[
                    "mip_gap_target"
                ]
            )
            != float(
                config.exact.mip_gap
            )
        ):
            raise AssertionError(
                "Exact mip-gap target was "
                "not preserved."
            )

    consolidated_path = (
        run_dir
        / "summary"
        / (
            "benchmark_outputs_"
            "consolidate.csv"
        )
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
            "Expected 3 consolidated rows."
        )

    if {
        float(row["lambda"])
        for row in rows
    } != {0.0, 0.5, 1.0}:
        raise AssertionError(
            "Unexpected lambdas in "
            "consolidated output."
        )

    manifest = json.loads(
        (
            run_dir / "manifest.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    if (
        manifest[
            "lambda_route_comparison"
        ]
        != (
            "charts/"
            "exact_lambda_"
            "route_comparison.png"
        )
    ):
        raise AssertionError(
            "Manifest does not reference "
            "the lambda comparison chart."
        )

    print(
        "[PASS] Resolved config provenance is correct"
    )
    print(
        "[PASS] Exact time limit and mip-gap target are preserved"
    )
    print(
        "[PASS] Exact weighted runs keep the common artifact contract"
    )
    print(
        "[PASS] All lambda route maps are combined into one comparison chart"
    )
    print(
        "[PASS] Exact consolidated output contains one row per lambda"
    )
    print(
        f"\nRun directory: {run_dir}"
    )
    print(
        "\nEX-5.1 — EXACT PRODUCTION "
        "PROVENANCE AND COMPARISON PASSED"
    )


if __name__ == "__main__":
    main()
