from __future__ import annotations

from pathlib import Path
import csv
import json
import shutil

from main import run_from_config


def main() -> None:
    project_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    config_path = (
        project_root
        / "configs"
        / "main1_small_alns_smoke.yaml"
    )

    output_root = (
        project_root
        / "outputs"
        / "main1_alns_smoke"
    )

    if output_root.exists():
        shutil.rmtree(
            output_root
        )

    returned_root = run_from_config(
        config_path
    )

    if returned_root != output_root:
        raise AssertionError(
            "Unified launcher returned an "
            "unexpected output root."
        )

    manifest_path = (
        output_root
        / "launcher_manifest.json"
    )
    consolidated_path = (
        output_root
        / "benchmark_outputs_consolidate.csv"
    )
    summary_path = (
        output_root
        / "lambda_summary.csv"
    )

    for path in (
        manifest_path,
        consolidated_path,
        summary_path,
    ):
        if (
            not path.exists()
            or path.stat().st_size <= 0
        ):
            raise AssertionError(
                f"Missing MAIN-1 artifact: {path}"
            )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    if (
        manifest["experiment_type"]
        != "alns_multiobjective"
    ):
        raise AssertionError(
            "Launcher experiment type is incorrect."
        )

    if (
        manifest["scope_lock"][
            "paper_faithful_alns"
        ]
        is not True
    ):
        raise AssertionError(
            "Paper-faithful scope lock is missing."
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
            "Expected 3 MAIN-1 ALNS smoke runs."
        )

    if {
        float(row["lambda"])
        for row in rows
    } != {0.0, 0.5, 1.0}:
        raise AssertionError(
            "Unexpected lambda values in "
            "MAIN-1 smoke output."
        )

    if {
        int(row["seed"])
        for row in rows
    } != {2026}:
        raise AssertionError(
            "Unexpected seed values in "
            "MAIN-1 smoke output."
        )

    if not all(
        str(
            row["validation_pass"]
        ).lower()
        == "true"
        for row in rows
    ):
        raise AssertionError(
            "At least one MAIN-1 smoke run "
            "failed validation."
        )

    print(
        "[PASS] Unified launcher reads an ALNS YAML config"
    )
    print(
        "[PASS] Unified launcher calls the existing paper ALNS batch runner"
    )
    print(
        "[PASS] Unified launcher consolidates ALNS outputs"
    )
    print(
        "[PASS] Unified launcher preserves normalization and scope lock"
    )
    print(
        "[PASS] MAIN-1 smoke output contains 3 valid lambda runs"
    )
    print(
        "\nMAIN-1 — UNIFIED ALNS LAUNCHER PASSED"
    )


if __name__ == "__main__":
    main()
