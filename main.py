from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from core import load_config, load_instance
from experiments.run_exact_multiobjective import (
    run_exact_multiobjective,
)
from experiments.run_micro_validation import (
    run_micro_validation,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def _resolve_path(
    value: str | Path,
    *,
    base_dir: Path,
) -> Path:
    path = Path(value)

    if not path.is_absolute():
        path = (base_dir / path).resolve()

    return path


def _load_raw_yaml(
    config_path: Path,
) -> dict[str, Any]:
    payload = yaml.safe_load(
        config_path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(payload, dict):
        raise ValueError(
            "Experiment config must contain a YAML mapping."
        )

    return payload


def _experiment_type(
    payload: dict[str, Any],
) -> str:
    """
    Resolve the experiment type from either supported schema.

    New unified schema:
        experiment:
          type: alns_multiobjective

    Legacy project schema:
        experiment_type: exact_multiobjective
    """
    value = None

    experiment = payload.get("experiment")

    if isinstance(experiment, dict):
        value = experiment.get("type")
    elif isinstance(experiment, str):
        value = experiment

    if value is None:
        value = payload.get("experiment_type")

    if value is None:
        value = payload.get("type")

    if value is None:
        raise ValueError(
            "Config must define either "
            "'experiment.type' or 'experiment_type'."
        )

    return str(value).strip()


def _run_command(
    command: list[str],
) -> None:
    print(
        "\n[MAIN-1] Executing:\n"
        + " ".join(command)
        + "\n"
    )

    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            "Experiment command failed with "
            f"exit code {completed.returncode}."
        )


def _comma_join(
    values: list[Any],
) -> str:
    return ",".join(
        str(value)
        for value in values
    )


def _run_micro_validation(
    config_path: Path,
) -> Path:
    """
    Preserve the legacy micro-validation workflow.

    This branch uses the existing project config loader,
    instance loader, and micro-validation runner.
    """
    config = load_config(
        config_path
    )

    instance_path = _resolve_path(
        config.instance_path,
        base_dir=PROJECT_ROOT,
    )
    config.instance_path = str(
        instance_path
    )

    output_root = _resolve_path(
        config.output.root_dir,
        base_dir=PROJECT_ROOT,
    )
    config.output.root_dir = str(
        output_root
    )

    instance = load_instance(
        instance_path
    )

    run_dir = run_micro_validation(
        config,
        instance,
    )

    print(
        "\n[MAIN-1] Micro-validation completed."
    )
    print(
        f"[MAIN-1] Run directory: {run_dir}"
    )

    return Path(run_dir)


def _run_exact(
    config_path: Path,
) -> Path:
    config = load_config(
        config_path
    )

    instance_path = _resolve_path(
        config.instance_path,
        base_dir=PROJECT_ROOT,
    )
    config.instance_path = str(
        instance_path
    )

    output_root = _resolve_path(
        config.output.root_dir,
        base_dir=PROJECT_ROOT,
    )
    config.output.root_dir = str(
        output_root
    )

    instance = load_instance(
        instance_path
    )

    run_dir = run_exact_multiobjective(
        config,
        instance,
    )

    print(
        "\n[MAIN-1] Exact experiment completed."
    )
    print(
        f"[MAIN-1] Run directory: {run_dir}"
    )

    return Path(run_dir)


def _validate_alns_config(
    payload: dict[str, Any],
) -> dict[str, Any]:
    required_top_level = {
        "instance",
        "alns",
        "normalization",
        "emission",
        "output",
    }

    missing = (
        required_top_level
        - set(payload)
    )

    if missing:
        raise ValueError(
            "ALNS config is missing sections: "
            f"{sorted(missing)}"
        )

    instance = payload["instance"]
    alns = payload["alns"]
    normalization = payload[
        "normalization"
    ]
    emission = payload["emission"]
    output = payload["output"]

    if not isinstance(instance, dict):
        raise TypeError(
            "instance section must be a mapping."
        )
    if not isinstance(alns, dict):
        raise TypeError(
            "alns section must be a mapping."
        )

    required_instance = {
        "path",
        "id",
    }
    required_alns = {
        "lambdas",
        "seeds",
        "iterations",
    }
    required_normalization = {
        "cost_min",
        "cost_max",
        "emission_min",
        "emission_max",
    }
    required_emission = {
        "dv_factor",
        "od_factor",
    }

    for section_name, section, required in (
        (
            "instance",
            instance,
            required_instance,
        ),
        (
            "alns",
            alns,
            required_alns,
        ),
        (
            "normalization",
            normalization,
            required_normalization,
        ),
        (
            "emission",
            emission,
            required_emission,
        ),
    ):
        absent = required - set(section)

        if absent:
            raise ValueError(
                f"{section_name} section is missing: "
                f"{sorted(absent)}"
            )

    if "root_dir" not in output:
        raise ValueError(
            "output.root_dir is required."
        )

    lambdas = [
        float(value)
        for value in alns["lambdas"]
    ]
    seeds = [
        int(value)
        for value in alns["seeds"]
    ]
    iterations = int(
        alns["iterations"]
    )

    if not lambdas:
        raise ValueError(
            "At least one lambda is required."
        )

    if not seeds:
        raise ValueError(
            "At least one seed is required."
        )

    if iterations <= 0:
        raise ValueError(
            "ALNS iterations must be positive."
        )

    return {
        "instance_path": str(
            instance["path"]
        ),
        "instance_id": str(
            instance["id"]
        ),
        "lambdas": lambdas,
        "seeds": seeds,
        "iterations": iterations,
        "cost_min": float(
            normalization["cost_min"]
        ),
        "cost_max": float(
            normalization["cost_max"]
        ),
        "emission_min": float(
            normalization[
                "emission_min"
            ]
        ),
        "emission_max": float(
            normalization[
                "emission_max"
            ]
        ),
        "dv_factor": float(
            emission["dv_factor"]
        ),
        "od_factor": float(
            emission["od_factor"]
        ),
        "output_root": str(
            output["root_dir"]
        ),
        "consolidate": bool(
            output.get(
                "consolidate",
                True,
            )
        ),
    }


def _run_alns(
    payload: dict[str, Any],
) -> Path:
    resolved = _validate_alns_config(
        payload
    )

    instance_path = _resolve_path(
        resolved["instance_path"],
        base_dir=PROJECT_ROOT,
    )
    output_root = _resolve_path(
        resolved["output_root"],
        base_dir=PROJECT_ROOT,
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        sys.executable,
        "-m",
        "experiments.run_paper_alns_batch",
        "--instance",
        str(instance_path),
        "--instance-id",
        resolved["instance_id"],
        "--lambdas",
        _comma_join(
            resolved["lambdas"]
        ),
        "--seeds",
        _comma_join(
            resolved["seeds"]
        ),
        "--iterations",
        str(resolved["iterations"]),
        "--cost-min",
        str(resolved["cost_min"]),
        "--cost-max",
        str(resolved["cost_max"]),
        "--emission-min",
        str(resolved["emission_min"]),
        "--emission-max",
        str(resolved["emission_max"]),
        "--dv-emission-factor",
        str(resolved["dv_factor"]),
        "--od-emission-factor",
        str(resolved["od_factor"]),
        "--output-root",
        str(output_root),
    ]

    _run_command(command)

    if resolved["consolidate"]:
        _run_command(
            [
                sys.executable,
                "-m",
                (
                    "experiments."
                    "consolidate_paper_alns_results"
                ),
                "--output-root",
                str(output_root),
            ]
        )

    launcher_manifest = {
        "launcher": "MAIN-1",
        "experiment_type": (
            "alns_multiobjective"
        ),
        "instance_id": (
            resolved["instance_id"]
        ),
        "instance_path": str(
            instance_path
        ),
        "lambdas": (
            resolved["lambdas"]
        ),
        "seeds": resolved["seeds"],
        "iterations": (
            resolved["iterations"]
        ),
        "normalization": {
            "cost_min": (
                resolved["cost_min"]
            ),
            "cost_max": (
                resolved["cost_max"]
            ),
            "emission_min": (
                resolved["emission_min"]
            ),
            "emission_max": (
                resolved["emission_max"]
            ),
        },
        "emission_factors": {
            "dv": (
                resolved["dv_factor"]
            ),
            "od": (
                resolved["od_factor"]
            ),
        },
        "scope_lock": {
            "paper_faithful_alns": True,
            "objective_extension": (
                "scalar_F_lambda"
            ),
            "enhanced_alns": False,
            "fallback": False,
            "operator_substitution": False,
            "operator_resampling": False,
        },
        "output_root": str(
            output_root
        ),
        "consolidated": (
            resolved["consolidate"]
        ),
    }

    manifest_path = (
        output_root
        / "launcher_manifest.json"
    )
    manifest_path.write_text(
        json.dumps(
            launcher_manifest,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "\n[MAIN-1] ALNS experiment completed."
    )
    print(
        f"[MAIN-1] Output root: {output_root}"
    )
    print(
        f"[MAIN-1] Manifest: {manifest_path}"
    )

    return output_root


def run_from_config(
    config_path: Path,
) -> Path:
    payload = _load_raw_yaml(
        config_path
    )

    experiment_type = (
        _experiment_type(payload)
    )

    if experiment_type == "micro_validation":
        return _run_micro_validation(
            config_path
        )

    if experiment_type in {
        "exact_multiobjective",
        "exact",
    }:
        return _run_exact(
            config_path
        )

    if experiment_type in {
        "alns_multiobjective",
        "alns",
    }:
        return _run_alns(
            payload
        )

    raise ValueError(
        "Unsupported experiment type: "
        f"{experiment_type}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Unified exact/ALNS experiment launcher."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help=(
            "Path to an exact or ALNS YAML config."
        ),
    )

    arguments = parser.parse_args()

    config_path = _resolve_path(
        arguments.config,
        base_dir=PROJECT_ROOT,
    )

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config does not exist: {config_path}"
        )

    run_from_config(
        config_path
    )


if __name__ == "__main__":
    main()
