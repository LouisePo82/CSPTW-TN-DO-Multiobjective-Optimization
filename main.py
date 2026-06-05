from __future__ import annotations
import argparse
from pathlib import Path
from core import load_config, load_instance
from experiments.run_exact_multiobjective import run_exact_multiobjective
from experiments.run_micro_validation import run_micro_validation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML configuration.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    project_root = Path(__file__).resolve().parent
    instance_path = Path(config.instance_path)
    if not instance_path.is_absolute():
        instance_path = project_root / instance_path
    config.instance_path = str(instance_path)

    output_root = Path(config.output.root_dir)
    if not output_root.is_absolute():
        config.output.root_dir = str(project_root / output_root)

    instance = load_instance(config.instance_path)

    if config.experiment_type == "micro_validation":
        run_dir = run_micro_validation(config, instance)
    elif config.experiment_type == "exact_multiobjective":
        run_dir = run_exact_multiobjective(config, instance)
    else:
        raise ValueError(f"Unsupported experiment_type: {config.experiment_type}")

    print(f"PASS: experiment completed. Outputs: {run_dir}")

if __name__ == "__main__":
    main()
