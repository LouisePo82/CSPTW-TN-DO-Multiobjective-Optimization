from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml

@dataclass
class ObjectiveConfig:
    mode: str = "weighted"
    emission_factor_dv: float = 3.0
    emission_factor_od: float = 1.0
    lambda_values: list[float] = field(default_factory=lambda: [0.0, 0.5, 1.0])
    epsilon_levels: int = 21

@dataclass
class ExactConfig:
    time_limit_sec: int = 300
    mip_gap: float = 0.0
    enable_output: bool = False
    require_optimal: bool = True

@dataclass
class ALNSConfig:
    runs: int = 30
    iterations: int = 30000
    seed: int = 42
    cooling_rate: float = 0.9994
    segment_length: int = 300
    reaction_factor: float = 0.1

@dataclass
class OutputConfig:
    root_dir: str = "outputs"
    save_solution_details: bool = True
    save_charts: bool = True
    save_instance_snapshot: bool = True

@dataclass
class ExperimentConfig:
    experiment_name: str
    instance_path: str
    solver: str
    experiment_type: str
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    exact: ExactConfig = field(default_factory=ExactConfig)
    alns: ALNSConfig = field(default_factory=ALNSConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    raw: dict[str, Any] = field(default_factory=dict)

def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ExperimentConfig(
        experiment_name=raw["experiment_name"],
        instance_path=raw["instance_path"],
        solver=raw["solver"],
        experiment_type=raw["experiment_type"],
        objective=ObjectiveConfig(**raw.get("objective", {})),
        exact=ExactConfig(**raw.get("exact", {})),
        alns=ALNSConfig(**raw.get("alns", {})),
        output=OutputConfig(**raw.get("output", {})),
        raw=raw,
    )
