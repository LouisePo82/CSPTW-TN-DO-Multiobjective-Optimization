from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any

@dataclass
class Solution:
    status: str
    solver_name: str
    objective_mode: str
    lambda_value: float | None = None
    cost: float | None = None
    emission: float | None = None
    objective: float | None = None
    dv_distance: float | None = None
    od_extra_distance: float | None = None
    dv_routes: dict[str, list[str]] = field(default_factory=dict)
    od_routes: dict[str, list[str]] = field(default_factory=dict)
    assignments: dict[str, dict[str, Any]] = field(default_factory=dict)
    arrival_times: dict[str, Any] = field(default_factory=dict)
    vehicle_loads: dict[str, float] = field(default_factory=dict)
    tn_demands: dict[str, float] = field(default_factory=dict)
    adp_loads: dict[str, dict[str, float]] = field(default_factory=dict)
    runtime_sec: float = 0.0
    optimality_gap: float | None = None
    validator_pass: bool | None = None
    validation_errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def routes(self):
        return {**self.dv_routes, **self.od_routes}

    def to_dict(self) -> dict:
        return asdict(self)
