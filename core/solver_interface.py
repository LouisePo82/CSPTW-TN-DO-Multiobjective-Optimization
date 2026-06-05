from __future__ import annotations
from typing import Protocol
from .solution import Solution

class SolverInterface(Protocol):
    name: str
    def solve(self, instance: dict, objective_config: dict, solver_config: dict) -> Solution:
        ...
