from __future__ import annotations
from core.solution import Solution

class ALNSSolver:
    """
    Plug-in boundary for the paper-faithful ALNS implementation.

    The architecture is complete, but the algorithm is intentionally not
    presented as validated until Algorithms 1-4, all destroy/repair operators,
    local search, adaptive weights and acceptance logic are implemented and
    benchmarked against exact ground truth.
    """
    name = "alns"

    def solve(self, instance: dict, objective_config: dict, solver_config: dict) -> Solution:
        raise NotImplementedError(
            "ALNS plug-in is scaffolded but not yet validated. "
            "Implement alns_main.py and operators, then run exact-vs-ALNS tests."
        )
