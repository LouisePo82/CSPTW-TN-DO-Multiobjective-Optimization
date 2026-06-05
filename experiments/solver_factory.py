from exact_solver import ExactSolver
from alns_solver import ALNSSolver

def create_solver(name: str):
    if name == "exact":
        return ExactSolver()
    if name == "alns":
        return ALNSSolver()
    raise ValueError(f"Unknown solver: {name}")
