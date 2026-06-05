from core import load_config, load_instance
from exact_solver import ExactSolver
from alns_solver import ALNSSolver

def smoke_test():
    assert ExactSolver.name == "exact"
    assert ALNSSolver.name == "alns"
