# CSP Multi-objective Refactored Repository

This repository freezes the validated exact MILP layer and reorganizes the project into a plug-in architecture.

## Dependency direction

```text
main -> experiments -> solver plugins -> core
reporting/visualization -> core solution schema
```

`core` never imports `exact_solver` or `alns_solver`.

## Current validation status

The exact solver is based on the validated v5 model:
- manually enumerable micro instance;
- zero-gap SCIP optimum;
- independent validator;
- TN synchronization;
- six regression tests;
- weighted-sum and epsilon-constraint exact analysis.

The ALNS package is intentionally scaffolded but raises `NotImplementedError` until the paper-faithful implementation is completed and benchmarked. It is not falsely presented as validated.

## Run micro validation

```bash
python main.py --config configs/micro_exact.yaml
```

## Run exact multi-objective experiment

```bash
python main.py --config configs/small_exact_multiobjective.yaml
```

## Standard output hierarchy

```text
outputs/<timestamp>_<experiment_name>/
├── manifest.json
├── config_used.yaml
├── instance_snapshot/
├── summary/
├── solutions/
├── charts/
└── logs/
```

## ALNS implementation order

1. `solution_state.py`
2. `initial_solution.py` — Algorithm 1
3. `od_insertion.py` — Algorithms 3 and 4
4. `destroy_operators.py` — 10 operators
5. `repair_operators.py`
6. `local_search.py`
7. `adaptive_weights.py`
8. `acceptance.py`
9. `alns_main.py`
10. exact-vs-ALNS gap tests
