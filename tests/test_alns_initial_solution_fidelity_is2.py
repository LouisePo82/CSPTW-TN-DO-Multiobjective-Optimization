from __future__ import annotations

from collections import Counter
from pathlib import Path
import csv
import json
import statistics
from typing import Any

from core.instance_loader import load_instance
from alns_solver.paper_initial_solution import (
    construct_paper_initial_solution,
)


SEEDS = list(range(100))
EMISSION_FACTORS = (3.0, 1.0)


def _canonical_assignment_signature(
    assignments: dict[str, dict[str, Any]],
) -> tuple:
    return tuple(
        (
            customer,
            tuple(
                sorted(
                    (
                        key,
                        str(value),
                    )
                    for key, value
                    in assignment.items()
                )
            ),
        )
        for customer, assignment
        in sorted(assignments.items())
    )


def _canonical_route_signature(
    routes: dict[str, list[str]],
) -> tuple:
    return tuple(
        (
            vehicle,
            tuple(route),
        )
        for vehicle, route
        in sorted(routes.items())
    )


def _phase1_assignment_signature(
    phase1_assignments: list[dict[str, Any]],
) -> tuple:
    """
    Canonical Phase-1 decision signature.

    Sort by customer so that a different processing order does not
    masquerade as a different assignment decision.
    """
    return tuple(
        sorted(
            (
                item["customer"],
                item["driver"],
                item["pickup"],
                int(item["position"]),
            )
            for item in phase1_assignments
        )
    )


def _phase1_execution_signature(
    phase1_assignments: list[dict[str, Any]],
) -> tuple:
    """
    Ordered execution trace used to confirm that seed-dependent processing
    paths differ even when the final assignment mapping converges.
    """
    return tuple(
        (
            item["customer"],
            item["driver"],
            item["pickup"],
            int(item["position"]),
        )
        for item in phase1_assignments
    )


def _complete_solution_signature(
    state,
) -> tuple:
    return (
        _canonical_assignment_signature(
            state.assignments
        ),
        _canonical_route_signature(
            state.dv_routes
        ),
        _canonical_route_signature(
            state.od_routes
        ),
    )


def _round_metric(
    value: float,
    digits: int = 10,
) -> float:
    return round(float(value), digits)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    instance = load_instance(
        project_root
        / "data"
        / "small"
        / "instance_001"
    )

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    phase1_orders = set()
    phase1_execution_signatures = set()
    phase1_assignment_signatures = set()
    final_assignment_signatures = set()
    dv_route_signatures = set()
    od_route_signatures = set()
    complete_solution_signatures = set()
    objective_signatures = set()

    tn_usage_count = 0
    depot_pickup_count = 0
    depot_fallback_count = 0
    unassigned_fallback_count = 0

    tn_usage_by_seed: list[int] = []
    depot_pickup_by_seed: list[int] = []

    for seed in SEEDS:
        try:
            result = construct_paper_initial_solution(
                instance,
                seed=seed,
                lambda_value=0.0,
                cost_bounds=None,
                emission_bounds=None,
                emission_factors=EMISSION_FACTORS,
                strategy_2_mode="paper_random_dv",
            )

            state = result.state
            trace = result.trace

            solution = state.to_core_solution(
                instance=instance,
                lambda_value=0.0,
                objective_mode="cost",
                emission_factors=EMISSION_FACTORS,
                require_complete=True,
                metadata={
                    "seed": seed,
                    "test": (
                        "initial_solution_fidelity_is2"
                    ),
                },
            )

            if not solution.validator_pass:
                failures.append(
                    {
                        "seed": seed,
                        "errors": list(
                            solution.validation_errors
                        ),
                    }
                )
                continue

            if not trace.final_validator_pass:
                failures.append(
                    {
                        "seed": seed,
                        "errors": [
                            "Trace final_validator_pass is false."
                        ],
                    }
                )
                continue

            if trace.final_validation_errors:
                failures.append(
                    {
                        "seed": seed,
                        "errors": list(
                            trace.final_validation_errors
                        ),
                    }
                )
                continue

            phase1_order_signature = tuple(
                trace.phase1_customer_order
            )
            phase1_execution_signature = (
                _phase1_execution_signature(
                    trace.phase1_assignments
                )
            )
            phase1_assign_signature = (
                _phase1_assignment_signature(
                    trace.phase1_assignments
                )
            )
            final_assign_signature = (
                _canonical_assignment_signature(
                    state.assignments
                )
            )
            dv_signature = (
                _canonical_route_signature(
                    state.dv_routes
                )
            )
            od_signature = (
                _canonical_route_signature(
                    state.od_routes
                )
            )
            complete_signature = (
                _complete_solution_signature(state)
            )
            objective_signature = (
                _round_metric(solution.cost),
                _round_metric(solution.emission),
            )

            phase1_orders.add(
                phase1_order_signature
            )
            phase1_execution_signatures.add(
                phase1_execution_signature
            )
            phase1_assignment_signatures.add(
                phase1_assign_signature
            )
            final_assignment_signatures.add(
                final_assign_signature
            )
            dv_route_signatures.add(
                dv_signature
            )
            od_route_signatures.add(
                od_signature
            )
            complete_solution_signatures.add(
                complete_signature
            )
            objective_signatures.add(
                objective_signature
            )

            used_tns = sorted(
                {
                    assignment["pickup"]
                    for assignment
                    in state.assignments.values()
                    if (
                        assignment.get("mode")
                        == "OD_HOME"
                        and assignment.get("pickup")
                        in instance["tns"]
                    )
                }
            )

            depot_pickup_drivers = sorted(
                {
                    assignment["driver"]
                    for assignment
                    in state.assignments.values()
                    if (
                        assignment.get("mode")
                        == "OD_HOME"
                        and assignment.get("pickup")
                        == instance["start_depot"]
                    )
                }
            )

            used_tn_flag = int(bool(used_tns))
            depot_pickup_flag = int(
                bool(depot_pickup_drivers)
            )

            tn_usage_count += used_tn_flag
            depot_pickup_count += (
                depot_pickup_flag
            )
            depot_fallback_count += len(
                trace.depot_fallbacks
            )
            unassigned_fallback_count += len(
                trace.unassigned_fallbacks
            )

            tn_usage_by_seed.append(
                used_tn_flag
            )
            depot_pickup_by_seed.append(
                depot_pickup_flag
            )

            rows.append(
                {
                    "seed": seed,
                    "status": solution.status,
                    "cost": solution.cost,
                    "emission": solution.emission,
                    "objective": solution.objective,
                    "dv_distance": (
                        solution.dv_distance
                    ),
                    "od_extra_distance": (
                        solution.od_extra_distance
                    ),
                    "phase1_order": "|".join(
                        trace.phase1_customer_order
                    ),
                    "phase1_execution_signature": (
                        repr(
                            phase1_execution_signature
                        )
                    ),
                    "phase1_assignment_signature": (
                        repr(
                            phase1_assign_signature
                        )
                    ),
                    "final_assignment_signature": (
                        repr(
                            final_assign_signature
                        )
                    ),
                    "dv_route_signature": repr(
                        dv_signature
                    ),
                    "od_route_signature": repr(
                        od_signature
                    ),
                    "complete_solution_signature": (
                        repr(
                            complete_signature
                        )
                    ),
                    "used_tns": "|".join(
                        used_tns
                    ),
                    "depot_pickup_drivers": "|".join(
                        depot_pickup_drivers
                    ),
                    "phase2_depot_fallbacks": len(
                        trace.depot_fallbacks
                    ),
                    "phase2_unassigned_fallbacks": len(
                        trace.unassigned_fallbacks
                    ),
                    "phase3_insertion_order": "|".join(
                        trace.phase3_insertion_order
                    ),
                    "validator_pass": (
                        solution.validator_pass
                    ),
                }
            )

        except Exception as exc:
            failures.append(
                {
                    "seed": seed,
                    "errors": [
                        f"{type(exc).__name__}: {exc}"
                    ],
                }
            )

    output_dir = (
        project_root
        / "outputs"
        / "alns_initial_solution_fidelity_tests"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_dir
        / "initial_solution_is2_100_seeds.csv"
    )

    if rows:
        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(rows[0]),
            )
            writer.writeheader()
            writer.writerows(rows)

    cost_values = [
        float(row["cost"])
        for row in rows
    ]
    emission_values = [
        float(row["emission"])
        for row in rows
    ]

    phase1_order_counts = Counter(
        row["phase1_order"]
        for row in rows
    )

    report = {
        "total_seeds": len(SEEDS),
        "valid_solutions": len(rows),
        "failures": len(failures),
        "pass_rate": (
            len(rows) / len(SEEDS)
        ),
        "diversity": {
            "unique_phase1_orders": len(
                phase1_orders
            ),
            "phase1_order_frequencies": dict(
                sorted(
                    phase1_order_counts.items()
                )
            ),
            "unique_phase1_execution_signatures": len(
                phase1_execution_signatures
            ),
            "unique_phase1_assignment_signatures": len(
                phase1_assignment_signatures
            ),
            "unique_final_assignment_signatures": len(
                final_assignment_signatures
            ),
            "unique_dv_route_signatures": len(
                dv_route_signatures
            ),
            "unique_od_route_signatures": len(
                od_route_signatures
            ),
            "unique_complete_solution_signatures": len(
                complete_solution_signatures
            ),
            "unique_objective_pairs": len(
                objective_signatures
            ),
        },
        "usage": {
            "tn_usage_seed_count": (
                tn_usage_count
            ),
            "depot_pickup_seed_count": (
                depot_pickup_count
            ),
            "phase2_depot_fallback_event_count": (
                depot_fallback_count
            ),
            "phase2_unassigned_fallback_event_count": (
                unassigned_fallback_count
            ),
        },
        "cost_statistics": {
            "min": (
                min(cost_values)
                if cost_values
                else None
            ),
            "mean": (
                statistics.mean(cost_values)
                if cost_values
                else None
            ),
            "max": (
                max(cost_values)
                if cost_values
                else None
            ),
            "std": (
                statistics.pstdev(cost_values)
                if len(cost_values) > 1
                else 0.0
            ),
        },
        "emission_statistics": {
            "min": (
                min(emission_values)
                if emission_values
                else None
            ),
            "mean": (
                statistics.mean(
                    emission_values
                )
                if emission_values
                else None
            ),
            "max": (
                max(emission_values)
                if emission_values
                else None
            ),
            "std": (
                statistics.pstdev(
                    emission_values
                )
                if len(emission_values) > 1
                else 0.0
            ),
        },
        "interpretation": {
            "phase1_randomization_verified": (
                len(phase1_orders) >= 2
                and len(phase1_execution_signatures) >= 2
            ),
            "final_convergence_observed": (
                len(complete_solution_signatures) == 1
            ),
            "note": (
                "Algorithm 1 randomizes the Type-1 processing order. "
                "On this small controlled instance, different execution "
                "paths may still converge to one final greedy solution. "
                "Final structural diversity is therefore diagnostic, not "
                "a correctness requirement."
            ),
        },
        "failure_details": failures,
    }

    report_path = (
        output_dir
        / "initial_solution_is2_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            report,
            indent=2,
        )
    )
    print(f"\nCSV saved to: {csv_path}")
    print(f"Report saved to: {report_path}")

    if failures:
        raise SystemExit(
            "IS-2 FAILED: "
            f"{len(failures)} of "
            f"{len(SEEDS)} seeds were invalid."
        )

    if len(rows) != len(SEEDS):
        raise SystemExit(
            "IS-2 FAILED: not all seeds produced "
            "a complete valid solution."
        )

    if len(phase1_orders) < 2:
        raise SystemExit(
            "IS-2 FAILED: Phase-1 customer order "
            "did not vary across seeds."
        )

    if (
        len(phase1_execution_signatures)
        < 2
    ):
        raise SystemExit(
            "IS-2 FAILED: seed-dependent Phase-1 execution "
            "paths did not vary across seeds."
        )

    print(
        "\n[PASS] 100/100 paper initial solutions "
        "are complete and valid."
    )
    print(
        "[PASS] Phase-1 customer order varies "
        "across seeds."
    )
    print(
        "[PASS] Phase-1 execution traces vary "
        "across seeds."
    )

    if len(phase1_assignment_signatures) == 1:
        print(
            "[INFO] Canonical Phase-1 assignments converge "
            "to one mapping on this instance."
        )
    else:
        print(
            "[INFO] Canonical Phase-1 assignments vary "
            "across seeds."
        )

    if len(complete_solution_signatures) == 1:
        print(
            "[INFO] Final solutions converge to one structural "
            "signature on this small controlled instance."
        )
    else:
        print(
            "[INFO] Final structural solutions vary across seeds."
        )

    print(
        "[PASS] IS-2 separates execution-path diversity, "
        "decision diversity, structural diversity, and "
        "objective diversity."
    )
    print(
        "\nINITIAL SOLUTION FIDELITY IS-2 — "
        "SEED DIVERSITY PASSED"
    )


if __name__ == "__main__":
    main()
