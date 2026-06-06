from __future__ import annotations

from pathlib import Path
import csv
import json
import math
from typing import Any

from core.instance_loader import load_instance
from core.objective import recompute_objectives
from alns_solver.paper_initial_solution import (
    construct_paper_initial_solution,
)


SEEDS = list(range(100))
EMISSION_FACTORS = (3.0, 1.0)
TOLERANCE = 1e-8


def _customer_occurrences(
    customer: str,
    dv_routes: dict[str, list[str]],
    od_routes: dict[str, list[str]],
) -> int:
    return sum(
        route.count(customer)
        for route in dv_routes.values()
    ) + sum(
        route.count(customer)
        for route in od_routes.values()
    )


def _assigned_od_customers(
    assignments: dict[str, dict[str, Any]],
    driver: str,
) -> list[str]:
    return sorted(
        customer
        for customer, assignment in assignments.items()
        if assignment.get("mode") == "OD_HOME"
        and assignment.get("driver") == driver
    )


def _assigned_tn_customers(
    assignments: dict[str, dict[str, Any]],
    tn: str,
) -> list[str]:
    return sorted(
        customer
        for customer, assignment in assignments.items()
        if assignment.get("mode") == "OD_HOME"
        and assignment.get("pickup") == tn
    )


def _assert_close(
    actual: float,
    expected: float,
    label: str,
    tolerance: float = TOLERANCE,
) -> None:
    if not math.isclose(
        float(actual),
        float(expected),
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise AssertionError(
            f"{label}: actual={actual}, expected={expected}"
        )


def _check_customer_coverage(
    state,
    instance: dict,
) -> None:
    expected = set(instance["customers"])
    assigned = set(state.assignments)

    if assigned != expected:
        raise AssertionError(
            "Customer assignment coverage mismatch. "
            f"Assigned={sorted(assigned)}, "
            f"expected={sorted(expected)}"
        )

    if state.unassigned_customers:
        raise AssertionError(
            "Unassigned customers remain: "
            f"{sorted(state.unassigned_customers)}"
        )

    for customer in instance["customers"]:
        assignment = state.assignments[customer]
        mode = assignment.get("mode")
        occurrences = _customer_occurrences(
            customer,
            state.dv_routes,
            state.od_routes,
        )

        if mode in {"DV_HOME", "OD_HOME"}:
            if occurrences != 1:
                raise AssertionError(
                    f"{customer}: expected exactly one route visit, "
                    f"found {occurrences}."
                )
        elif mode == "ADP":
            if occurrences != 0:
                raise AssertionError(
                    f"{customer}: ADP customer must not appear "
                    f"as a route node; found {occurrences} visits."
                )
        else:
            raise AssertionError(
                f"{customer}: unknown assignment mode {mode}."
            )


def _check_mode_eligibility(
    state,
    instance: dict,
) -> None:
    for customer, assignment in state.assignments.items():
        customer_type = int(
            instance["nodes"][customer]["customer_type"]
        )
        mode = assignment.get("mode")

        if customer_type == 1 and mode not in {
            "DV_HOME",
            "OD_HOME",
        }:
            raise AssertionError(
                f"{customer} Type 1 cannot use mode {mode}."
            )

        if customer_type == 2 and mode != "ADP":
            raise AssertionError(
                f"{customer} Type 2 must use ADP, got {mode}."
            )

        if customer_type == 3 and mode not in {
            "DV_HOME",
            "OD_HOME",
            "ADP",
        }:
            raise AssertionError(
                f"{customer} Type 3 cannot use mode {mode}."
            )

        if mode == "ADP":
            adp = assignment.get("adp")
            if instance["gamma"].get((customer, adp), 0) != 1:
                raise AssertionError(
                    f"{customer} is incompatible with ADP {adp}."
                )


def _check_assignment_route_consistency(
    state,
    instance: dict,
) -> None:
    for customer, assignment in state.assignments.items():
        mode = assignment.get("mode")

        if mode == "DV_HOME":
            vehicle = assignment.get("vehicle")
            if customer not in state.dv_routes.get(vehicle, []):
                raise AssertionError(
                    f"{customer} missing from DV route {vehicle}."
                )

        elif mode == "ADP":
            vehicle = assignment.get("vehicle")
            adp = assignment.get("adp")
            if adp not in state.dv_routes.get(vehicle, []):
                raise AssertionError(
                    f"ADP {adp} for {customer} missing "
                    f"from DV route {vehicle}."
                )

        elif mode == "OD_HOME":
            driver = assignment.get("driver")
            pickup = assignment.get("pickup")
            route = state.od_routes.get(driver, [])

            if not route:
                raise AssertionError(
                    f"{customer}: OD route {driver} is inactive."
                )

            if route[1] != pickup:
                raise AssertionError(
                    f"{customer}: assignment pickup {pickup} "
                    f"does not match OD route pickup {route[1]}."
                )

            if customer not in route:
                raise AssertionError(
                    f"{customer} missing from OD route {driver}."
                )


def _check_route_endpoints(
    state,
    instance: dict,
) -> None:
    for vehicle in instance["dvs"]:
        route = state.dv_routes.get(vehicle, [])
        if not route:
            continue

        if route[0] != instance["start_depot"]:
            raise AssertionError(
                f"{vehicle}: invalid DV start {route[0]}."
            )
        if route[-1] != instance["end_depot"]:
            raise AssertionError(
                f"{vehicle}: invalid DV end {route[-1]}."
            )

    for driver in instance["ods"]:
        route = state.od_routes.get(driver, [])
        if not route:
            continue

        info = instance["vehicles"][driver]

        if route[0] != info["origin"]:
            raise AssertionError(
                f"{driver}: invalid OD origin {route[0]}."
            )
        if route[-1] != info["destination"]:
            raise AssertionError(
                f"{driver}: invalid OD destination {route[-1]}."
            )
        if route[1] not in instance["pickup_points"]:
            raise AssertionError(
                f"{driver}: invalid pickup point {route[1]}."
            )


def _check_capacities(
    state,
    solution,
    instance: dict,
) -> None:
    for vehicle in instance["dvs"]:
        load = float(
            solution.vehicle_loads.get(vehicle, 0.0)
        )
        capacity = float(
            instance["vehicles"][vehicle]["capacity"]
        )

        if load > capacity + TOLERANCE:
            raise AssertionError(
                f"{vehicle}: DV load {load} exceeds "
                f"capacity {capacity}."
            )

    for driver in instance["ods"]:
        customer_count = len(
            _assigned_od_customers(
                state.assignments,
                driver,
            )
        )
        capacity = int(
            instance["vehicles"][driver]["capacity"]
        )

        if customer_count > capacity:
            raise AssertionError(
                f"{driver}: serves {customer_count} customers, "
                f"capacity is {capacity}."
            )


def _check_tn_supply_and_sync(
    state,
    solution,
    instance: dict,
) -> None:
    fixed_tns = {
        tn
        for tn in instance["tns"]
        if _assigned_tn_customers(
            state.assignments,
            tn,
        )
    }

    for tn in fixed_tns:
        visiting_dvs = [
            vehicle
            for vehicle, route in state.dv_routes.items()
            if tn in route
        ]

        if len(visiting_dvs) != 1:
            raise AssertionError(
                f"{tn}: expected exactly one supplying DV, "
                f"found {visiting_dvs}."
            )

        completion = float(
            solution.arrival_times["tn_completion"][tn]
        )

        for driver in instance["ods"]:
            customers = [
                customer
                for customer in _assigned_od_customers(
                    state.assignments,
                    driver,
                )
                if state.assignments[customer].get("pickup")
                == tn
            ]

            if not customers:
                continue

            pickup_time = float(
                solution.arrival_times["od_pickup"][driver][tn]
            )

            if pickup_time + TOLERANCE < completion:
                raise AssertionError(
                    f"{driver} reaches {tn} at {pickup_time} "
                    f"before DV completion {completion}."
                )


def _check_fixed_tn_trace(
    state,
    trace,
) -> None:
    for tn, info in trace.fixed_tn_positions.items():
        vehicle = info["vehicle"]
        position = int(info["position"])
        route = state.dv_routes.get(vehicle, [])

        if len(route) <= position:
            raise AssertionError(
                f"{tn}: fixed position {position} is out of range."
            )

        if route[position] != tn:
            raise AssertionError(
                f"{tn}: final route moved fixed TN from "
                f"{vehicle}[{position}]."
            )


def _check_trace_consistency(
    state,
    trace,
    instance: dict,
) -> None:
    if set(trace.phase1_customer_order) != set(
        instance["type1"]
    ):
        raise AssertionError(
            "Phase-1 trace does not contain exactly Type-1 customers."
        )

    phase1_customers = {
        item["customer"]
        for item in trace.phase1_assignments
    }

    if not phase1_customers.issubset(
        set(instance["type1"])
    ):
        raise AssertionError(
            "Phase-1 trace assigned a non-Type-1 customer."
        )

    if set(trace.phase1_unassigned) & phase1_customers:
        raise AssertionError(
            "A Phase-1 customer is both assigned and unassigned."
        )

    if set(trace.phase3_insertion_order) != set(
        trace.phase3_initial_unassigned
    ):
        raise AssertionError(
            "Phase-3 insertion order does not cover exactly "
            "the Phase-3 initial unassigned set."
        )

    if len(trace.phase3_insertion_order) != len(
        set(trace.phase3_insertion_order)
    ):
        raise AssertionError(
            "Phase-3 trace contains duplicate customer insertions."
        )

    if not trace.final_validator_pass:
        raise AssertionError(
            "Trace reports final validator failure."
        )

    if trace.final_validation_errors:
        raise AssertionError(
            "Trace contains final validation errors: "
            f"{trace.final_validation_errors}"
        )


def _check_objective_recomputation(
    state,
    solution,
    instance: dict,
) -> None:
    recomputed = recompute_objectives(
        instance,
        state.dv_routes,
        state.od_routes,
        emission_factors=EMISSION_FACTORS,
    )

    _assert_close(
        solution.dv_distance,
        recomputed["dv_distance"],
        "DV distance",
    )
    _assert_close(
        solution.od_extra_distance,
        recomputed["od_extra_distance"],
        "OD extra distance",
    )
    _assert_close(
        solution.cost,
        recomputed["cost"],
        "Cost",
    )
    _assert_close(
        solution.emission,
        recomputed["emission"],
        "Emission",
    )

    expected_cost = (
        float(solution.dv_distance)
        + float(instance["rho"])
        * float(solution.od_extra_distance)
    )
    _assert_close(
        solution.cost,
        expected_cost,
        "Cost formula",
    )

    expected_emission = (
        float(EMISSION_FACTORS[0])
        * float(solution.dv_distance)
        + float(EMISSION_FACTORS[1])
        * float(solution.od_extra_distance)
    )
    _assert_close(
        solution.emission,
        expected_emission,
        "Emission formula",
    )

    _assert_close(
        solution.objective,
        solution.cost,
        "Lambda=0 objective",
    )


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

    check_names = [
        "customer_coverage",
        "mode_eligibility",
        "assignment_route_consistency",
        "route_endpoints",
        "capacities",
        "tn_supply_and_synchronization",
        "fixed_tn_trace",
        "trace_consistency",
        "objective_recomputation",
        "shared_validator",
    ]

    passed_counts = {
        name: 0
        for name in check_names
    }

    for seed in SEEDS:
        seed_checks: dict[str, bool] = {
            name: False
            for name in check_names
        }

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
                        "initial_solution_fidelity_is3"
                    ),
                },
            )

            _check_customer_coverage(
                state,
                instance,
            )
            seed_checks["customer_coverage"] = True

            _check_mode_eligibility(
                state,
                instance,
            )
            seed_checks["mode_eligibility"] = True

            _check_assignment_route_consistency(
                state,
                instance,
            )
            seed_checks[
                "assignment_route_consistency"
            ] = True

            _check_route_endpoints(
                state,
                instance,
            )
            seed_checks["route_endpoints"] = True

            _check_capacities(
                state,
                solution,
                instance,
            )
            seed_checks["capacities"] = True

            _check_tn_supply_and_sync(
                state,
                solution,
                instance,
            )
            seed_checks[
                "tn_supply_and_synchronization"
            ] = True

            _check_fixed_tn_trace(
                state,
                trace,
            )
            seed_checks["fixed_tn_trace"] = True

            _check_trace_consistency(
                state,
                trace,
                instance,
            )
            seed_checks["trace_consistency"] = True

            _check_objective_recomputation(
                state,
                solution,
                instance,
            )
            seed_checks[
                "objective_recomputation"
            ] = True

            if not solution.validator_pass:
                raise AssertionError(
                    "Shared validator failed: "
                    f"{solution.validation_errors}"
                )
            seed_checks["shared_validator"] = True

            for name, passed in seed_checks.items():
                if passed:
                    passed_counts[name] += 1

            rows.append(
                {
                    "seed": seed,
                    "status": solution.status,
                    "cost": solution.cost,
                    "emission": solution.emission,
                    "objective": solution.objective,
                    "dv_distance": solution.dv_distance,
                    "od_extra_distance": (
                        solution.od_extra_distance
                    ),
                    "assigned_customer_count": len(
                        state.assignments
                    ),
                    "unassigned_customer_count": len(
                        state.unassigned_customers
                    ),
                    "fixed_tn_count": len(
                        trace.fixed_tn_positions
                    ),
                    "phase1_order": "|".join(
                        trace.phase1_customer_order
                    ),
                    "phase3_insertion_order": "|".join(
                        trace.phase3_insertion_order
                    ),
                    **seed_checks,
                }
            )

        except Exception as exc:
            failures.append(
                {
                    "seed": seed,
                    "errors": [
                        f"{type(exc).__name__}: {exc}"
                    ],
                    "completed_checks": [
                        name
                        for name, passed
                        in seed_checks.items()
                        if passed
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
        / "initial_solution_is3_100_seeds.csv"
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

    report = {
        "total_seeds": len(SEEDS),
        "valid_solutions": len(rows),
        "failures": len(failures),
        "pass_rate": (
            len(rows) / len(SEEDS)
        ),
        "independent_check_pass_counts": (
            passed_counts
        ),
        "all_checks_passed_for_all_seeds": (
            len(rows) == len(SEEDS)
            and all(
                count == len(SEEDS)
                for count in passed_counts.values()
            )
        ),
        "failure_details": failures,
    }

    report_path = (
        output_dir
        / "initial_solution_is3_report.json"
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
            "IS-3 FAILED: "
            f"{len(failures)} of "
            f"{len(SEEDS)} seeds failed independent checks."
        )

    if not report[
        "all_checks_passed_for_all_seeds"
    ]:
        raise SystemExit(
            "IS-3 FAILED: at least one independent check "
            "did not pass for all seeds."
        )

    print(
        "\n[PASS] 100/100 solutions have complete, "
        "exactly-once customer coverage."
    )
    print(
        "[PASS] 100/100 solutions satisfy customer-mode "
        "eligibility and ADP compatibility."
    )
    print(
        "[PASS] 100/100 assignments match their DV/OD routes."
    )
    print(
        "[PASS] 100/100 solutions satisfy DV and OD capacity."
    )
    print(
        "[PASS] 100/100 TN assignments have exactly one "
        "supplying DV and valid synchronization."
    )
    print(
        "[PASS] 100/100 solutions preserve fixed TN positions."
    )
    print(
        "[PASS] 100/100 phase traces are internally consistent."
    )
    print(
        "[PASS] 100/100 cost, emission, distance, and "
        "lambda=0 objective values recompute exactly."
    )
    print(
        "[PASS] 100/100 solutions pass the shared validator."
    )
    print(
        "\nINITIAL SOLUTION FIDELITY IS-3 — "
        "FULL VALIDITY PASSED"
    )


if __name__ == "__main__":
    main()
