from __future__ import annotations

from pathlib import Path
import json

from core.instance_loader import load_instance
from alns_solver.paper_alns_main import (
    build_ml1_paper_initial_state,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]

    instance_path = (
        root
        / "data"
        / "solomon"
        / "derived"
        / "r101_25"
    )

    instance = load_instance(
        instance_path
    )

    state = build_ml1_paper_initial_state(
        instance,
        seed=2026,
        lambda_value=0.5,
        cost_bounds=None,
        emission_bounds=None,
        emission_factors=(3.0, 1.0),
    )

    solution = state.to_core_solution(
        instance=instance,
        lambda_value=0.5,
        objective_mode="weighted",
        cost_bounds=None,
        emission_bounds=None,
        emission_factors=(3.0, 1.0),
        require_complete=True,
        metadata={
            "validation_scope": (
                "solomon_r101_25_sr2"
            ),
        },
    )

    assert not state.unassigned_customers, (
        "Initial solution left unassigned customers: "
        f"{sorted(state.unassigned_customers)}"
    )

    assert (
        set(state.assignments)
        == set(instance["customers"])
    ), (
        "Assignment set does not cover all customers."
    )

    assert solution.validator_pass, (
        "Shared validator rejected the R101-25 "
        f"initial solution: {solution.validation_errors}"
    )

    active_dv_routes = {
        vehicle: route
        for vehicle, route in state.dv_routes.items()
        if route
    }

    active_od_routes = {
        driver: route
        for driver, route in state.od_routes.items()
        if route
    }

    assignment_modes: dict[str, int] = {}

    for assignment in state.assignments.values():
        mode = str(
            assignment.get("mode", "UNKNOWN")
        )
        assignment_modes[mode] = (
            assignment_modes.get(mode, 0)
            + 1
        )

    report = {
        "instance_id": instance["metadata"].get(
            "instance_id"
        ),
        "customer_count": len(
            instance["customers"]
        ),
        "seed": 2026,
        "lambda_value": 0.5,
        "validator_pass": bool(
            solution.validator_pass
        ),
        "validation_errors": list(
            solution.validation_errors
        ),
        "unassigned_customers": sorted(
            state.unassigned_customers
        ),
        "assignment_count": len(
            state.assignments
        ),
        "assignment_modes": assignment_modes,
        "active_dv_routes": active_dv_routes,
        "active_od_routes": active_od_routes,
        "metrics": {
            "cost": float(solution.cost),
            "emission": float(
                solution.emission
            ),
            "F_lambda": float(
                solution.objective
            ),
            "dv_distance": float(
                solution.dv_distance
            ),
            "od_extra_distance": float(
                solution.od_extra_distance
            ),
        },
    }

    output_dir = (
        root
        / "outputs"
        / "solomon_feasibility_tests"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        output_dir
        / "r101_25_initial_feasibility_sr2.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "[PASS] R101-25 paper initial solution is constructed"
    )
    print(
        "[PASS] All 25 customers receive assignments"
    )
    print(
        "[PASS] No customer remains unassigned"
    )
    print(
        "[PASS] Shared validator accepts the initial solution"
    )
    print(
        f"[INFO] Assignment modes: {assignment_modes}"
    )
    print(
        f"[INFO] Active DV routes: "
        f"{len(active_dv_routes)}"
    )
    print(
        f"[INFO] Active OD routes: "
        f"{len(active_od_routes)}"
    )
    print(
        f"[INFO] Report saved to: {report_path}"
    )
    print(
        "\nSR-2 — R101-25 INITIAL FEASIBILITY PASSED"
    )


if __name__ == "__main__":
    main()
