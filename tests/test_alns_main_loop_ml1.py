from __future__ import annotations

from pathlib import Path
import json

from alns_solver.paper_alns_main import (
    build_ml1_paper_initial_state,
    run_one_paper_alns_candidate_pipeline,
)
from core.instance_loader import load_instance


def state_signature(state):
    return {
        "dv_routes": {
            key: list(value)
            for key, value in sorted(
                state.dv_routes.items()
            )
        },
        "od_routes": {
            key: list(value)
            for key, value in sorted(
                state.od_routes.items()
            )
        },
        "assignments": {
            key: dict(value)
            for key, value in sorted(
                state.assignments.items()
            )
        },
        "unassigned": sorted(
            state.unassigned_customers
        ),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    instance = load_instance(
        root / "data" / "small" / "instance_001"
    )

    lambda_value = 0.5
    cost_bounds = (0.0, 100.0)
    emission_bounds = (0.0, 300.0)
    emission_factors = (3.0, 1.0)

    current_state = build_ml1_paper_initial_state(
        instance,
        seed=42,
        lambda_value=lambda_value,
        cost_bounds=cost_bounds,
        emission_bounds=emission_bounds,
        emission_factors=emission_factors,
    )

    current_solution = current_state.to_core_solution(
        instance=instance,
        lambda_value=lambda_value,
        objective_mode="weighted",
        cost_bounds=cost_bounds,
        emission_bounds=emission_bounds,
        emission_factors=emission_factors,
        require_complete=True,
    )

    if not current_solution.validator_pass:
        raise AssertionError(
            "Controlled current state is invalid."
        )

    before_signature = state_signature(
        current_state
    )

    result = run_one_paper_alns_candidate_pipeline(
        current_state,
        instance,
        best_objective=float(
            current_solution.objective
        ),
        lambda_value=lambda_value,
        cost_bounds=cost_bounds,
        emission_bounds=emission_bounds,
        emission_factors=emission_factors,
        destroy_seed=7,
        strategy_2_seed=7,
    )

    after_signature = state_signature(
        current_state
    )

    if before_signature != after_signature:
        raise AssertionError(
            "ML-1 mutated the current state."
        )

    print("[PASS] ML-1 preserves the input current state")

    if result.destroy_operator != "route_removal":
        raise AssertionError(
            "ML-1 did not use controlled route removal."
        )

    if (
        result.destroy_result.operator_name
        != "paper_route_removal"
    ):
        raise AssertionError(
            "Destroy result is not paper route removal."
        )

    if (
        result.destroy_result.removed_route[
            "route_type"
        ]
        != "DV"
    ):
        raise AssertionError(
            "Route removal did not select a DV route."
        )

    if not result.destroy_result.removed_customers:
        raise AssertionError(
            "Route removal removed no customers."
        )

    print("[PASS] ML-1 uses paper DV-route removal")
    print("[PASS] Destroy stage creates a non-empty removed set")

    if (
        result.repair_result.operator_name
        != "best_insertion"
    ):
        raise AssertionError(
            "ML-1 did not use best insertion."
        )

    if not result.repair_result.validator_pass:
        raise AssertionError(
            "Repair result failed shared validator."
        )

    if result.repair_result.state.unassigned_customers:
        raise AssertionError(
            "Repair left unassigned customers."
        )

    print("[PASS] ML-1 uses paper-aligned best insertion")
    print("[PASS] Repair returns a complete valid state")

    if (
        result.local_search_mode
        != "paper_local_search"
    ):
        raise AssertionError(
            "ML-1 did not use paper local search."
        )

    if not result.local_search_result.metadata[
        "paper_faithful"
    ]:
        raise AssertionError(
            "Local search is not labelled paper-faithful."
        )

    if result.local_search_result.metadata[
        "enhanced"
    ]:
        raise AssertionError(
            "Enhanced local search entered ML-1."
        )

    if (
        result.local_search_result.metadata[
            "delta_ls"
        ]
        != 0.1
    ):
        raise AssertionError(
            "Paper local-search delta is not 0.1."
        )

    print("[PASS] ML-1 uses paper local-search mode only")
    print("[PASS] No enhanced local-search behavior is enabled")

    if not result.validator_pass:
        raise AssertionError(
            "Final ML-1 candidate is invalid."
        )

    if result.validation_errors:
        raise AssertionError(
            "Final candidate has validation errors."
        )

    if result.candidate_state.unassigned_customers:
        raise AssertionError(
            "Final candidate is incomplete."
        )

    print("[PASS] Final candidate is complete and valid")

    shared_metrics = result.candidate_state.evaluate(
        instance=instance,
        lambda_value=lambda_value,
        cost_bounds=cost_bounds,
        emission_bounds=emission_bounds,
        emission_factors=emission_factors,
    )

    if abs(
        result.candidate_objective
        - shared_metrics["objective"]
    ) > 1e-10:
        raise AssertionError(
            "Candidate objective differs from shared F_lambda."
        )

    if abs(
        result.candidate_cost
        - shared_metrics["cost"]
    ) > 1e-10:
        raise AssertionError(
            "Candidate cost differs from shared core."
        )

    if abs(
        result.candidate_emission
        - shared_metrics["emission"]
    ) > 1e-10:
        raise AssertionError(
            "Candidate emission differs from shared core."
        )

    print("[PASS] Candidate objective uses shared scalar F_lambda")
    print("[PASS] Cost and emission are recomputed by shared core")

    metadata = result.metadata

    if not metadata["paper_faithful"]:
        raise AssertionError(
            "ML-1 is not labelled paper-faithful."
        )

    if metadata["enhanced"]:
        raise AssertionError(
            "ML-1 is incorrectly labelled enhanced."
        )

    if metadata["acceptance_applied"]:
        raise AssertionError(
            "SA acceptance was applied in ML-1."
        )

    if metadata["adaptive_weights_applied"]:
        raise AssertionError(
            "Adaptive weights were applied in ML-1."
        )

    if metadata["temperature_cooling_applied"]:
        raise AssertionError(
            "Temperature cooling was applied in ML-1."
        )

    print("[PASS] ML-1 contains no SA acceptance side effect")
    print("[PASS] ML-1 contains no adaptive-weight side effect")
    print("[PASS] ML-1 contains no temperature-cooling side effect")

    repeated = run_one_paper_alns_candidate_pipeline(
        current_state,
        instance,
        best_objective=float(
            current_solution.objective
        ),
        lambda_value=lambda_value,
        cost_bounds=cost_bounds,
        emission_bounds=emission_bounds,
        emission_factors=emission_factors,
        destroy_seed=7,
        strategy_2_seed=7,
    )

    if (
        state_signature(result.candidate_state)
        != state_signature(
            repeated.candidate_state
        )
    ):
        raise AssertionError(
            "Fixed seed did not reproduce candidate structure."
        )

    if abs(
        result.candidate_objective
        - repeated.candidate_objective
    ) > 1e-12:
        raise AssertionError(
            "Fixed seed did not reproduce candidate objective."
        )

    print("[PASS] Fixed seeds reproduce the ML-1 candidate")

    report = {
        "pipeline": {
            "destroy_operator": (
                result.destroy_operator
            ),
            "repair_operator": (
                result.repair_operator
            ),
            "local_search_mode": (
                result.local_search_mode
            ),
            "removed_customers": (
                result.destroy_result
                .removed_customers
            ),
            "repair_insertion_order": (
                result.repair_result
                .insertion_order
            ),
        },
        "candidate": {
            "objective": (
                result.candidate_objective
            ),
            "cost": result.candidate_cost,
            "emission": (
                result.candidate_emission
            ),
            "dv_distance": (
                result.candidate_dv_distance
            ),
            "od_extra_distance": (
                result.candidate_od_extra_distance
            ),
            "validator_pass": (
                result.validator_pass
            ),
        },
        "fidelity": result.metadata,
        "reproducible": True,
    }

    output_dir = (
        root
        / "outputs"
        / "alns_main_loop_fidelity_tests"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "alns_main_loop_ml1_report.json"
    )
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nReport saved to: {output_path}")
    print(
        "\nALNS MAIN LOOP FIDELITY ML-1 — "
        "ONE-ITERATION CANDIDATE PIPELINE PASSED"
    )


if __name__ == "__main__":
    main()
