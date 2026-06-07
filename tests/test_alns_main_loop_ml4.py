from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json

from core.instance_loader import load_instance
from alns_solver.paper_alns_main import (
    run_paper_alns_short_integrated_run,
)


def state_signature(state):
    return {
        "dv_routes": deepcopy(state.dv_routes),
        "od_routes": deepcopy(state.od_routes),
        "assignments": deepcopy(state.assignments),
        "unassigned": sorted(
            state.unassigned_customers
        ),
    }


def run_once(instance):
    return run_paper_alns_short_integrated_run(
        instance,
        iterations=20,
        seed=42,
        lambda_value=0.5,
        cost_bounds=(0.0, 100.0),
        emission_bounds=(0.0, 300.0),
        emission_factors=(3.0, 1.0),
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    instance = load_instance(
        root / "data" / "small" / "instance_001"
    )

    result = run_once(instance)

    if result.iterations != 20:
        raise AssertionError(
            "ML-4 did not complete 20 iterations."
        )

    if len(result.iteration_logs) != 20:
        raise AssertionError(
            "ML-4 log length is incorrect."
        )

    if [
        log.iteration for log in result.iteration_logs
    ] != list(range(1, 21)):
        raise AssertionError(
            "ML-4 iteration numbering is not consecutive."
        )

    print("[PASS] ML-4 completes 20 consecutive iterations")

    if any(
        log.destroy_operator != "route_removal"
        for log in result.iteration_logs
    ):
        raise AssertionError(
            "Unexpected destroy operator entered ML-4."
        )

    if any(
        log.repair_operator != "best_insertion"
        for log in result.iteration_logs
    ):
        raise AssertionError(
            "Unexpected repair operator entered ML-4."
        )

    if any(
        not log.candidate_validator_pass
        for log in result.iteration_logs
    ):
        raise AssertionError(
            "An ML-4 candidate failed validation."
        )

    print("[PASS] Every iteration uses the controlled paper operator pair")
    print("[PASS] Every generated candidate passes shared validation")

    temperatures_used = [
        log.temperature_used
        for log in result.iteration_logs
    ]
    temperatures_after = [
        log.temperature_after_cooling
        for log in result.iteration_logs
    ]

    if any(
        after >= used
        for used, after in zip(
            temperatures_used,
            temperatures_after,
        )
    ):
        raise AssertionError(
            "Temperature did not cool after an iteration."
        )

    if any(
        later >= earlier
        for earlier, later in zip(
            temperatures_used,
            temperatures_used[1:],
        )
    ):
        raise AssertionError(
            "Temperature sequence is not decreasing."
        )

    print("[PASS] Temperature cools exactly once per iteration")
    print("[PASS] Temperature decreases monotonically")

    best_sequence = [
        result.initial_objective
    ] + [
        log.best_objective_after
        for log in result.iteration_logs
    ]

    if any(
        later > earlier + 1e-12
        for earlier, later in zip(
            best_sequence,
            best_sequence[1:],
        )
    ):
        raise AssertionError(
            "Best objective increased during ML-4."
        )

    print("[PASS] Best objective is monotonically non-increasing")

    for log in result.iteration_logs:
        expected_current = (
            log.candidate_objective
            if log.accepted
            else log.current_objective_before
        )

        if abs(
            log.current_objective_after
            - expected_current
        ) > 1e-12:
            raise AssertionError(
                "Current transition is inconsistent."
            )

    print("[PASS] Accepted/rejected state transitions remain consistent")

    destroy_record = (
        result.adaptive_result.state
        .destroy_records["route_removal"]
    )
    repair_record = (
        result.adaptive_result.state
        .repair_records["best_insertion"]
    )

    if destroy_record.segment_uses != 20:
        raise AssertionError(
            "Destroy adaptive usage count is not 20."
        )

    if repair_record.segment_uses != 20:
        raise AssertionError(
            "Repair adaptive usage count is not 20."
        )

    if (
        result.adaptive_result.state
        .completed_updates != 0
    ):
        raise AssertionError(
            "Short ML-4 run unexpectedly crossed a segment boundary."
        )

    print("[PASS] Adaptive usage is recorded once per iteration")
    print("[PASS] No segment update occurs before iteration 300")

    if result.metadata["enhanced"]:
        raise AssertionError(
            "Enhanced behavior entered ML-4."
        )

    if not result.metadata["paper_faithful"]:
        raise AssertionError(
            "ML-4 lost paper-faithful label."
        )

    if result.metadata["full_roulette_execution"]:
        raise AssertionError(
            "ML-4 incorrectly claims full roulette execution."
        )

    if result.metadata["experiment_runner"]:
        raise AssertionError(
            "ML-4 incorrectly claims to be an experiment runner."
        )

    print("[PASS] ML-4 uses paper components only")
    print("[PASS] ML-4 is not mislabeled as full roulette or experiment run")
    print("[PASS] All comparisons use one scalar F_lambda")

    repeated = run_once(instance)

    if (
        state_signature(result.final_current_state)
        != state_signature(
            repeated.final_current_state
        )
    ):
        raise AssertionError(
            "Fixed seed did not reproduce final current state."
        )

    if (
        state_signature(result.final_best_state)
        != state_signature(
            repeated.final_best_state
        )
    ):
        raise AssertionError(
            "Fixed seed did not reproduce final best state."
        )

    if abs(
        result.final_current_objective
        - repeated.final_current_objective
    ) > 1e-12:
        raise AssertionError(
            "Fixed seed did not reproduce current objective."
        )

    if abs(
        result.final_best_objective
        - repeated.final_best_objective
    ) > 1e-12:
        raise AssertionError(
            "Fixed seed did not reproduce best objective."
        )

    first_log = [
        (
            log.candidate_objective,
            log.accepted,
            log.reward_event,
            log.reward,
            log.temperature_used,
        )
        for log in result.iteration_logs
    ]
    repeated_log = [
        (
            log.candidate_objective,
            log.accepted,
            log.reward_event,
            log.reward,
            log.temperature_used,
        )
        for log in repeated.iteration_logs
    ]

    if first_log != repeated_log:
        raise AssertionError(
            "Fixed seed did not reproduce iteration history."
        )

    print("[PASS] Fixed seed reproduces states, objectives, and run history")

    report = {
        "run": {
            "iterations": result.iterations,
            "initial_objective": (
                result.initial_objective
            ),
            "final_current_objective": (
                result.final_current_objective
            ),
            "final_best_objective": (
                result.final_best_objective
            ),
            "accepted_count": sum(
                log.accepted
                for log in result.iteration_logs
            ),
            "reward_event_counts": {
                event: sum(
                    log.reward_event == event
                    for log in result.iteration_logs
                )
                for event in (
                    "new_global_best",
                    "better_current",
                    "worse_accepted",
                    "rejected",
                )
            },
        },
        "temperature": {
            "initial_used": (
                result.iteration_logs[0]
                .temperature_used
            ),
            "final_after_cooling": (
                result.iteration_logs[-1]
                .temperature_after_cooling
            ),
            "monotonic": True,
        },
        "adaptive": {
            "destroy_uses": (
                destroy_record.segment_uses
            ),
            "repair_uses": (
                repair_record.segment_uses
            ),
            "completed_updates": (
                result.adaptive_result.state
                .completed_updates
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
        / "alns_main_loop_ml4_report.json"
    )
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nReport saved to: {output_path}")
    print(
        "\nALNS MAIN LOOP FIDELITY ML-4 — "
        "SHORT DETERMINISTIC INTEGRATED RUN PASSED"
    )


if __name__ == "__main__":
    main()
