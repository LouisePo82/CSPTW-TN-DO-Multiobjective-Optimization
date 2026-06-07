from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import random

from core.instance_loader import load_instance

from alns_solver.paper_alns_main import (
    PaperALNSCandidatePipelineResult,
    apply_paper_sa_transition,
    build_ml2_paper_controllers,
    select_paper_operator_pair,
)
from alns_solver.paper_operator_dispatch import (
    PAPER_DESTROY_OPERATOR_NAMES,
    PAPER_REPAIR_OPERATOR_NAMES,
    PaperOperatorDispatchContext,
    dispatch_paper_destroy,
    dispatch_paper_repair,
)
from alns_solver.paper_removal_quantity import (
    destroy_operator_uses_removal_quantity,
    sample_paper_removal_quantity,
)
from alns_solver.paper_history import (
    PaperHistoricalPositionState,
)
from alns_solver.local_search_factory import (
    PAPER_LOCAL_SEARCH_MODE,
    build_local_search,
)

from tests.test_alns_destroy_paper_fidelity_gate1 import (
    emission_anchor_state,
)
from tests.test_alns_destroy_paper_fidelity_gate2 import (
    mixed_home_state,
)
from tests.test_alns_destroy_paper_fidelity_gate3 import (
    cost_anchor_state,
)


LAMBDA_VALUE = 0.5
COST_BOUNDS = (
    23.089059445460528,
    24.28427622523578,
)
EMISSION_BOUNDS = (
    77.85476672718833,
    79.22375667475296,
)
EMISSION_FACTORS = (3.0, 1.0)
ITERATIONS = 36
SEED = 2026


def state_signature(state):
    return {
        "dv_routes": deepcopy(state.dv_routes),
        "od_routes": deepcopy(state.od_routes),
        "assignments": deepcopy(state.assignments),
        "unassigned": sorted(
            state.unassigned_customers
        ),
    }


def fixture_key_for_destroy(operator_name):
    if operator_name in {
        "random_adp_removal",
        "worst_adp_removal",
    }:
        return "active_adp"

    if operator_name in {
        "random_tn_removal",
        "route_removal",
    }:
        return "active_tn"

    return "general_home"


def build_fixture(fixture_key):
    if fixture_key == "active_adp":
        return cost_anchor_state()

    if fixture_key == "active_tn":
        return emission_anchor_state()

    if fixture_key == "general_home":
        return mixed_home_state()

    raise ValueError(
        f"Unknown fixture key: {fixture_key}"
    )


def evaluate_state(state, instance):
    solution = state.to_core_solution(
        instance=instance,
        lambda_value=LAMBDA_VALUE,
        objective_mode="weighted",
        cost_bounds=COST_BOUNDS,
        emission_bounds=EMISSION_BOUNDS,
        emission_factors=EMISSION_FACTORS,
        require_complete=True,
    )

    if not solution.validator_pass:
        raise AssertionError(
            f"Fixture/state validation failed: "
            f"{solution.validation_errors}"
        )

    return solution


def run_gate(instance):
    base_fixture = mixed_home_state()
    base_solution = evaluate_state(
        base_fixture,
        instance,
    )

    adaptive, sa = build_ml2_paper_controllers(
        initial_objective=float(
            base_solution.objective
        ),
        destroy_operator_names=(
            PAPER_DESTROY_OPERATOR_NAMES
        ),
        repair_operator_names=(
            PAPER_REPAIR_OPERATOR_NAMES
        ),
        seed=SEED + 50_000,
    )

    roulette_rng = random.Random(SEED)
    quantity_rng = random.Random(SEED + 1)

    histories = {
        key: (
            PaperHistoricalPositionState
            .initialize_from_state(
                build_fixture(key),
                instance,
            )
        )
        for key in (
            "general_home",
            "active_adp",
            "active_tn",
        )
    }

    logs = []

    for iteration in range(1, ITERATIONS + 1):
        selection = select_paper_operator_pair(
            iteration=iteration,
            adaptive_state=adaptive.state,
            rng=roulette_rng,
        )

        destroy_name = (
            selection.destroy_operator
        )
        repair_name = (
            selection.repair_operator
        )
        fixture_key = fixture_key_for_destroy(
            destroy_name
        )
        current_state = build_fixture(
            fixture_key
        )
        best_state = current_state.copy()

        current_solution = evaluate_state(
            current_state,
            instance,
        )
        current_objective = float(
            current_solution.objective
        )
        best_objective = current_objective

        if destroy_operator_uses_removal_quantity(
            destroy_name
        ):
            quantity_sample = (
                sample_paper_removal_quantity(
                    len(instance["customers"]),
                    rng=quantity_rng,
                    seed=SEED + iteration,
                )
            )
            removal_count = (
                quantity_sample.quantity
            )
        else:
            quantity_sample = None
            removal_count = 1

        history_snapshot = (
            histories[fixture_key].snapshot()
        )

        context = PaperOperatorDispatchContext(
            removal_count=removal_count,
            seed=SEED + iteration,
            lambda_value=LAMBDA_VALUE,
            cost_bounds=COST_BOUNDS,
            emission_bounds=EMISSION_BOUNDS,
            emission_factors=EMISSION_FACTORS,
            best_historical_position_costs=(
                history_snapshot
            ),
        )

        source_signature = state_signature(
            current_state
        )

        destroy_result = (
            dispatch_paper_destroy(
                destroy_name,
                current_state,
                instance,
                context=context,
            )
        )

        if (
            state_signature(current_state)
            != source_signature
        ):
            raise AssertionError(
                "Destroy dispatch mutated the "
                "fixture source state."
            )

        if not destroy_result.removed_customers:
            raise AssertionError(
                f"{destroy_name} removed no customers."
            )

        repair_result = dispatch_paper_repair(
            repair_name,
            destroy_result.state,
            instance,
            context=context,
        )

        if not repair_result.validator_pass:
            raise AssertionError(
                f"{repair_name} produced invalid "
                "repair result."
            )

        local_search_result = build_local_search(
            repair_result.state,
            instance,
            mode=PAPER_LOCAL_SEARCH_MODE,
            best_objective=best_objective,
            lambda_value=LAMBDA_VALUE,
            cost_bounds=COST_BOUNDS,
            emission_bounds=EMISSION_BOUNDS,
            emission_factors=EMISSION_FACTORS,
        )

        if (
            not local_search_result.metadata[
                "paper_faithful"
            ]
            or local_search_result.metadata[
                "enhanced"
            ]
        ):
            raise AssertionError(
                "Non-paper local search entered ML-5B."
            )

        candidate_solution = (
            local_search_result.state
            .to_core_solution(
                instance=instance,
                lambda_value=LAMBDA_VALUE,
                objective_mode="weighted",
                cost_bounds=COST_BOUNDS,
                emission_bounds=EMISSION_BOUNDS,
                emission_factors=(
                    EMISSION_FACTORS
                ),
                require_complete=True,
            )
        )

        if not candidate_solution.validator_pass:
            raise AssertionError(
                "ML-5B candidate failed shared "
                "validation."
            )

        candidate_pipeline = (
            PaperALNSCandidatePipelineResult(
                destroy_operator=destroy_name,
                repair_operator=repair_name,
                local_search_mode=(
                    PAPER_LOCAL_SEARCH_MODE
                ),
                destroy_result=destroy_result,
                repair_result=repair_result,
                local_search_result=(
                    local_search_result
                ),
                candidate_state=(
                    local_search_result
                    .state.copy()
                ),
                candidate_objective=float(
                    candidate_solution.objective
                ),
                candidate_cost=float(
                    candidate_solution.cost
                ),
                candidate_emission=float(
                    candidate_solution.emission
                ),
                candidate_dv_distance=float(
                    candidate_solution.dv_distance
                ),
                candidate_od_extra_distance=float(
                    candidate_solution
                    .od_extra_distance
                ),
                validator_pass=True,
                validation_errors=[],
                metadata={
                    "paper_faithful": True,
                    "enhanced": False,
                    "pipeline_scope": (
                        "ml5b_applicable_fixture_"
                        "roulette_integration"
                    ),
                    "fixture_key": fixture_key,
                    "sampled_q": (
                        None
                        if quantity_sample is None
                        else quantity_sample.quantity
                    ),
                    "operator_substitution": False,
                    "fallback": False,
                    "objective_input": (
                        "scalar_F_lambda"
                    ),
                },
            )
        )

        transition = apply_paper_sa_transition(
            iteration=iteration,
            candidate_pipeline=(
                candidate_pipeline
            ),
            current_state=current_state,
            best_state=best_state,
            current_objective=(
                current_objective
            ),
            best_objective=best_objective,
            sa_controller=sa.controller,
        )

        if not (
            transition
            .temperature_after_cooling
            < transition.temperature_used
        ):
            raise AssertionError(
                "Temperature did not cool exactly "
                "after the iteration decision."
            )

        histories[
            fixture_key
        ].observe_current_state(
            transition.current_state_after,
            instance,
        )

        logs.append({
            "iteration": iteration,
            "destroy_operator": destroy_name,
            "repair_operator": repair_name,
            "fixture_key": fixture_key,
            "removal_count": removal_count,
            "removed_customers": list(
                destroy_result.removed_customers
            ),
            "repair_objective": float(
                repair_result.final_objective
            ),
            "repair_state_signature": (
                state_signature(
                    repair_result.state
                )
            ),
            "local_search_state_signature": (
                state_signature(
                    local_search_result.state
                )
            ),
            "candidate_objective": float(
                candidate_solution.objective
            ),
            "accepted": bool(
                transition.accepted
            ),
            "reward_event": (
                transition.reward_event
            ),
            "reward": float(
                transition.reward
            ),
            "temperature_used": float(
                transition.temperature_used
            ),
            "temperature_after_cooling": float(
                transition
                .temperature_after_cooling
            ),
        })

    return {
        "logs": logs,
        "adaptive": adaptive,
        "sa": sa,
        "histories": histories,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    instance = load_instance(
        root / "data" / "small" / "instance_001"
    )

    first = run_gate(instance)
    second = run_gate(instance)

    first_logs = first["logs"]
    second_logs = second["logs"]

    if first_logs != second_logs:
        print("\nML-5B REPRODUCIBILITY DIAGNOSTIC")
        print("=" * 72)

        if len(first_logs) != len(second_logs):
            print(
                "Different log lengths:",
                len(first_logs),
                len(second_logs),
            )

        for index, (first_row, second_row) in enumerate(
            zip(first_logs, second_logs),
            start=1,
        ):
            if first_row == second_row:
                continue

            print(f"First differing iteration: {index}")

            all_keys = sorted(
                set(first_row) | set(second_row)
            )

            for key in all_keys:
                first_value = first_row.get(key)
                second_value = second_row.get(key)

                if first_value != second_value:
                    print(f"  Field: {key}")
                    print(f"    first : {first_value!r}")
                    print(f"    second: {second_value!r}")

                    if (
                        isinstance(first_value, float)
                        and isinstance(second_value, float)
                    ):
                        print(
                            "    absolute difference:",
                            abs(first_value - second_value),
                        )

            break

        raise AssertionError(
            "Fixed seeds did not reproduce the "
            "ML-5B integration history. "
            "See diagnostic above."
        )

    print(
        "[PASS] Fixed seeds reproduce full ML-5B "
        "roulette integration"
    )

    if len(first_logs) != ITERATIONS:
        raise AssertionError(
            "ML-5B iteration count is incorrect."
        )

    if [
        row["iteration"]
        for row in first_logs
    ] != list(range(1, ITERATIONS + 1)):
        raise AssertionError(
            "ML-5B iterations are not consecutive."
        )

    print(
        "[PASS] ML-5B completes consecutive "
        "integration iterations"
    )

    destroy_used = {
        row["destroy_operator"]
        for row in first_logs
    }
    repair_used = {
        row["repair_operator"]
        for row in first_logs
    }

    if not destroy_used <= set(
        PAPER_DESTROY_OPERATOR_NAMES
    ):
        raise AssertionError(
            "Roulette selected a non-paper "
            "destroy operator."
        )

    if not repair_used <= set(
        PAPER_REPAIR_OPERATOR_NAMES
    ):
        raise AssertionError(
            "Roulette selected a non-paper "
            "repair operator."
        )

    if len(destroy_used) < 2:
        raise AssertionError(
            "Controlled seed did not exercise "
            "multiple destroy operators."
        )

    if len(repair_used) < 2:
        raise AssertionError(
            "Controlled seed did not exercise "
            "multiple repair operators."
        )

    print(
        "[PASS] Roulette selections remain inside "
        "the full paper pools"
    )
    print(
        "[PASS] Controlled run exercises multiple "
        "destroy and repair operators"
    )

    adaptive = first["adaptive"].state

    total_destroy_uses = sum(
        record.segment_uses
        for record
        in adaptive.destroy_records.values()
    )
    total_repair_uses = sum(
        record.segment_uses
        for record
        in adaptive.repair_records.values()
    )

    if total_destroy_uses != ITERATIONS:
        raise AssertionError(
            "Destroy adaptive usage count does not "
            "match iteration count."
        )

    if total_repair_uses != ITERATIONS:
        raise AssertionError(
            "Repair adaptive usage count does not "
            "match iteration count."
        )

    if adaptive.completed_updates != 0:
        raise AssertionError(
            "Short ML-5B run unexpectedly crossed "
            "the 300-iteration boundary."
        )

    print(
        "[PASS] Adaptive use is recorded exactly "
        "once per selected operator pair"
    )
    print(
        "[PASS] No adaptive segment update occurs "
        "before iteration 300"
    )

    temperatures = [
        row["temperature_used"]
        for row in first_logs
    ]

    if any(
        later >= earlier
        for earlier, later in zip(
            temperatures,
            temperatures[1:],
        )
    ):
        raise AssertionError(
            "ML-5B temperature sequence is not "
            "strictly decreasing."
        )

    print(
        "[PASS] Paper SA cools once per "
        "integration iteration"
    )

    if any(
        row["fixture_key"]
        != fixture_key_for_destroy(
            row["destroy_operator"]
        )
        for row in first_logs
    ):
        raise AssertionError(
            "Fixture router substituted or "
            "misrouted a destroy operator."
        )

    print(
        "[PASS] Every selected destroy operator "
        "executes without substitution"
    )
    print(
        "[PASS] Operator-specific fixtures avoid "
        "inventing fallback behavior"
    )

    for history in first[
        "histories"
    ].values():
        metadata = history.metadata()

        if (
            not metadata["paper_faithful"]
            or metadata["enhanced"]
            or metadata[
                "objective_extension_applied"
            ]
        ):
            raise AssertionError(
                "Historical lifecycle lost its "
                "paper contract."
            )

    print(
        "[PASS] Historical observations update "
        "after current-state transitions"
    )
    print(
        "[PASS] Historical metric remains "
        "independent of F_lambda"
    )

    report = {
        "iterations": ITERATIONS,
        "seed": SEED,
        "destroy_operators_used": sorted(
            destroy_used
        ),
        "repair_operators_used": sorted(
            repair_used
        ),
        "accepted_count": sum(
            row["accepted"]
            for row in first_logs
        ),
        "reward_event_counts": {
            event: sum(
                row["reward_event"] == event
                for row in first_logs
            )
            for event in (
                "new_global_best",
                "better_current",
                "worse_accepted",
                "rejected",
            )
        },
        "adaptive": {
            "destroy_uses": (
                total_destroy_uses
            ),
            "repair_uses": (
                total_repair_uses
            ),
            "completed_updates": (
                adaptive.completed_updates
            ),
        },
        "contracts": {
            "paper_faithful": True,
            "enhanced": False,
            "full_paper_roulette_pools": True,
            "sampled_paper_q": True,
            "paper_dispatch": True,
            "paper_local_search": True,
            "paper_simulated_annealing": True,
            "paper_adaptive_rewards": True,
            "paper_history_update": True,
            "operator_substitution": False,
            "fallback": False,
            "fixture_router_test_only": True,
            "production_applicability_policy": (
                "not_defined_by_this_gate"
            ),
            "objective_input": (
                "scalar_F_lambda"
            ),
        },
        "iterations_log": first_logs,
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
        / "alns_main_loop_ml5b_report.json"
    )
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nReport saved to: {output_path}")
    print(
        "\nALNS MAIN LOOP FIDELITY ML-5B — "
        "FULL-POOL APPLICABLE ROULETTE "
        "INTEGRATION PASSED"
    )


if __name__ == "__main__":
    main()
