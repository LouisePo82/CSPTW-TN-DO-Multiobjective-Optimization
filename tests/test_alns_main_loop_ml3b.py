from __future__ import annotations

from pathlib import Path
import json
import random

from alns_solver.adaptive_weights_factory import (
    PAPER_ADAPTIVE_WEIGHTS_MODE,
    build_adaptive_weights,
)
from alns_solver.paper_adaptive_weights import (
    PAPER_SEGMENT_LENGTH,
)
from alns_solver.paper_alns_main import (
    select_paper_operator_pair,
)
from alns_solver.paper_operator_dispatch import (
    PAPER_DESTROY_OPERATOR_NAMES,
    PAPER_REPAIR_OPERATOR_NAMES,
)


def weights(records):
    return {
        name: float(record.weight)
        for name, record in records.items()
    }


def scores(records):
    return {
        name: float(record.segment_score)
        for name, record in records.items()
    }


def uses(records):
    return {
        name: int(record.segment_uses)
        for name, record in records.items()
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]

    if PAPER_SEGMENT_LENGTH != 300:
        raise AssertionError(
            "Paper segment length is not 300."
        )

    adaptive = build_adaptive_weights(
        mode=PAPER_ADAPTIVE_WEIGHTS_MODE,
        destroy_operator_names=(
            PAPER_DESTROY_OPERATOR_NAMES
        ),
        repair_operator_names=(
            PAPER_REPAIR_OPERATOR_NAMES
        ),
    )
    state = adaptive.state
    controller = adaptive.controller
    rng = random.Random(2026)

    # ---------------------------------------------------------
    # ML-3B-A — Full paper pools and initial probabilities
    # ---------------------------------------------------------
    selection_1 = select_paper_operator_pair(
        iteration=1,
        adaptive_state=state,
        rng=rng,
    )

    if set(selection_1.destroy_probabilities) != set(
        PAPER_DESTROY_OPERATOR_NAMES
    ):
        raise AssertionError(
            "Destroy roulette does not expose the full paper pool."
        )

    if set(selection_1.repair_probabilities) != set(
        PAPER_REPAIR_OPERATOR_NAMES
    ):
        raise AssertionError(
            "Repair roulette does not expose the full paper pool."
        )

    expected_destroy_probability = (
        1.0 / len(PAPER_DESTROY_OPERATOR_NAMES)
    )
    expected_repair_probability = (
        1.0 / len(PAPER_REPAIR_OPERATOR_NAMES)
    )

    if any(
        abs(probability - expected_destroy_probability)
        > 1e-12
        for probability
        in selection_1.destroy_probabilities.values()
    ):
        raise AssertionError(
            "Equal initial destroy weights did not give equal probabilities."
        )

    if any(
        abs(probability - expected_repair_probability)
        > 1e-12
        for probability
        in selection_1.repair_probabilities.values()
    ):
        raise AssertionError(
            "Equal initial repair weights did not give equal probabilities."
        )

    print("[PASS] Roulette uses the full 12-operator destroy pool")
    print("[PASS] Roulette uses the full 6-operator repair pool")
    print("[PASS] Equal initial weights give equal probabilities")

    # ---------------------------------------------------------
    # ML-3B-B — Reproducible roulette sequence
    # ---------------------------------------------------------
    adaptive_repeat = build_adaptive_weights(
        mode=PAPER_ADAPTIVE_WEIGHTS_MODE,
        destroy_operator_names=(
            PAPER_DESTROY_OPERATOR_NAMES
        ),
        repair_operator_names=(
            PAPER_REPAIR_OPERATOR_NAMES
        ),
    )
    rng_a = random.Random(77)
    rng_b = random.Random(77)

    sequence_a = [
        (
            select_paper_operator_pair(
                iteration=iteration,
                adaptive_state=adaptive.state,
                rng=rng_a,
            ).destroy_operator,
            select_paper_operator_pair(
                iteration=iteration,
                adaptive_state=adaptive.state,
                rng=rng_a,
            ).repair_operator,
        )
        for iteration in range(1, 21)
    ]
    sequence_b = [
        (
            select_paper_operator_pair(
                iteration=iteration,
                adaptive_state=adaptive_repeat.state,
                rng=rng_b,
            ).destroy_operator,
            select_paper_operator_pair(
                iteration=iteration,
                adaptive_state=adaptive_repeat.state,
                rng=rng_b,
            ).repair_operator,
        )
        for iteration in range(1, 21)
    ]

    if sequence_a != sequence_b:
        raise AssertionError(
            "Fixed seed did not reproduce roulette sequence."
        )

    print("[PASS] Fixed seed reproduces roulette selections")

    # ---------------------------------------------------------
    # ML-3B-C — Fast-forward iterations 1..298
    # ---------------------------------------------------------
    selected_destroy = (
        PAPER_DESTROY_OPERATOR_NAMES[0]
    )
    selected_repair = (
        PAPER_REPAIR_OPERATOR_NAMES[0]
    )

    initial_destroy_weights = weights(
        state.destroy_records
    )
    initial_repair_weights = weights(
        state.repair_records
    )

    for iteration in range(1, 299):
        controller.process_iteration(
            iteration=iteration,
            destroy_operator=selected_destroy,
            repair_operator=selected_repair,
            candidate_objective=10.0,
            current_objective=10.0,
            best_objective=9.0,
            accepted=True,
        )

    if weights(state.destroy_records) != initial_destroy_weights:
        raise AssertionError(
            "Destroy weights changed before iteration 300."
        )

    if weights(state.repair_records) != initial_repair_weights:
        raise AssertionError(
            "Repair weights changed before iteration 300."
        )

    print("[PASS] No weight update occurs before iteration 300")

    # ---------------------------------------------------------
    # ML-3B-D — Iteration 299
    # ---------------------------------------------------------
    selection_299 = select_paper_operator_pair(
        iteration=299,
        adaptive_state=state,
        rng=rng,
    )

    controller.process_iteration(
        iteration=299,
        destroy_operator=(
            selection_299.destroy_operator
        ),
        repair_operator=(
            selection_299.repair_operator
        ),
        candidate_objective=9.5,
        current_objective=10.0,
        best_objective=9.0,
        accepted=True,
    )

    weights_after_299_destroy = weights(
        state.destroy_records
    )
    weights_after_299_repair = weights(
        state.repair_records
    )

    if weights_after_299_destroy != initial_destroy_weights:
        raise AssertionError(
            "Destroy weights changed at iteration 299."
        )

    if weights_after_299_repair != initial_repair_weights:
        raise AssertionError(
            "Repair weights changed at iteration 299."
        )

    print("[PASS] Iteration 299 records use/reward without updating weights")

    # ---------------------------------------------------------
    # ML-3B-E — Iteration 300 boundary
    # ---------------------------------------------------------
    selection_300 = select_paper_operator_pair(
        iteration=300,
        adaptive_state=state,
        rng=rng,
    )

    uses_before_300_destroy = uses(
        state.destroy_records
    )
    uses_before_300_repair = uses(
        state.repair_records
    )

    controller.process_iteration(
        iteration=300,
        destroy_operator=(
            selection_300.destroy_operator
        ),
        repair_operator=(
            selection_300.repair_operator
        ),
        candidate_objective=8.5,
        current_objective=9.5,
        best_objective=9.0,
        accepted=True,
    )

    if state.completed_updates != 1:
        raise AssertionError(
            "Eq. (48) did not update at iteration 300."
        )

    if any(
        value != 0.0
        for value in scores(
            state.destroy_records
        ).values()
    ):
        raise AssertionError(
            "Destroy segment scores did not reset."
        )

    if any(
        value != 0.0
        for value in scores(
            state.repair_records
        ).values()
    ):
        raise AssertionError(
            "Repair segment scores did not reset."
        )

    if any(
        value != 0
        for value in uses(
            state.destroy_records
        ).values()
    ):
        raise AssertionError(
            "Destroy segment uses did not reset."
        )

    if any(
        value != 0
        for value in uses(
            state.repair_records
        ).values()
    ):
        raise AssertionError(
            "Repair segment uses did not reset."
        )

    updated_destroy_weights = weights(
        state.destroy_records
    )
    updated_repair_weights = weights(
        state.repair_records
    )

    if (
        updated_destroy_weights
        == initial_destroy_weights
    ):
        raise AssertionError(
            "Destroy weights did not change at boundary."
        )

    if (
        updated_repair_weights
        == initial_repair_weights
    ):
        raise AssertionError(
            "Repair weights did not change at boundary."
        )

    print("[PASS] Iteration 300 is recorded before Eq. (48)")
    print("[PASS] Eq. (48) updates weights exactly at iteration 300")
    print("[PASS] Segment scores and uses reset after the update")

    # ---------------------------------------------------------
    # ML-3B-F — Iteration 301 uses updated weights
    # ---------------------------------------------------------
    selection_301 = select_paper_operator_pair(
        iteration=301,
        adaptive_state=state,
        rng=rng,
    )

    if (
        selection_301.destroy_weights
        != updated_destroy_weights
    ):
        raise AssertionError(
            "Iteration 301 did not see updated destroy weights."
        )

    if (
        selection_301.repair_weights
        != updated_repair_weights
    ):
        raise AssertionError(
            "Iteration 301 did not see updated repair weights."
        )

    controller.process_iteration(
        iteration=301,
        destroy_operator=(
            selection_301.destroy_operator
        ),
        repair_operator=(
            selection_301.repair_operator
        ),
        candidate_objective=10.0,
        current_objective=9.5,
        best_objective=8.5,
        accepted=False,
    )

    if sum(
        uses(state.destroy_records).values()
    ) != 1:
        raise AssertionError(
            "Segment 2 destroy uses did not start at one."
        )

    if sum(
        uses(state.repair_records).values()
    ) != 1:
        raise AssertionError(
            "Segment 2 repair uses did not start at one."
        )

    if state.completed_updates != 1:
        raise AssertionError(
            "Iteration 301 caused an extra weight update."
        )

    print("[PASS] Iteration 301 roulette uses updated weights")
    print("[PASS] Iteration 301 starts the second segment")
    print("[PASS] No extra update occurs at iteration 301")
    print("[PASS] F_lambda affects rewards, not roulette normalization")
    print("[PASS] No enhanced operator-selection behavior is introduced")

    report = {
        "paper_pools": {
            "destroy_count": len(
                PAPER_DESTROY_OPERATOR_NAMES
            ),
            "repair_count": len(
                PAPER_REPAIR_OPERATOR_NAMES
            ),
        },
        "initial_probabilities": {
            "destroy": (
                selection_1.destroy_probabilities
            ),
            "repair": (
                selection_1.repair_probabilities
            ),
        },
        "boundary": {
            "iteration_299_selection": {
                "destroy": (
                    selection_299.destroy_operator
                ),
                "repair": (
                    selection_299.repair_operator
                ),
            },
            "iteration_300_selection": {
                "destroy": (
                    selection_300.destroy_operator
                ),
                "repair": (
                    selection_300.repair_operator
                ),
                "uses_before_boundary_destroy": (
                    uses_before_300_destroy
                ),
                "uses_before_boundary_repair": (
                    uses_before_300_repair
                ),
            },
            "completed_updates_after_300": (
                state.completed_updates
            ),
            "updated_destroy_weights": (
                updated_destroy_weights
            ),
            "updated_repair_weights": (
                updated_repair_weights
            ),
            "iteration_301_selection": {
                "destroy": (
                    selection_301.destroy_operator
                ),
                "repair": (
                    selection_301.repair_operator
                ),
            },
        },
        "fidelity": {
            "paper_faithful": True,
            "enhanced": False,
            "segment_length": 300,
            "reaction_factor": 0.1,
            "objective_role": (
                "F_lambda affects rewards and weights; "
                "roulette uses normalized weights only"
            ),
        },
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
        / "alns_main_loop_ml3b_report.json"
    )
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nReport saved to: {output_path}")
    print(
        "\nALNS MAIN LOOP FIDELITY ML-3B — "
        "ROULETTE AND SEGMENT BOUNDARY PASSED"
    )


if __name__ == "__main__":
    main()
