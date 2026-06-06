from __future__ import annotations

from pathlib import Path
import json

from alns_solver.paper_adaptive_weights import (
    PaperAdaptiveWeightState,
    PaperAdaptiveWeightController,
)


TOLERANCE = 1e-12


def assert_close(
    actual: float,
    expected: float,
    label: str,
) -> None:
    if abs(
        float(actual) - float(expected)
    ) > TOLERANCE:
        raise AssertionError(
            f"{label}: actual={actual}, expected={expected}"
        )


def process_rejected(
    controller: PaperAdaptiveWeightController,
    *,
    iteration: int,
    destroy_operator: str,
    repair_operator: str,
):
    return controller.process_iteration(
        iteration=iteration,
        destroy_operator=destroy_operator,
        repair_operator=repair_operator,
        candidate_objective=11.0,
        current_objective=10.0,
        best_objective=9.0,
        accepted=False,
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]

    state = PaperAdaptiveWeightState.create(
        (
            "destroy_a",
            "destroy_b",
            "destroy_unused",
        ),
        (
            "repair_a",
            "repair_b",
        ),
        initial_weight=1.0,
    )
    controller = PaperAdaptiveWeightController(
        state=state
    )

    report = {}

    # ---------------------------------------------------------
    # AW-3A — Sequential iteration contract
    # ---------------------------------------------------------
    try:
        controller.process_iteration(
            iteration=2,
            destroy_operator="destroy_a",
            repair_operator="repair_a",
            candidate_objective=10.0,
            current_objective=10.0,
            best_objective=9.0,
            accepted=False,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Non-consecutive iteration was accepted."
        )

    print("[PASS] Segment controller requires consecutive iterations")

    # ---------------------------------------------------------
    # AW-3B — Iterations 1-296: rejected events still count uses
    # ---------------------------------------------------------
    for iteration in range(1, 297):
        result = process_rejected(
            controller,
            iteration=iteration,
            destroy_operator="destroy_a",
            repair_operator="repair_a",
        )

        if result.reward != 0.0:
            raise AssertionError(
                "Rejected iteration received non-zero reward."
            )

        if result.update_applied:
            raise AssertionError(
                "Weight update occurred before iteration 300."
            )

    if state.destroy_records[
        "destroy_a"
    ].segment_uses != 296:
        raise AssertionError(
            "Rejected destroy uses were not counted."
        )

    if state.repair_records[
        "repair_a"
    ].segment_uses != 296:
        raise AssertionError(
            "Rejected repair uses were not counted."
        )

    print("[PASS] Rejected iterations receive reward zero")
    print("[PASS] Rejected iterations still increment usage counts")
    print("[PASS] No update occurs before the segment boundary")

    # ---------------------------------------------------------
    # AW-3C — Reward priority and same reward for operator pair
    # ---------------------------------------------------------
    result_297 = controller.process_iteration(
        iteration=297,
        destroy_operator="destroy_a",
        repair_operator="repair_a",
        candidate_objective=8.0,
        current_objective=10.0,
        best_objective=9.0,
        accepted=True,
    )
    if (
        result_297.event != "new_global_best"
        or result_297.reward != 33.0
    ):
        raise AssertionError(
            "Iteration 297 did not receive new-best reward."
        )

    result_298 = controller.process_iteration(
        iteration=298,
        destroy_operator="destroy_a",
        repair_operator="repair_a",
        candidate_objective=9.5,
        current_objective=10.0,
        best_objective=9.0,
        accepted=True,
    )
    if (
        result_298.event != "better_current"
        or result_298.reward != 15.0
    ):
        raise AssertionError(
            "Iteration 298 did not receive better-current reward."
        )

    result_299 = controller.process_iteration(
        iteration=299,
        destroy_operator="destroy_b",
        repair_operator="repair_b",
        candidate_objective=11.0,
        current_objective=10.0,
        best_objective=9.0,
        accepted=True,
    )
    if (
        result_299.event != "worse_accepted"
        or result_299.reward != 9.0
    ):
        raise AssertionError(
            "Iteration 299 did not receive worse-accepted reward."
        )

    print("[PASS] New global best has reward priority sigma1=33")
    print("[PASS] Better current receives sigma2=15")
    print("[PASS] Worse accepted receives sigma3=9")

    # ---------------------------------------------------------
    # AW-3D — Iteration 300 is recorded before update
    # ---------------------------------------------------------
    result_300 = process_rejected(
        controller,
        iteration=300,
        destroy_operator="destroy_b",
        repair_operator="repair_b",
    )

    if not result_300.update_applied:
        raise AssertionError(
            "Iteration 300 did not trigger weight update."
        )

    if result_300.completed_updates != 1:
        raise AssertionError(
            "First completed-update count is incorrect."
        )

    before = result_300.update_event["before"]

    if before["destroy"]["destroy_a"][
        "segment_uses"
    ] != 298:
        raise AssertionError(
            "Destroy A usage count before update is incorrect."
        )

    if before["destroy"]["destroy_b"][
        "segment_uses"
    ] != 2:
        raise AssertionError(
            "Boundary iteration was not included in segment 1."
        )

    if before["repair"]["repair_b"][
        "segment_uses"
    ] != 2:
        raise AssertionError(
            "Repair boundary use was not included in segment 1."
        )

    expected_a = (
        0.9 * 1.0
        + 0.1 * (48.0 / 298.0)
    )
    expected_b = (
        0.9 * 1.0
        + 0.1 * (9.0 / 2.0)
    )

    assert_close(
        state.destroy_records[
            "destroy_a"
        ].weight,
        expected_a,
        "Destroy A segment-1 weight",
    )
    assert_close(
        state.repair_records[
            "repair_a"
        ].weight,
        expected_a,
        "Repair A segment-1 weight",
    )
    assert_close(
        state.destroy_records[
            "destroy_b"
        ].weight,
        expected_b,
        "Destroy B segment-1 weight",
    )
    assert_close(
        state.repair_records[
            "repair_b"
        ].weight,
        expected_b,
        "Repair B segment-1 weight",
    )
    assert_close(
        state.destroy_records[
            "destroy_unused"
        ].weight,
        1.0,
        "Unused destroy weight",
    )

    for record in (
        list(state.destroy_records.values())
        + list(state.repair_records.values())
    ):
        if (
            record.segment_uses != 0
            or record.segment_score != 0.0
        ):
            raise AssertionError(
                f"{record.name}: segment statistics not reset."
            )

    print("[PASS] Iteration 300 belongs to the first segment")
    print("[PASS] Selected destroy/repair pair receives same reward")
    print("[PASS] Eq. (48) runs after recording boundary iteration")
    print("[PASS] Segment statistics reset after update")

    # ---------------------------------------------------------
    # AW-3E — New segment starts at 301 and updates at 600
    # ---------------------------------------------------------
    for iteration in range(301, 601):
        result = process_rejected(
            controller,
            iteration=iteration,
            destroy_operator="destroy_unused",
            repair_operator="repair_b",
        )

        if (
            iteration < 600
            and result.update_applied
        ):
            raise AssertionError(
                "Second segment updated before iteration 600."
            )

    if not result.update_applied:
        raise AssertionError(
            "Iteration 600 did not trigger second update."
        )

    if state.completed_updates != 2:
        raise AssertionError(
            "Second completed-update count is incorrect."
        )

    # Used with average reward zero, therefore Eq. (48) decays
    # only the old-weight component.
    assert_close(
        state.destroy_records[
            "destroy_unused"
        ].weight,
        0.9,
        "Second-segment zero-average destroy weight",
    )
    assert_close(
        state.repair_records[
            "repair_b"
        ].weight,
        0.9 * expected_b,
        "Second-segment zero-average repair weight",
    )

    # Operators unused in segment 2 preserve their segment-1 weights.
    assert_close(
        state.destroy_records[
            "destroy_a"
        ].weight,
        expected_a,
        "Unused segment-2 destroy A preserved",
    )
    assert_close(
        state.destroy_records[
            "destroy_b"
        ].weight,
        expected_b,
        "Unused segment-2 destroy B preserved",
    )

    print("[PASS] Segment 2 starts with iteration 301")
    print("[PASS] Second update occurs exactly at iteration 600")
    print("[PASS] Used zero-reward operators follow Eq. (48)")
    print("[PASS] Unused segment-2 operators preserve old weights")

    # ---------------------------------------------------------
    # AW-3F — Scalar F_lambda compatibility
    # ---------------------------------------------------------
    scalar_state = PaperAdaptiveWeightState.create(
        ("destroy_scalar",),
        ("repair_scalar",),
    )
    scalar_controller = (
        PaperAdaptiveWeightController(
            state=scalar_state
        )
    )

    # Example: cost may be worse, but F_lambda is better.
    # Controller intentionally receives only F_lambda values.
    scalar_result = scalar_controller.process_iteration(
        iteration=1,
        destroy_operator="destroy_scalar",
        repair_operator="repair_scalar",
        candidate_objective=0.425,
        current_objective=0.500,
        best_objective=0.450,
        accepted=True,
    )

    if (
        scalar_result.event
        != "new_global_best"
        or scalar_result.reward != 33.0
    ):
        raise AssertionError(
            "Scalar F_lambda improvement was misclassified."
        )

    print("[PASS] Controller uses one scalar F_lambda objective")
    print("[PASS] No separate cost/emission reward logic is introduced")

    report["segment_1"] = {
        "expected_destroy_a_weight": expected_a,
        "expected_destroy_b_weight": expected_b,
        "boundary_update": result_300.update_event,
    }
    report["segment_2"] = {
        "completed_updates": state.completed_updates,
        "final_state": state.snapshot(),
    }
    report["reward_events"] = {
        "iteration_297": {
            "event": result_297.event,
            "reward": result_297.reward,
        },
        "iteration_298": {
            "event": result_298.event,
            "reward": result_298.reward,
        },
        "iteration_299": {
            "event": result_299.event,
            "reward": result_299.reward,
        },
        "iteration_300": {
            "event": result_300.event,
            "reward": result_300.reward,
            "update_applied": (
                result_300.update_applied
            ),
        },
    }
    report["multiobjective_scalar_example"] = {
        "candidate_F_lambda": 0.425,
        "current_F_lambda": 0.500,
        "best_F_lambda": 0.450,
        "event": scalar_result.event,
        "reward": scalar_result.reward,
    }

    output_dir = (
        root
        / "outputs"
        / "alns_adaptive_weights_fidelity_tests"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path = (
        output_dir
        / "adaptive_weights_aw3_report.json"
    )
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nReport saved to: {output_path}")
    print(
        "\nADAPTIVE WEIGHTS FIDELITY AW-3 — "
        "SEGMENT CONTROLLER PASSED"
    )


if __name__ == "__main__":
    main()
