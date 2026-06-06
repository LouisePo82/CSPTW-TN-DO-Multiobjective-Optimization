from __future__ import annotations
from pathlib import Path
import json
from alns_solver.paper_adaptive_weights import (
    PAPER_SEGMENT_LENGTH,
    PAPER_REACTION_FACTOR,
    PAPER_REWARD_NEW_GLOBAL_BEST,
    PAPER_REWARD_BETTER_CURRENT,
    PAPER_REWARD_WORSE_ACCEPTED,
    PAPER_REWARD_REJECTED,
    PaperAdaptiveWeightState,
    classify_paper_reward,
)

TOLERANCE = 1e-12

def assert_close(actual: float, expected: float, label: str) -> None:
    if abs(float(actual) - float(expected)) > TOLERANCE:
        raise AssertionError(f"{label}: actual={actual}, expected={expected}")

def main() -> None:
    root = Path(__file__).resolve().parents[1]
    destroy_names = (
        "random_customer_removal",
        "related_removal",
        "route_removal",
    )
    repair_names = ("best_insertion", "regret_2")
    state = PaperAdaptiveWeightState.create(
        destroy_names, repair_names, initial_weight=1.0
    )
    report = {}

    if PAPER_SEGMENT_LENGTH != 300:
        raise AssertionError("Paper segment length must equal 300.")
    assert_close(PAPER_REACTION_FACTOR, 0.1, "Paper reaction factor")
    assert_close(PAPER_REWARD_NEW_GLOBAL_BEST, 33.0, "Sigma 1")
    assert_close(PAPER_REWARD_BETTER_CURRENT, 15.0, "Sigma 2")
    assert_close(PAPER_REWARD_WORSE_ACCEPTED, 9.0, "Sigma 3")
    assert_close(PAPER_REWARD_REJECTED, 0.0, "Rejected reward")
    print("[PASS] Paper segment length eta_s=300")
    print("[PASS] Paper reaction factor zeta=0.1")
    print("[PASS] Paper rewards are 33, 15, 9, and 0")

    for record in state.destroy_records.values():
        assert_close(record.weight, 1.0, "Destroy initial weight")
        assert_close(record.segment_score, 0.0, "Destroy initial score")
        if record.segment_uses != 0:
            raise AssertionError("Destroy initial usage count is not zero.")
    for record in state.repair_records.values():
        assert_close(record.weight, 1.0, "Repair initial weight")
        assert_close(record.segment_score, 0.0, "Repair initial score")
        if record.segment_uses != 0:
            raise AssertionError("Repair initial usage count is not zero.")
    print("[PASS] Destroy operators start with equal weights")
    print("[PASS] Repair operators start with equal weights")
    print("[PASS] Destroy and repair pools are stored separately")

    cases = [
        (dict(candidate_objective=8.0,current_objective=10.0,best_objective=9.0,accepted=True),("new_global_best",33.0)),
        (dict(candidate_objective=9.5,current_objective=10.0,best_objective=9.0,accepted=True),("better_current",15.0)),
        (dict(candidate_objective=11.0,current_objective=10.0,best_objective=9.0,accepted=True),("worse_accepted",9.0)),
        (dict(candidate_objective=11.0,current_objective=10.0,best_objective=9.0,accepted=False),("rejected",0.0)),
        (dict(candidate_objective=10.0,current_objective=10.0,best_objective=9.0,accepted=True),("rejected",0.0)),
    ]
    classification_results = []
    for arguments, expected in cases:
        actual = classify_paper_reward(**arguments)
        if actual != expected:
            raise AssertionError(
                f"Reward classification mismatch: actual={actual}, expected={expected}"
            )
        classification_results.append({**arguments, "event": actual[0], "reward": actual[1]})
    print("[PASS] New global best receives sigma1=33")
    print("[PASS] Better current receives sigma2=15")
    print("[PASS] Worse accepted receives sigma3=9")
    print("[PASS] Rejected/equal candidate receives zero")
    print("[PASS] Reward classification accepts one scalar F_lambda")

    state.record_event(
        destroy_operator="random_customer_removal",
        repair_operator="best_insertion",
        event="new_global_best",
    )
    state.record_event(
        destroy_operator="random_customer_removal",
        repair_operator="best_insertion",
        event="better_current",
    )
    state.record_event(
        destroy_operator="related_removal",
        repair_operator="regret_2",
        event="worse_accepted",
    )

    random_destroy = state.destroy_records["random_customer_removal"]
    best_repair = state.repair_records["best_insertion"]
    if random_destroy.segment_uses != 2 or best_repair.segment_uses != 2:
        raise AssertionError("Usage accumulation is incorrect.")
    assert_close(random_destroy.segment_score, 48.0, "Destroy cumulative score")
    assert_close(best_repair.segment_score, 48.0, "Repair cumulative score")
    print("[PASS] Selected destroy operator accumulates score/use")
    print("[PASS] Selected repair operator receives same reward")

    if state.should_update(299):
        raise AssertionError("Weights updated before segment boundary.")
    if not state.should_update(300):
        raise AssertionError("Weights did not update at iteration 300.")
    if state.should_update(301):
        raise AssertionError("Weights updated after segment boundary.")
    if not state.should_update(600):
        raise AssertionError("Weights did not update at iteration 600.")
    print("[PASS] Weight update occurs only at 300, 600, ...")

    before = state.snapshot()
    update_event = state.update_weights(iteration=300)
    expected_random = 0.9 * 1.0 + 0.1 * (48.0 / 2.0)
    expected_related = 0.9 * 1.0 + 0.1 * 9.0

    assert_close(
        state.destroy_records["random_customer_removal"].weight,
        expected_random,
        "Eq. 48 random destroy",
    )
    assert_close(
        state.repair_records["best_insertion"].weight,
        expected_random,
        "Eq. 48 best repair",
    )
    assert_close(
        state.destroy_records["related_removal"].weight,
        expected_related,
        "Eq. 48 related destroy",
    )
    assert_close(
        state.repair_records["regret_2"].weight,
        expected_related,
        "Eq. 48 regret repair",
    )
    assert_close(
        state.destroy_records["route_removal"].weight,
        1.0,
        "Unused operator preserved weight",
    )
    for record in list(state.destroy_records.values()) + list(state.repair_records.values()):
        assert_close(record.segment_score, 0.0, "Segment score reset")
        if record.segment_uses != 0:
            raise AssertionError(f"{record.name}: usage count was not reset.")
    if state.completed_updates != 1:
        raise AssertionError("Completed update counter is incorrect.")

    print("[PASS] Eq. (48) updates used operators exactly")
    print("[PASS] Unused operator preserves its old weight")
    print("[PASS] Segment scores and usage counts reset")
    print("[PASS] Updated weights remain positive")

    report["paper_constants"] = {
        "segment_length": PAPER_SEGMENT_LENGTH,
        "reaction_factor": PAPER_REACTION_FACTOR,
        "reward_new_global_best": PAPER_REWARD_NEW_GLOBAL_BEST,
        "reward_better_current": PAPER_REWARD_BETTER_CURRENT,
        "reward_worse_accepted": PAPER_REWARD_WORSE_ACCEPTED,
        "reward_rejected": PAPER_REWARD_REJECTED,
    }
    report["reward_classification"] = classification_results
    report["before_update"] = before
    report["update_event"] = update_event
    report["expected_weights"] = {
        "random_customer_removal": expected_random,
        "best_insertion": expected_random,
        "related_removal": expected_related,
        "regret_2": expected_related,
        "route_removal": 1.0,
    }

    output_dir = root / "outputs" / "alns_adaptive_weights_fidelity_tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "adaptive_weights_aw1_report.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport saved to: {output_path}")
    print(
        "\nADAPTIVE WEIGHTS FIDELITY AW-1 — "
        "STATE, REWARDS, AND EQ. (48) PASSED"
    )

if __name__ == "__main__":
    main()
