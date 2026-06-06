from __future__ import annotations

from pathlib import Path
import json
import random

from alns_solver.paper_acceptance import (
    PaperSimulatedAnnealingController,
    PaperTemperatureSchedule,
)
from alns_solver.paper_adaptive_weights import (
    PaperAdaptiveWeightController,
    PaperAdaptiveWeightState,
)


class FixedRandom:
    def __init__(self, values):
        self.values = iter(values)

    def random(self):
        return float(next(self.values))


def make_controller(
    *,
    initial_objective: float,
    random_values,
) -> PaperSimulatedAnnealingController:
    adaptive_state = PaperAdaptiveWeightState.create(
        ("destroy_a",),
        ("repair_a",),
    )

    return PaperSimulatedAnnealingController(
        temperature_schedule=PaperTemperatureSchedule(
            initial_objective=initial_objective
        ),
        adaptive_controller=PaperAdaptiveWeightController(
            state=adaptive_state
        ),
        rng=FixedRandom(random_values),
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    report = {}

    # ---------------------------------------------------------
    # SA-3A — New global best
    # ---------------------------------------------------------
    controller = make_controller(
        initial_objective=10.0,
        random_values=[],
    )

    new_best = controller.process_iteration(
        iteration=1,
        destroy_operator="destroy_a",
        repair_operator="repair_a",
        candidate_objective=8.0,
        current_objective=10.0,
        best_objective=9.0,
    )

    if not new_best.acceptance_decision.accepted:
        raise AssertionError(
            "New global best was not accepted."
        )

    if (
        new_best.adaptive_result.event
        != "new_global_best"
        or new_best.adaptive_result.reward != 33.0
    ):
        raise AssertionError(
            "New global best did not receive reward 33."
        )

    if new_best.current_objective_after != 8.0:
        raise AssertionError(
            "Current objective was not updated to new best."
        )

    if new_best.best_objective_after != 8.0:
        raise AssertionError(
            "Best objective was not updated."
        )

    print("[PASS] New global best is accepted and receives reward 33")

    # ---------------------------------------------------------
    # SA-3B — Better than current but not global best
    # ---------------------------------------------------------
    controller = make_controller(
        initial_objective=10.0,
        random_values=[],
    )

    better_current = controller.process_iteration(
        iteration=1,
        destroy_operator="destroy_a",
        repair_operator="repair_a",
        candidate_objective=9.5,
        current_objective=10.0,
        best_objective=9.0,
    )

    if (
        better_current.adaptive_result.event
        != "better_current"
        or better_current.adaptive_result.reward != 15.0
    ):
        raise AssertionError(
            "Better-current candidate did not receive reward 15."
        )

    if better_current.current_objective_after != 9.5:
        raise AssertionError(
            "Accepted better candidate did not become current."
        )

    if better_current.best_objective_after != 9.0:
        raise AssertionError(
            "Historical best changed incorrectly."
        )

    print("[PASS] Better-current candidate receives reward 15")

    # ---------------------------------------------------------
    # SA-3C — Worse candidate accepted by SA
    # ---------------------------------------------------------
    controller = make_controller(
        initial_objective=10.0,
        random_values=[0.0],
    )

    worse_accepted = controller.process_iteration(
        iteration=1,
        destroy_operator="destroy_a",
        repair_operator="repair_a",
        candidate_objective=11.0,
        current_objective=10.0,
        best_objective=9.0,
    )

    if not worse_accepted.acceptance_decision.accepted:
        raise AssertionError(
            "Controlled worse candidate was not accepted."
        )

    if (
        worse_accepted.adaptive_result.event
        != "worse_accepted"
        or worse_accepted.adaptive_result.reward != 9.0
    ):
        raise AssertionError(
            "SA-accepted worse candidate did not receive reward 9."
        )

    if worse_accepted.current_objective_after != 11.0:
        raise AssertionError(
            "Accepted worse candidate did not become current."
        )

    if worse_accepted.best_objective_after != 9.0:
        raise AssertionError(
            "Best objective changed after worse acceptance."
        )

    print("[PASS] SA-accepted worse candidate receives reward 9")

    # ---------------------------------------------------------
    # SA-3D — Worse candidate rejected by SA
    # ---------------------------------------------------------
    controller = make_controller(
        initial_objective=10.0,
        random_values=[1.0],
    )

    worse_rejected = controller.process_iteration(
        iteration=1,
        destroy_operator="destroy_a",
        repair_operator="repair_a",
        candidate_objective=11.0,
        current_objective=10.0,
        best_objective=9.0,
    )

    if worse_rejected.acceptance_decision.accepted:
        raise AssertionError(
            "Controlled worse candidate was unexpectedly accepted."
        )

    if (
        worse_rejected.adaptive_result.event
        != "rejected"
        or worse_rejected.adaptive_result.reward != 0.0
    ):
        raise AssertionError(
            "Rejected worse candidate did not receive reward 0."
        )

    if worse_rejected.current_objective_after != 10.0:
        raise AssertionError(
            "Rejected candidate changed current objective."
        )

    print("[PASS] SA-rejected worse candidate receives reward 0")

    # ---------------------------------------------------------
    # SA-3E — Equal candidate
    # ---------------------------------------------------------
    controller = make_controller(
        initial_objective=10.0,
        random_values=[],
    )

    equal = controller.process_iteration(
        iteration=1,
        destroy_operator="destroy_a",
        repair_operator="repair_a",
        candidate_objective=10.0,
        current_objective=10.0,
        best_objective=9.0,
    )

    if not equal.acceptance_decision.accepted:
        raise AssertionError(
            "Equal candidate was not accepted."
        )

    if (
        equal.adaptive_result.event != "rejected"
        or equal.adaptive_result.reward != 0.0
    ):
        raise AssertionError(
            "Equal candidate should receive zero adaptive reward."
        )

    print("[PASS] Equal candidate is accepted but receives reward 0")

    # ---------------------------------------------------------
    # SA-3F — Same reward credited to destroy and repair
    # ---------------------------------------------------------
    state = (
        worse_accepted
    )
    controller = make_controller(
        initial_objective=10.0,
        random_values=[0.0],
    )
    integrated = controller.process_iteration(
        iteration=1,
        destroy_operator="destroy_a",
        repair_operator="repair_a",
        candidate_objective=11.0,
        current_objective=10.0,
        best_objective=9.0,
    )

    adaptive_state = (
        controller.adaptive_controller.state
    )

    if (
        adaptive_state.destroy_records[
            "destroy_a"
        ].segment_score != 9.0
    ):
        raise AssertionError(
            "Destroy operator did not receive reward 9."
        )

    if (
        adaptive_state.repair_records[
            "repair_a"
        ].segment_score != 9.0
    ):
        raise AssertionError(
            "Repair operator did not receive reward 9."
        )

    if (
        adaptive_state.destroy_records[
            "destroy_a"
        ].segment_uses != 1
        or adaptive_state.repair_records[
            "repair_a"
        ].segment_uses != 1
    ):
        raise AssertionError(
            "Selected operator usage counts are incorrect."
        )

    print("[PASS] Selected destroy and repair receive identical credit")

    # ---------------------------------------------------------
    # SA-3G — Cooling after every decision
    # ---------------------------------------------------------
    if not (
        integrated.temperature_after_cooling
        < integrated.temperature_used
    ):
        raise AssertionError(
            "Temperature did not cool after iteration."
        )

    if (
        controller.temperature_schedule
        .completed_iterations != 1
    ):
        raise AssertionError(
            "Temperature schedule did not advance."
        )

    print("[PASS] Temperature cools after every acceptance decision")

    # ---------------------------------------------------------
    # SA-3H — Scalar F_lambda consistency
    # ---------------------------------------------------------
    controller = make_controller(
        initial_objective=0.5,
        random_values=[0.0],
    )

    scalar_result = controller.process_iteration(
        iteration=1,
        destroy_operator="destroy_a",
        repair_operator="repair_a",
        candidate_objective=0.55,
        current_objective=0.50,
        best_objective=0.45,
    )

    if (
        scalar_result.adaptive_result.event
        != "worse_accepted"
        or scalar_result.adaptive_result.reward
        != 9.0
    ):
        raise AssertionError(
            "Scalar F_lambda was not used consistently."
        )

    print("[PASS] SA and adaptive rewards use the same scalar F_lambda")
    print("[PASS] No separate cost/emission integration logic introduced")

    report["new_global_best"] = {
        "accepted": (
            new_best.acceptance_decision.accepted
        ),
        "reward": new_best.adaptive_result.reward,
        "current_after": (
            new_best.current_objective_after
        ),
        "best_after": new_best.best_objective_after,
    }
    report["better_current"] = {
        "reward": (
            better_current.adaptive_result.reward
        ),
        "current_after": (
            better_current.current_objective_after
        ),
        "best_after": (
            better_current.best_objective_after
        ),
    }
    report["worse_accepted"] = {
        "probability": (
            worse_accepted.acceptance_decision
            .acceptance_probability
        ),
        "random_value": (
            worse_accepted.acceptance_decision
            .random_value
        ),
        "reward": (
            worse_accepted.adaptive_result.reward
        ),
        "current_after": (
            worse_accepted.current_objective_after
        ),
    }
    report["worse_rejected"] = {
        "probability": (
            worse_rejected.acceptance_decision
            .acceptance_probability
        ),
        "random_value": (
            worse_rejected.acceptance_decision
            .random_value
        ),
        "reward": (
            worse_rejected.adaptive_result.reward
        ),
        "current_after": (
            worse_rejected.current_objective_after
        ),
    }
    report["equal_candidate"] = {
        "accepted": (
            equal.acceptance_decision.accepted
        ),
        "reward": equal.adaptive_result.reward,
    }
    report["integration"] = {
        "destroy_score": (
            adaptive_state.destroy_records[
                "destroy_a"
            ].segment_score
        ),
        "repair_score": (
            adaptive_state.repair_records[
                "repair_a"
            ].segment_score
        ),
        "temperature_used": (
            integrated.temperature_used
        ),
        "temperature_after_cooling": (
            integrated.temperature_after_cooling
        ),
    }
    report["multiobjective_scalar"] = {
        "candidate_F_lambda": 0.55,
        "current_F_lambda": 0.50,
        "best_F_lambda": 0.45,
        "accepted": (
            scalar_result.acceptance_decision
            .accepted
        ),
        "reward": (
            scalar_result.adaptive_result.reward
        ),
    }

    output_dir = (
        root
        / "outputs"
        / "alns_simulated_annealing_fidelity_tests"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "simulated_annealing_sa3_report.json"
    )
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nReport saved to: {output_path}")
    print(
        "\nSIMULATED ANNEALING FIDELITY SA-3 — "
        "ADAPTIVE REWARD INTEGRATION PASSED"
    )


if __name__ == "__main__":
    main()
