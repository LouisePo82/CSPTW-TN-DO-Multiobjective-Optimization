from __future__ import annotations

from pathlib import Path
import json

from alns_solver.adaptive_weights_factory import (
    ENHANCED_ADAPTIVE_WEIGHTS_MODE,
    PAPER_ADAPTIVE_WEIGHTS_MODE,
    SUPPORTED_ADAPTIVE_WEIGHT_MODES,
    build_adaptive_weights,
)
from alns_solver.paper_adaptive_weights import (
    PAPER_REACTION_FACTOR,
    PAPER_SEGMENT_LENGTH,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]

    destroy_names = (
        "destroy_a",
        "destroy_b",
    )
    repair_names = (
        "repair_a",
        "repair_b",
    )

    report = {}

    # ---------------------------------------------------------
    # AW-4A — Supported mode registry
    # ---------------------------------------------------------
    if SUPPORTED_ADAPTIVE_WEIGHT_MODES != {
        PAPER_ADAPTIVE_WEIGHTS_MODE,
        ENHANCED_ADAPTIVE_WEIGHTS_MODE,
    }:
        raise AssertionError(
            "Adaptive-weight mode registry is incorrect."
        )

    try:
        build_adaptive_weights(
            mode="unknown_mode",
            destroy_operator_names=destroy_names,
            repair_operator_names=repair_names,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Unsupported adaptive-weight mode was accepted."
        )

    print("[PASS] Adaptive-weight factory exposes exactly two modes")
    print("[PASS] Unsupported adaptive-weight mode is rejected")

    # ---------------------------------------------------------
    # AW-4B — Paper mode metadata and fixed parameters
    # ---------------------------------------------------------
    paper = build_adaptive_weights(
        mode=PAPER_ADAPTIVE_WEIGHTS_MODE,
        destroy_operator_names=destroy_names,
        repair_operator_names=repair_names,
    )

    if paper.metadata["paper_faithful"] is not True:
        raise AssertionError(
            "Paper mode is not labelled paper-faithful."
        )

    if paper.metadata["enhanced"] is not False:
        raise AssertionError(
            "Paper mode is incorrectly labelled enhanced."
        )

    if (
        paper.state.segment_length
        != PAPER_SEGMENT_LENGTH
    ):
        raise AssertionError(
            "Paper segment length is not 300."
        )

    if abs(
        paper.state.reaction_factor
        - PAPER_REACTION_FACTOR
    ) > 1e-12:
        raise AssertionError(
            "Paper reaction factor is not 0.1."
        )

    if paper.metadata["rewards"] != {
        "new_global_best": 33.0,
        "better_current": 15.0,
        "worse_accepted": 9.0,
        "rejected": 0.0,
    }:
        raise AssertionError(
            "Paper reward metadata is incorrect."
        )

    if (
        paper.metadata[
            "separate_cost_emission_rewards"
        ]
        is not False
    ):
        raise AssertionError(
            "Paper mode introduced separate objective rewards."
        )

    if (
        paper.metadata[
            "independent_state_per_lambda_run"
        ]
        is not True
    ):
        raise AssertionError(
            "Paper mode does not require independent lambda runs."
        )

    print("[PASS] Paper mode is labelled paper-faithful")
    print("[PASS] Paper mode fixes segment length at 300")
    print("[PASS] Paper mode fixes reaction factor at 0.1")
    print("[PASS] Paper mode fixes rewards at 33, 15, 9, and 0")
    print("[PASS] Paper mode consumes one scalar F_lambda objective")

    # ---------------------------------------------------------
    # AW-4C — Paper overrides rejected
    # ---------------------------------------------------------
    invalid_paper_overrides = [
        {
            "segment_length": 100,
        },
        {
            "reaction_factor": 0.3,
        },
        {
            "rewards": {
                "new_global_best": 10.0,
                "better_current": 5.0,
                "worse_accepted": 1.0,
                "rejected": 0.0,
            },
        },
    ]

    for override in invalid_paper_overrides:
        try:
            build_adaptive_weights(
                mode=PAPER_ADAPTIVE_WEIGHTS_MODE,
                destroy_operator_names=destroy_names,
                repair_operator_names=repair_names,
                **override,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"Paper override was accepted: {override}"
            )

    print("[PASS] Paper parameter overrides are rejected")

    # ---------------------------------------------------------
    # AW-4D — Enhanced mode permits explicit overrides
    # ---------------------------------------------------------
    enhanced_rewards = {
        "new_global_best": 20.0,
        "better_current": 10.0,
        "worse_accepted": 5.0,
        "rejected": 0.0,
    }

    enhanced = build_adaptive_weights(
        mode=ENHANCED_ADAPTIVE_WEIGHTS_MODE,
        destroy_operator_names=destroy_names,
        repair_operator_names=repair_names,
        initial_weight=2.0,
        segment_length=100,
        reaction_factor=0.3,
        rewards=enhanced_rewards,
    )

    if enhanced.metadata["paper_faithful"]:
        raise AssertionError(
            "Enhanced mode is incorrectly labelled paper-faithful."
        )

    if enhanced.metadata["enhanced"] is not True:
        raise AssertionError(
            "Enhanced mode lost enhanced=True."
        )

    if enhanced.state.segment_length != 100:
        raise AssertionError(
            "Enhanced segment-length override was not applied."
        )

    if abs(
        enhanced.state.reaction_factor - 0.3
    ) > 1e-12:
        raise AssertionError(
            "Enhanced reaction-factor override was not applied."
        )

    for record in (
        list(enhanced.state.destroy_records.values())
        + list(enhanced.state.repair_records.values())
    ):
        if record.weight != 2.0:
            raise AssertionError(
                "Enhanced initial-weight override was not applied."
            )

    if enhanced.metadata["rewards"] != enhanced_rewards:
        raise AssertionError(
            "Enhanced reward override metadata is incorrect."
        )

    print("[PASS] Enhanced mode is explicitly non-paper")
    print("[PASS] Enhanced mode allows sensitivity overrides")
    print("[PASS] Enhanced mode remains separate from main experiments")

    # ---------------------------------------------------------
    # AW-4E — Independent state per lambda run
    # ---------------------------------------------------------
    lambda_0 = build_adaptive_weights(
        mode=PAPER_ADAPTIVE_WEIGHTS_MODE,
        destroy_operator_names=destroy_names,
        repair_operator_names=repair_names,
    )
    lambda_1 = build_adaptive_weights(
        mode=PAPER_ADAPTIVE_WEIGHTS_MODE,
        destroy_operator_names=destroy_names,
        repair_operator_names=repair_names,
    )

    lambda_0.state.destroy_records[
        "destroy_a"
    ].weight = 9.0

    if (
        lambda_1.state.destroy_records[
            "destroy_a"
        ].weight
        != 1.0
    ):
        raise AssertionError(
            "Adaptive state leaked between lambda runs."
        )

    print("[PASS] Each lambda run receives independent adaptive state")

    # ---------------------------------------------------------
    # AW-4F — Paper controller remains operational
    # ---------------------------------------------------------
    result = paper.controller.process_iteration(
        iteration=1,
        destroy_operator="destroy_a",
        repair_operator="repair_a",
        candidate_objective=0.40,
        current_objective=0.50,
        best_objective=0.45,
        accepted=True,
    )

    if (
        result.event != "new_global_best"
        or result.reward != 33.0
    ):
        raise AssertionError(
            "Paper factory controller did not preserve AW-3 behavior."
        )

    print("[PASS] Factory paper mode preserves validated AW-3 controller")

    report["paper_mode"] = {
        "metadata": paper.metadata,
        "snapshot": paper.state.snapshot(),
    }
    report["enhanced_mode"] = {
        "metadata": enhanced.metadata,
        "snapshot": enhanced.state.snapshot(),
    }
    report["separation"] = {
        "paper_faithful": (
            paper.metadata["paper_faithful"]
        ),
        "enhanced_paper_faithful": (
            enhanced.metadata["paper_faithful"]
        ),
        "paper_overrides_rejected": True,
        "independent_lambda_state": True,
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
        / "adaptive_weights_aw4_report.json"
    )
    output_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nReport saved to: {output_path}")
    print(
        "\nADAPTIVE WEIGHTS FIDELITY AW-4 — "
        "PAPER AND ENHANCED MODE SEPARATION PASSED"
    )


if __name__ == "__main__":
    main()
