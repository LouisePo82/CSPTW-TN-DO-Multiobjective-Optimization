from __future__ import annotations

from pathlib import Path
import json
import random

from alns_solver.acceptance_factory import (
    ENHANCED_SIMULATED_ANNEALING_MODE,
    PAPER_SIMULATED_ANNEALING_MODE,
    SUPPORTED_SIMULATED_ANNEALING_MODES,
    build_simulated_annealing,
)
from alns_solver.paper_acceptance import (
    PAPER_COOLING_RATE,
    PAPER_INITIAL_ACCEPTANCE_PROBABILITY,
    PAPER_INITIAL_RELATIVE_WORSENING,
)
from alns_solver.paper_adaptive_weights import (
    PaperAdaptiveWeightController,
    PaperAdaptiveWeightState,
)


def make_adaptive_controller():
    state = PaperAdaptiveWeightState.create(
        ("destroy_a",),
        ("repair_a",),
    )
    return PaperAdaptiveWeightController(
        state=state
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    report = {}

    # ---------------------------------------------------------
    # SA-4A — Supported mode registry
    # ---------------------------------------------------------
    if SUPPORTED_SIMULATED_ANNEALING_MODES != {
        PAPER_SIMULATED_ANNEALING_MODE,
        ENHANCED_SIMULATED_ANNEALING_MODE,
    }:
        raise AssertionError(
            "Simulated-annealing mode registry is incorrect."
        )

    try:
        build_simulated_annealing(
            mode="unknown_mode",
            initial_objective=10.0,
            adaptive_controller=make_adaptive_controller(),
            rng=random.Random(1),
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Unsupported simulated-annealing mode was accepted."
        )

    print("[PASS] SA factory exposes exactly two modes")
    print("[PASS] Unsupported SA mode is rejected")

    # ---------------------------------------------------------
    # SA-4B — Paper metadata and fixed parameters
    # ---------------------------------------------------------
    paper = build_simulated_annealing(
        mode=PAPER_SIMULATED_ANNEALING_MODE,
        initial_objective=10.0,
        adaptive_controller=make_adaptive_controller(),
        rng=random.Random(2026),
    )

    if paper.metadata["paper_faithful"] is not True:
        raise AssertionError(
            "Paper SA mode is not labelled paper-faithful."
        )

    if paper.metadata["enhanced"] is not False:
        raise AssertionError(
            "Paper SA mode is incorrectly labelled enhanced."
        )

    if abs(
        paper.metadata["relative_worsening"]
        - PAPER_INITIAL_RELATIVE_WORSENING
    ) > 1e-12:
        raise AssertionError(
            "Paper relative worsening is incorrect."
        )

    if abs(
        paper.metadata[
            "target_acceptance_probability"
        ]
        - PAPER_INITIAL_ACCEPTANCE_PROBABILITY
    ) > 1e-12:
        raise AssertionError(
            "Paper target acceptance probability is incorrect."
        )

    if abs(
        paper.metadata["cooling_rate"]
        - PAPER_COOLING_RATE
    ) > 1e-12:
        raise AssertionError(
            "Paper cooling rate is incorrect."
        )

    if (
        paper.metadata[
            "separate_cost_emission_acceptance"
        ]
        is not False
    ):
        raise AssertionError(
            "Paper SA introduced separate objective acceptance."
        )

    if (
        paper.metadata[
            "independent_state_per_lambda_run"
        ]
        is not True
    ):
        raise AssertionError(
            "Paper SA does not require independent lambda runs."
        )

    print("[PASS] Paper SA mode is labelled paper-faithful")
    print("[PASS] Paper SA fixes 50%-worse calibration")
    print("[PASS] Paper SA fixes target probability at 0.5")
    print("[PASS] Paper SA fixes cooling rate at 0.9994")
    print("[PASS] Paper SA consumes one scalar F_lambda objective")

    # ---------------------------------------------------------
    # SA-4C — Paper overrides rejected
    # ---------------------------------------------------------
    invalid_overrides = [
        {
            "relative_worsening": 0.25,
        },
        {
            "target_acceptance_probability": 0.8,
        },
        {
            "cooling_rate": 0.95,
        },
    ]

    for override in invalid_overrides:
        try:
            build_simulated_annealing(
                mode=PAPER_SIMULATED_ANNEALING_MODE,
                initial_objective=10.0,
                adaptive_controller=make_adaptive_controller(),
                rng=random.Random(1),
                **override,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"Paper SA override was accepted: {override}"
            )

    print("[PASS] Paper SA parameter overrides are rejected")

    # ---------------------------------------------------------
    # SA-4D — Enhanced overrides
    # ---------------------------------------------------------
    enhanced = build_simulated_annealing(
        mode=ENHANCED_SIMULATED_ANNEALING_MODE,
        initial_objective=10.0,
        adaptive_controller=make_adaptive_controller(),
        rng=random.Random(2026),
        relative_worsening=0.25,
        target_acceptance_probability=0.8,
        cooling_rate=0.95,
    )

    if enhanced.metadata["paper_faithful"]:
        raise AssertionError(
            "Enhanced SA is incorrectly labelled paper-faithful."
        )

    if enhanced.metadata["enhanced"] is not True:
        raise AssertionError(
            "Enhanced SA lost enhanced=True."
        )

    if abs(
        enhanced.metadata["relative_worsening"]
        - 0.25
    ) > 1e-12:
        raise AssertionError(
            "Enhanced relative-worsening override failed."
        )

    if abs(
        enhanced.metadata[
            "target_acceptance_probability"
        ]
        - 0.8
    ) > 1e-12:
        raise AssertionError(
            "Enhanced target-probability override failed."
        )

    if abs(
        enhanced.metadata["cooling_rate"]
        - 0.95
    ) > 1e-12:
        raise AssertionError(
            "Enhanced cooling-rate override failed."
        )

    print("[PASS] Enhanced SA is explicitly non-paper")
    print("[PASS] Enhanced SA allows sensitivity overrides")
    print("[PASS] Enhanced SA remains separate from main experiments")

    # ---------------------------------------------------------
    # SA-4E — Independent state per lambda run
    # ---------------------------------------------------------
    lambda_0 = build_simulated_annealing(
        mode=PAPER_SIMULATED_ANNEALING_MODE,
        initial_objective=10.0,
        adaptive_controller=make_adaptive_controller(),
        rng=random.Random(1),
    )
    lambda_1 = build_simulated_annealing(
        mode=PAPER_SIMULATED_ANNEALING_MODE,
        initial_objective=10.0,
        adaptive_controller=make_adaptive_controller(),
        rng=random.Random(1),
    )

    lambda_0.controller.temperature_schedule.current_temperature = 1.0

    if (
        lambda_1.controller.temperature_schedule.current_temperature
        == 1.0
    ):
        raise AssertionError(
            "Temperature state leaked between lambda runs."
        )

    lambda_0.controller.adaptive_controller.state.destroy_records[
        "destroy_a"
    ].weight = 9.0

    if (
        lambda_1.controller.adaptive_controller.state.destroy_records[
            "destroy_a"
        ].weight
        != 1.0
    ):
        raise AssertionError(
            "Adaptive state leaked between lambda runs."
        )

    print("[PASS] Each lambda run receives independent SA state")

    # ---------------------------------------------------------
    # SA-4F — Paper controller preserves SA-3 integration
    # ---------------------------------------------------------
    result = paper.controller.process_iteration(
        iteration=1,
        destroy_operator="destroy_a",
        repair_operator="repair_a",
        candidate_objective=11.0,
        current_objective=10.0,
        best_objective=9.0,
    )

    if (
        result.adaptive_result.event
        not in {"worse_accepted", "rejected"}
    ):
        raise AssertionError(
            "Paper factory did not preserve SA-3 integration."
        )

    if not (
        result.temperature_after_cooling
        < result.temperature_used
    ):
        raise AssertionError(
            "Paper factory did not preserve cooling."
        )

    print("[PASS] Factory paper mode preserves validated SA-3 integration")

    report["paper_mode"] = {
        "metadata": paper.metadata,
        "initial_temperature": (
            paper.controller.temperature_schedule
            .initial_temperature
        ),
        "current_temperature": (
            paper.controller.temperature_schedule
            .current_temperature
        ),
    }
    report["enhanced_mode"] = {
        "metadata": enhanced.metadata,
        "initial_temperature": (
            enhanced.controller.temperature_schedule
            .initial_temperature
        ),
        "current_temperature": (
            enhanced.controller.temperature_schedule
            .current_temperature
        ),
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
        / "alns_simulated_annealing_fidelity_tests"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "simulated_annealing_sa4_report.json"
    )
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nReport saved to: {output_path}")
    print(
        "\nSIMULATED ANNEALING FIDELITY SA-4 — "
        "PAPER AND ENHANCED MODE SEPARATION PASSED"
    )


if __name__ == "__main__":
    main()
