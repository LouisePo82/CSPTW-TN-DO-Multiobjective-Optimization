from pathlib import Path

from core.instance_loader import load_instance
from alns_solver.paper_alns_main import (
    build_ml1_paper_initial_state,
)
from alns_solver.paper_production_runner import (
    PaperALNSRunConfig,
    run_paper_alns_production,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]

    instance = load_instance(
        root / "data" / "small" / "instance_001"
    )

    config = PaperALNSRunConfig(
        "instance_001",
        0.5,
        2026,
        301,
        (0.0, 100.0),
        (0.0, 300.0),
        (3.0, 1.0),
    )

    initial_state = build_ml1_paper_initial_state(
        instance,
        seed=config.run_seed,
        lambda_value=config.lambda_value,
        cost_bounds=config.cost_bounds,
        emission_bounds=config.emission_bounds,
        emission_factors=config.emission_factors,
    )

    result = run_paper_alns_production(
        instance=instance,
        initial_state=initial_state,
        config=config,
    )

    history = result.iteration_history

    # Sequential production run.
    assert len(history) == 301
    assert [
        row["iteration"]
        for row in history
    ] == list(range(1, 302))

    print(
        "[PASS] Sequential current state runs "
        "for 301 consecutive iterations"
    )

    # Validation throughout the run.
    assert result.best_solution.validator_pass
    assert result.final_current_solution.validator_pass
    assert all(
        row["validation_pass"]
        for row in history
    )

    print(
        "[PASS] Every production iteration "
        "retains a valid current state"
    )

    # Best objective must never become worse.
    best_values = [
        float(row["best_objective_after"])
        for row in history
    ]

    assert all(
        later <= earlier + 1e-12
        for earlier, later in zip(
            best_values,
            best_values[1:],
        )
    )

    print(
        "[PASS] Best objective is monotonically "
        "non-increasing"
    )

    # SA cooling must happen exactly once per iteration.
    temperatures_used = [
        float(row["temperature_used"])
        for row in history
    ]
    temperatures_after = [
        float(row["temperature_after_cooling"])
        for row in history
    ]

    assert all(
        cooled < used
        for used, cooled in zip(
            temperatures_used,
            temperatures_after,
        )
    )

    assert all(
        abs(
            next_used - previous_cooled
        ) <= 1e-12
        for previous_cooled, next_used in zip(
            temperatures_after,
            temperatures_used[1:],
        )
    )

    print(
        "[PASS] Paper SA cools exactly once "
        "per iteration"
    )

    # Explicit boundary records.
    row_300 = history[299]
    row_301 = history[300]

    assert row_300["iteration"] == 300
    assert row_301["iteration"] == 301

    print(
        "[PASS] Iteration 300 adaptive boundary "
        "is reached"
    )
    print(
        "[PASS] Iteration 301 continues with "
        "persistent production state"
    )

    # After the update at iteration 300, segment counters reset.
    # Iteration 301 contributes exactly one use in each pool.
    destroy_segment_uses = sum(
        int(row["segment_uses"])
        for row in result.operator_statistics
        if row["pool"] == "destroy"
    )
    repair_segment_uses = sum(
        int(row["segment_uses"])
        for row in result.operator_statistics
        if row["pool"] == "repair"
    )

    assert destroy_segment_uses == 1
    assert repair_segment_uses == 1

    print(
        "[PASS] Segment statistics reset at 300 "
        "and iteration 301 starts segment 2"
    )

    # Production metadata must preserve scope lock.
    assert result.metadata["paper_faithful"] is True
    assert result.metadata["enhanced"] is False
    assert result.metadata["fallback"] is False
    assert result.metadata["operator_substitution"] is False
    assert result.metadata["operator_resampling"] is False
    assert result.metadata["objective_input"] == "scalar_F_lambda"

    print(
        "[PASS] Production run remains paper-faithful "
        "with scalar F_lambda only"
    )
    print(
        "[PASS] No enhanced ALNS, fallback, "
        "substitution, or resampling is introduced"
    )

    print(
        "\nML-6.3 — PAPER PRODUCTION SEGMENT "
        "BOUNDARY PASSED"
    )


if __name__ == "__main__":
    main()
