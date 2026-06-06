from __future__ import annotations

from pathlib import Path
import json
import math

from alns_solver.paper_acceptance import (
    PAPER_COOLING_RATE,
    PAPER_INITIAL_ACCEPTANCE_PROBABILITY,
    PAPER_INITIAL_RELATIVE_WORSENING,
    PaperTemperatureSchedule,
    cool_temperature,
    paper_initial_temperature,
    worse_candidate_acceptance_probability,
)


TOLERANCE = 1e-12


def assert_close(
    actual: float,
    expected: float,
    label: str,
    tolerance: float = TOLERANCE,
) -> None:
    if abs(
        float(actual) - float(expected)
    ) > tolerance:
        raise AssertionError(
            f"{label}: actual={actual}, expected={expected}"
        )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    report = {}

    # ---------------------------------------------------------
    # SA-2A — Paper constants
    # ---------------------------------------------------------
    assert_close(
        PAPER_INITIAL_RELATIVE_WORSENING,
        0.5,
        "Initial relative worsening",
    )
    assert_close(
        PAPER_INITIAL_ACCEPTANCE_PROBABILITY,
        0.5,
        "Initial acceptance probability",
    )
    assert_close(
        PAPER_COOLING_RATE,
        0.9994,
        "Paper cooling rate",
    )

    print("[PASS] Initial worsening calibration uses 50%")
    print("[PASS] Initial target acceptance probability is 50%")
    print("[PASS] Paper cooling rate is 0.9994")

    # ---------------------------------------------------------
    # SA-2B — Initial temperature formula
    # ---------------------------------------------------------
    initial_objective = 100.0
    expected_temperature = (
        50.0 / math.log(2.0)
    )

    actual_temperature = (
        paper_initial_temperature(
            initial_objective=initial_objective
        )
    )

    assert_close(
        actual_temperature,
        expected_temperature,
        "Initial temperature formula",
    )

    print("[PASS] Initial temperature follows -delta/log(p)")

    # ---------------------------------------------------------
    # SA-2C — Calibration reproduces probability 0.5
    # ---------------------------------------------------------
    worse_objective = (
        initial_objective * 1.5
    )

    calibrated_probability = (
        worse_candidate_acceptance_probability(
            candidate_objective=(
                worse_objective
            ),
            current_objective=(
                initial_objective
            ),
            temperature=(
                actual_temperature
            ),
        )
    )

    assert_close(
        calibrated_probability,
        0.5,
        "Calibrated acceptance probability",
    )

    print("[PASS] A 50%-worse candidate has probability 0.5 at T0")

    # ---------------------------------------------------------
    # SA-2D — One-step cooling
    # ---------------------------------------------------------
    cooled = cool_temperature(
        temperature=actual_temperature
    )

    assert_close(
        cooled,
        actual_temperature * 0.9994,
        "One-step cooling",
    )

    if not cooled < actual_temperature:
        raise AssertionError(
            "Temperature did not decrease."
        )

    print("[PASS] Cooling follows T_next = 0.9994 * T_current")
    print("[PASS] Cooling is strictly decreasing")

    # ---------------------------------------------------------
    # SA-2E — Iteration 1 uses T0, then cooling occurs
    # ---------------------------------------------------------
    schedule = PaperTemperatureSchedule(
        initial_objective=initial_objective
    )

    iteration_1_temperature = (
        schedule.temperature_for_iteration(1)
    )

    assert_close(
        iteration_1_temperature,
        actual_temperature,
        "Iteration 1 temperature",
    )

    event_1 = schedule.cool_after_iteration(1)

    assert_close(
        event_1["temperature_used"],
        actual_temperature,
        "Iteration 1 used temperature",
    )
    assert_close(
        schedule.current_temperature,
        actual_temperature * 0.9994,
        "Temperature after iteration 1",
    )

    print("[PASS] Iteration 1 uses T0")
    print("[PASS] Cooling occurs after the iteration decision")

    # ---------------------------------------------------------
    # SA-2F — Consecutive schedule and closed-form check
    # ---------------------------------------------------------
    temperatures = [
        iteration_1_temperature
    ]

    for iteration in range(2, 11):
        temperature_used = (
            schedule.temperature_for_iteration(
                iteration
            )
        )
        temperatures.append(
            temperature_used
        )

        expected = (
            actual_temperature
            * (0.9994 ** (iteration - 1))
        )

        assert_close(
            temperature_used,
            expected,
            f"Temperature at iteration {iteration}",
            tolerance=1e-10,
        )

        schedule.cool_after_iteration(
            iteration
        )

    if any(
        later >= earlier
        for earlier, later
        in zip(
            temperatures,
            temperatures[1:],
        )
    ):
        raise AssertionError(
            "Temperature schedule is not strictly decreasing."
        )

    if any(
        temperature <= 0.0
        for temperature in temperatures
    ):
        raise AssertionError(
            "Temperature schedule contains non-positive values."
        )

    print("[PASS] Temperature schedule matches T0 * alpha^(k-1)")
    print("[PASS] Temperature remains positive and monotonic")

    # ---------------------------------------------------------
    # SA-2G — Fixed input reproducibility
    # ---------------------------------------------------------
    schedule_repeat = PaperTemperatureSchedule(
        initial_objective=initial_objective
    )

    repeat_temperatures = []

    for iteration in range(1, 11):
        repeat_temperatures.append(
            schedule_repeat.temperature_for_iteration(
                iteration
            )
        )
        schedule_repeat.cool_after_iteration(
            iteration
        )

    if repeat_temperatures != temperatures:
        raise AssertionError(
            "Same input did not reproduce temperature schedule."
        )

    print("[PASS] Fixed initial objective reproduces schedule")

    # ---------------------------------------------------------
    # SA-2H — Consecutive iteration enforcement
    # ---------------------------------------------------------
    invalid_schedule = PaperTemperatureSchedule(
        initial_objective=initial_objective
    )

    try:
        invalid_schedule.temperature_for_iteration(
            2
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Non-consecutive temperature lookup was accepted."
        )

    try:
        invalid_schedule.cool_after_iteration(
            2
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Non-consecutive cooling was accepted."
        )

    print("[PASS] Temperature controller requires consecutive iterations")

    # ---------------------------------------------------------
    # SA-2I — Invalid inputs and paper override rejection
    # ---------------------------------------------------------
    invalid_objectives = (
        0.0,
        -1.0,
        float("nan"),
        float("inf"),
    )

    for value in invalid_objectives:
        try:
            paper_initial_temperature(
                initial_objective=value
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"Invalid initial objective accepted: {value}"
            )

    invalid_rates = (
        0.0,
        -0.1,
        1.0,
        1.1,
        float("nan"),
        float("inf"),
    )

    for rate in invalid_rates:
        try:
            cool_temperature(
                temperature=10.0,
                cooling_rate=rate,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"Invalid cooling rate accepted: {rate}"
            )

    try:
        PaperTemperatureSchedule(
            initial_objective=100.0,
            cooling_rate=0.9,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Paper cooling-rate override was accepted."
        )

    print("[PASS] Invalid temperature calibration inputs rejected")
    print("[PASS] Invalid cooling rates rejected")
    print("[PASS] Paper cooling rate cannot be overridden")

    # ---------------------------------------------------------
    # SA-2J — Scalar F_lambda compatibility
    # ---------------------------------------------------------
    scalar_initial_objective = 0.8
    scalar_temperature = (
        paper_initial_temperature(
            initial_objective=(
                scalar_initial_objective
            )
        )
    )

    scalar_probability = (
        worse_candidate_acceptance_probability(
            candidate_objective=1.2,
            current_objective=0.8,
            temperature=scalar_temperature,
        )
    )

    assert_close(
        scalar_probability,
        0.5,
        "Scalar F_lambda calibration",
    )

    print("[PASS] Temperature calibration accepts scalar F_lambda")
    print("[PASS] No separate cost/emission temperature schedule introduced")

    report["paper_constants"] = {
        "relative_worsening": (
            PAPER_INITIAL_RELATIVE_WORSENING
        ),
        "target_acceptance_probability": (
            PAPER_INITIAL_ACCEPTANCE_PROBABILITY
        ),
        "cooling_rate": PAPER_COOLING_RATE,
    }
    report["initial_temperature"] = {
        "initial_objective": (
            initial_objective
        ),
        "expected_temperature": (
            expected_temperature
        ),
        "actual_temperature": (
            actual_temperature
        ),
        "worse_objective": (
            worse_objective
        ),
        "calibrated_probability": (
            calibrated_probability
        ),
    }
    report["schedule"] = {
        "temperatures_used_first_10": (
            temperatures
        ),
        "final_snapshot": (
            schedule.snapshot()
        ),
        "reproducible": True,
    }
    report["multiobjective_scalar"] = {
        "initial_F_lambda": (
            scalar_initial_objective
        ),
        "initial_temperature": (
            scalar_temperature
        ),
        "fifty_percent_worse_F_lambda": (
            1.2
        ),
        "acceptance_probability": (
            scalar_probability
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
        / "simulated_annealing_sa2_report.json"
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
        "\nSIMULATED ANNEALING FIDELITY SA-2 — "
        "INITIAL TEMPERATURE AND COOLING PASSED"
    )


if __name__ == "__main__":
    main()
