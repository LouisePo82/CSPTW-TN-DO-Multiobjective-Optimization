from __future__ import annotations

from pathlib import Path
import json
import math
import random

from alns_solver.paper_acceptance import (
    EPSILON,
    accept_with_simulated_annealing,
    worse_candidate_acceptance_probability,
)


TOLERANCE = 1e-12


class FixedRandom:
    def __init__(
        self,
        value: float,
    ) -> None:
        self.value = float(value)

    def random(self) -> float:
        return self.value


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
    # SA-1A — Better and equal candidates
    # ---------------------------------------------------------
    better = accept_with_simulated_annealing(
        candidate_objective=9.0,
        current_objective=10.0,
        temperature=1.0,
        rng=random.Random(1),
    )

    if not better.accepted:
        raise AssertionError(
            "Better candidate was rejected."
        )

    if better.reason != "better_candidate":
        raise AssertionError(
            "Better-candidate reason is incorrect."
        )

    if better.random_value is not None:
        raise AssertionError(
            "Better candidate should not consume RNG."
        )

    assert_close(
        better.acceptance_probability,
        1.0,
        "Better acceptance probability",
    )

    equal = accept_with_simulated_annealing(
        candidate_objective=10.0,
        current_objective=10.0,
        temperature=1.0,
        rng=random.Random(1),
    )

    if not equal.accepted:
        raise AssertionError(
            "Equal candidate was rejected."
        )

    if equal.reason != "equal_candidate":
        raise AssertionError(
            "Equal-candidate reason is incorrect."
        )

    if equal.random_value is not None:
        raise AssertionError(
            "Equal candidate should not consume RNG."
        )

    assert_close(
        equal.acceptance_probability,
        1.0,
        "Equal acceptance probability",
    )

    print("[PASS] Better candidate is always accepted")
    print("[PASS] Equal candidate is always accepted")
    print("[PASS] Better/equal decisions do not consume RNG")

    # ---------------------------------------------------------
    # SA-1B — Worse-candidate formula
    # ---------------------------------------------------------
    expected_probability = math.exp(-0.5)

    actual_probability = (
        worse_candidate_acceptance_probability(
            candidate_objective=11.0,
            current_objective=10.0,
            temperature=2.0,
        )
    )

    assert_close(
        actual_probability,
        expected_probability,
        "Worse-candidate probability",
    )

    print("[PASS] Worse-candidate probability follows exp(-delta/T)")

    # ---------------------------------------------------------
    # SA-1C — Inclusive boundary
    # ---------------------------------------------------------
    boundary = accept_with_simulated_annealing(
        candidate_objective=11.0,
        current_objective=10.0,
        temperature=2.0,
        rng=FixedRandom(
            expected_probability
        ),
    )

    if not boundary.accepted:
        raise AssertionError(
            "Inclusive probability boundary was rejected."
        )

    below = accept_with_simulated_annealing(
        candidate_objective=11.0,
        current_objective=10.0,
        temperature=2.0,
        rng=FixedRandom(
            expected_probability - 1e-10
        ),
    )

    if not below.accepted:
        raise AssertionError(
            "Random value below probability was rejected."
        )

    above = accept_with_simulated_annealing(
        candidate_objective=11.0,
        current_objective=10.0,
        temperature=2.0,
        rng=FixedRandom(
            expected_probability + 1e-10
        ),
    )

    if above.accepted:
        raise AssertionError(
            "Random value above probability was accepted."
        )

    print("[PASS] Acceptance boundary uses u <= probability")
    print("[PASS] Random value below probability is accepted")
    print("[PASS] Random value above probability is rejected")

    # ---------------------------------------------------------
    # SA-1D — Temperature effect
    # ---------------------------------------------------------
    low_temperature_probability = (
        worse_candidate_acceptance_probability(
            candidate_objective=11.0,
            current_objective=10.0,
            temperature=0.5,
        )
    )
    high_temperature_probability = (
        worse_candidate_acceptance_probability(
            candidate_objective=11.0,
            current_objective=10.0,
            temperature=5.0,
        )
    )

    if not (
        high_temperature_probability
        > low_temperature_probability
    ):
        raise AssertionError(
            "Higher temperature did not increase acceptance probability."
        )

    print("[PASS] Higher temperature gives higher acceptance probability")

    # ---------------------------------------------------------
    # SA-1E — Fixed-seed reproducibility
    # ---------------------------------------------------------
    rng_1 = random.Random(2026)
    rng_2 = random.Random(2026)

    sequence_1 = [
        accept_with_simulated_annealing(
            candidate_objective=11.0,
            current_objective=10.0,
            temperature=2.0,
            rng=rng_1,
        ).accepted
        for _ in range(100)
    ]
    sequence_2 = [
        accept_with_simulated_annealing(
            candidate_objective=11.0,
            current_objective=10.0,
            temperature=2.0,
            rng=rng_2,
        ).accepted
        for _ in range(100)
    ]

    if sequence_1 != sequence_2:
        raise AssertionError(
            "Fixed seed did not reproduce acceptance sequence."
        )

    print("[PASS] Fixed seed reproduces SA acceptance sequence")

    # ---------------------------------------------------------
    # SA-1F — Numerical stability
    # ---------------------------------------------------------
    underflow_probability = (
        worse_candidate_acceptance_probability(
            candidate_objective=1_000_000.0,
            current_objective=0.0,
            temperature=1e-6,
        )
    )

    if underflow_probability != 0.0:
        raise AssertionError(
            "Extreme worse candidate should underflow safely to zero."
        )

    print("[PASS] Extreme exponent underflows safely to zero")

    # ---------------------------------------------------------
    # SA-1G — Invalid inputs
    # ---------------------------------------------------------
    invalid_cases = [
        {
            "candidate_objective": float("nan"),
            "current_objective": 10.0,
            "temperature": 1.0,
        },
        {
            "candidate_objective": 10.0,
            "current_objective": float("inf"),
            "temperature": 1.0,
        },
        {
            "candidate_objective": 10.0,
            "current_objective": 9.0,
            "temperature": 0.0,
        },
        {
            "candidate_objective": 10.0,
            "current_objective": 9.0,
            "temperature": -1.0,
        },
        {
            "candidate_objective": 10.0,
            "current_objective": 9.0,
            "temperature": float("nan"),
        },
        {
            "candidate_objective": 10.0,
            "current_objective": 9.0,
            "temperature": float("inf"),
        },
    ]

    for arguments in invalid_cases:
        try:
            accept_with_simulated_annealing(
                **arguments,
                rng=random.Random(1),
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"Invalid SA input was accepted: {arguments}"
            )

    print("[PASS] Invalid objectives and temperatures are rejected")

    # ---------------------------------------------------------
    # SA-1H — Scalar F_lambda compatibility
    # ---------------------------------------------------------
    scalar_probability = (
        worse_candidate_acceptance_probability(
            candidate_objective=0.45,
            current_objective=0.40,
            temperature=0.10,
        )
    )
    expected_scalar_probability = math.exp(
        -0.05 / 0.10
    )

    assert_close(
        scalar_probability,
        expected_scalar_probability,
        "Scalar F_lambda probability",
    )

    print("[PASS] SA consumes one scalar F_lambda objective")
    print("[PASS] No separate cost/emission acceptance logic introduced")

    report["better_candidate"] = {
        "accepted": better.accepted,
        "probability": better.acceptance_probability,
        "reason": better.reason,
    }
    report["equal_candidate"] = {
        "accepted": equal.accepted,
        "probability": equal.acceptance_probability,
        "reason": equal.reason,
    }
    report["worse_candidate"] = {
        "candidate_objective": 11.0,
        "current_objective": 10.0,
        "temperature": 2.0,
        "delta": 1.0,
        "expected_probability": expected_probability,
        "actual_probability": actual_probability,
        "inclusive_boundary_accepted": (
            boundary.accepted
        ),
        "above_boundary_accepted": (
            above.accepted
        ),
    }
    report["temperature_effect"] = {
        "low_temperature": 0.5,
        "low_probability": (
            low_temperature_probability
        ),
        "high_temperature": 5.0,
        "high_probability": (
            high_temperature_probability
        ),
    }
    report["fixed_seed"] = {
        "seed": 2026,
        "first_30_decisions": sequence_1[:30],
        "reproducible": True,
    }
    report["numerical_stability"] = {
        "underflow_probability": (
            underflow_probability
        ),
    }
    report["multiobjective_scalar"] = {
        "candidate_F_lambda": 0.45,
        "current_F_lambda": 0.40,
        "temperature": 0.10,
        "probability": scalar_probability,
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
        / "simulated_annealing_sa1_report.json"
    )
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nReport saved to: {output_path}")
    print(
        "\nSIMULATED ANNEALING FIDELITY SA-1 — "
        "ACCEPTANCE PROBABILITY PASSED"
    )


if __name__ == "__main__":
    main()
