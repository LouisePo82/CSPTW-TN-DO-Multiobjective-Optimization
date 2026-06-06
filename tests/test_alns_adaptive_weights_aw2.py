from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import math
import random

from alns_solver.paper_adaptive_weights import (
    OperatorWeightRecord,
    PaperAdaptiveWeightState,
    operator_probabilities,
    select_destroy_operator,
    select_repair_operator,
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


def main() -> None:
    root = Path(__file__).resolve().parents[1]

    state = PaperAdaptiveWeightState.create(
        (
            "random_customer_removal",
            "related_removal",
            "route_removal",
        ),
        (
            "best_insertion",
            "regret_2",
        ),
        initial_weight=1.0,
    )

    report = {}

    # ---------------------------------------------------------
    # AW-2A — Equal initial probabilities
    # ---------------------------------------------------------
    destroy_initial = operator_probabilities(
        state.destroy_records
    )
    repair_initial = operator_probabilities(
        state.repair_records
    )

    for probability in destroy_initial.values():
        assert_close(
            probability,
            1.0 / 3.0,
            "Equal destroy probability",
        )

    for probability in repair_initial.values():
        assert_close(
            probability,
            1.0 / 2.0,
            "Equal repair probability",
        )

    assert_close(
        sum(destroy_initial.values()),
        1.0,
        "Destroy probability sum",
    )
    assert_close(
        sum(repair_initial.values()),
        1.0,
        "Repair probability sum",
    )

    print("[PASS] Equal destroy weights give equal probabilities")
    print("[PASS] Equal repair weights give equal probabilities")
    print("[PASS] Destroy and repair pools normalize separately")

    # ---------------------------------------------------------
    # AW-2B — Controlled post-update weights
    # ---------------------------------------------------------
    state.destroy_records[
        "random_customer_removal"
    ].weight = 3.3
    state.destroy_records[
        "related_removal"
    ].weight = 1.8
    state.destroy_records[
        "route_removal"
    ].weight = 1.0

    state.repair_records[
        "best_insertion"
    ].weight = 3.3
    state.repair_records[
        "regret_2"
    ].weight = 1.8

    destroy_probabilities = operator_probabilities(
        state.destroy_records
    )
    repair_probabilities = operator_probabilities(
        state.repair_records
    )

    expected_destroy = {
        "random_customer_removal": 3.3 / 6.1,
        "related_removal": 1.8 / 6.1,
        "route_removal": 1.0 / 6.1,
    }
    expected_repair = {
        "best_insertion": 3.3 / 5.1,
        "regret_2": 1.8 / 5.1,
    }

    for name, expected in (
        expected_destroy.items()
    ):
        assert_close(
            destroy_probabilities[name],
            expected,
            f"Destroy probability {name}",
        )

    for name, expected in (
        expected_repair.items()
    ):
        assert_close(
            repair_probabilities[name],
            expected,
            f"Repair probability {name}",
        )

    if not (
        destroy_probabilities[
            "random_customer_removal"
        ]
        > destroy_probabilities[
            "related_removal"
        ]
        > destroy_probabilities[
            "route_removal"
        ]
    ):
        raise AssertionError(
            "Destroy probabilities do not follow weight ranking."
        )

    if not (
        repair_probabilities[
            "best_insertion"
        ]
        > repair_probabilities[
            "regret_2"
        ]
    ):
        raise AssertionError(
            "Repair probabilities do not follow weight ranking."
        )

    print("[PASS] Probability equals weight divided by pool total")
    print("[PASS] Higher weight gives higher selection probability")

    # ---------------------------------------------------------
    # AW-2C — Fixed-seed reproducibility and pool membership
    # ---------------------------------------------------------
    destroy_rng_1 = random.Random(42)
    destroy_rng_2 = random.Random(42)
    repair_rng_1 = random.Random(42)
    repair_rng_2 = random.Random(42)

    destroy_sequence_1 = [
        select_destroy_operator(
            state,
            rng=destroy_rng_1,
        )
        for _ in range(100)
    ]
    destroy_sequence_2 = [
        select_destroy_operator(
            state,
            rng=destroy_rng_2,
        )
        for _ in range(100)
    ]

    repair_sequence_1 = [
        select_repair_operator(
            state,
            rng=repair_rng_1,
        )
        for _ in range(100)
    ]
    repair_sequence_2 = [
        select_repair_operator(
            state,
            rng=repair_rng_2,
        )
        for _ in range(100)
    ]

    if (
        destroy_sequence_1
        != destroy_sequence_2
    ):
        raise AssertionError(
            "Destroy selection is not reproducible."
        )

    if (
        repair_sequence_1
        != repair_sequence_2
    ):
        raise AssertionError(
            "Repair selection is not reproducible."
        )

    if not set(
        destroy_sequence_1
    ).issubset(
        state.destroy_records
    ):
        raise AssertionError(
            "Destroy selection returned operator outside destroy pool."
        )

    if not set(
        repair_sequence_1
    ).issubset(
        state.repair_records
    ):
        raise AssertionError(
            "Repair selection returned operator outside repair pool."
        )

    print("[PASS] Fixed seed reproduces destroy sequence")
    print("[PASS] Fixed seed reproduces repair sequence")
    print("[PASS] Roulette selection stays inside the requested pool")

    # ---------------------------------------------------------
    # AW-2D — Empirical frequency follows theoretical ranking
    # ---------------------------------------------------------
    sample_size = 50_000

    destroy_counts = Counter(
        select_destroy_operator(
            state,
            rng=random.Random(2026),
        )
        for _ in range(sample_size)
    )

    # Use one RNG stream rather than restarting every draw.
    destroy_rng = random.Random(2026)
    destroy_counts = Counter(
        select_destroy_operator(
            state,
            rng=destroy_rng,
        )
        for _ in range(sample_size)
    )

    repair_rng = random.Random(2026)
    repair_counts = Counter(
        select_repair_operator(
            state,
            rng=repair_rng,
        )
        for _ in range(sample_size)
    )

    destroy_frequencies = {
        name: count / sample_size
        for name, count
        in destroy_counts.items()
    }
    repair_frequencies = {
        name: count / sample_size
        for name, count
        in repair_counts.items()
    }

    for name, expected in (
        destroy_probabilities.items()
    ):
        if abs(
            destroy_frequencies.get(name, 0.0)
            - expected
        ) > 0.015:
            raise AssertionError(
                f"Destroy empirical frequency for {name} "
                "is too far from theoretical probability."
            )

    for name, expected in (
        repair_probabilities.items()
    ):
        if abs(
            repair_frequencies.get(name, 0.0)
            - expected
        ) > 0.015:
            raise AssertionError(
                f"Repair empirical frequency for {name} "
                "is too far from theoretical probability."
            )

    print("[PASS] Empirical destroy frequencies match probabilities")
    print("[PASS] Empirical repair frequencies match probabilities")

    # ---------------------------------------------------------
    # AW-2E — Invalid weights rejected
    # ---------------------------------------------------------
    invalid_values = (
        0.0,
        -1.0,
        float("nan"),
        float("inf"),
        float("-inf"),
    )

    for value in invalid_values:
        records = {
            "bad_operator": OperatorWeightRecord(
                name="bad_operator",
                weight=1.0,
            )
        }
        records[
            "bad_operator"
        ].weight = value

        try:
            operator_probabilities(
                records
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"Invalid weight {value} was accepted."
            )

    try:
        operator_probabilities({})
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Empty operator pool was accepted."
        )

    print("[PASS] Zero, negative, NaN, and infinite weights rejected")
    print("[PASS] Empty operator pool rejected")

    report["initial_probabilities"] = {
        "destroy": destroy_initial,
        "repair": repair_initial,
    }
    report["updated_probabilities"] = {
        "destroy": destroy_probabilities,
        "repair": repair_probabilities,
    }
    report["fixed_seed_sequences"] = {
        "destroy_first_20": (
            destroy_sequence_1[:20]
        ),
        "repair_first_20": (
            repair_sequence_1[:20]
        ),
    }
    report["empirical_frequencies"] = {
        "sample_size": sample_size,
        "destroy": destroy_frequencies,
        "repair": repair_frequencies,
    }
    report["invalid_weight_policy"] = {
        "rejected_values": [
            str(value)
            for value in invalid_values
        ],
        "empty_pool_rejected": True,
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
        / "adaptive_weights_aw2_report.json"
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
        "\nADAPTIVE WEIGHTS FIDELITY AW-2 — "
        "ROULETTE-WHEEL SELECTION PASSED"
    )


if __name__ == "__main__":
    main()
