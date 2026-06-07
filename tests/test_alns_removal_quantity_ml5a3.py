from __future__ import annotations

from pathlib import Path
import json
import random

from alns_solver.paper_operator_dispatch import (
    PAPER_DESTROY_OPERATOR_NAMES,
)
from alns_solver.paper_removal_quantity import (
    PAPER_COUNT_BASED_DESTROY_OPERATORS,
    PAPER_REMOVAL_LOWER_FRACTION,
    PAPER_REMOVAL_UPPER_FRACTION,
    PAPER_STRUCTURAL_DESTROY_OPERATORS,
    destroy_operator_uses_removal_quantity,
    paper_removal_quantity_bounds,
    sample_paper_removal_quantity,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]

    if PAPER_REMOVAL_LOWER_FRACTION != 0.10:
        raise AssertionError(
            "Paper lower removal fraction is not 0.10."
        )

    if PAPER_REMOVAL_UPPER_FRACTION != 0.40:
        raise AssertionError(
            "Paper upper removal fraction is not 0.40."
        )

    print("[PASS] Paper removal fractions are fixed at 0.10 and 0.40")

    # Small project instance: |N| = 6 -> q in [1, 2].
    small_bounds = paper_removal_quantity_bounds(6)

    if (
        small_bounds.minimum != 1
        or small_bounds.maximum != 2
    ):
        raise AssertionError(
            "Bounds for six customers must be [1, 2]."
        )

    # Larger examples independently verify rounding.
    examples = {
        1: (1, 1),
        6: (1, 2),
        10: (1, 4),
        11: (2, 4),
        25: (3, 10),
        100: (10, 40),
    }

    for customer_count, expected in examples.items():
        bounds = paper_removal_quantity_bounds(
            customer_count
        )
        actual = (bounds.minimum, bounds.maximum)

        if actual != expected:
            raise AssertionError(
                f"|N|={customer_count}: expected "
                f"{expected}, received {actual}."
            )

    print("[PASS] Integer bounds use ceil(0.1|N|) and floor(0.4|N|)")
    print("[PASS] Removal quantity is always at least one")

    try:
        paper_removal_quantity_bounds(0)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Zero customer count was accepted."
        )

    try:
        paper_removal_quantity_bounds(-5)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Negative customer count was accepted."
        )

    print("[PASS] Invalid customer counts are rejected")

    # Reproducibility.
    rng_a = random.Random(2026)
    rng_b = random.Random(2026)

    sequence_a = [
        sample_paper_removal_quantity(
            25,
            rng=rng_a,
            seed=2026,
        ).quantity
        for _ in range(100)
    ]
    sequence_b = [
        sample_paper_removal_quantity(
            25,
            rng=rng_b,
            seed=2026,
        ).quantity
        for _ in range(100)
    ]

    if sequence_a != sequence_b:
        raise AssertionError(
            "Fixed seed did not reproduce q sequence."
        )

    print("[PASS] Fixed seed reproduces removal-quantity sequence")

    bounds_25 = paper_removal_quantity_bounds(25)

    if any(
        quantity < bounds_25.minimum
        or quantity > bounds_25.maximum
        for quantity in sequence_a
    ):
        raise AssertionError(
            "Sampled q fell outside paper bounds."
        )

    if not (
        bounds_25.minimum in sequence_a
        and bounds_25.maximum in sequence_a
    ):
        raise AssertionError(
            "Inclusive sampler did not reach both bounds."
        )

    print("[PASS] Every sampled q lies inside the inclusive paper interval")
    print("[PASS] Both paper interval endpoints are reachable")

    # Pool classification must partition all 12 paper destroy operators.
    count_based = set(
        PAPER_COUNT_BASED_DESTROY_OPERATORS
    )
    structural = set(
        PAPER_STRUCTURAL_DESTROY_OPERATORS
    )
    full_pool = set(PAPER_DESTROY_OPERATOR_NAMES)

    if count_based & structural:
        raise AssertionError(
            "Count-based and structural pools overlap."
        )

    if count_based | structural != full_pool:
        raise AssertionError(
            "Removal-quantity classification does not "
            "partition the full paper destroy pool."
        )

    if len(count_based) != 8:
        raise AssertionError(
            "Expected eight count-based destroy operators."
        )

    if len(structural) != 4:
        raise AssertionError(
            "Expected four structural destroy operators."
        )

    for operator_name in count_based:
        if not destroy_operator_uses_removal_quantity(
            operator_name
        ):
            raise AssertionError(
                f"{operator_name} should consume q."
            )

    for operator_name in structural:
        if destroy_operator_uses_removal_quantity(
            operator_name
        ):
            raise AssertionError(
                f"{operator_name} must keep structural semantics."
            )

    print("[PASS] Exactly eight destroy operators consume sampled q")
    print("[PASS] Four structural removals do not consume sampled q")

    try:
        destroy_operator_uses_removal_quantity(
            "unknown_operator"
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Unknown operator was accepted."
        )

    print("[PASS] Unknown destroy operator names are rejected")

    sample = sample_paper_removal_quantity(
        6,
        rng=random.Random(7),
        seed=7,
    )

    if sample.metadata["lambda_dependent"]:
        raise AssertionError(
            "Removal quantity became lambda-dependent."
        )

    if sample.metadata["enhanced"]:
        raise AssertionError(
            "Enhanced removal quantity entered paper mode."
        )

    print("[PASS] Removal quantity is independent of F_lambda")
    print("[PASS] No enhanced removal-quantity tuning is introduced")

    report = {
        "paper_fractions": {
            "minimum": (
                PAPER_REMOVAL_LOWER_FRACTION
            ),
            "maximum": (
                PAPER_REMOVAL_UPPER_FRACTION
            ),
        },
        "integer_rule": (
            "ceil_lower_floor_upper"
        ),
        "examples": {
            str(customer_count): {
                "minimum": expected[0],
                "maximum": expected[1],
            }
            for customer_count, expected
            in examples.items()
        },
        "operator_classification": {
            "count_based": sorted(count_based),
            "structural": sorted(structural),
        },
        "sample_for_six_customers": {
            "quantity": sample.quantity,
            "minimum": sample.bounds.minimum,
            "maximum": sample.bounds.maximum,
        },
        "fidelity": sample.metadata,
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
        / "alns_main_loop_ml5a3_report.json"
    )
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nReport saved to: {output_path}")
    print(
        "\nALNS MAIN LOOP FIDELITY ML-5A.3 — "
        "PAPER REMOVAL QUANTITY PASSED"
    )


if __name__ == "__main__":
    main()
