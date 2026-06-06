from __future__ import annotations

from pathlib import Path
import json

from core.instance_loader import load_instance
from alns_solver.initial_solution_factory import (
    PAPER_MODE,
    ENHANCED_MODE,
    SUPPORTED_INITIAL_SOLUTION_MODES,
    build_initial_solution,
)

EMISSION_FACTORS = (3.0, 1.0)
TOLERANCE = 1e-8
EXPECTED_PAPER_COST = 24.33466340567672
EXPECTED_PAPER_EMISSION = 78.10670262939306
EXPECTED_ENHANCED_COST = 23.089059445460528
EXPECTED_ENHANCED_EMISSION = 79.22375667475296


def signature(state) -> tuple:
    return (
        tuple((k, tuple(v)) for k, v in sorted(state.dv_routes.items())),
        tuple((k, tuple(v)) for k, v in sorted(state.od_routes.items())),
        tuple(
            (c, tuple(sorted(a.items())))
            for c, a in sorted(state.assignments.items())
        ),
    )


def assert_close(actual: float, expected: float, label: str) -> None:
    if abs(float(actual) - float(expected)) > TOLERANCE:
        raise AssertionError(
            f"{label}: actual={actual}, expected={expected}"
        )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    instance = load_instance(root / "data" / "small" / "instance_001")
    report = {}

    if SUPPORTED_INITIAL_SOLUTION_MODES != {PAPER_MODE, ENHANCED_MODE}:
        raise AssertionError("Mode registry is incorrect.")

    try:
        build_initial_solution(instance, mode="unknown_mode")
    except ValueError:
        pass
    else:
        raise AssertionError("Unknown mode was not rejected.")

    print("[PASS] Initial-solution factory exposes exactly two modes")
    print("[PASS] Unsupported mode is rejected")

    paper_1 = build_initial_solution(
        instance,
        mode=PAPER_MODE,
        seed=0,
        lambda_value=0.0,
        emission_factors=EMISSION_FACTORS,
        strategy_2_mode="paper_random_dv",
    )
    paper_2 = build_initial_solution(
        instance,
        mode=PAPER_MODE,
        seed=0,
        lambda_value=0.0,
        emission_factors=EMISSION_FACTORS,
        strategy_2_mode="paper_random_dv",
    )

    if paper_1.trace is None:
        raise AssertionError("Paper mode must return Algorithm-1 trace.")
    if paper_1.metadata["construction_mode"] != PAPER_MODE:
        raise AssertionError("Paper mode label is incorrect.")
    if not paper_1.metadata["paper_faithful"]:
        raise AssertionError("Paper mode is not marked paper-faithful.")
    if paper_1.metadata["enhanced"]:
        raise AssertionError("Paper mode is incorrectly marked enhanced.")
    if signature(paper_1.state) != signature(paper_2.state):
        raise AssertionError("Paper mode is not reproducible.")

    paper_solution = paper_1.state.to_core_solution(
        instance=instance,
        lambda_value=0.0,
        objective_mode="cost",
        emission_factors=EMISSION_FACTORS,
        require_complete=True,
        metadata=paper_1.metadata,
    )
    if not paper_solution.validator_pass:
        raise AssertionError(
            f"Paper mode invalid: {paper_solution.validation_errors}"
        )
    assert_close(paper_solution.cost, EXPECTED_PAPER_COST, "Paper cost")
    assert_close(
        paper_solution.emission,
        EXPECTED_PAPER_EMISSION,
        "Paper emission",
    )

    print("[PASS] Paper mode returns three-phase Algorithm-1 trace")
    print("[PASS] Paper mode metadata is labeled paper-faithful")
    print("[PASS] Paper mode is reproducible for a fixed seed")
    print("[PASS] Paper mode matches validated small-instance baseline")
    print("[PASS] Paper mode passes shared validator")

    enhanced_1 = build_initial_solution(
        instance,
        mode=ENHANCED_MODE,
        seed=0,
        max_attempts=100,
    )
    enhanced_2 = build_initial_solution(
        instance,
        mode=ENHANCED_MODE,
        seed=0,
        max_attempts=100,
    )

    if enhanced_1.trace is not None:
        raise AssertionError("Enhanced mode must not expose paper trace.")
    if enhanced_1.metadata["construction_mode"] != ENHANCED_MODE:
        raise AssertionError("Enhanced mode label is incorrect.")
    if enhanced_1.metadata["paper_faithful"]:
        raise AssertionError("Enhanced mode is incorrectly paper-faithful.")
    if not enhanced_1.metadata["enhanced"]:
        raise AssertionError("Enhanced mode is not marked enhanced.")
    if signature(enhanced_1.state) != signature(enhanced_2.state):
        raise AssertionError("Enhanced mode is not reproducible.")

    enhanced_solution = enhanced_1.state.to_core_solution(
        instance=instance,
        lambda_value=0.0,
        objective_mode="cost",
        emission_factors=EMISSION_FACTORS,
        require_complete=True,
        metadata=enhanced_1.metadata,
    )
    if not enhanced_solution.validator_pass:
        raise AssertionError(
            f"Enhanced mode invalid: {enhanced_solution.validation_errors}"
        )
    assert_close(
        enhanced_solution.cost,
        EXPECTED_ENHANCED_COST,
        "Enhanced cost",
    )
    assert_close(
        enhanced_solution.emission,
        EXPECTED_ENHANCED_EMISSION,
        "Enhanced emission",
    )

    print("[PASS] Enhanced mode uses generic constructor")
    print("[PASS] Enhanced mode is explicitly not paper-faithful")
    print("[PASS] Enhanced mode is reproducible for a fixed seed")
    print("[PASS] Enhanced mode matches generic baseline")
    print("[PASS] Enhanced mode passes shared validator")

    if signature(paper_1.state) == signature(enhanced_1.state):
        raise AssertionError(
            "Controlled instance did not distinguish both modes."
        )

    print("[PASS] Paper and enhanced modes have distinct labels")
    print("[PASS] Paper trace cannot be confused with enhanced mode")
    print("[PASS] Controlled instance distinguishes both constructors")

    report["paper_mode"] = {
        "metadata": paper_1.metadata,
        "cost": paper_solution.cost,
        "emission": paper_solution.emission,
        "validator_pass": paper_solution.validator_pass,
        "phase1_order": paper_1.trace.phase1_customer_order,
        "fixed_tn_positions": paper_1.trace.fixed_tn_positions,
    }
    report["enhanced_mode"] = {
        "metadata": enhanced_1.metadata,
        "cost": enhanced_solution.cost,
        "emission": enhanced_solution.emission,
        "validator_pass": enhanced_solution.validator_pass,
    }
    report["separation"] = {
        "paper_mode_label": PAPER_MODE,
        "enhanced_mode_label": ENHANCED_MODE,
        "same_structural_solution": False,
        "paper_cost_minus_enhanced_cost": (
            paper_solution.cost - enhanced_solution.cost
        ),
        "paper_emission_minus_enhanced_emission": (
            paper_solution.emission - enhanced_solution.emission
        ),
    }

    output_dir = (
        root / "outputs" / "alns_initial_solution_fidelity_tests"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "initial_solution_is4_report.json"
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nReport saved to: {output_path}")
    print(
        "\nINITIAL SOLUTION FIDELITY IS-4 — "
        "PAPER AND ENHANCED MODE SEPARATION PASSED"
    )


if __name__ == "__main__":
    main()
