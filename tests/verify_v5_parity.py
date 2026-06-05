from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


TOLERANCE = 1e-6

EXPECTED = {
    "micro_cost": 11.478708664619075,
    "cost_anchor_cost": 23.089059445460528,
    "cost_anchor_emission": 79.22375667475296,
    "emission_anchor_cost": 24.28427622523578,
    "emission_anchor_emission": 77.85476672718833,
}

MICRO_OUTPUT_SUFFIX = "_micro_exact_validation"
SMALL_OUTPUT_SUFFIX = "_small_exact_multiobjective"


def assert_close(
    label: str,
    actual: float,
    expected: float,
    tolerance: float = TOLERANCE,
) -> None:
    difference = abs(actual - expected)

    if difference > tolerance:
        raise AssertionError(
            f"{label} mismatch:\n"
            f"  actual     = {actual}\n"
            f"  expected   = {expected}\n"
            f"  difference = {difference}\n"
            f"  tolerance  = {tolerance}"
        )

    print(
        f"[PASS] {label}: "
        f"actual={actual:.12f}, "
        f"expected={expected:.12f}, "
        f"difference={difference:.3e}"
    )


def find_latest_output_folder(
    outputs_dir: Path,
    suffix: str,
) -> Path:
    if not outputs_dir.exists():
        raise FileNotFoundError(f"Outputs directory does not exist: {outputs_dir}")

    matching_folders = [
        path
        for path in outputs_dir.iterdir()
        if path.is_dir() and path.name.endswith(suffix)
    ]

    if not matching_folders:
        raise FileNotFoundError(
            f"No output folder ending with '{suffix}' was found under:\n{outputs_dir}"
        )

    return max(
        matching_folders,
        key=lambda path: path.stat().st_mtime,
    )


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file does not exist:\n{path}")

    return json.loads(path.read_text(encoding="utf-8"))


def read_payoff_table(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(f"Payoff table does not exist:\n{path}")

    rows: dict[str, dict[str, float]] = {}

    with path.open(
        mode="r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            anchor = row["anchor"].strip()

            rows[anchor] = {
                "cost": float(row["cost"]),
                "emission": float(row["emission"]),
            }

    required_anchors = {"cost", "emission"}

    if set(rows) != required_anchors:
        raise AssertionError(
            "Payoff table must contain exactly the cost and "
            f"emission anchors. Found: {set(rows)}"
        )

    return rows


def validate_solution_ground_truth(
    label: str,
    solution: dict,
) -> None:
    status = solution.get("status")

    if status != "OPTIMAL":
        raise AssertionError(f"{label}: expected OPTIMAL, received {status}")

    if not solution.get("validator_pass", False):
        raise AssertionError(
            f"{label}: validator failed:\n{solution.get('validation_errors', [])}"
        )

    optimality_gap = float(solution.get("optimality_gap", float("inf")))

    if optimality_gap > TOLERANCE:
        raise AssertionError(f"{label}: optimality gap is too large: {optimality_gap}")

    print(f"[PASS] {label}: status=OPTIMAL, validator_pass=True, gap={optimality_gap}")


def validate_micro_result(
    micro_output: Path,
) -> None:
    print("\n========================================")
    print("CHECK 1 — MICRO EXACT PARITY")
    print("========================================")

    solution_path = micro_output / "solutions" / "micro_exact" / "solution.json"

    solution = read_json(solution_path)

    validate_solution_ground_truth(
        "micro exact solution",
        solution,
    )

    assert_close(
        "micro_cost",
        float(solution["cost"]),
        EXPECTED["micro_cost"],
    )

    expected_assignments = {
        "C1": "OD_HOME",
        "C2": "ADP",
        "C3": "DV_HOME",
    }

    actual_assignments = {
        customer: assignment.get("mode")
        for customer, assignment in solution["assignments"].items()
    }

    if actual_assignments != expected_assignments:
        raise AssertionError(
            "Micro assignment mismatch:\n"
            f"  actual   = {actual_assignments}\n"
            f"  expected = {expected_assignments}"
        )

    c1_pickup = solution["assignments"]["C1"].get("pickup")

    if c1_pickup != "TN1":
        raise AssertionError(
            "C1 must be served by OD1 through TN1 in the "
            f"micro ground truth. Actual pickup: {c1_pickup}"
        )

    if "TN1" not in solution["dv_routes"]["DV1"]:
        raise AssertionError("DV1 must visit TN1 in the micro solution.")

    if "TN1" not in solution["od_routes"]["OD1"]:
        raise AssertionError("OD1 must visit TN1 in the micro solution.")

    print("[PASS] micro delivery assignments")
    print("[PASS] micro TN synchronization structure")


def validate_small_result(
    small_output: Path,
) -> None:
    print("\n========================================")
    print("CHECK 2 — SMALL EXACT PARITY")
    print("========================================")

    payoff_path = small_output / "summary" / "payoff_table.csv"

    payoff = read_payoff_table(payoff_path)

    assert_close(
        "cost_anchor_cost",
        payoff["cost"]["cost"],
        EXPECTED["cost_anchor_cost"],
    )

    assert_close(
        "cost_anchor_emission",
        payoff["cost"]["emission"],
        EXPECTED["cost_anchor_emission"],
    )

    assert_close(
        "emission_anchor_cost",
        payoff["emission"]["cost"],
        EXPECTED["emission_anchor_cost"],
    )

    assert_close(
        "emission_anchor_emission",
        payoff["emission"]["emission"],
        EXPECTED["emission_anchor_emission"],
    )

    cost_solution = read_json(
        small_output / "solutions" / "cost_anchor" / "solution.json"
    )

    emission_solution = read_json(
        small_output / "solutions" / "emission_anchor" / "solution.json"
    )

    validate_solution_ground_truth(
        "small cost anchor",
        cost_solution,
    )

    validate_solution_ground_truth(
        "small emission anchor",
        emission_solution,
    )

    if cost_solution["od_routes"]["OD1"][1] != "S":
        raise AssertionError("Cost anchor OD1 should pick up at depot S.")

    if emission_solution["od_routes"]["OD1"][1] != "TN1":
        raise AssertionError("Emission anchor OD1 should pick up at TN1.")

    print("[PASS] cost-anchor route structure")
    print("[PASS] emission-anchor TN route structure")


def validate_nondominated_set(
    small_output: Path,
) -> None:
    print("\n========================================")
    print("CHECK 3 — NONDOMINATED SET PARITY")
    print("========================================")

    path = small_output / "summary" / "nondominated_solutions.csv"

    if not path.exists():
        raise FileNotFoundError(f"Nondominated results file does not exist:\n{path}")

    with path.open(
        mode="r",
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    actual_points = {
        (
            round(float(row["cost"]), 8),
            round(float(row["emission"]), 8),
        )
        for row in rows
    }

    expected_points = {
        (
            round(EXPECTED["cost_anchor_cost"], 8),
            round(EXPECTED["cost_anchor_emission"], 8),
        ),
        (
            round(EXPECTED["emission_anchor_cost"], 8),
            round(EXPECTED["emission_anchor_emission"], 8),
        ),
    }

    if actual_points != expected_points:
        raise AssertionError(
            "Nondominated set mismatch:\n"
            f"  actual   = {actual_points}\n"
            f"  expected = {expected_points}"
        )

    if len(actual_points) != 2:
        raise AssertionError(
            "Expected exactly two unique nondominated points, "
            f"but found {len(actual_points)}."
        )

    print("[PASS] exact nondominated set contains 2 points")
    print("[PASS] exact nondominated points match v5")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    outputs_dir = project_root / "outputs"

    micro_output = find_latest_output_folder(
        outputs_dir,
        MICRO_OUTPUT_SUFFIX,
    )

    small_output = find_latest_output_folder(
        outputs_dir,
        SMALL_OUTPUT_SUFFIX,
    )

    print("Using output folders:")
    print(f"  Micro: {micro_output}")
    print(f"  Small: {small_output}")

    validate_micro_result(micro_output)
    validate_small_result(small_output)
    validate_nondominated_set(small_output)

    print("\n========================================")
    print("MIGRATION PARITY TEST PASSED")
    print("========================================")
    print(
        "The refactored repository reproduces the validated "
        "v5 exact ground-truth results."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\n========================================")
        print("MIGRATION PARITY TEST FAILED")
        print("========================================")
        print(f"{type(error).__name__}: {error}")
        sys.exit(1)
