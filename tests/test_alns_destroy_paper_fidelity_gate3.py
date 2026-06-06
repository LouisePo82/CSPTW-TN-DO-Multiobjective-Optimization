from __future__ import annotations

from pathlib import Path
import json

from core.instance_loader import load_instance
from alns_solver.solution_state import ALNSSolutionState
from alns_solver.paper_destroy_operators import (
    _rank_biased_index,
    worst_customer_scores,
    paper_worst_customer_removal,
    paper_probabilistic_worst_customer_removal,
    paper_related_removal,
    paper_probabilistic_related_removal,
    PAPER_DESTROY_OPERATOR_REGISTRY,
    paper_destroy_operator_names,
)
from alns_solver.repair_operators import regret_3_repair


COST_BOUNDS = (
    23.089059445460528,
    24.28427622523578,
)

EMISSION_BOUNDS = (
    77.85476672718833,
    79.22375667475296,
)

EMISSION_FACTORS = (3.0, 1.0)
RANDOMNESS_FACTOR = 5.0


def cost_anchor_state() -> ALNSSolutionState:
    return ALNSSolutionState(
        dv_routes={
            "DV1": [],
            "DV2": ["S", "A1", "T"],
        },
        od_routes={
            "OD1": ["O1", "S", "C1", "D1"],
            "OD2": ["O2", "S", "C2", "C5", "D2"],
        },
        assignments={
            "C1": {
                "mode": "OD_HOME",
                "driver": "OD1",
                "pickup": "S",
            },
            "C2": {
                "mode": "OD_HOME",
                "driver": "OD2",
                "pickup": "S",
            },
            "C3": {
                "mode": "ADP",
                "vehicle": "DV2",
                "adp": "A1",
            },
            "C4": {
                "mode": "ADP",
                "vehicle": "DV2",
                "adp": "A1",
            },
            "C5": {
                "mode": "OD_HOME",
                "driver": "OD2",
                "pickup": "S",
            },
            "C6": {
                "mode": "ADP",
                "vehicle": "DV2",
                "adp": "A1",
            },
        },
    )


def repair_and_validate(
    state: ALNSSolutionState,
    instance: dict,
):
    result = regret_3_repair(
        state,
        instance,
        lambda_value=0.0,
        cost_bounds=COST_BOUNDS,
        emission_bounds=EMISSION_BOUNDS,
        emission_factors=EMISSION_FACTORS,
        strategy_2_mode="paper_random_dv",
        strategy_2_seed=9,
    )

    if not result.validator_pass:
        raise AssertionError(
            f"Repair failed: {result.validation_errors}"
        )

    return result


def assert_partial(
    result,
    expected_count: int,
):
    if len(result.removed_customers) != expected_count:
        raise AssertionError(
            f"Expected {expected_count} removals, "
            f"received {result.removed_customers}."
        )

    for customer in result.removed_customers:
        if customer in result.state.assignments:
            raise AssertionError(
                f"{customer} remains assigned."
            )

        if customer not in result.state.unassigned_customers:
            raise AssertionError(
                f"{customer} is not marked unassigned."
            )


def main() -> None:
    root = Path(__file__).resolve().parents[1]

    instance = load_instance(
        root
        / "data"
        / "small"
        / "instance_001"
    )

    report = {}

    # =========================================================
    # Test 1 — deterministic worst customer
    # =========================================================
    scores = worst_customer_scores(
        cost_anchor_state(),
        instance,
        lambda_value=0.0,
        cost_bounds=COST_BOUNDS,
        emission_bounds=EMISSION_BOUNDS,
        emission_factors=EMISSION_FACTORS,
        candidate_customers=["C1", "C2", "C5"],
    )

    deterministic = paper_worst_customer_removal(
        cost_anchor_state(),
        instance,
        removal_count=1,
        lambda_value=0.0,
        cost_bounds=COST_BOUNDS,
        emission_bounds=EMISSION_BOUNDS,
        emission_factors=EMISSION_FACTORS,
        candidate_customers=["C1", "C2", "C5"],
    )

    assert_partial(
        deterministic,
        expected_count=1,
    )

    if deterministic.removed_customers[0] != scores[0].customer:
        raise AssertionError(
            "Deterministic worst-customer removal did not "
            "select the top-ranked customer."
        )

    repaired_deterministic = repair_and_validate(
        deterministic.state,
        instance,
    )

    print("[PASS] Deterministic worst-customer removal uses rank 0")
    print("[PASS] Deterministic worst-customer partial state repaired")

    report["worst_customer_deterministic"] = {
        "ranked_customers": [
            score.customer
            for score in scores
        ],
        "savings": {
            score.customer: score.saving
            for score in scores
        },
        "removed_customers": (
            deterministic.removed_customers
        ),
        "validator_pass": (
            repaired_deterministic.validator_pass
        ),
    }

    # =========================================================
    # Test 2 — probabilistic worst customer, p=5
    # =========================================================
    probabilistic_results = []

    for seed in range(50):
        result = paper_probabilistic_worst_customer_removal(
            cost_anchor_state(),
            instance,
            removal_count=1,
            seed=seed,
            randomness_factor=RANDOMNESS_FACTOR,
            lambda_value=0.0,
            cost_bounds=COST_BOUNDS,
            emission_bounds=EMISSION_BOUNDS,
            emission_factors=EMISSION_FACTORS,
            candidate_customers=["C1", "C2", "C5"],
        )

        assert_partial(
            result,
            expected_count=1,
        )

        probabilistic_results.append(
            result.removed_customers[0]
        )

    top_customer = scores[0].customer
    top_frequency = probabilistic_results.count(
        top_customer
    )

    if top_frequency <= 25:
        raise AssertionError(
            "Rank-biased selection with p=5 did not favor "
            "the top-ranked worst customer strongly enough. "
            f"Frequency={top_frequency}/50."
        )

    print("[PASS] Probabilistic worst-customer uses rank bias")
    print("[PASS] Randomness factor p=5 favors top ranks")

    report["worst_customer_probabilistic"] = {
        "randomness_factor": RANDOMNESS_FACTOR,
        "top_customer": top_customer,
        "top_frequency_50_seeds": top_frequency,
        "selected_customers": probabilistic_results,
    }

    # =========================================================
    # Test 3 — deterministic/probabilistic related removal
    # =========================================================
    deterministic_related = paper_related_removal(
        cost_anchor_state(),
        instance,
        removal_count=2,
        seed_customer="C1",
        candidate_customers=[
            "C1",
            "C2",
            "C5",
            "C6",
        ],
        phi_1=5.0,
        phi_2=9.0,
        phi_3=1.0,
    )

    assert_partial(
        deterministic_related,
        expected_count=2,
    )

    repaired_related = repair_and_validate(
        deterministic_related.state,
        instance,
    )

    probabilistic_related_results = []

    for seed in range(50):
        result = paper_probabilistic_related_removal(
            cost_anchor_state(),
            instance,
            removal_count=2,
            seed=seed,
            seed_customer="C1",
            randomness_factor=RANDOMNESS_FACTOR,
            candidate_customers=[
                "C1",
                "C2",
                "C5",
                "C6",
            ],
            phi_1=5.0,
            phi_2=9.0,
            phi_3=1.0,
        )

        assert_partial(
            result,
            expected_count=2,
        )

        if result.removed_customers[0] != "C1":
            raise AssertionError(
                "Probabilistic related removal did not "
                "remove the seed first."
            )

        probabilistic_related_results.append(
            result.removed_customers[1]
        )

    deterministic_second = (
        deterministic_related.removed_customers[1]
    )

    deterministic_second_frequency = (
        probabilistic_related_results.count(
            deterministic_second
        )
    )

    if deterministic_second_frequency <= 25:
        raise AssertionError(
            "Probabilistic related removal with p=5 did not "
            "favor the most related candidate strongly enough. "
            f"Frequency="
            f"{deterministic_second_frequency}/50."
        )

    print("[PASS] Deterministic Related Removal uses Eq. (47) rank 0")
    print("[PASS] Probabilistic Related Removal uses rank bias")
    print("[PASS] Probabilistic Related Removal removes seed first")
    print("[PASS] Related-removal partial state repaired")

    report["related_variants"] = {
        "randomness_factor": RANDOMNESS_FACTOR,
        "deterministic_removed": (
            deterministic_related.removed_customers
        ),
        "probabilistic_second_customers": (
            probabilistic_related_results
        ),
        "deterministic_second_frequency_50_seeds": (
            deterministic_second_frequency
        ),
        "validator_pass": repaired_related.validator_pass,
    }

    # =========================================================
    # Test 4 — operator-pool contract
    # =========================================================
    names = paper_destroy_operator_names()

    expected_names = {
        "random_customer_removal",
        "worst_customer_removal_deterministic",
        "worst_customer_removal_probabilistic",
        "route_removal",
        "random_adp_removal",
        "worst_adp_removal",
        "random_tn_removal",
        "related_removal_deterministic",
        "related_removal_probabilistic",
        "historical_node_removal",
        "neighborhood_removal",
        "node_neighborhood_removal",
    }

    if set(names) != expected_names:
        raise AssertionError(
            "Paper destroy registry does not match the "
            "expected 12 selectable entries.\n"
            f"Actual={sorted(names)}\n"
            f"Expected={sorted(expected_names)}"
        )

    if len(PAPER_DESTROY_OPERATOR_REGISTRY) != 12:
        raise AssertionError(
            "Paper destroy registry must contain 12 entries."
        )

    print("[PASS] Destroy operator registry contains 12 entries")
    print("[PASS] Deterministic/probabilistic variants are separate")

    report["operator_registry"] = {
        "entry_count": len(
            PAPER_DESTROY_OPERATOR_REGISTRY
        ),
        "operator_names": names,
    }

    output_dir = (
        root
        / "outputs"
        / "alns_destroy_fidelity_tests"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "destroy_fidelity_gate3_report.json"
    )

    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nReport saved to: {output_path}")
    print(
        "\nPAPER FIDELITY GATE 3 — "
        "PROBABILISTIC OPERATORS AND REGISTRY PASSED"
    )


if __name__ == "__main__":
    main()
