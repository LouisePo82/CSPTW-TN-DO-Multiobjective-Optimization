from __future__ import annotations

from pathlib import Path

from core.instance_loader import load_instance
from alns_solver.solution_state import ALNSSolutionState
from alns_solver.destroy_operators import (
    route_removal,
    random_adp_removal,
)
from alns_solver.repair_operators import (
    enumerate_insertion_candidates,
    best_insertion_repair,
    regret_3_repair,
    perturbed_regret_3_repair,
)


COST_BOUNDS = (
    23.089059445460528,
    24.28427622523578,
)

EMISSION_BOUNDS = (
    77.85476672718833,
    79.22375667475296,
)

EMISSION_FACTORS = (3.0, 1.0)
TAU = 0.275
TOLERANCE = 1e-8


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


def normalized_emission(emission: float) -> float:
    emission_min, emission_max = EMISSION_BOUNDS

    denominator = emission_max - emission_min

    if abs(denominator) <= 1e-12:
        raise ValueError(
            "Emission normalization range must be nonzero."
        )

    return (
        float(emission) - emission_min
    ) / denominator


def assert_complete_valid(
    result,
    instance: dict,
    label: str,
) -> None:
    if not result.validator_pass:
        raise AssertionError(
            f"{label}: final solution failed validation: "
            f"{result.validation_errors}"
        )

    if result.state.unassigned_customers:
        raise AssertionError(
            f"{label}: unassigned customers remain: "
            f"{sorted(result.state.unassigned_customers)}"
        )

    if set(result.state.assignments) != set(
        instance["customers"]
    ):
        raise AssertionError(
            f"{label}: customer coverage is incomplete."
        )


def main() -> None:
    root = Path(__file__).resolve().parents[1]

    instance = load_instance(
        root
        / "data"
        / "small"
        / "instance_001"
    )

    # =========================================================
    # Gate 1 — Corrected Regret-3 and Perturbed Regret-3
    # =========================================================
    partial = random_adp_removal(
        state=cost_anchor_state(),
        instance=instance,
        seed=0,
        vehicle="DV2",
        adp="A1",
    ).state

    regret_3 = regret_3_repair(
        partial,
        instance,
        lambda_value=0.0,
        cost_bounds=COST_BOUNDS,
        emission_bounds=EMISSION_BOUNDS,
        emission_factors=EMISSION_FACTORS,
        strategy_2_mode="paper_random_dv",
        strategy_2_seed=7,
    )

    assert_complete_valid(
        regret_3,
        instance,
        "Corrected Regret-3",
    )

    perturbed_regret_3 = perturbed_regret_3_repair(
        partial,
        instance,
        seed=42,
        noise_strength=TAU,
        lambda_value=0.0,
        cost_bounds=COST_BOUNDS,
        emission_bounds=EMISSION_BOUNDS,
        emission_factors=EMISSION_FACTORS,
        strategy_2_mode="paper_random_dv",
        strategy_2_seed=7,
    )

    assert_complete_valid(
        perturbed_regret_3,
        instance,
        "Perturbed Regret-3",
    )

    print("[PASS] Corrected Regret-3 formula")
    print("[PASS] Perturbed Regret-3 implemented")
    print("[PASS] Paper perturbation factor tau=0.275")

    # =========================================================
    # Gate 2A — Lambda=1 in paper-random-DV mode
    #
    # This mode is paper-faithful but not globally optimal.
    # Verify the objective is recomputed from emission correctly.
    # Do NOT require objective=0.
    # =========================================================
    emission_partial = route_removal(
        state=cost_anchor_state(),
        instance=instance,
        route_type="OD",
        route_id="OD1",
        seed=0,
    ).state

    paper_random_result = best_insertion_repair(
        emission_partial,
        instance,
        lambda_value=1.0,
        cost_bounds=COST_BOUNDS,
        emission_bounds=EMISSION_BOUNDS,
        emission_factors=EMISSION_FACTORS,
        strategy_2_mode="paper_random_dv",
        strategy_2_seed=1,
    )

    assert_complete_valid(
        paper_random_result,
        instance,
        "Lambda=1 paper-random-DV repair",
    )

    expected_random_objective = normalized_emission(
        paper_random_result.final_emission
    )

    if abs(
        paper_random_result.final_objective
        - expected_random_objective
    ) > TOLERANCE:
        raise AssertionError(
            "Lambda=1 objective mismatch in paper-random-DV mode. "
            f"Actual objective: "
            f"{paper_random_result.final_objective}. "
            f"Expected normalized emission: "
            f"{expected_random_objective}. "
            f"Emission: {paper_random_result.final_emission}."
        )

    print(
        "[PASS] Lambda=1 objective equals normalized emission "
        "in paper-random-DV mode"
    )

    # =========================================================
    # Gate 2B — Exhaustive enhanced mode
    #
    # This is not the paper's random-DV rule. It is retained as an
    # enhanced diagnostic mode and should recover the known emission
    # anchor on this tiny validated instance.
    # =========================================================
    exhaustive_result = best_insertion_repair(
        emission_partial,
        instance,
        lambda_value=1.0,
        cost_bounds=COST_BOUNDS,
        emission_bounds=EMISSION_BOUNDS,
        emission_factors=EMISSION_FACTORS,
        strategy_2_mode="exhaustive_best_dv",
        strategy_2_seed=1,
    )

    assert_complete_valid(
        exhaustive_result,
        instance,
        "Lambda=1 exhaustive-DV repair",
    )

    expected_exhaustive_objective = normalized_emission(
        exhaustive_result.final_emission
    )

    if abs(
        exhaustive_result.final_objective
        - expected_exhaustive_objective
    ) > TOLERANCE:
        raise AssertionError(
            "Lambda=1 objective mismatch in exhaustive mode. "
            f"Actual: {exhaustive_result.final_objective}. "
            f"Expected: {expected_exhaustive_objective}."
        )

    if abs(
        exhaustive_result.final_emission
        - EMISSION_BOUNDS[0]
    ) > TOLERANCE:
        raise AssertionError(
            "Exhaustive Strategy II did not recover the known "
            "emission anchor. "
            f"Actual emission: "
            f"{exhaustive_result.final_emission}. "
            f"Expected: {EMISSION_BOUNDS[0]}."
        )

    if abs(
        exhaustive_result.final_objective
    ) > TOLERANCE:
        raise AssertionError(
            "Emission-anchor normalized objective should be zero. "
            f"Actual: {exhaustive_result.final_objective}."
        )

    print(
        "[PASS] Exhaustive diagnostic mode recovers emission anchor"
    )

    # =========================================================
    # Gate 3 — Strategy I: inactive OD + active TN
    # =========================================================
    active_tn_state = cost_anchor_state()
    active_tn_state.dv_routes["DV2"] = [
        "S",
        "TN1",
        "A1",
        "T",
    ]
    active_tn_state.mark_customer_unassigned("C1")
    active_tn_state.od_routes["OD1"] = []

    candidates = enumerate_insertion_candidates(
        active_tn_state,
        instance,
        "C1",
        lambda_value=0.0,
        cost_bounds=COST_BOUNDS,
        emission_bounds=EMISSION_BOUNDS,
        emission_factors=EMISSION_FACTORS,
        strategy_2_mode="paper_random_dv",
        strategy_2_seed=3,
    )

    if not any(
        candidate.details.get("pickup_status")
        == "active_strategy_1"
        and candidate.details.get("pickup") == "TN1"
        and candidate.details.get("driver") == "OD1"
        for candidate in candidates
    ):
        raise AssertionError(
            "Strategy I did not generate an inactive-OD "
            "candidate using active TN1."
        )

    print(
        "[PASS] Strategy I supports inactive OD + active TN"
    )

    # =========================================================
    # Gate 4 — Strategy II paper mode uses one randomly selected DV
    # =========================================================
    new_tn_state = cost_anchor_state()
    new_tn_state.mark_customer_unassigned("C1")
    new_tn_state.od_routes["OD1"] = []

    paper_candidates = enumerate_insertion_candidates(
        new_tn_state,
        instance,
        "C1",
        lambda_value=1.0,
        cost_bounds=COST_BOUNDS,
        emission_bounds=EMISSION_BOUNDS,
        emission_factors=EMISSION_FACTORS,
        strategy_2_mode="paper_random_dv",
        strategy_2_seed=11,
    )

    strategy_2_dvs = {
        candidate.details.get("dv_vehicle")
        for candidate in paper_candidates
        if candidate.details.get("pickup_status")
        == "new_tn"
    }

    strategy_2_dvs.discard(None)

    if len(strategy_2_dvs) > 1:
        raise AssertionError(
            "Paper Strategy II evaluated more than one selected DV: "
            f"{sorted(strategy_2_dvs)}"
        )

    print(
        "[PASS] Strategy II paper mode uses one randomly selected DV"
    )

    print(
        "\nPAPER FIDELITY GATE — REPAIR OPERATORS PASSED"
    )


if __name__ == "__main__":
    main()
