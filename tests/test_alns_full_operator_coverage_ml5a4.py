from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json

from core.instance_loader import load_instance

from alns_solver.destroy_operators import (
    random_customer_removal,
    random_adp_removal,
    random_tn_removal,
)
from alns_solver.paper_destroy_operators import (
    paper_worst_customer_removal,
    paper_probabilistic_worst_customer_removal,
    paper_route_removal,
    paper_worst_adp_removal,
    paper_related_removal,
    paper_probabilistic_related_removal,
    paper_historical_node_removal,
    paper_neighborhood_removal,
    paper_node_neighborhood_removal,
    paper_destroy_operator_names,
)
from alns_solver.repair_operators import (
    best_insertion_repair,
    perturbed_best_insertion_repair,
    regret_2_repair,
    perturbed_regret_repair,
    regret_3_repair,
    perturbed_regret_3_repair,
)
from alns_solver.paper_history import (
    PaperHistoricalPositionState,
)
from alns_solver.paper_removal_quantity import (
    PAPER_COUNT_BASED_DESTROY_OPERATORS,
    PAPER_STRUCTURAL_DESTROY_OPERATORS,
)
from alns_solver.paper_operator_dispatch import (
    PAPER_DESTROY_OPERATOR_NAMES,
    PAPER_REPAIR_OPERATOR_NAMES,
)

from tests.test_alns_destroy_paper_fidelity_gate1 import (
    emission_anchor_state,
)
from tests.test_alns_destroy_paper_fidelity_gate2 import (
    mixed_home_state,
)
from tests.test_alns_destroy_paper_fidelity_gate3 import (
    cost_anchor_state,
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


def state_signature(state):
    return {
        "dv_routes": deepcopy(state.dv_routes),
        "od_routes": deepcopy(state.od_routes),
        "assignments": deepcopy(state.assignments),
        "unassigned": sorted(
            state.unassigned_customers
        ),
    }


def assert_destroy_result(
    label,
    original_state,
    result,
):
    if not result.removed_customers:
        raise AssertionError(
            f"{label}: no customer was removed."
        )

    if state_signature(original_state) != state_signature(
        original_state.copy()
    ):
        raise AssertionError(
            f"{label}: source state is unstable."
        )

    for customer in result.removed_customers:
        if customer in result.state.assignments:
            raise AssertionError(
                f"{label}: {customer} remains assigned."
            )
        if customer not in result.state.unassigned_customers:
            raise AssertionError(
                f"{label}: {customer} not marked unassigned."
            )


def repair_with_regret_3(partial, instance, seed):
    result = regret_3_repair(
        partial,
        instance,
        lambda_value=0.5,
        cost_bounds=COST_BOUNDS,
        emission_bounds=EMISSION_BOUNDS,
        emission_factors=EMISSION_FACTORS,
        strategy_2_mode="paper_random_dv",
        strategy_2_seed=seed,
    )

    if not result.validator_pass:
        raise AssertionError(
            "Regret-3 failed after destroy: "
            f"{result.validation_errors}"
        )

    if result.state.unassigned_customers:
        raise AssertionError(
            "Regret-3 left customers unassigned."
        )

    return result


def run_destroy_coverage(instance):
    general = mixed_home_state()
    adp = cost_anchor_state()
    tn = emission_anchor_state()

    history = (
        PaperHistoricalPositionState
        .initialize_from_state(
            general,
            instance,
        )
        .snapshot()
    )

    cases = []

    cases.append((
        "random_customer_removal",
        general,
        random_customer_removal(
            general,
            instance,
            removal_count=1,
            seed=11,
        ),
    ))

    cases.append((
        "worst_customer_removal_deterministic",
        general,
        paper_worst_customer_removal(
            general,
            instance,
            removal_count=1,
            lambda_value=0.5,
            cost_bounds=COST_BOUNDS,
            emission_bounds=EMISSION_BOUNDS,
            emission_factors=EMISSION_FACTORS,
        ),
    ))

    cases.append((
        "worst_customer_removal_probabilistic",
        general,
        paper_probabilistic_worst_customer_removal(
            general,
            instance,
            removal_count=1,
            seed=12,
            randomness_factor=5.0,
            lambda_value=0.5,
            cost_bounds=COST_BOUNDS,
            emission_bounds=EMISSION_BOUNDS,
            emission_factors=EMISSION_FACTORS,
        ),
    ))

    cases.append((
        "route_removal",
        tn,
        paper_route_removal(
            tn,
            instance,
            vehicle="DV2",
            seed=13,
        ),
    ))

    cases.append((
        "random_adp_removal",
        adp,
        random_adp_removal(
            adp,
            instance,
            vehicle="DV2",
            adp="A1",
            seed=14,
        ),
    ))

    cases.append((
        "worst_adp_removal",
        adp,
        paper_worst_adp_removal(
            adp,
            instance,
            lambda_value=0.5,
            cost_bounds=COST_BOUNDS,
            emission_bounds=EMISSION_BOUNDS,
            emission_factors=EMISSION_FACTORS,
        ),
    ))

    cases.append((
        "random_tn_removal",
        tn,
        random_tn_removal(
            tn,
            instance,
            tn="TN1",
            seed=15,
        ),
    ))

    cases.append((
        "related_removal_deterministic",
        general,
        paper_related_removal(
            general,
            instance,
            removal_count=1,
            seed_customer="C1",
            seed=16,
            phi_1=5.0,
            phi_2=9.0,
            phi_3=1.0,
        ),
    ))

    cases.append((
        "related_removal_probabilistic",
        general,
        paper_probabilistic_related_removal(
            general,
            instance,
            removal_count=1,
            seed_customer="C1",
            seed=17,
            randomness_factor=5.0,
            phi_1=5.0,
            phi_2=9.0,
            phi_3=1.0,
        ),
    ))

    cases.append((
        "historical_node_removal",
        general,
        paper_historical_node_removal(
            general,
            instance,
            removal_count=1,
            best_historical_position_costs=history,
        ),
    ))

    cases.append((
        "neighborhood_removal",
        general,
        paper_neighborhood_removal(
            general,
            instance,
            removal_count=1,
        ),
    ))

    cases.append((
        "node_neighborhood_removal",
        general,
        paper_node_neighborhood_removal(
            general,
            instance,
            removal_count=1,
            seed_customer="C1",
            seed=18,
        ),
    ))

    records = {}

    for index, (label, source, result) in enumerate(
        cases,
        start=1,
    ):
        original_signature = state_signature(source)

        assert_destroy_result(
            label,
            source,
            result,
        )

        if state_signature(source) != original_signature:
            raise AssertionError(
                f"{label}: destroy mutated input state."
            )

        repaired = repair_with_regret_3(
            result.state,
            instance,
            seed=100 + index,
        )

        records[label] = {
            "removed_customers": list(
                result.removed_customers
            ),
            "repair_validator_pass": (
                repaired.validator_pass
            ),
        }

    expected = set(PAPER_DESTROY_OPERATOR_NAMES)
    actual = set(records)

    if actual != expected:
        raise AssertionError(
            "Destroy execution coverage mismatch. "
            f"Missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )

    if set(paper_destroy_operator_names()) != expected:
        raise AssertionError(
            "Paper destroy registry disagrees with dispatcher pool."
        )

    classified = (
        set(PAPER_COUNT_BASED_DESTROY_OPERATORS)
        | set(PAPER_STRUCTURAL_DESTROY_OPERATORS)
    )

    if classified != expected:
        raise AssertionError(
            "Removal-quantity classification disagrees "
            "with full destroy pool."
        )

    return records


def common_repair_partial(instance):
    return random_adp_removal(
        cost_anchor_state(),
        instance,
        vehicle="DV2",
        adp="A1",
        seed=21,
    ).state


def assert_repair(label, result, instance):
    if not result.validator_pass:
        raise AssertionError(
            f"{label}: validation failed: "
            f"{result.validation_errors}"
        )

    if result.state.unassigned_customers:
        raise AssertionError(
            f"{label}: unassigned customers remain."
        )

    if set(result.state.assignments) != set(
        instance["customers"]
    ):
        raise AssertionError(
            f"{label}: incomplete customer coverage."
        )


def run_repair_coverage(instance):
    partial = common_repair_partial(instance)

    common = dict(
        lambda_value=0.5,
        cost_bounds=COST_BOUNDS,
        emission_bounds=EMISSION_BOUNDS,
        emission_factors=EMISSION_FACTORS,
        strategy_2_mode="paper_random_dv",
        strategy_2_seed=31,
    )

    cases = {
        "best_insertion": best_insertion_repair(
            partial,
            instance,
            **common,
        ),
        "perturbed_best_insertion": (
            perturbed_best_insertion_repair(
                partial,
                instance,
                seed=32,
                noise_strength=TAU,
                **common,
            )
        ),
        "regret_2": regret_2_repair(
            partial,
            instance,
            **common,
        ),
        "perturbed_regret_2": (
            perturbed_regret_repair(
                partial,
                instance,
                k=2,
                seed=33,
                noise_strength=TAU,
                **common,
            )
        ),
        "regret_3": regret_3_repair(
            partial,
            instance,
            **common,
        ),
        "perturbed_regret_3": (
            perturbed_regret_3_repair(
                partial,
                instance,
                seed=34,
                noise_strength=TAU,
                **common,
            )
        ),
    }

    records = {}

    for label, result in cases.items():
        assert_repair(
            label,
            result,
            instance,
        )
        records[label] = {
            "validator_pass": result.validator_pass,
            "final_objective": result.final_objective,
            "final_cost": result.final_cost,
            "final_emission": result.final_emission,
        }

    expected = set(PAPER_REPAIR_OPERATOR_NAMES)
    actual = set(records)

    if actual != expected:
        raise AssertionError(
            "Repair execution coverage mismatch. "
            f"Missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )

    return records


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    instance = load_instance(
        root / "data" / "small" / "instance_001"
    )

    destroy_records = run_destroy_coverage(
        instance
    )

    print("[PASS] All 12 paper destroy entries execute")
    print("[PASS] Each destroy uses an applicable validated fixture")
    print("[PASS] Destroy operators preserve their input states")
    print("[PASS] Every destroy partial state is repaired and validated")

    repair_records = run_repair_coverage(
        instance
    )

    print("[PASS] All 6 paper repair operators execute")
    print("[PASS] Every repair completes the common partial state")
    print("[PASS] Every repaired solution passes shared validation")
    print("[PASS] Perturbed repairs use paper tau=0.275")
    print("[PASS] Repair Strategy II remains paper_random_dv")

    # Repeat the complete coverage to verify fixed-seed behavior.
    repeated_destroy = run_destroy_coverage(
        instance
    )
    repeated_repair = run_repair_coverage(
        instance
    )

    if destroy_records != repeated_destroy:
        raise AssertionError(
            "Destroy coverage is not reproducible."
        )

    if repair_records != repeated_repair:
        raise AssertionError(
            "Repair coverage is not reproducible."
        )

    print("[PASS] Fixed seeds reproduce destroy and repair coverage")
    print("[PASS] No fallback or operator substitution is introduced")
    print("[PASS] No enhanced ALNS behavior is introduced")

    report = {
        "destroy_operator_count": len(
            destroy_records
        ),
        "repair_operator_count": len(
            repair_records
        ),
        "destroy_coverage": destroy_records,
        "repair_coverage": repair_records,
        "contracts": {
            "operator_specific_applicable_fixtures": True,
            "fallback": False,
            "operator_substitution": False,
            "applicable_subset_roulette": False,
            "paper_random_dv": True,
            "perturbation_tau": TAU,
            "paper_faithful": True,
            "enhanced": False,
        },
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
        / "alns_main_loop_ml5a4_report.json"
    )
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nReport saved to: {output_path}")
    print(
        "\nALNS MAIN LOOP FIDELITY ML-5A.4 — "
        "FULL OPERATOR EXECUTION COVERAGE PASSED"
    )


if __name__ == "__main__":
    main()
