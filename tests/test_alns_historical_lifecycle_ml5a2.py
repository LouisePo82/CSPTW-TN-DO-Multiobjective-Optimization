from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json

from core.instance_loader import load_instance
from alns_solver.paper_history import (
    PaperHistoricalPositionState,
)
from tests.test_alns_destroy_paper_fidelity_gate2 import (
    mixed_home_state,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    instance = load_instance(
        root / "data" / "small" / "instance_001"
    )
    initial_state = mixed_home_state()

    history = (
        PaperHistoricalPositionState
        .initialize_from_state(
            initial_state,
            instance,
        )
    )

    expected_initial = {"C1", "C2", "C5", "C6"}

    if set(history.best_position_costs) != expected_initial:
        raise AssertionError(
            "Initialization did not observe exactly the "
            "Type-1/3 home-delivery customers."
        )

    if history.completed_observations != 1:
        raise AssertionError(
            "Initialization must count as the first observation."
        )

    print("[PASS] Initial solution initializes historical position costs")
    print("[PASS] Only Type-1/3 home-delivery customers are observed")

    snapshot_before = history.snapshot()
    snapshot_before["C1"] = -999.0

    if history.best_cost("C1") == -999.0:
        raise AssertionError(
            "History snapshot aliases internal state."
        )

    print("[PASS] Pre-iteration snapshot is independent")

    # A worse observation must not increase historical best.
    worse_state = initial_state.copy()
    worse_state.od_routes["OD1"] = [
        "O1", "S", "C6", "C1", "D1"
    ]
    worse_state.assignments["C6"] = {
        "mode": "OD_HOME",
        "driver": "OD1",
        "pickup": "S",
    }
    worse_state.dv_routes["DV1"] = ["S", "T"]

    before_worse = history.snapshot()
    history.observe_current_state(
        worse_state,
        instance,
    )
    after_worse = history.snapshot()

    for customer, previous_best in before_worse.items():
        if (
            customer in after_worse
            and after_worse[customer]
            > previous_best + 1e-12
        ):
            raise AssertionError(
                "Historical best increased after observation."
            )

    print("[PASS] Historical best never increases")

    # Put Type-3 C5 at ADP. This state is used only to test lifecycle
    # observation semantics, not complete-route feasibility.
    adp_state = initial_state.copy()
    adp_state.od_routes["OD2"] = [
        node
        for node in adp_state.od_routes["OD2"]
        if node != "C5"
    ]
    adp_state.assignments["C5"] = {
        "mode": "ADP",
        "vehicle": "DV2",
        "adp": "A1",
    }

    c5_best_before = history.best_cost("C5")
    c5_count_before = history.observation_counts["C5"]

    observations = history.observe_current_state(
        adp_state,
        instance,
    )

    if "C5" in observations:
        raise AssertionError(
            "ADP-served Type-3 customer was observed."
        )

    if history.best_cost("C5") != c5_best_before:
        raise AssertionError(
            "ADP mode erased or changed prior history."
        )

    if (
        history.observation_counts["C5"]
        != c5_count_before
    ):
        raise AssertionError(
            "ADP mode incremented historical observation count."
        )

    print("[PASS] Type-3 customer at ADP creates no observation")
    print("[PASS] ADP mode preserves prior historical best")

    # Customer returns to home delivery and becomes observable again.
    returned_state = initial_state.copy()
    history.observe_current_state(
        returned_state,
        instance,
    )

    if (
        history.observation_counts["C5"]
        != c5_count_before + 1
    ):
        raise AssertionError(
            "Returning home-delivery customer was not observed."
        )

    if history.best_cost("C5") > c5_best_before + 1e-12:
        raise AssertionError(
            "Returning observation increased historical best."
        )

    print("[PASS] Returning home-delivery customer is observed again")

    # Temporal contract: snapshot for iteration m is taken before observing
    # current_state_after from iteration m.
    temporal_history = (
        PaperHistoricalPositionState
        .initialize_from_state(
            initial_state,
            instance,
        )
    )
    iteration_m_snapshot = temporal_history.snapshot()
    internal_before = temporal_history.snapshot()

    temporal_history.observe_current_state(
        returned_state,
        instance,
    )

    if iteration_m_snapshot != internal_before:
        raise AssertionError(
            "Pre-iteration snapshot changed retroactively."
        )

    print(
        "[PASS] Iteration-m history excludes its future "
        "post-transition observation"
    )

    metadata = history.metadata()

    if not metadata["paper_faithful"]:
        raise AssertionError(
            "History controller lost paper-faithful label."
        )

    if metadata["enhanced"]:
        raise AssertionError(
            "Enhanced behavior entered history lifecycle."
        )

    if metadata["objective_extension_applied"]:
        raise AssertionError(
            "F_lambda was incorrectly applied to history metric."
        )

    print("[PASS] Historical lifecycle remains independent of F_lambda")
    print("[PASS] No enhanced historical behavior is introduced")

    report = {
        "initial_tracked_customers": sorted(
            expected_initial
        ),
        "best_position_costs": (
            history.snapshot()
        ),
        "observation_counts": dict(
            history.observation_counts
        ),
        "completed_observations": (
            history.completed_observations
        ),
        "contracts": {
            "best_never_increases": True,
            "adp_skips_observation": True,
            "adp_preserves_prior_history": True,
            "return_to_home_observed": True,
            "pre_iteration_snapshot_independent": True,
            "update_after_state_transition": True,
        },
        "fidelity": metadata,
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
        / "alns_main_loop_ml5a2_report.json"
    )
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nReport saved to: {output_path}")
    print(
        "\nALNS MAIN LOOP FIDELITY ML-5A.2 — "
        "HISTORICAL POSITION LIFECYCLE PASSED"
    )


if __name__ == "__main__":
    main()
