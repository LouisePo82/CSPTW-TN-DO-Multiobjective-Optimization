from __future__ import annotations

from pathlib import Path
import shutil

from core.instance_loader import load_instance
from alns_solver.paper_alns_main import (
    build_ml1_paper_initial_state,
)
from reporting.route_visualization import (
    plot_solution_routes,
    plot_vehicle_routes,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]

    instance_path = (
        root
        / "data"
        / "solomon"
        / "derived"
        / "r101_25"
    )

    output_dir = (
        root
        / "outputs"
        / "solomon_feasibility_tests"
        / "r101_25_visualization_sr3"
    )

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    instance = load_instance(
        instance_path
    )

    state = build_ml1_paper_initial_state(
        instance,
        seed=2026,
        lambda_value=0.5,
        cost_bounds=None,
        emission_bounds=None,
        emission_factors=(3.0, 1.0),
    )

    solution = state.to_core_solution(
        instance=instance,
        lambda_value=0.5,
        objective_mode="weighted",
        cost_bounds=None,
        emission_bounds=None,
        emission_factors=(3.0, 1.0),
        require_complete=True,
        metadata={
            "source": (
                "solomon_r101_25_sr3"
            ),
        },
    )

    assert solution.validator_pass

    combined_path = (
        output_dir
        / "best_route_map.png"
    )

    generated_combined = plot_solution_routes(
        instance=instance,
        state=state,
        output_path=combined_path,
        instance_id="r101_25_derived",
        lambda_value=0.5,
        seed=2026,
    )

    vehicle_dir = (
        output_dir
        / "vehicle_routes"
    )

    generated_vehicle_maps = (
        plot_vehicle_routes(
            instance=instance,
            state=state,
            output_dir=vehicle_dir,
            instance_id="r101_25_derived",
            lambda_value=0.5,
            seed=2026,
        )
    )

    assert generated_combined.exists()
    assert generated_combined.stat().st_size > 0

    assert generated_vehicle_maps
    assert all(
        path.exists()
        and path.stat().st_size > 0
        for path in generated_vehicle_maps
    )

    print(
        "[PASS] R101-25 initial solution is accepted by shared visualization"
    )
    print(
        "[PASS] Combined Solomon-derived route map is created"
    )
    print(
        "[PASS] Per-vehicle Solomon-derived route maps are created"
    )
    print(
        "[PASS] Shared visualization module is reused without solver changes"
    )
    print(
        f"[INFO] Combined map: {generated_combined}"
    )
    print(
        f"[INFO] Vehicle maps: "
        f"{len(generated_vehicle_maps)}"
    )
    print(
        "\nSR-3 — R101-25 SHARED ROUTE VISUALIZATION PASSED"
    )


if __name__ == "__main__":
    main()
