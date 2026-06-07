from __future__ import annotations

from pathlib import Path
from collections import defaultdict
import csv
import math
import statistics

import matplotlib.pyplot as plt


def _read_csv(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def _write_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> Path:
    if not rows:
        raise ValueError("Cannot write an empty CSV.")

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)

    return path


def create_lambda_summary(
    consolidated_csv: Path,
    output_csv: Path,
) -> list[dict[str, object]]:
    rows = _read_csv(consolidated_csv)

    grouped: dict[
        float,
        list[dict[str, str]],
    ] = {}

    for row in rows:
        grouped.setdefault(
            float(row["lambda"]),
            [],
        ).append(row)

    summary_rows: list[dict[str, object]] = []

    for lambda_value in sorted(grouped):
        group = grouped[lambda_value]

        costs = [
            float(row["best_cost"])
            for row in group
        ]
        emissions = [
            float(row["best_emission"])
            for row in group
        ]
        objectives = [
            float(row["best_F_lambda"])
            for row in group
        ]
        runtimes = [
            float(row["runtime_seconds"])
            for row in group
        ]
        valid_flags = [
            str(row["validation_pass"]).lower()
            == "true"
            for row in group
        ]

        summary_rows.append(
            {
                "lambda": lambda_value,
                "n_runs": len(group),
                "mean_cost": statistics.fmean(costs),
                "std_cost": (
                    statistics.stdev(costs)
                    if len(costs) > 1
                    else 0.0
                ),
                "min_cost": min(costs),
                "max_cost": max(costs),
                "mean_emission": statistics.fmean(
                    emissions
                ),
                "std_emission": (
                    statistics.stdev(emissions)
                    if len(emissions) > 1
                    else 0.0
                ),
                "min_emission": min(emissions),
                "max_emission": max(emissions),
                "mean_F_lambda": statistics.fmean(
                    objectives
                ),
                "std_F_lambda": (
                    statistics.stdev(objectives)
                    if len(objectives) > 1
                    else 0.0
                ),
                "mean_runtime_seconds": (
                    statistics.fmean(runtimes)
                ),
                "std_runtime_seconds": (
                    statistics.stdev(runtimes)
                    if len(runtimes) > 1
                    else 0.0
                ),
                "min_runtime_seconds": min(runtimes),
                "max_runtime_seconds": max(runtimes),
                "valid_run_rate": (
                    sum(valid_flags)
                    / len(valid_flags)
                ),
            }
        )

    _write_csv(
        output_csv,
        summary_rows,
    )

    return summary_rows


def _coordinate_group_key(
    x_value: float,
    y_value: float,
    *,
    digits: int = 10,
) -> tuple[float, float]:
    """
    Group visually identical trade-off points.

    Rounding is reporting-only and does not alter the
    underlying experimental results.
    """
    return (
        round(x_value, digits),
        round(y_value, digits),
    )


def _format_lambda_group(
    lambda_values: list[float],
) -> str:
    values = ", ".join(
        f"{value:g}"
        for value in sorted(lambda_values)
    )
    return f"λ = {values}"


def plot_cost_emission_tradeoff(
    rows: list[dict[str, object]],
    output_path: Path,
) -> Path:
    grouped_points: dict[
        tuple[float, float],
        list[float],
    ] = defaultdict(list)

    exact_coordinates: dict[
        tuple[float, float],
        tuple[float, float],
    ] = {}

    for row in rows:
        mean_cost = float(row["mean_cost"])
        mean_emission = float(
            row["mean_emission"]
        )
        lambda_value = float(row["lambda"])

        key = _coordinate_group_key(
            mean_cost,
            mean_emission,
        )
        grouped_points[key].append(
            lambda_value
        )
        exact_coordinates.setdefault(
            key,
            (
                mean_cost,
                mean_emission,
            ),
        )

    figure, axis = plt.subplots(
        figsize=(9, 6.5)
    )

    for point_index, key in enumerate(
        sorted(grouped_points)
    ):
        x_value, y_value = (
            exact_coordinates[key]
        )
        lambda_values = grouped_points[key]

        axis.scatter(
            [x_value],
            [y_value],
            s=90,
            zorder=3,
        )

        horizontal_offset = 8
        vertical_offset = (
            10
            if point_index % 2 == 0
            else -18
        )

        axis.annotate(
            _format_lambda_group(
                lambda_values
            ),
            (x_value, y_value),
            xytext=(
                horizontal_offset,
                vertical_offset,
            ),
            textcoords="offset points",
            fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": "0.80",
                "alpha": 0.90,
            },
            zorder=4,
        )

    axis.set_title(
        "Cost–emission trade-off"
    )
    axis.set_xlabel(
        "Mean total cost"
    )
    axis.set_ylabel(
        "Mean emission proxy"
    )
    axis.grid(
        True,
        alpha=0.25,
    )

    figure.tight_layout()
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)

    return output_path


def _metric_std_column(
    metric: str,
) -> str | None:
    mapping = {
        "mean_cost": "std_cost",
        "mean_emission": "std_emission",
        "mean_F_lambda": "std_F_lambda",
        "mean_runtime_seconds": (
            "std_runtime_seconds"
        ),
    }

    return mapping.get(metric)


def plot_metric_by_lambda(
    rows: list[dict[str, object]],
    metric: str,
    output_path: Path,
    ylabel: str,
) -> Path:
    ordered = sorted(
        rows,
        key=lambda row: float(row["lambda"]),
    )

    x_values = [
        float(row["lambda"])
        for row in ordered
    ]
    y_values = [
        float(row[metric])
        for row in ordered
    ]

    std_column = _metric_std_column(
        metric
    )

    y_errors: list[float] | None = None

    if (
        std_column is not None
        and all(
            std_column in row
            for row in ordered
        )
    ):
        candidate_errors = [
            float(row[std_column])
            for row in ordered
        ]

        if any(
            not math.isclose(
                value,
                0.0,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            for value in candidate_errors
        ):
            y_errors = candidate_errors

    figure, axis = plt.subplots(
        figsize=(8.5, 5.5)
    )

    if y_errors is None:
        axis.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=2.0,
        )
    else:
        axis.errorbar(
            x_values,
            y_values,
            yerr=y_errors,
            marker="o",
            linewidth=2.0,
            capsize=4,
        )

    axis.set_title(
        f"{ylabel} by λ"
    )
    axis.set_xlabel("λ")
    axis.set_ylabel(ylabel)
    axis.grid(
        True,
        alpha=0.25,
    )

    figure.tight_layout()
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)

    return output_path
