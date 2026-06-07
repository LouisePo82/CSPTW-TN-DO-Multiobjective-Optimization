from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch


# ============================================================
# Visual conventions
# ============================================================

DV_ROUTE_COLOR = "#2b6cb0"  # blue
OD_ROUTE_COLOR = "#2f9e44"  # green
ADP_LINK_COLOR = "0.70"  # gray dotted assignment line

TYPE1_COLOR = "#a9c1e8"  # Type 1 customer
TYPE2_COLOR = "#8ea7d9"  # Type 2 customer
TYPE3_COLOR = "#c9c9c9"  # Type 3 customer

TN_ACTIVE_FACE = "#f2b27c"
TN_ACTIVE_EDGE = "#d3873b"

ADP_ACTIVE_FACE = "#5b8dd1"
ADP_ACTIVE_EDGE = "#3d6ca9"

OD_ENDPOINT_EDGE = "#4f81bd"
OD_ORIGIN_FACE = "#5f89c8"
OD_DEST_FACE = "white"

INACTIVE_EDGE = "0.60"
INACTIVE_FACE = "white"


# ============================================================
# Instance / state helpers
# ============================================================


def _node_coordinates(
    instance: dict,
) -> dict[str, tuple[float, float]]:
    """
    Read coordinates without modifying the instance.
    """
    for key in (
        "coordinates",
        "coords",
        "node_coordinates",
    ):
        mapping = instance.get(key)

        if isinstance(mapping, dict) and mapping:
            result: dict[str, tuple[float, float]] = {}

            for node, value in mapping.items():
                if isinstance(value, dict):
                    x = value.get("x")
                    y = value.get("y")
                    if x is None or y is None:
                        continue
                    result[str(node)] = (float(x), float(y))
                else:
                    result[str(node)] = (
                        float(value[0]),
                        float(value[1]),
                    )

            if result:
                return result

    result: dict[str, tuple[float, float]] = {}

    for node, data in instance.get("nodes", {}).items():
        if isinstance(data, dict) and "x" in data and "y" in data:
            result[str(node)] = (
                float(data["x"]),
                float(data["y"]),
            )

    if not result:
        raise KeyError("No coordinates found in instance.")

    return result


def _explicit_node_type(
    instance: dict,
    node: str,
) -> str | None:
    data = instance.get("nodes", {}).get(node)

    if not isinstance(data, dict):
        return None

    raw = data.get("type") or data.get("node_type") or data.get("kind")

    if raw is None:
        return None

    value = str(raw).strip().lower()

    if value in {"depot", "start_depot", "end_depot"}:
        return "depot"
    if value in {"tn", "transshipment", "transshipment_node"}:
        return "tn"
    if value in {
        "adp",
        "alternative_delivery_point",
        "pickup_point",
    }:
        return "adp"
    if value in {"customer"}:
        return "customer"
    if value in {"od_origin", "origin"}:
        return "od_origin"
    if value in {"od_destination", "destination"}:
        return "od_destination"
    if value in {"dv_terminal", "terminal"}:
        return "dv_terminal"

    return None


def _node_type(
    instance: dict,
    node: str,
) -> str:
    explicit = _explicit_node_type(instance, node)

    if explicit is not None:
        return explicit

    upper = str(node).upper()

    if upper == "S":
        return "depot"
    if upper == "T":
        return "dv_terminal"
    if upper.startswith("TN"):
        return "tn"
    if upper.startswith("A"):
        return "adp"
    if upper.startswith("C"):
        return "customer"
    if upper.startswith("O") and len(upper) > 1 and upper[1:].isdigit():
        return "od_origin"
    if upper.startswith("D") and len(upper) > 1 and upper[1:].isdigit():
        return "od_destination"

    return "other"


def _display_label(
    instance: dict,
    node: str,
) -> str:
    kind = _node_type(instance, node)
    upper = str(node).upper()

    if kind == "depot":
        return "D"

    if kind == "dv_terminal":
        return "T"

    if kind == "tn":
        return str(node)

    if kind == "adp":
        suffix = upper[1:] if upper.startswith("A") else str(node)
        return f"ADP{suffix}"

    if kind in {"od_origin", "od_destination"}:
        return upper

    return str(node)


def _customer_record(
    instance: dict,
    customer: str,
) -> dict[str, Any]:
    customers = instance.get("customers")

    if isinstance(customers, dict):
        record = customers.get(customer)
        if isinstance(record, dict):
            return record

    nodes = instance.get("nodes")

    if isinstance(nodes, dict):
        record = nodes.get(customer)
        if isinstance(record, dict):
            return record

    return {}


def _normalize_customer_type(value: Any) -> int | None:
    if isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value if value in {1, 2, 3} else None

    if isinstance(value, float) and value.is_integer():
        integer = int(value)
        return integer if integer in {1, 2, 3} else None

    if isinstance(value, str):
        clean = value.strip().upper()

        direct = {
            "1": 1,
            "2": 2,
            "3": 3,
            "T1": 1,
            "T2": 2,
            "T3": 3,
            "TYPE1": 1,
            "TYPE2": 2,
            "TYPE3": 3,
            "TYPE 1": 1,
            "TYPE 2": 2,
            "TYPE 3": 3,
            "CUSTOMER_TYPE_1": 1,
            "CUSTOMER_TYPE_2": 2,
            "CUSTOMER_TYPE_3": 3,
        }

        return direct.get(clean)

    return None


def _customer_type(
    instance: dict,
    customer: str,
) -> int:
    record = _customer_record(instance, customer)

    for key in (
        "customer_type",
        "cust_type",
        "delivery_type",
        "type_id",
        "type",
    ):
        if key not in record:
            continue

        parsed = _normalize_customer_type(record[key])

        if parsed is not None:
            return parsed

    return 1


def _customer_color(
    customer_type: int,
) -> str:
    if customer_type == 1:
        return TYPE1_COLOR
    if customer_type == 2:
        return TYPE2_COLOR
    return TYPE3_COLOR


def _active_infrastructure(
    instance: dict,
    state,
) -> tuple[set[str], set[str]]:
    active_adps: set[str] = set()
    active_tns: set[str] = set()

    assignments = getattr(state, "assignments", {})

    for info in assignments.values():
        if not isinstance(info, dict):
            continue

        if str(info.get("mode", "")).upper() == "ADP":
            adp = info.get("adp")
            if isinstance(adp, str):
                active_adps.add(adp)

        pickup = info.get("pickup")
        if isinstance(pickup, str) and _node_type(instance, pickup) == "tn":
            active_tns.add(pickup)

    for route_group in (
        getattr(state, "dv_routes", {}),
        getattr(state, "od_routes", {}),
    ):
        for route in route_group.values():
            for node in route:
                if _node_type(instance, node) == "tn":
                    active_tns.add(node)

    return active_adps, active_tns


# ============================================================
# Title helpers
# ============================================================


def _format_lambda_value(
    lambda_value: float | None,
) -> str | None:
    if lambda_value is None:
        return None

    return f"{float(lambda_value):g}"


def _compose_solution_title(
    *,
    instance_id: str | None,
    lambda_value: float | None,
    seed: int | None,
    title: str | None = None,
) -> str:
    """
    Build the combined-map title from explicit run metadata.

    ``title`` remains available as a deliberate full-title override,
    but metadata is never parsed back from a display string.
    """
    if title is not None:
        return title

    lambda_text = _format_lambda_value(lambda_value)

    if lambda_text is None:
        line_1 = "Crowd-Shipping Route Map"
    else:
        line_1 = (
            "Crowd-Shipping Route Map — "
            f"Scenario λ = {lambda_text}"
        )

    subtitle_parts: list[str] = []

    if instance_id is not None:
        subtitle_parts.append(str(instance_id))

    if seed is not None:
        subtitle_parts.append(f"seed = {int(seed)}")

    subtitle_parts.append(
        "Paper-faithful ALNS best solution"
    )

    return f"{line_1}\n{' | '.join(subtitle_parts)}"


def _compose_vehicle_title(
    *,
    fleet_name: str,
    vehicle: str,
    instance_id: str | None,
    lambda_value: float | None,
    seed: int | None,
) -> str:
    line_1 = f"{fleet_name.upper()} route — {vehicle}"

    metadata_parts: list[str] = []

    if instance_id is not None:
        metadata_parts.append(str(instance_id))

    lambda_text = _format_lambda_value(lambda_value)
    if lambda_text is not None:
        metadata_parts.append(f"λ = {lambda_text}")

    if seed is not None:
        metadata_parts.append(f"seed = {int(seed)}")

    if not metadata_parts:
        return line_1

    return f"{line_1}\n{' | '.join(metadata_parts)}"


# ============================================================
# Drawing helpers
# ============================================================


def _draw_route_arrow(
    ax,
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    *,
    color: str,
    linestyle: str,
    linewidth: float,
    shrink_a: float = 18,
    shrink_b: float = 18,
    zorder: int = 4,
) -> None:
    arrow = FancyArrowPatch(
        start_xy,
        end_xy,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=linewidth,
        linestyle=linestyle,
        color=color,
        alpha=0.95,
        shrinkA=shrink_a,
        shrinkB=shrink_b,
        zorder=zorder,
    )
    ax.add_patch(arrow)


def _draw_route(
    ax,
    *,
    route: Iterable[str],
    coordinates: dict[str, tuple[float, float]],
    color: str,
    linestyle: str,
    linewidth: float,
) -> None:
    valid_nodes = [node for node in route if node in coordinates]

    if len(valid_nodes) < 2:
        return

    for origin, destination in zip(
        valid_nodes,
        valid_nodes[1:],
    ):
        _draw_route_arrow(
            ax,
            coordinates[origin],
            coordinates[destination],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
        )


def _draw_adp_assignments(
    ax,
    *,
    instance: dict,
    state,
    coordinates: dict[str, tuple[float, float]],
) -> None:
    for customer, assignment in sorted(getattr(state, "assignments", {}).items()):
        if not isinstance(assignment, dict):
            continue

        if str(assignment.get("mode", "")).upper() != "ADP":
            continue

        adp = assignment.get("adp")

        if customer not in coordinates or adp not in coordinates:
            continue

        customer_xy = coordinates[customer]
        adp_xy = coordinates[adp]

        ax.plot(
            [customer_xy[0], adp_xy[0]],
            [customer_xy[1], adp_xy[1]],
            color=ADP_LINK_COLOR,
            linestyle=":",
            linewidth=1.7,
            alpha=0.9,
            zorder=1,
        )

        middle_x = (customer_xy[0] + adp_xy[0]) / 2.0
        middle_y = (customer_xy[1] + adp_xy[1]) / 2.0

        ax.annotate(
            f"{customer} → {_display_label(instance, adp)}",
            (middle_x, middle_y),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            color="0.52",
            zorder=2,
        )


def _node_marker_style(
    *,
    instance: dict,
    node: str,
    active_adps: set[str],
    active_tns: set[str],
) -> dict[str, Any]:
    kind = _node_type(instance, node)

    style = {
        "marker": "o",
        "face_color": "0.75",
        "edge_color": "white",
        "size": 550,
        "linewidth": 1.2,
        "alpha": 1.0,
        "text_color": "black",
        "zorder": 8,
    }

    if kind == "depot":
        style.update(
            {
                "marker": "o",
                "face_color": "black",
                "edge_color": "black",
                "size": 750,
                "text_color": "white",
                "zorder": 50,
            }
        )
    elif kind == "dv_terminal":
        style.update(
            {
                "marker": "s",
                "face_color": "black",
                "edge_color": "black",
                "size": 650,
                "text_color": "white",
                "zorder": 10,
            }
        )
    elif kind == "tn":
        if node in active_tns:
            style.update(
                {
                    "marker": "o",
                    "face_color": TN_ACTIVE_FACE,
                    "edge_color": TN_ACTIVE_EDGE,
                    "size": 650,
                    "text_color": "black",
                    "zorder": 10,
                }
            )
        else:
            style.update(
                {
                    "marker": "o",
                    "face_color": INACTIVE_FACE,
                    "edge_color": INACTIVE_EDGE,
                    "size": 650,
                    "text_color": "0.40",
                    "alpha": 0.88,
                    "zorder": 9,
                }
            )
    elif kind == "adp":
        if node in active_adps:
            style.update(
                {
                    "marker": "s",
                    "face_color": ADP_ACTIVE_FACE,
                    "edge_color": ADP_ACTIVE_EDGE,
                    "size": 650,
                    "text_color": "white",
                    "zorder": 10,
                }
            )
        else:
            style.update(
                {
                    "marker": "s",
                    "face_color": INACTIVE_FACE,
                    "edge_color": INACTIVE_EDGE,
                    "size": 650,
                    "text_color": "0.40",
                    "alpha": 0.88,
                    "zorder": 9,
                }
            )
    elif kind == "od_origin":
        style.update(
            {
                "marker": "^",
                "face_color": OD_ORIGIN_FACE,
                "edge_color": OD_ENDPOINT_EDGE,
                "size": 750,
                "text_color": "white",
                "zorder": 10,
            }
        )
    elif kind == "od_destination":
        style.update(
            {
                "marker": "^",
                "face_color": OD_DEST_FACE,
                "edge_color": OD_ENDPOINT_EDGE,
                "size": 750,
                "text_color": OD_ENDPOINT_EDGE,
                "zorder": 10,
            }
        )
    elif kind == "customer":
        c_type = _customer_type(instance, node)
        style.update(
            {
                "marker": "o",
                "face_color": _customer_color(c_type),
                "edge_color": "white",
                "size": 550,
                "text_color": "black",
                "zorder": 9,
            }
        )

    return style


def _draw_nodes(
    ax,
    *,
    instance: dict,
    coordinates: dict[str, tuple[float, float]],
    active_adps: set[str],
    active_tns: set[str],
) -> None:
    depot_nodes: list[tuple[str, tuple[float, float]]] = []
    other_nodes: list[tuple[str, tuple[float, float]]] = []

    depot_coords: set[tuple[float, float]] = set()
    for node, coord in coordinates.items():
        if _node_type(instance, node) == "depot":
            depot_coords.add(coord)
            depot_nodes.append((node, coord))
        else:
            other_nodes.append((node, coord))

    for node, (x, y) in other_nodes:
        if (x, y) in depot_coords:
            continue

        style = _node_marker_style(
            instance=instance,
            node=node,
            active_adps=active_adps,
            active_tns=active_tns,
        )

        ax.scatter(
            [x],
            [y],
            marker=style["marker"],
            s=style["size"],
            color=style["face_color"],
            edgecolors=style["edge_color"],
            linewidths=style["linewidth"],
            alpha=style["alpha"],
            zorder=style["zorder"],
        )

        label = _display_label(instance, node)
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=8.0,
            fontweight="bold",
            color=style["text_color"],
            clip_on=False,
            zorder=style["zorder"] + 2,
        )

        kind = _node_type(instance, node)
        if kind == "tn" and node not in active_tns:
            ax.annotate(
                "(inactive)",
                (x, y),
                xytext=(0, 14),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
                color="0.45",
                zorder=style["zorder"] + 2,
            )
        if kind == "adp" and node not in active_adps:
            ax.annotate(
                "(inactive)",
                (x, y),
                xytext=(0, 14),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
                color="0.45",
                zorder=style["zorder"] + 2,
            )

    for node, (x, y) in depot_nodes:
        style = _node_marker_style(
            instance=instance,
            node=node,
            active_adps=active_adps,
            active_tns=active_tns,
        )

        ax.scatter(
            [x],
            [y],
            marker=style["marker"],
            s=style["size"],
            color=style["face_color"],
            edgecolors=style["edge_color"],
            linewidths=style["linewidth"],
            alpha=1.0,
            zorder=style["zorder"],
        )

        label = _display_label(instance, node)
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=9.0,
            fontweight="bold",
            color=style["text_color"],
            clip_on=False,
            zorder=style["zorder"] + 2,
        )


def _legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=11,
            label="Depot",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=TN_ACTIVE_FACE,
            markeredgecolor=TN_ACTIVE_EDGE,
            markersize=11,
            label="Transshipment Node",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=ADP_ACTIVE_FACE,
            markeredgecolor=ADP_ACTIVE_EDGE,
            markersize=11,
            label="Alternative Delivery Point",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=TYPE1_COLOR,
            markeredgecolor="white",
            markersize=10,
            label="Customer Type 1 — Home delivery",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=TYPE2_COLOR,
            markeredgecolor="white",
            markersize=10,
            label="Customer Type 2 — ADP pickup",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=TYPE3_COLOR,
            markeredgecolor="white",
            markersize=10,
            label="Customer Type 3 — Flexible option",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor=OD_ORIGIN_FACE,
            markeredgecolor=OD_ENDPOINT_EDGE,
            markersize=11,
            label="Origin node of an Occasional Driver",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="white",
            markeredgecolor=OD_ENDPOINT_EDGE,
            markersize=11,
            label="Destination node of an Occasional Driver",
        ),
        Line2D(
            [0],
            [0],
            color=ADP_LINK_COLOR,
            linewidth=1.8,
            linestyle=":",
            label="Customer-to-ADP assignment",
        ),
        Line2D(
            [0],
            [0],
            color=DV_ROUTE_COLOR,
            linewidth=2.4,
            linestyle="-",
            label="Route of Dedicated Vehicles",
        ),
        Line2D(
            [0],
            [0],
            color=OD_ROUTE_COLOR,
            linewidth=2.2,
            linestyle="--",
            label="Route of Occasional Drivers",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="white",
            markeredgecolor=INACTIVE_EDGE,
            markersize=10,
            label="Inactive TN / ADP",
        ),
    ]


# ============================================================
# Public API
# ============================================================


def plot_solution_routes(
    *,
    instance: dict,
    state,
    output_path: Path,
    instance_id: str | None = None,
    lambda_value: float | None = None,
    seed: int | None = None,
    title: str | None = None,
    annotate_sequence: bool = False,
) -> Path:
    coordinates = _node_coordinates(instance)
    active_adps, active_tns = _active_infrastructure(
        instance,
        state,
    )

    fig, ax = plt.subplots(figsize=(17.5, 10.5))

    # 1) Customer-to-ADP assignments first.
    _draw_adp_assignments(
        ax,
        instance=instance,
        state=state,
        coordinates=coordinates,
    )

    # 2) Dedicated vehicle routes.
    for _, route in sorted(getattr(state, "dv_routes", {}).items()):
        if route:
            _draw_route(
                ax,
                route=route,
                coordinates=coordinates,
                color=DV_ROUTE_COLOR,
                linestyle="-",
                linewidth=2.4,
            )

    # 3) Occasional driver routes.
    od_routes = getattr(state, "od_routes", {})

    for _, route in sorted(od_routes.items()):
        if route:
            _draw_route(
                ax,
                route=route,
                coordinates=coordinates,
                color=OD_ROUTE_COLOR,
                linestyle="--",
                linewidth=2.2,
            )

    # 4) Nodes on top of routes.
    _draw_nodes(
        ax,
        instance=instance,
        coordinates=coordinates,
        active_adps=active_adps,
        active_tns=active_tns,
    )

    # 5) Final layout.
    ax.set_title(
        _compose_solution_title(
            instance_id=instance_id,
            lambda_value=lambda_value,
            seed=seed,
            title=title,
        ),
        fontsize=16,
        pad=18,
    )
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.set_aspect("equal", adjustable="datalim")

    ax.legend(
        handles=_legend_handles(),
        title="Terminology",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        fontsize=9,
        title_fontsize=11,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    return output_path


def plot_vehicle_routes(
    *,
    instance: dict,
    state,
    output_dir: Path,
    instance_id: str | None = None,
    lambda_value: float | None = None,
    seed: int | None = None,
) -> list[Path]:
    coordinates = _node_coordinates(instance)
    active_adps, active_tns = _active_infrastructure(
        instance,
        state,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated: list[Path] = []

    route_groups = (
        (
            "dv",
            getattr(state, "dv_routes", {}),
            DV_ROUTE_COLOR,
            "-",
            2.4,
        ),
        (
            "od",
            getattr(state, "od_routes", {}),
            OD_ROUTE_COLOR,
            "--",
            2.2,
        ),
    )

    for fleet_name, routes, color, linestyle, linewidth in route_groups:
        for vehicle, route in sorted(routes.items()):
            if not route:
                continue

            fig, ax = plt.subplots(figsize=(14.0, 9.0))

            if fleet_name == "dv":
                assignments = getattr(state, "assignments", {})

                for customer, assignment in assignments.items():
                    if not isinstance(assignment, dict):
                        continue

                    if str(assignment.get("mode", "")).upper() != "ADP":
                        continue

                    if assignment.get("vehicle") != vehicle:
                        continue

                    adp = assignment.get("adp")

                    if customer not in coordinates or adp not in coordinates:
                        continue

                    customer_xy = coordinates[customer]
                    adp_xy = coordinates[adp]

                    ax.plot(
                        [customer_xy[0], adp_xy[0]],
                        [customer_xy[1], adp_xy[1]],
                        color=ADP_LINK_COLOR,
                        linestyle=":",
                        linewidth=1.7,
                        alpha=0.9,
                        zorder=1,
                    )

            _draw_route(
                ax,
                route=route,
                coordinates=coordinates,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
            )

            _draw_nodes(
                ax,
                instance=instance,
                coordinates=coordinates,
                active_adps=active_adps,
                active_tns=active_tns,
            )

            ax.set_title(
                _compose_vehicle_title(
                    fleet_name=fleet_name,
                    vehicle=vehicle,
                    instance_id=instance_id,
                    lambda_value=lambda_value,
                    seed=seed,
                ),
                fontsize=14,
                pad=14,
            )
            ax.set_xlabel("X coordinate")
            ax.set_ylabel("Y coordinate")
            ax.grid(True, linestyle=":", alpha=0.35)
            ax.set_aspect("equal", adjustable="datalim")

            ax.legend(
                handles=_legend_handles(),
                title="Terminology",
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                frameon=True,
                fontsize=8,
                title_fontsize=10,
            )

            fig.tight_layout()

            path = output_dir / f"{fleet_name}_{vehicle}_route.png"

            fig.savefig(
                path,
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)
            generated.append(path)

    return generated
