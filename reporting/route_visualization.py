from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import re
import textwrap

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.offsetbox import VPacker, TextArea, AnnotationBbox


# ============================================================
# Visual conventions
# ============================================================

DV_ROUTE_COLOR = "#2b6cb0"  # blue
OD_ROUTE_COLOR = "#2f9e44"  # green
ADP_LINK_COLOR = "0.68"  # light gray dotted assignment line

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

# Customer-outline semantics. Fill color still represents customer type.
DV_HOME_EDGE = DV_ROUTE_COLOR
OD_HOME_EDGE = OD_ROUTE_COLOR
ADP_MODE_EDGE = "0.38"
UNASSIGNED_EDGE = "white"

# Declutter / readability settings.
SHOW_INACTIVE_LABELS = False
SHOW_ASSIGNMENT_EDGE_LABELS = False
SHOW_ROUTE_SUMMARY = True
SHOW_SERVICE_MODE_KEY = True
SHOW_OD_SEQUENCE_ON_COMBINED = True
SHOW_SEQUENCE_ON_VEHICLE_MAPS = True

ASSIGNMENT_LINE_ALPHA = 0.34
ASSIGNMENT_LINE_WIDTH = 1.10
DV_ROUTE_LINE_WIDTH = 2.60
OD_ROUTE_LINE_WIDTH = 2.45
NODE_LABEL_FONT_SIZE = 8.0
SEQUENCE_FONT_SIZE = 7.2
ROUTE_SUMMARY_FONT_SIZE = 8.2


# ============================================================
# Instance / state helpers
# ============================================================


def _node_coordinates(
    instance: dict,
) -> dict[str, tuple[float, float]]:
    """Read node coordinates without modifying the instance."""
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
    node: str,  # ĐÃ SỬA: Thêm chính xác tham số cấu trúc node ở đây
) -> str | None:
    data = instance.get("nodes", {}).get(node)

    if not isinstance(data, dict):
        return None

    raw = data.get("type") or data.get("node_type") or data.get("kind")

    if raw is None:
        return None

    value = str(raw).strip().lower()

    if value in {
        "depot",
        "start_depot",
        "end_depot",
        "depot_start",
        "depot_end",
    }:
        return "depot"
    if value in {"tn", "transshipment", "transshipment_node"}:
        return "tn"
    if value in {
        "adp",
        "alternative_delivery_point",
        "pickup_point",
    }:
        return "adp"
    if value == "customer":
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
    """
    ĐỒNG BỘ HOÁ TERMINOLOGY TOÀN DIỆN:
    - S và T đều chuyển thành 'D' để khớp với vòng tròn Depot duy nhất trên hình vẽ.
    - Giữ nguyên TN1, TN2... (và ép viết hoa cho đẹp mắt).
    - Chuyển đổi A2 thành ADP2, A1 thành ADP1 thống nhất.
    """
    kind = _node_type(instance, node)
    upper = str(node).upper()

    if kind == "depot" or kind == "dv_terminal":
        return "D"
    if kind == "tn":
        return upper
    if kind == "adp":
        if upper.startswith("ADP"):
            suffix = upper[3:]
        elif upper.startswith("A"):
            suffix = upper[1:]
        else:
            suffix = upper
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


def _customer_color(customer_type: int) -> str:
    if customer_type == 1:
        return TYPE1_COLOR
    if customer_type == 2:
        return TYPE2_COLOR
    return TYPE3_COLOR


def _assignment_mode_map(state) -> dict[str, str]:
    result: dict[str, str] = {}

    for customer, info in getattr(state, "assignments", {}).items():
        if not isinstance(info, dict):
            continue
        result[str(customer)] = str(info.get("mode", "")).upper()

    return result


def _customer_service_edge(
    customer: str,
    assignment_modes: dict[str, str],
) -> tuple[str, float]:
    mode = assignment_modes.get(customer, "")

    if mode == "DV_HOME":
        return DV_HOME_EDGE, 2.4
    if mode == "OD_HOME":
        return OD_HOME_EDGE, 2.6
    if mode == "ADP":
        return ADP_MODE_EDGE, 2.4

    return UNASSIGNED_EDGE, 1.2


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


def _format_lambda_value(lambda_value: float | None) -> str | None:
    if lambda_value is None:
        return None
    return f"{float(lambda_value):g}"


def _extract_meta_from_title(
    title: str | None,
) -> tuple[str | None, float | None, int | None]:
    if not title:
        return None, None, None

    instance_match = re.search(
        r"([A-Za-z0-9]+(?:[_\-][A-Za-z0-9]+)+)",
        title,
        flags=re.IGNORECASE,
    )
    lambda_match = re.search(
        r"λ\s*=\s*([0-9.]+)",
        title,
        flags=re.IGNORECASE,
    )
    seed_match = re.search(
        r"seed\s*=\s*([0-9]+)",
        title,
        flags=re.IGNORECASE,
    )

    instance_id = instance_match.group(1) if instance_match else None
    lambda_value = float(lambda_match.group(1)) if lambda_match else None
    seed_value = int(seed_match.group(1)) if seed_match else None

    return instance_id, lambda_value, seed_value


def _compose_solution_title(
    *,
    instance_id: str | None,
    lambda_value: float | None,
    seed: int | None,
    title: str | None = None,
) -> str:
    if title is not None:
        if "\n" in title or "Crowd-Shipping Route Map" in title:
            return title

        legacy_instance, legacy_lambda, legacy_seed = _extract_meta_from_title(title)
        instance_id = instance_id or legacy_instance
        lambda_value = lambda_value if lambda_value is not None else legacy_lambda
        seed = seed if seed is not None else legacy_seed

    lambda_text = _format_lambda_value(lambda_value)

    if lambda_text is None:
        line_1 = "Crowd-Shipping Route Map"
    else:
        line_1 = f"Crowd-Shipping Route Map — Scenario λ = {lambda_text}"

    subtitle_parts: list[str] = []

    if instance_id is not None:
        subtitle_parts.append(str(instance_id))
    if seed is not None:
        subtitle_parts.append(f"seed = {int(seed)}")

    subtitle_parts.append("Paper-faithful ALNS best solution")
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
# Route-summary helpers
# ============================================================


def _compact_route(instance: dict, route: Iterable[str], max_nodes: int = 12) -> str:
    nodes = [_display_label(instance, node) for node in route]

    if len(nodes) <= max_nodes:
        return " → ".join(nodes)

    head_count = max_nodes // 2
    tail_count = max_nodes - head_count
    return " → ".join(nodes[:head_count] + ["…"] + nodes[-tail_count:])


def _draw_route_summary(ax, instance: dict, state) -> None:
    if not SHOW_ROUTE_SUMMARY:
        return

    children_elements = []

    # Tiêu đề 1: Active route sequences (In đậm)
    children_elements.append(
        TextArea(
            "Active route sequences",
            textprops=dict(fontweight="bold", fontsize=ROUTE_SUMMARY_FONT_SIZE),
        )
    )

    # Thêm dữ liệu lộ trình OD xe vệ tinh
    for vehicle, route in sorted(getattr(state, "od_routes", {}).items()):
        if route:
            route_str = _compact_route(instance, route, max_nodes=10)
            wrapped_lines = textwrap.wrap(route_str, width=34)
            for i, w_line in enumerate(wrapped_lines):
                lbl = f"{vehicle}: {w_line}" if i == 0 else f"    {w_line}"
                children_elements.append(
                    TextArea(lbl, textprops=dict(fontsize=ROUTE_SUMMARY_FONT_SIZE))
                )

    # Thêm dữ liệu lộ trình DV xe công ty
    for vehicle, route in sorted(getattr(state, "dv_routes", {}).items()):
        if route:
            route_str = _compact_route(instance, route, max_nodes=10)
            wrapped_lines = textwrap.wrap(route_str, width=34)
            for i, w_line in enumerate(wrapped_lines):
                lbl = f"{vehicle}: {w_line}" if i == 0 else f"    {w_line}"
                children_elements.append(
                    TextArea(lbl, textprops=dict(fontsize=ROUTE_SUMMARY_FONT_SIZE))
                )

    # Tiêu đề 2: Customer outline (In đậm) và chú thích màu sắc hình khối
    if SHOW_SERVICE_MODE_KEY:
        children_elements.append(
            TextArea("", textprops=dict(fontsize=ROUTE_SUMMARY_FONT_SIZE))
        )
        children_elements.append(
            TextArea(
                "Customer outline",
                textprops=dict(fontweight="bold", fontsize=ROUTE_SUMMARY_FONT_SIZE),
            )
        )
        children_elements.append(
            TextArea(
                "blue = DV home",
                textprops=dict(fontsize=ROUTE_SUMMARY_FONT_SIZE),
            )
        )
        children_elements.append(
            TextArea(
                "green = OD home",
                textprops=dict(fontsize=ROUTE_SUMMARY_FONT_SIZE),
            )
        )
        children_elements.append(
            TextArea(
                "gray = ADP pickup",
                textprops=dict(fontsize=ROUTE_SUMMARY_FONT_SIZE),
            )
        )

    v_box_layout = VPacker(children=children_elements, align="left", pad=0, sep=2.5)

    anchored_box = AnnotationBbox(
        v_box_layout,
        (1.025, 0.37),
        xycoords="axes fraction",
        boxcoords="axes fraction",
        box_alignment=(
            0.0,
            1.0,
        ),  # ĐÃ SỬA: Sửa chính xác thuộc tính định vị hộp nhận diện của Matplotlib
        bboxprops={
            "boxstyle": "round,pad=0.55",
            "facecolor": "white",
            "edgecolor": "0.80",
            "alpha": 0.97,
        },
        frameon=True,
        clip_on=False,
        zorder=100,
    )
    ax.add_artist(anchored_box)


# ============================================================
# Drawing helpers
# ============================================================


def _curvature_for_route(
    *,
    fleet_name: str,
    route_index: int,
) -> float:
    if fleet_name.lower() == "od":
        values = (0.16, -0.16, 0.22, -0.22)
    else:
        values = (0.055, -0.055, 0.09, -0.09, 0.12, -0.12)

    return values[route_index % len(values)]


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
    rad_val: float = 0.0,
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
        alpha=0.96,
        shrinkA=shrink_a,
        shrinkB=shrink_b,
        connectionstyle=f"arc3,rad={rad_val}",
        zorder=zorder,
    )
    ax.add_patch(arrow)


def _sequence_label_position(
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    rad_val: float,
) -> tuple[float, float]:
    x1, y1 = start_xy
    x2, y2 = end_xy
    mx = (x1 + x2) / 2.0
    my = (y1 + y2) / 2.0

    dx = x2 - x1
    dy = y2 - y1
    norm = max((dx * dx + dy * dy) ** 0.5, 1e-9)

    offset = 0.8 + 3.2 * abs(rad_val)
    direction = 1.0 if rad_val >= 0 else -1.0

    return (
        mx + direction * (-dy / norm) * offset,
        my + direction * (dx / norm) * offset,
    )


def _draw_route(
    ax,
    *,
    route: Iterable[str],
    coordinates: dict[str, tuple[float, float]],
    color: str,
    linestyle: str,
    linewidth: float,
    fleet_name: str,
    route_index: int,
    annotate_sequence: bool,
) -> None:
    """Vẽ tuyến đường của phương tiện (Không nhận tham số instance thừa)."""
    valid_nodes = [node for node in route if node in coordinates]

    if len(valid_nodes) < 2:
        return

    rad_val = _curvature_for_route(
        fleet_name=fleet_name,
        route_index=route_index,
    )

    for segment_index, (origin, destination) in enumerate(
        zip(valid_nodes, valid_nodes[1:]),
        start=1,
    ):
        start_xy = coordinates[origin]
        end_xy = coordinates[destination]

        _draw_route_arrow(
            ax,
            start_xy,
            end_xy,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            rad_val=rad_val,
            zorder=5 if fleet_name.lower() == "od" else 4,
        )

        if annotate_sequence:
            label_xy = _sequence_label_position(
                start_xy,
                end_xy,
                rad_val,
            )
            ax.text(
                label_xy[0],
                label_xy[1],
                str(segment_index),
                ha="center",
                va="center",
                fontsize=SEQUENCE_FONT_SIZE,
                fontweight="bold",
                color=color,
                bbox={
                    "boxstyle": "circle,pad=0.20",
                    "facecolor": "white",
                    "edgecolor": color,
                    "linewidth": 1.0,
                    "alpha": 0.96,
                },
                clip_on=False,
                zorder=20,
            )


def _draw_adp_assignments(
    ax,
    *,
    instance: dict,
    state,
    coordinates: dict[str, tuple[float, float]],
    vehicle_filter: str | None = None,
) -> None:
    for customer, assignment in sorted(getattr(state, "assignments", {}).items()):
        if not isinstance(assignment, dict):
            continue
        if str(assignment.get("mode", "")).upper() != "ADP":
            continue
        if vehicle_filter is not None and assignment.get("vehicle") != vehicle_filter:
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
            linewidth=ASSIGNMENT_LINE_WIDTH,
            alpha=ASSIGNMENT_LINE_ALPHA,
            zorder=1,
        )

        if SHOW_ASSIGNMENT_EDGE_LABELS:
            midpoint = (
                (customer_xy[0] + adp_xy[0]) / 2.0,
                (customer_xy[1] + adp_xy[1]) / 2.0,
            )
            ax.text(
                midpoint[0],
                midpoint[1],
                f"{customer} → {_display_label(instance, str(adp))}",
                fontsize=6.8,
                color="0.50",
                ha="center",
                va="center",
                zorder=2,
            )


def _node_marker_style(
    *,
    instance: dict,
    node: str,
    active_adps: set[str],
    active_tns: set[str],
    assignment_modes: dict[str, str],
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
                "size": 760,
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
                    "alpha": 0.65,
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
                    "size": 680,
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
                    "alpha": 0.65,
                    "zorder": 9,
                }
            )
    elif kind == "od_origin":
        style.update(
            {
                "marker": "^",
                "face_color": OD_ORIGIN_FACE,
                "edge_color": OD_ENDPOINT_EDGE,
                "size": 760,
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
                "size": 760,
                "text_color": OD_ENDPOINT_EDGE,
                "zorder": 10,
            }
        )
    elif kind == "customer":
        c_type = _customer_type(instance, node)
        edge_color, edge_width = _customer_service_edge(
            node,
            assignment_modes,
        )
        style.update(
            {
                "marker": "o",
                "face_color": _customer_color(c_type),
                "edge_color": edge_color,
                "size": 575,
                "linewidth": edge_width,
                "text_color": "black",
                "zorder": 11,
            }
        )

    return style


def _draw_nodes(
    ax,
    *,
    instance: dict,
    state,
    coordinates: dict[str, tuple[float, float]],
    active_adps: set[str],
    active_tns: set[str],
) -> None:
    assignment_modes = _assignment_mode_map(state)
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
            assignment_modes=assignment_modes,
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
            fontsize=NODE_LABEL_FONT_SIZE,
            fontweight="bold",
            color=style["text_color"],
            clip_on=False,
            zorder=style["zorder"] + 2,
        )

    for node, (x, y) in depot_nodes:
        style = _node_marker_style(
            instance=instance,
            node=node,
            active_adps=active_adps,
            active_tns=active_tns,
            assignment_modes=assignment_modes,
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

        ax.text(
            x,
            y,
            _display_label(instance, node),
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
            linewidth=ASSIGNMENT_LINE_WIDTH,
            linestyle=":",
            alpha=ASSIGNMENT_LINE_ALPHA,
            label="Customer-to-ADP assignment",
        ),
        Line2D(
            [0],
            [0],
            color=DV_ROUTE_COLOR,
            linewidth=DV_ROUTE_LINE_WIDTH,
            linestyle="-",
            label="Route of Dedicated Vehicles",
        ),
        Line2D(
            [0],
            [0],
            color=OD_ROUTE_COLOR,
            linewidth=OD_ROUTE_LINE_WIDTH,
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
    active_adps, active_tns = _active_infrastructure(instance, state)

    fig, ax = plt.subplots(figsize=(19.0, 11.2))

    _draw_adp_assignments(
        ax,
        instance=instance,
        state=state,
        coordinates=coordinates,
    )

    for route_index, (_, route) in enumerate(
        sorted(getattr(state, "dv_routes", {}).items())
    ):
        if route:
            _draw_route(
                ax,
                route=route,
                coordinates=coordinates,
                color=DV_ROUTE_COLOR,
                linestyle="-",
                linewidth=DV_ROUTE_LINE_WIDTH,
                fleet_name="dv",
                route_index=route_index,
                annotate_sequence=annotate_sequence,
            )

    for route_index, (_, route) in enumerate(
        sorted(getattr(state, "od_routes", {}).items())
    ):
        if route:
            _draw_route(
                ax,
                route=route,
                coordinates=coordinates,
                color=OD_ROUTE_COLOR,
                linestyle="--",
                linewidth=OD_ROUTE_LINE_WIDTH,
                fleet_name="od",
                route_index=route_index,
                annotate_sequence=(annotate_sequence or SHOW_OD_SEQUENCE_ON_COMBINED),
            )

    _draw_nodes(
        ax,
        instance=instance,
        state=state,
        coordinates=coordinates,
        active_adps=active_adps,
        active_tns=active_tns,
    )

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
    ax.grid(True, linestyle=":", alpha=0.28)
    ax.set_aspect("equal", adjustable="datalim")

    ax.legend(
        handles=_legend_handles(),
        title="Terminology",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.98),
        frameon=True,
        fontsize=8.6,
        title_fontsize=10.5,
    )
    _draw_route_summary(ax, instance, state)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
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
    """Draw one clear route map per active DV/OD route."""
    coordinates = _node_coordinates(instance)
    active_adps, active_tns = _active_infrastructure(instance, state)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []

    route_groups = (
        (
            "dv",
            getattr(state, "dv_routes", {}),
            DV_ROUTE_COLOR,
            "-",
            DV_ROUTE_LINE_WIDTH,
        ),
        (
            "od",
            getattr(state, "od_routes", {}),
            OD_ROUTE_COLOR,
            "--",
            OD_ROUTE_LINE_WIDTH,
        ),
    )

    for fleet_name, routes, color, linestyle, linewidth in route_groups:
        active_items = [
            (vehicle, route) for vehicle, route in sorted(routes.items()) if route
        ]

        for route_index, (vehicle, route) in enumerate(active_items):
            fig, ax = plt.subplots(figsize=(15.5, 9.4))

            if fleet_name == "dv":
                _draw_adp_assignments(
                    ax,
                    instance=instance,
                    state=state,
                    coordinates=coordinates,
                    vehicle_filter=vehicle,
                )

            _draw_route(
                ax,
                route=route,
                coordinates=coordinates,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                fleet_name=fleet_name,
                route_index=route_index,
                annotate_sequence=SHOW_SEQUENCE_ON_VEHICLE_MAPS,
            )

            _draw_nodes(
                ax,
                instance=instance,
                state=state,
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
            ax.grid(True, linestyle=":", alpha=0.28)
            ax.set_aspect("equal", adjustable="datalim")

            ax.legend(
                handles=_legend_handles(),
                title="Terminology",
                loc="upper left",
                bbox_to_anchor=(1.02, 0.98),
                frameon=True,
                fontsize=8,
                title_fontsize=9.5,
            )

            ax.text(
                1.025,
                0.37,
                f"Route sequence\n{vehicle}: {_compact_route(instance, route, 14)}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.3,
                linespacing=1.35,
                bbox={
                    "boxstyle": "round,pad=0.50",
                    "facecolor": "white",
                    "edgecolor": "0.80",
                    "alpha": 0.97,
                },
                clip_on=False,
                zorder=100,
            )

            fig.tight_layout()
            path = output_dir / f"{fleet_name}_{vehicle}_route.png"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            generated.append(path)

    return generated
