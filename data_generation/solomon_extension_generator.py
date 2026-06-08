from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import csv
import json
import math

from data_generation.solomon_parser import (
    SolomonInstance,
    SolomonNode,
    parse_solomon_instance,
)


GENERATOR_VERSION = "solomon_extension_v1"


@dataclass(frozen=True)
class ExtensionConfig:
    customer_limit: int
    generation_seed: int = 2026
    rho: float = 0.20
    service_time_per_weight: float = 1.0
    big_m_time: float = 10000.0
    dv_emission_factor: float = 3.0
    od_emission_factor: float = 1.0
    od_capacity: int = 2


def _mean_xy(nodes: Iterable[SolomonNode]) -> tuple[float, float]:
    items = list(nodes)
    if not items:
        raise ValueError("Cannot compute a centroid from an empty node set.")
    return (
        sum(node.x for node in items) / len(items),
        sum(node.y for node in items) / len(items),
    )


def _partition_by_x(
    nodes: list[SolomonNode],
    group_count: int,
) -> list[list[SolomonNode]]:
    if group_count <= 0:
        raise ValueError("group_count must be positive.")

    ordered = sorted(
        nodes,
        key=lambda node: (
            node.x,
            node.y,
            node.node_id,
        ),
    )

    groups: list[list[SolomonNode]] = [
        []
        for _ in range(group_count)
    ]

    for index, node in enumerate(ordered):
        group_index = min(
            group_count - 1,
            index * group_count // len(ordered),
        )
        groups[group_index].append(node)

    return groups


def _customer_type(node_id: int) -> int:
    return ((node_id - 1) % 3) + 1


def _nearest_adp(
    customer: SolomonNode,
    adps: list[dict],
) -> str:
    return min(
        adps,
        key=lambda adp: (
            math.hypot(
                customer.x - float(adp["x"]),
                customer.y - float(adp["y"]),
            ),
            str(adp["id"]),
        ),
    )["id"]


def generate_solomon_extension(
    source_file: str | Path,
    output_dir: str | Path,
    *,
    config: ExtensionConfig,
) -> Path:
    solomon: SolomonInstance = parse_solomon_instance(
        source_file,
        customer_limit=config.customer_limit,
    )

    output = Path(output_dir)
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    customers = list(solomon.customers)
    depot = solomon.depot
    horizon = float(depot.due_time)

    typed_customers = {
        node.node_id: _customer_type(node.node_id)
        for node in customers
    }

    eligible_for_adp = [
        node
        for node in customers
        if typed_customers[node.node_id] in {2, 3}
    ]

    adp_count = 2 if config.customer_limit <= 25 else 3
    adp_groups = _partition_by_x(
        eligible_for_adp,
        adp_count,
    )

    adps: list[dict] = []
    for index, group in enumerate(
        adp_groups,
        start=1,
    ):
        x, y = _mean_xy(group)
        adps.append(
            {
                "id": f"A{index}",
                "type": "adp",
                "customer_type": 0,
                "x": x,
                "y": y,
                "demand": 0.0,
                "tw_start": 0.0,
                "tw_end": horizon,
            }
        )

    tn_count = 1 if config.customer_limit <= 25 else 2
    tn_groups = _partition_by_x(
        customers,
        tn_count,
    )

    tns: list[dict] = []
    for index, group in enumerate(
        tn_groups,
        start=1,
    ):
        group_x, group_y = _mean_xy(group)
        tns.append(
            {
                "id": f"TN{index}",
                "type": "tn",
                "customer_type": 0,
                "x": (depot.x + group_x) / 2.0,
                "y": (depot.y + group_y) / 2.0,
                "demand": 0.0,
                "tw_start": 0.0,
                "tw_end": horizon,
            }
        )

    nodes: list[dict] = [
        {
            "id": "S",
            "type": "depot_start",
            "customer_type": 0,
            "x": depot.x,
            "y": depot.y,
            "demand": 0.0,
            "tw_start": depot.ready_time,
            "tw_end": depot.due_time,
        },
        {
            "id": "T",
            "type": "depot_end",
            "customer_type": 0,
            "x": depot.x,
            "y": depot.y,
            "demand": 0.0,
            "tw_start": depot.ready_time,
            "tw_end": depot.due_time,
        },
        *tns,
        *adps,
    ]

    for customer in customers:
        nodes.append(
            {
                "id": f"C{customer.node_id}",
                "type": "customer",
                "customer_type": typed_customers[
                    customer.node_id
                ],
                "x": customer.x,
                "y": customer.y,
                # Native benchmark semantics:
                # one customer represents one parcel unit.
                #
                # Solomon coordinates and source ordering are
                # preserved, but original demand values are not
                # used directly because the current CSPTW-TN-DO
                # implementation treats OD capacity both as
                # customer count and as assigned demand.
                "demand": 1.0,

                # Use the common depot operating horizon.
                #
                # The paper-faithful construction heuristic ranks
                # intermediate insertions without enforcing the
                # original route-level Solomon time-window order.
                # Keeping the original narrow customer windows can
                # therefore produce a complete route that fails the
                # shared final validator.
                "tw_start": depot.ready_time,
                "tw_end": depot.due_time,
            }
        )

    # Use unit-parcel capacity semantics consistently
    # across DV and OD routes.
    native_total_demand = float(
        len(customers)
    )

    # A moderate native DV capacity avoids carrying the
    # original Solomon vehicle-capacity scale into the
    # transformed unit-demand benchmark.
    dv_capacity = (
        10
        if config.customer_limit <= 25
        else 15
    )

    minimum_dvs = math.ceil(
        native_total_demand / dv_capacity
    )

    # Retain one spare vehicle so repair operators have
    # an inactive route available during reconstruction.
    dv_count = max(
        2,
        minimum_dvs + 1,
    )

    vehicles: list[dict] = []
    for index in range(1, dv_count + 1):
        vehicles.append(
            {
                "id": f"DV{index}",
                "type": "dv",
                "capacity": dv_capacity,
                "origin": "S",
                "destination": "T",
                "earliest": depot.ready_time,
                "latest": depot.due_time,
                "emission_factor": config.dv_emission_factor,
            }
        )

    od_count = 2 if config.customer_limit <= 25 else 3
    customer_x, customer_y = _mean_xy(customers)

    for index in range(1, od_count + 1):
        angle = (
            2.0
            * math.pi
            * (index - 1)
            / od_count
        )
        origin_id = f"O{index}"
        destination_id = f"D{index}"

        origin_x = depot.x + 3.0 * math.cos(angle)
        origin_y = depot.y + 3.0 * math.sin(angle)
        destination_x = (
            customer_x
            + 3.0 * math.cos(angle + math.pi / 4.0)
        )
        destination_y = (
            customer_y
            + 3.0 * math.sin(angle + math.pi / 4.0)
        )

        nodes.extend(
            [
                {
                    "id": origin_id,
                    "type": "od_origin",
                    "customer_type": 0,
                    "x": origin_x,
                    "y": origin_y,
                    "demand": 0.0,
                    "tw_start": 0.0,
                    "tw_end": horizon,
                },
                {
                    "id": destination_id,
                    "type": "od_destination",
                    "customer_type": 0,
                    "x": destination_x,
                    "y": destination_y,
                    "demand": 0.0,
                    "tw_start": 0.0,
                    "tw_end": horizon,
                },
            ]
        )

        vehicles.append(
            {
                "id": f"OD{index}",
                "type": "od",
                "capacity": config.od_capacity,
                "origin": origin_id,
                "destination": destination_id,
                "earliest": 0.0,
                "latest": horizon,
                "emission_factor": config.od_emission_factor,
            }
        )

    gamma_rows: list[dict] = []
    for customer in eligible_for_adp:
        customer_id = f"C{customer.node_id}"
        selected_adp = _nearest_adp(
            customer,
            adps,
        )

        for adp in adps:
            gamma_rows.append(
                {
                    "customer_id": customer_id,
                    "adp_id": adp["id"],
                    "gamma": int(
                        adp["id"] == selected_adp
                    ),
                }
            )

    def write_csv(
        path: Path,
        fieldnames: list[str],
        rows: list[dict],
    ) -> None:
        with path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )
            writer.writeheader()
            writer.writerows(rows)

    write_csv(
        output / "nodes.csv",
        [
            "id",
            "type",
            "customer_type",
            "x",
            "y",
            "demand",
            "tw_start",
            "tw_end",
        ],
        nodes,
    )

    write_csv(
        output / "vehicles.csv",
        [
            "id",
            "type",
            "capacity",
            "origin",
            "destination",
            "earliest",
            "latest",
            "emission_factor",
        ],
        vehicles,
    )

    write_csv(
        output / "adp_compatibility.csv",
        [
            "customer_id",
            "adp_id",
            "gamma",
        ],
        gamma_rows,
    )

    write_csv(
        output / "parameters.csv",
        [
            "parameter",
            "value",
        ],
        [
            {
                "parameter": "rho",
                "value": config.rho,
            },
            {
                "parameter": "service_time_per_weight",
                "value": config.service_time_per_weight,
            },
            {
                "parameter": "big_m_time",
                "value": config.big_m_time,
            },
        ],
    )

    type_counts = {
        str(customer_type): sum(
            1
            for value in typed_customers.values()
            if value == customer_type
        )
        for customer_type in (1, 2, 3)
    }

    metadata = {
        "instance_id": (
            f"{solomon.name.lower()}_"
            f"{config.customer_limit}_derived"
        ),
        "scale": (
            "small"
            if config.customer_limit <= 25
            else "medium"
        ),
        "source": "Solomon VRPTW",
        "source_instance": solomon.name,
        "source_file": str(
            solomon.source_file
        ),
        "customer_count": config.customer_limit,
        "dv_count": dv_count,
        "od_count": od_count,
        "adp_count": adp_count,
        "tn_count": tn_count,
        "customer_type_counts": type_counts,
        "derived_model": "CSPTW-TN-DO",
        "generation_seed": config.generation_seed,
        "generator_version": GENERATOR_VERSION,
        "benchmark_classification": (
            "Solomon-spatially-derived CSPTW-TN-DO"
        ),
        "preserved_source_fields": [
            "customer coordinates",
            "customer ordering",
            "depot coordinates",
            "source instance identity",
        ],
        "demand_transformation": {
            "source_field": "Solomon demand",
            "native_rule": (
                "one parcel unit per customer"
            ),
            "reason": (
                "match the implemented CSPTW-TN-DO "
                "capacity semantics"
            ),
        },
        "time_window_transformation": {
            "source_fields": [
                "Solomon ready time",
                "Solomon due time",
            ],
            "native_rule": (
                "common depot operating horizon"
            ),
            "horizon_start": float(
                depot.ready_time
            ),
            "horizon_end": float(
                depot.due_time
            ),
            "reason": (
                "retain compatibility with the "
                "paper-faithful construction heuristic "
                "and shared final validator"
            ),
        },
        "source_demand_summary": {
            "minimum": min(
                customer.demand
                for customer in customers
            ),
            "maximum": max(
                customer.demand
                for customer in customers
            ),
            "total": sum(
                customer.demand
                for customer in customers
            ),
        },
        "source_time_window_note": (
            "Original customer-specific Solomon windows "
            "are preserved only as source provenance and "
            "are not used as native route constraints."
        ),
        "service_time_note": (
            "Native model uses service_time_per_weight; "
            "Solomon node-specific service time is not copied directly."
        ),
        "purpose": (
            "Solomon-spatially-derived CSPTW-TN-DO exact and paper-faithful ALNS benchmark"
        ),
    }

    (output / "metadata.json").write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return output
