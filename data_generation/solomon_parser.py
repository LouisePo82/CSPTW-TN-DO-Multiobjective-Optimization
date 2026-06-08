from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json
import re


@dataclass(frozen=True)
class SolomonNode:
    node_id: int
    x: float
    y: float
    demand: float
    ready_time: float
    due_time: float
    service_time: float


@dataclass(frozen=True)
class SolomonInstance:
    name: str
    vehicle_count: int
    vehicle_capacity: float
    depot: SolomonNode
    customers: tuple[SolomonNode, ...]
    source_file: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "vehicle_count": self.vehicle_count,
            "vehicle_capacity": self.vehicle_capacity,
            "depot": asdict(self.depot),
            "customers": [
                asdict(customer)
                for customer in self.customers
            ],
            "customer_count": len(self.customers),
            "source_file": self.source_file,
        }


def _numeric_rows(
    lines: list[str],
) -> list[list[float]]:
    rows: list[list[float]] = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            continue

        parts = stripped.split()

        if len(parts) < 7:
            continue

        if not re.fullmatch(
            r"[+-]?\d+",
            parts[0],
        ):
            continue

        try:
            values = [
                float(value)
                for value in parts[:7]
            ]
        except ValueError:
            continue

        rows.append(values)

    return rows


def _vehicle_parameters(
    lines: list[str],
) -> tuple[int, float]:
    for index, line in enumerate(lines):
        if line.strip().upper() != "VEHICLE":
            continue

        for candidate in lines[
            index + 1:index + 6
        ]:
            parts = candidate.split()

            if len(parts) < 2:
                continue

            try:
                vehicle_count = int(
                    float(parts[0])
                )
                vehicle_capacity = float(
                    parts[1]
                )
            except ValueError:
                continue

            return (
                vehicle_count,
                vehicle_capacity,
            )

    raise ValueError(
        "Could not find Solomon vehicle parameters."
    )


def parse_solomon_instance(
    source_file: str | Path,
    *,
    customer_limit: int | None = None,
) -> SolomonInstance:
    path = Path(source_file)

    if not path.exists():
        raise FileNotFoundError(
            f"Solomon source file not found: {path}"
        )

    lines = path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()

    if not lines:
        raise ValueError(
            f"Solomon source file is empty: {path}"
        )

    name = lines[0].strip().upper()

    vehicle_count, vehicle_capacity = (
        _vehicle_parameters(lines)
    )

    rows = _numeric_rows(lines)

    if len(rows) < 2:
        raise ValueError(
            "Expected at least one depot and "
            "one customer row."
        )

    nodes = tuple(
        SolomonNode(
            node_id=int(row[0]),
            x=float(row[1]),
            y=float(row[2]),
            demand=float(row[3]),
            ready_time=float(row[4]),
            due_time=float(row[5]),
            service_time=float(row[6]),
        )
        for row in rows
    )

    depot_candidates = [
        node
        for node in nodes
        if node.node_id == 0
    ]

    if len(depot_candidates) != 1:
        raise ValueError(
            "Expected exactly one depot with node_id 0."
        )

    depot = depot_candidates[0]

    customers = tuple(
        node
        for node in nodes
        if node.node_id != 0
    )

    if customer_limit is not None:
        if customer_limit <= 0:
            raise ValueError(
                "customer_limit must be positive."
            )

        if customer_limit > len(customers):
            raise ValueError(
                f"Requested {customer_limit} customers, "
                f"but source contains only "
                f"{len(customers)}."
            )

        customers = customers[
            :customer_limit
        ]

    return SolomonInstance(
        name=name,
        vehicle_count=vehicle_count,
        vehicle_capacity=vehicle_capacity,
        depot=depot,
        customers=customers,
        source_file=(
            "data/solomon/raw/solomon-100/In/"
            f"{path.name}"
        ),
    )


def save_solomon_json(
    instance: SolomonInstance,
    output_file: str | Path,
) -> Path:
    path = Path(output_file)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            instance.to_dict(),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return path
