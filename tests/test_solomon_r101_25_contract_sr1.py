from __future__ import annotations

from pathlib import Path
import json
import shutil

from core.instance_loader import load_instance
from data_generation.solomon_extension_generator import (
    ExtensionConfig,
    generate_solomon_extension,
)
from data_generation.solomon_parser import (
    parse_solomon_instance,
    save_solomon_json,
)


def main() -> None:
    project_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    source = (
        project_root
        / "data"
        / "solomon"
        / "raw"
        / "solomon-100"
        / "In"
        / "r101.txt"
    )

    parsed_output = (
        project_root
        / "data"
        / "solomon"
        / "derived"
        / "parsed"
        / "r101_25.json"
    )

    output = (
        project_root
        / "data"
        / "solomon"
        / "derived"
        / "r101_25"
    )

    solomon = parse_solomon_instance(
        source,
        customer_limit=25,
    )

    assert solomon.name.upper() == "R101"
    assert len(solomon.customers) == 25
    assert solomon.source_file == (
        "data/solomon/raw/"
        "solomon-100/In/r101.txt"
    )

    save_solomon_json(
        solomon,
        parsed_output,
    )

    if output.exists():
        shutil.rmtree(output)

    generate_solomon_extension(
        source,
        output,
        config=ExtensionConfig(
            customer_limit=25,
            generation_seed=2026,
        ),
    )

    instance = load_instance(output)

    required_files = {
        "nodes.csv",
        "vehicles.csv",
        "adp_compatibility.csv",
        "parameters.csv",
        "metadata.json",
    }

    actual_files = {
        path.name
        for path in output.iterdir()
        if path.is_file()
    }

    assert actual_files == required_files
    assert parsed_output.exists()
    assert parsed_output.stat().st_size > 0

    assert len(instance["customers"]) == 25
    assert len(instance["type1"]) == 9
    assert len(instance["type2"]) == 8
    assert len(instance["type3"]) == 8

    all_customers = set(
        instance["customers"]
    )

    all_typed = (
        set(instance["type1"])
        | set(instance["type2"])
        | set(instance["type3"])
    )

    assert all_typed == all_customers

    assert not (
        set(instance["type1"])
        & set(instance["type2"])
    )
    assert not (
        set(instance["type1"])
        & set(instance["type3"])
    )
    assert not (
        set(instance["type2"])
        & set(instance["type3"])
    )

    assert len(instance["adps"]) == 2
    assert len(instance["tns"]) == 1
    assert len(instance["ods"]) == 2
    assert len(instance["dvs"]) >= 2

    assert instance["start_depot"] == "S"
    assert instance["end_depot"] == "T"

    assert instance["pickup_points"] == [
        "S",
        "TN1",
    ]

    eligible = (
        set(instance["type2"])
        | set(instance["type3"])
    )

    for customer in eligible:
        compatible_count = sum(
            instance["gamma"].get(
                (customer, adp),
                0,
            )
            for adp in instance["adps"]
        )

        assert compatible_count == 1, (
            customer,
            compatible_count,
        )

    for customer in instance["type1"]:
        assert all(
            (customer, adp)
            not in instance["gamma"]
            for adp in instance["adps"]
        )

    depot_node = instance["nodes"][
        instance["start_depot"]
    ]

    for customer in instance["customers"]:
        node = instance["nodes"][customer]

        assert node["demand"] == 1.0
        assert (
            node["tw_start"]
            == depot_node["tw_start"]
        )
        assert (
            node["tw_end"]
            == depot_node["tw_end"]
        )

    first_native = instance["nodes"]["C1"]
    last_native = instance["nodes"]["C25"]

    assert first_native["x"] == 41.0
    assert first_native["y"] == 49.0

    assert last_native["x"] == 65.0
    assert last_native["y"] == 20.0

    metadata = json.loads(
        (output / "metadata.json").read_text(
            encoding="utf-8"
        )
    )

    assert (
        metadata["instance_id"]
        == "r101_25_derived"
    )
    assert metadata["source"] == "Solomon VRPTW"
    assert metadata["source_instance"] == "R101"
    assert metadata["source_file"] == (
        "data/solomon/raw/"
        "solomon-100/In/r101.txt"
    )
    assert metadata["customer_count"] == 25
    assert metadata["generation_seed"] == 2026

    assert (
        metadata["generator_version"]
        == "solomon_extension_v1"
    )

    assert (
        metadata["benchmark_classification"]
        == "Solomon-spatially-derived CSPTW-TN-DO"
    )

    assert (
        metadata["demand_transformation"][
            "native_rule"
        ]
        == "one parcel unit per customer"
    )

    assert (
        metadata["time_window_transformation"][
            "native_rule"
        ]
        == "common depot operating horizon"
    )

    assert (
        metadata["time_window_transformation"][
            "horizon_start"
        ]
        == 0.0
    )

    assert (
        metadata["time_window_transformation"][
            "horizon_end"
        ]
        == 230.0
    )

    print(
        "[PASS] R101-25 native instance files are generated"
    )
    print(
        "[PASS] Parsed R101-25 provenance JSON is exported"
    )
    print(
        "[PASS] Repository loader accepts the derived instance"
    )
    print(
        "[PASS] R101 coordinates and customer ordering are preserved"
    )
    print(
        "[PASS] Unit-demand transformation is applied"
    )
    print(
        "[PASS] Common depot-horizon transformation is applied"
    )
    print(
        "[PASS] Customer types are deterministic and exhaustive"
    )
    print(
        "[PASS] ADP compatibility is valid for Type 2/3 customers"
    )
    print(
        "[PASS] DV, OD, TN, ADP, depot, and pickup contracts are present"
    )
    print(
        "[PASS] Relative Solomon provenance is preserved"
    )
    print(
        "\nSR-1 — R101-25 STRUCTURAL CONTRACT PASSED"
    )


if __name__ == "__main__":
    main()
