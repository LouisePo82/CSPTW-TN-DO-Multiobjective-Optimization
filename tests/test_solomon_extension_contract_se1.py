from __future__ import annotations

from pathlib import Path
import json
import shutil

from core.instance_loader import load_instance
from data_generation.solomon_extension_generator import (
    ExtensionConfig,
    generate_solomon_extension,
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
        / "c101.txt"
    )

    output = (
        project_root
        / "data"
        / "solomon"
        / "derived"
        / "c101_25"
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

    assert {
        path.name
        for path in output.iterdir()
        if path.is_file()
    } == required_files

    assert len(instance["customers"]) == 25
    assert len(instance["type1"]) == 9
    assert len(instance["type2"]) == 8
    assert len(instance["type3"]) == 8

    all_customers = set(instance["customers"])
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

    assert instance["start_depot"] == "S"
    assert instance["end_depot"] == "T"
    assert instance["pickup_points"] == [
        "S",
        "TN1",
    ]

    metadata = json.loads(
        (output / "metadata.json").read_text(
            encoding="utf-8"
        )
    )

    assert metadata["source"] == "Solomon VRPTW"
    assert metadata["source_instance"] == "C101"
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

    print(
        "[PASS] C101-25 native instance files are generated"
    )
    print(
        "[PASS] Repository loader accepts the derived instance"
    )
    print(
        "[PASS] Customer types are deterministic and exhaustive"
    )
    print(
        "[PASS] ADP compatibility gives one eligible ADP per Type 2/3 customer"
    )
    print(
        "[PASS] DV, OD, TN, ADP, depot, and pickup-point contracts are present"
    )
    print(
        "[PASS] Solomon provenance metadata is preserved"
    )
    print(
        "\nSE-1 — SOLOMON EXTENSION STRUCTURAL CONTRACT PASSED"
    )


if __name__ == "__main__":
    main()
