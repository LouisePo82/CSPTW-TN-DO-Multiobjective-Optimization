from pathlib import Path

from data_generation.solomon_parser import (
    parse_solomon_instance,
    save_solomon_json,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]

    c101_path = (
        root
        / "data"
        / "solomon"
        / "raw"
        / "solomon-100"
        / "In"
        / "c101.txt"
    )

    r101_path = (
        root
        / "data"
        / "solomon"
        / "raw"
        / "solomon-100"
        / "In"
        / "r101.txt"
    )

    c101_full = parse_solomon_instance(
        c101_path
    )
    r101_full = parse_solomon_instance(
        r101_path
    )

    assert c101_full.name == "C101"
    assert r101_full.name == "R101"

    assert c101_full.vehicle_count == 25
    assert c101_full.vehicle_capacity == 200.0

    assert len(c101_full.customers) == 100
    assert len(r101_full.customers) == 100

    assert c101_full.depot.node_id == 0
    assert c101_full.depot.x == 40.0
    assert c101_full.depot.y == 50.0

    c101_25 = parse_solomon_instance(
        c101_path,
        customer_limit=25,
    )
    c101_50 = parse_solomon_instance(
        c101_path,
        customer_limit=50,
    )

    assert len(c101_25.customers) == 25
    assert len(c101_50.customers) == 50

    assert [
        customer.node_id
        for customer in c101_25.customers
    ] == list(range(1, 26))

    assert [
        customer.node_id
        for customer in c101_50.customers
    ] == list(range(1, 51))

    output_path = save_solomon_json(
        c101_25,
        (
            root
            / "data"
            / "solomon"
            / "derived"
            / "parsed"
            / "c101_25.json"
        ),
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    print(
        "[PASS] C101 and R101 Solomon files parse successfully"
    )
    print(
        "[PASS] Depot and vehicle parameters are recovered"
    )
    print(
        "[PASS] Full source contains 100 customers"
    )
    print(
        "[PASS] Customer limits 25 and 50 preserve source ordering"
    )
    print(
        "[PASS] Parsed C101-25 JSON is exported"
    )
    print(
        "\nSP-1 — SOLOMON PARSER PASSED"
    )


if __name__ == "__main__":
    main()
