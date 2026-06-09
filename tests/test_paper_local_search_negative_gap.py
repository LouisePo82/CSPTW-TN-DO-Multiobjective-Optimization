from alns_solver.paper_local_search import (
    paper_local_search_eligible,
)


def main() -> None:
    delta = 0.1

    # Original non-negative paper behavior.
    assert paper_local_search_eligible(
        110.0,
        100.0,
        delta_ls=delta,
    )

    assert not paper_local_search_eligible(
        110.1,
        100.0,
        delta_ls=delta,
    )

    # Sign-safe extension for normalized objectives.
    best = -0.25911607831262573

    assert paper_local_search_eligible(
        best,
        best,
        delta_ls=delta,
    )

    assert paper_local_search_eligible(
        -0.24,
        best,
        delta_ls=delta,
    )

    assert not paper_local_search_eligible(
        -0.22,
        best,
        delta_ls=delta,
    )

    print(
        "[PASS] Positive paper gap behavior preserved"
    )
    print(
        "[PASS] Negative normalized objective supported"
    )


if __name__ == "__main__":
    main()
