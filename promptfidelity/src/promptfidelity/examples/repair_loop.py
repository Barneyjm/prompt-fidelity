"""
promptfidelity.examples.repair_loop -- mid-run self-inspection and repair,
end to end.

Self-contained: stdlib only, no network, no API keys. `discover()` is a fake
in-memory movie "backend" that does no real filtering -- this script only
cares about which ARGUMENTS reach it, since that's all book() ever looks at.

Narrative:
    1. Declare constraints, including one a naive first call will drop and
       one it will substitute.
    2. Make that first (imperfect) tool call, then check() the ledger and
       render it for audience="model" -- the text an agent loop would
       inject into its own context.
    3. A scripted "model repair": re-issue ONE complete, corrected call.
    4. check() again: nothing left unhonored, and conjunction_honored is
       True -- one call, not a patchwork of calls, satisfied everything.
    5. Print all four audience renderings of the final ledger.

Run directly:
    python -m promptfidelity.examples.repair_loop
"""

import promptfidelity as pf


@pf.instrument
def discover(**kwargs):
    """Fake backend: returns one canned result, args and all. No real
    filtering happens here -- @pf.instrument records the kwargs it was
    called with, which is the only thing book() ever diffs against."""
    return [{"title": "Solaris (1972)", "called_with": dict(kwargs)}]


def main() -> None:
    constraints = [
        pf.Constraint(id="genre", description="science fiction genre",
                      params={"with_genres": "878"}, p=0.08),
        pf.Constraint(id="rating", description="rated above 7",
                      params={"vote_average.gte": "7.0"}, p=0.1),
        pf.Constraint(id="pacing", description="slow, meditative pacing", p=0.1),
    ]

    print("1. Constraints declared:")
    for c in constraints:
        print(f"   - {c.id}: {c.description}  params={c.params or '<not expressible>'}")

    with pf.trace(constraints) as rec:
        print("\n2. First tool call (naive first attempt):")
        first_args = {"vote_average.gte": "6.5", "popularity.gte": "50"}
        print(f"   discover(**{first_args!r})")
        discover(**first_args)

        ledger = rec.check()
        print("\n   check() -> render('model')  [would be injected into the agent's context]:")
        for line in ledger.render("model").splitlines():
            print(f"   | {line}")
        assert ledger.unhonored, "expected genre (dropped) and rating (substituted)"

        print("\n3. Scripted model repair -- ONE complete corrective call:")
        repair_args = {"with_genres": "878", "vote_average.gte": "7.0"}
        print(f"   discover(**{repair_args!r})")
        discover(**repair_args)

    print("\n4. check() again:")
    final_ledger = rec.check()
    print(f"   unhonored: {final_ledger.unhonored}")
    print(f"   conjunction_honored: {final_ledger.conjunction_honored}")
    assert not final_ledger.unhonored, "the complete repair call should honor everything"
    assert final_ledger.conjunction_honored, (
        "one complete call verified genre+rating together -- no piecewise repair here"
    )

    print("\n5. Final ledger, rendered for all four audiences:")
    for audience in ("model", "engineer", "product", "executive"):
        print(f"\n--- {audience} ---")
        rendering = final_ledger.render(audience)
        print(rendering if rendering else "(empty string -- nothing left to inject)")


if __name__ == "__main__":
    main()
