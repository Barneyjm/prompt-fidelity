"""
Fleet simulation: 10,000 agent runs booked through the REAL promptfidelity
package. Zero LLM calls -- prompts are drawn from an archetype pool and
agent behavior is scripted per deploy version, but every ledger is booked
by pf.book() and every rollup number is a sum of real ledger entries.

The story arc built into the behavior model:
    v1.0 (days 1-10):  baseline bad habits -- rating floors relaxed, the
                       vote_count.lte param unsupported (dropped), an
                       imposed min-votes popularity floor on every call.
    v1.1 (days 11-20): "the fix" -- vote_count.lte supported, substitution
                       rate halved. Fidelity trend visibly improves.
    v1.2 (days 21-30): the regression that matters -- fidelity stays high,
                       but the deploy quietly adds an undisclosed
                       with_original_language=en filter. The headline
                       number says nothing; the imposed census catches it.

Sharded rollup: runs land in 10 shards, each shard aggregates
independently, then shards merge. Conservation makes the rollup exact, so
the dashboard can print a books-balance line across shards -- if it
didn't balance, data was lost.

Run: python examples/fleet_simulate.py   (writes examples/fleet_rollup.json)
"""

import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "promptfidelity", "src"))

import promptfidelity as pf

RNG = random.Random(20260704)
N_RUNS = 10_000
N_DAYS = 30
N_SHARDS = 10

# ---------------------------------------------------------------- archetypes
# (description, params, p) tuples per prompt archetype. Mechanical pool --
# in production these come from decomposition; here they're the fixture.
ARCHETYPES = [
    ("genre-era", [
        ("Sci-fi genre", {"with_genres": "878"}, 0.08),
        ("Released in the 1990s", {"primary_release_date.gte": "1990-01-01",
                                   "primary_release_date.lte": "1999-12-31"}, 0.10),
        ("Melancholy tone", {}, 0.2),
    ]),
    ("quality-bar", [
        ("Thriller genre", {"with_genres": "53"}, 0.10),
        ("Rated 8.0 or higher", {"vote_average.gte": "8.0"}, 0.05),
        ("Tense, propulsive pacing", {}, 0.15),
    ]),
    ("deep-cut", [
        ("Horror genre", {"with_genres": "27"}, 0.09),
        ("Obscure (under 500 votes)", {"vote_count.lte": "500"}, 0.45),
        ("Rated 7.5 or higher", {"vote_average.gte": "7.5"}, 0.10),
    ]),
    ("topical-search", [
        ("About space stations", {"query": "space station"}, 0.02),
        ("Released in the 1980s", {"primary_release_date.gte": "1980-01-01",
                                   "primary_release_date.lte": "1989-12-31"}, 0.10),
    ]),
    ("foreign-gem", [
        ("Japanese language", {"with_original_language": "ja"}, 0.05),
        ("Animation genre", {"with_genres": "16"}, 0.06),
        ("Obscure (under 500 votes)", {"vote_count.lte": "500"}, 0.45),
        ("Wistful, contemplative mood", {}, 0.2),
    ]),
    ("runtime-fit", [
        ("Comedy genre", {"with_genres": "35"}, 0.12),
        ("Under 100 minutes", {"with_runtime.lte": "100"}, 0.35),
        ("Feel-good tone", {}, 0.3),
    ]),
]

ADVISORY = {"query"}

# ------------------------------------------------------------ deploy behavior

def deploy_for_day(day: int) -> str:
    if day <= 10:
        return "v1.0"
    if day <= 20:
        return "v1.1"
    return "v1.2"


def simulate_call(constraints, deploy):
    """What the scripted agent ACTUALLY sends, given its deploy's habits."""
    args = {}
    for c in constraints:
        for name, value in c.params.items():
            if name == "vote_count.lte":
                # v1.0 doesn't support the param at all -> dropped
                if deploy == "v1.0":
                    continue
                args[name] = value
            elif name == "vote_average.gte":
                # rating floors above 7.0 get relaxed; v1.1+ halves the habit
                p_relax = 0.85 if deploy == "v1.0" else 0.40
                if float(value) > 7.0 and RNG.random() < p_relax:
                    args[name] = "7.0"
                else:
                    args[name] = value
            else:
                args[name] = value
    # imposed on every call, no constraint asked:
    args["vote_count.gte"] = "100"
    if deploy == "v1.2":
        # the quiet regression: English-only filter nobody asked for.
        # Skip when the user explicitly constrained language (param collision
        # would book as substitution, muddying the story -- and real agents
        # usually "respect" an explicit param while filtering everyone else).
        if "with_original_language" not in args:
            args["with_original_language"] = "en"
    return args


# ------------------------------------------------------------------ simulate

def main():
    shards = [defaultdict(float) for _ in range(N_SHARDS)]  # account -> bits
    shard_runs = [0] * N_SHARDS
    fid_histogram = [0] * 10                      # deciles
    daily = {d: {"fid_num": 0.0, "fid_den": 0.0, "runs": 0,
                 "deploy": deploy_for_day(d)} for d in range(1, N_DAYS + 1)}
    backlog = defaultdict(lambda: {"bits": 0.0, "runs": 0, "desc": ""})
    substitutions = defaultdict(lambda: {"count": 0, "eligible": 0})
    imposed_census = defaultdict(lambda: {"count": 0, "first_day": None,
                                          "deploys": set()})

    for i in range(N_RUNS):
        day = RNG.randint(1, N_DAYS)
        deploy = deploy_for_day(day)
        name, spec = ARCHETYPES[RNG.randrange(len(ARCHETYPES))]
        constraints = [pf.Constraint(id=f"c{j}", description=d, params=p, p=sp)
                       for j, (d, p, sp) in enumerate(spec)]

        args = simulate_call(constraints, deploy)
        ledger = pf.book(constraints, args, advisory_params=ADVISORY)

        # --- aggregate (all mechanical sums of real ledger entries) ---
        s = i % N_SHARDS
        shard_runs[s] += 1
        for e in ledger.entries:
            shards[s][e.account] += e.bits or 0.0
            if e.account in ("dropped", "substituted"):
                for pname in e.params:
                    backlog[pname]["bits"] += e.bits or 0.0
                    backlog[pname]["runs"] += 1
                    backlog[pname]["desc"] = e.description
            if e.account == "substituted":
                for pname in e.params:
                    substitutions[pname]["count"] += 1
        for c in constraints:
            for pname in c.params:
                if pname in substitutions:
                    substitutions[pname]["eligible"] += 1
        for imp in ledger.imposed:
            rec = imposed_census[f"{imp.description}={imp.value}"]
            rec["count"] += 1
            rec["deploys"].add(deploy)
            if rec["first_day"] is None or day < rec["first_day"]:
                rec["first_day"] = day

        f = ledger.fidelity
        fid_histogram[min(int(f * 10), 9)] += 1
        daily[day]["fid_num"] += ledger.verified_bits
        daily[day]["fid_den"] += ledger.total_bits
        daily[day]["runs"] += 1

    # --- merge shards + trial balance ---
    total = defaultdict(float)
    for sh in shards:
        for account, bits in sh.items():
            total[account] += bits
    grand_total = sum(total.values())
    shard_sum = sum(sum(sh.values()) for sh in shards)
    books_balance = abs(grand_total - shard_sum) < 1e-6

    fleet_fidelity = total["verified"] / grand_total
    below_half = sum(fid_histogram[:5])

    rollup = {
        "meta": {"runs": N_RUNS, "days": N_DAYS, "shards": N_SHARDS,
                 "generated": "2026-07-04", "engine": "promptfidelity "
                 + pf.__version__ + " (real book(), scripted agents, no LLM)"},
        "fleet": {
            "fidelity": round(fleet_fidelity, 4),
            "accounts": {k: round(v, 1) for k, v in sorted(total.items())},
            "total_bits": round(grand_total, 1),
            "below_half_runs": below_half,
            "books_balance": books_balance,
            "shard_totals": [round(sum(sh.values()), 1) for sh in shards],
        },
        "histogram": fid_histogram,
        "daily": [{"day": d, "deploy": v["deploy"], "runs": v["runs"],
                   "fidelity": round(v["fid_num"] / v["fid_den"], 4) if v["fid_den"] else None}
                  for d, v in sorted(daily.items())],
        "backlog": sorted(
            [{"param": k, "desc": v["desc"], "bits": round(v["bits"], 1), "runs": v["runs"]}
             for k, v in backlog.items()],
            key=lambda r: -r["bits"]),
        "substitution_hotspots": [
            {"param": k, "substituted": v["count"], "eligible": v["eligible"],
             "rate": round(v["count"] / v["eligible"], 3) if v["eligible"] else 0}
            for k, v in sorted(substitutions.items(), key=lambda kv: -kv[1]["count"])
            if v["count"]],
        "imposed_census": sorted(
            [{"filter": k, "count": v["count"], "first_day": v["first_day"],
              "deploys": sorted(v["deploys"])}
             for k, v in imposed_census.items()],
            key=lambda r: r["first_day"]),
    }

    out = os.path.join(os.path.dirname(__file__), "fleet_rollup.json")
    with open(out, "w") as fh:
        json.dump(rollup, fh, indent=1)

    print(f"{N_RUNS} runs booked through pf.book() across {N_SHARDS} shards")
    print(f"fleet fidelity: {fleet_fidelity:.1%}  |  books balance across shards: {books_balance}")
    print(f"runs below F=0.5: {below_half} ({below_half/N_RUNS:.0%})")
    print("daily trend (deploy):",
          " ".join(f"d{r['day']}:{r['fidelity']:.2f}" for r in rollup["daily"][::6]))
    print("\ntop demand backlog:")
    for r in rollup["backlog"][:3]:
        print(f"  {r['param']:<28} {r['bits']:>8.1f} unmet bits across {r['runs']} runs")
    print("\nimposed census:")
    for r in rollup["imposed_census"]:
        print(f"  {r['filter']:<32} first seen day {r['first_day']:>2}  "
              f"({', '.join(r['deploys'])})  n={r['count']}")
    print(f"\nrollup written to {out}")


if __name__ == "__main__":
    main()
