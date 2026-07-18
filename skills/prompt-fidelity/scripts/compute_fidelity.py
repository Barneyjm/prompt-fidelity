#!/usr/bin/env python3
"""
Compute a prompt fidelity report from a list of constraints.

Stdlib only — no installs required.

Usage:
    python3 compute_fidelity.py constraints.json
    echo '[{"description": "...", "type": "verified", "estimated_survival_rate": 0.1}]' \
        | python3 compute_fidelity.py
    python3 compute_fidelity.py --json --pool-size 1000000000 constraints.json

Input: a JSON array of constraint objects (or an object with a
"constraints" key and optional "pool_size" key), where each constraint has:
    description             str   human-readable constraint
    type                    str   "verified" or "inferred"
    estimated_survival_rate float fraction of the candidate pool that
                                  satisfies this constraint, in (0, 1)

Pool size: per-constraint bits are capped at log2(pool_size) — a constraint
cannot carry more information than it takes to identify a single row in the
candidate pool. Set it via --pool-size or a top-level "pool_size" key
(--pool-size wins if both are given). Without it, the cap defaults to
20 bits (a one-in-a-million pool, roughly the TMDb movie catalog).

Output: a formatted fidelity report (or JSON with --json).
"""

import json
import math
import sys

# Default per-constraint cap; 20 bits ~= a one-in-a-million survival rate.
DEFAULT_MAX_CONSTRAINT_BITS = 20.0
BAR_WIDTH = 20


def bits(survival_rate: float, max_bits: float) -> float:
    if survival_rate <= 0:
        return max_bits
    if survival_rate >= 1:
        return 0.0
    return min(-math.log2(survival_rate), max_bits)


def max_bits_for_pool(pool_size: float | None) -> float:
    if pool_size is None:
        return DEFAULT_MAX_CONSTRAINT_BITS
    if pool_size <= 1:
        raise ValueError(f"pool_size must be greater than 1, got {pool_size}")
    return math.log2(pool_size)


def analyze(constraints: list[dict], pool_size: float | None = None) -> dict:
    verified = [c for c in constraints if c.get("type") == "verified"]
    inferred = [c for c in constraints if c.get("type") == "inferred"]
    unknown = [c for c in constraints if c.get("type") not in ("verified", "inferred")]
    if unknown:
        names = ", ".join(repr(c.get("description", "?")) for c in unknown)
        raise ValueError(f'constraint "type" must be "verified" or "inferred": {names}')

    max_bits = max_bits_for_pool(pool_size)
    for c in constraints:
        c["bits"] = round(bits(float(c["estimated_survival_rate"]), max_bits), 2)

    verified_bits = sum(c["bits"] for c in verified)
    inferred_bits = sum(c["bits"] for c in inferred)
    total_bits = verified_bits + inferred_bits
    score = 1.0 if total_bits == 0 else verified_bits / total_bits

    return {
        "fidelity_score": round(score, 3),
        "pool_size": pool_size,
        "max_constraint_bits": round(max_bits, 2),
        "verified_bits": round(verified_bits, 2),
        "inferred_bits": round(inferred_bits, 2),
        "total_bits": round(total_bits, 2),
        "num_verified_constraints": len(verified),
        "num_inferred_constraints": len(inferred),
        "constraints": constraints,
    }


def render(report: dict) -> str:
    score = report["fidelity_score"]
    filled = round(score * BAR_WIDTH)
    bar = "█" * filled + "░" * (BAR_WIDTH - filled)
    rule = "═" * 50
    thin = "─" * 40

    lines = [
        rule,
        f"  PROMPT FIDELITY: {score * 100:.1f}%",
        rule,
        "",
        f"  [{bar}] {score * 100:.1f}%",
        "",
        "  Constraint Breakdown:",
        f"  {thin}",
    ]

    verified = [c for c in report["constraints"] if c["type"] == "verified"]
    inferred = [c for c in report["constraints"] if c["type"] == "inferred"]

    if verified:
        lines.append("")
        lines.append("  VERIFIED (checkable against a tool or data source):")
        for c in verified:
            lines.append(f"  ✓ {c['description']} ({c['bits']:.2f} bits)")
    if inferred:
        lines.append("")
        lines.append("  INFERRED (requires LLM judgment):")
        for c in inferred:
            lines.append(f"  ? {c['description']} ({c['bits']:.2f} bits)")

    def plural(n: int) -> str:
        return "constraint" if n == 1 else "constraints"

    n_v = report["num_verified_constraints"]
    n_i = report["num_inferred_constraints"]
    lines += [
        "",
        f"  {thin}",
        f"  Verified:  {report['verified_bits']:>7.2f} bits ({n_v} {plural(n_v)})",
        f"  Inferred:  {report['inferred_bits']:>7.2f} bits ({n_i} {plural(n_i)})",
        f"  Total:     {report['total_bits']:>7.2f} bits",
    ]
    if report["pool_size"] is not None:
        lines.append(
            f"  Pool:      {report['pool_size']:,.0f} rows "
            f"(per-constraint cap: {report['max_constraint_bits']:.1f} bits)"
        )
    lines.append(rule)
    return "\n".join(lines)


def main() -> int:
    args = sys.argv[1:]
    as_json = "--json" in args
    args = [a for a in args if a != "--json"]

    pool_size = None
    if "--pool-size" in args:
        i = args.index("--pool-size")
        try:
            pool_size = float(args[i + 1])
        except (IndexError, ValueError):
            print("error: --pool-size requires a numeric value", file=sys.stderr)
            return 1
        del args[i:i + 2]

    raw = open(args[0]).read() if args else sys.stdin.read()
    data = json.loads(raw)
    if isinstance(data, dict):
        constraints = data["constraints"]
        if pool_size is None:
            pool_size = data.get("pool_size")
    else:
        constraints = data

    try:
        report = analyze(constraints, pool_size=pool_size)
    except (KeyError, ValueError, TypeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(json.dumps(report, indent=2) if as_json else render(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
