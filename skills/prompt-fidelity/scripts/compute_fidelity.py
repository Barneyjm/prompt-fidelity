#!/usr/bin/env python3
"""
Compute a prompt fidelity report from a list of constraints.

Stdlib only — no installs required.

Usage:
    python3 compute_fidelity.py constraints.json
    echo '[{"description": "...", "type": "verified", "estimated_survival_rate": 0.1}]' \
        | python3 compute_fidelity.py
    python3 compute_fidelity.py --json constraints.json

Input: a JSON array of constraint objects (or an object with a
"constraints" key), where each constraint has:
    description             str   human-readable constraint
    type                    str   "verified" or "inferred"
    estimated_survival_rate float fraction of the candidate pool that
                                  satisfies this constraint, in (0, 1)

Output: a formatted fidelity report (or JSON with --json).
"""

import json
import math
import sys

# Cap per-constraint bits; 20 bits ~= a one-in-a-million survival rate.
MAX_CONSTRAINT_BITS = 20.0
BAR_WIDTH = 20


def bits(survival_rate: float) -> float:
    if survival_rate <= 0:
        return MAX_CONSTRAINT_BITS
    if survival_rate >= 1:
        return 0.0
    return min(-math.log2(survival_rate), MAX_CONSTRAINT_BITS)


def analyze(constraints: list[dict]) -> dict:
    verified = [c for c in constraints if c.get("type") == "verified"]
    inferred = [c for c in constraints if c.get("type") == "inferred"]
    unknown = [c for c in constraints if c.get("type") not in ("verified", "inferred")]
    if unknown:
        names = ", ".join(repr(c.get("description", "?")) for c in unknown)
        raise ValueError(f'constraint "type" must be "verified" or "inferred": {names}')

    for c in constraints:
        c["bits"] = round(bits(float(c["estimated_survival_rate"])), 2)

    verified_bits = sum(c["bits"] for c in verified)
    inferred_bits = sum(c["bits"] for c in inferred)
    total_bits = verified_bits + inferred_bits
    score = 1.0 if total_bits == 0 else verified_bits / total_bits

    return {
        "fidelity_score": round(score, 3),
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

    lines += [
        "",
        f"  {thin}",
        f"  Verified:  {report['verified_bits']:>7.2f} bits "
        f"({report['num_verified_constraints']} constraints)",
        f"  Inferred:  {report['inferred_bits']:>7.2f} bits "
        f"({report['num_inferred_constraints']} constraints)",
        f"  Total:     {report['total_bits']:>7.2f} bits",
        rule,
    ]
    return "\n".join(lines)


def main() -> int:
    args = sys.argv[1:]
    as_json = "--json" in args
    paths = [a for a in args if a != "--json"]

    raw = open(paths[0]).read() if paths else sys.stdin.read()
    data = json.loads(raw)
    constraints = data["constraints"] if isinstance(data, dict) else data

    try:
        report = analyze(constraints)
    except (KeyError, ValueError, TypeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(json.dumps(report, indent=2) if as_json else render(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
