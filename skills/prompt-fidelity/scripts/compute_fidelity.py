#!/usr/bin/env python3
"""
Compute a prompt-fidelity self-report from a list of constraints.

Stdlib only -- no installs required. This is the BEHAVIORAL tier: the
classification comes from the model following SKILL.md's point-at-the-
evidence discipline, and the report discloses that (`classification:
self-reported`). For mechanical booking against recorded tool calls --
no model judgment in the loop -- use the promptfidelity package in this
repo instead (pf.trace() / pf.book()).

Usage:
    python3 compute_fidelity.py constraints.json
    echo '[{"description": "...", "type": "verified", "evidence": "with_genres=878", "estimated_survival_rate": 0.1}]' \
        | python3 compute_fidelity.py
    python3 compute_fidelity.py --json constraints.json

Input: a JSON array of constraint objects (or an object with a
"constraints" key), where each constraint has:
    description             str   human-readable constraint
    type                    str   "verified" | "transmitted" | "unhonored" | "inferred"
    estimated_survival_rate float fraction of the candidate pool that
                                  satisfies this constraint, in (0, 1)
    evidence                str   (verified/transmitted/unhonored) the quoted
                                  tool-call argument that carried -- or
                                  altered/omitted -- this constraint

Types (see SKILL.md Step 2 for the point-at-the-evidence test):
    verified    -- an ENFORCED parameter carried it; results provably satisfy it
    transmitted -- delivered via a free-text/advisory input; content survived
                   (rewording is fine) but the backend does not enforce it
    unhonored   -- visibly altered en route, or never reached any call at all
    inferred    -- no tool input could express it; model judgment alone

Scores:
    auditability   = (verified + transmitted) / total   -- the headline:
                     how much left a mechanical evidence trail
    strict fidelity = verified / total                  -- deliberately harsher

Output: a formatted report block (or JSON with --json). Include it verbatim
in the response, unhonored disclosures first.
"""

import json
import math
import sys

TYPES = ("verified", "transmitted", "unhonored", "inferred")
EVIDENCE_TYPES = ("verified", "transmitted", "unhonored")

# Cap per-constraint bits; 20 bits ~= a one-in-a-million survival rate.
MAX_CONSTRAINT_BITS = 20.0
BAR_WIDTH = 20


def bits(survival_rate: float) -> float:
    if survival_rate <= 0:
        return MAX_CONSTRAINT_BITS
    if survival_rate >= 1:
        return 0.0
    return min(-math.log2(survival_rate), MAX_CONSTRAINT_BITS)


def analyze(constraints: list) -> dict:
    unknown = [c for c in constraints if c.get("type") not in TYPES]
    if unknown:
        names = ", ".join(repr(c.get("description", "?")) for c in unknown)
        raise ValueError(f'constraint "type" must be one of {TYPES}: {names}')
    missing_evidence = [
        c for c in constraints
        if c.get("type") in EVIDENCE_TYPES and not c.get("evidence")
    ]
    if missing_evidence:
        names = ", ".join(repr(c.get("description", "?")) for c in missing_evidence)
        raise ValueError(
            f"verified/transmitted/unhonored constraints need `evidence` -- the "
            f"quoted tool-call argument (SKILL.md Step 2). Missing on: {names}. "
            f"If you cannot quote the argument, the type is wrong."
        )

    for c in constraints:
        c["bits"] = round(bits(float(c["estimated_survival_rate"])), 2)

    by_type = {t: [c for c in constraints if c["type"] == t] for t in TYPES}
    type_bits = {t: round(sum(c["bits"] for c in group), 2)
                 for t, group in by_type.items()}
    total_bits = round(sum(type_bits.values()), 2)

    auditable_bits = type_bits["verified"] + type_bits["transmitted"]
    auditability = 1.0 if total_bits == 0 else auditable_bits / total_bits
    strict_fidelity = 1.0 if total_bits == 0 else type_bits["verified"] / total_bits

    return {
        "classification": "self-reported",
        "auditability": round(auditability, 3),
        "strict_fidelity": round(strict_fidelity, 3),
        "bits": type_bits,
        "total_bits": total_bits,
        "counts": {t: len(by_type[t]) for t in TYPES},
        "unhonored": [
            {"description": c["description"], "evidence": c.get("evidence", "")}
            for c in by_type["unhonored"]
        ],
        "constraints": constraints,
    }


def format_report(result: dict) -> str:
    filled = round(result["auditability"] * BAR_WIDTH)
    bar = "█" * filled + "░" * (BAR_WIDTH - filled)
    lines = [
        "─" * 54,
        f"  AUDITABILITY (self-reported): {result['auditability']:.1%}",
        f"  [{bar}]",
        f"  strict fidelity (verified only): {result['strict_fidelity']:.1%}",
        "",
    ]
    labels = {
        "verified": "VERIFIED    (enforced; provably satisfied)",
        "transmitted": "TRANSMITTED (delivered; not enforced)",
        "unhonored": "UNHONORED   (altered or dropped -- DISCLOSE)",
        "inferred": "INFERRED    (model judgment alone)",
    }
    for t in TYPES:
        group = [c for c in result["constraints"] if c["type"] == t]
        if not group:
            continue
        lines.append(f"  {labels[t]}: {result['bits'][t]} bits")
        for c in group:
            lines.append(f"    - {c['description']} ({c['bits']} bits)")
            if c.get("evidence"):
                lines.append(f"      evidence: {c['evidence']}")
        lines.append("")
    lines.append(
        f"  classification: self-reported (behavioral tier; for mechanical"
    )
    lines.append(
        f"  booking against recorded calls, use the promptfidelity package)"
    )
    lines.append("─" * 54)
    return "\n".join(lines)


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--json"]
    as_json = "--json" in sys.argv[1:]

    if args:
        with open(args[0]) as f:
            data = json.load(f)
    else:
        data = json.load(sys.stdin)

    constraints = data["constraints"] if isinstance(data, dict) else data

    try:
        result = analyze(constraints)
    except (ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2) if as_json else format_report(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
