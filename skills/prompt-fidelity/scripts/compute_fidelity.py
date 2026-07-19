#!/usr/bin/env python3
"""
Compute a prompt fidelity report from a list of constraints.

Stdlib only — no installs required.

Usage:
    python3 compute_fidelity.py constraints.json
    echo '[{"description": "...", "type": "verified", "estimated_survival_rate": 0.1}]' \
        | python3 compute_fidelity.py
    python3 compute_fidelity.py --json --pool-size 1e9 --joint-count 1100 constraints.json

Input: a JSON array of constraint objects (or an object with a
"constraints" key and optional "pool_size", "verified_joint_count",
"verified_joint_survival_rate" keys). Each constraint has:
    description             str   human-readable constraint
    type                    str   "verified", "inferred", or "injected"
    estimated_survival_rate float fraction of the candidate pool that
                                  satisfies this constraint, in (0, 1);
                                  optional for "injected"
    rate_source             str   "measured" (an exact or system-returned
                                  count against the actual data),
                                  "approximated" (system-derived but
                                  inexact — planner statistics, sampled
                                  counts, possibly stale), or "estimated"
                                  (a guess). Optional, but omitting it on
                                  verified/inferred constraints drops the
                                  provenance labels from the report; the
                                  CLI warns on stderr when that happens.

Constraint types:
    verified  — mechanically checkable against the data source; counts
                toward the fidelity numerator.
    inferred  — requires subjective judgment; counts toward the denominator.
    injected  — a filter the SYSTEM applied that the user never asked for
                (quality floors, top-N truncation, sampling, default sort
                order). Excluded from the fidelity score entirely, but
                always listed in the report: silent pool-narrowing is the
                main way a "100% fidelity" claim becomes dishonest.

Pool size: per-constraint bits are capped at log2(pool_size) — a constraint
cannot carry more information than it takes to identify a single row in the
candidate pool. Set it via --pool-size or a top-level "pool_size" key
(--pool-size wins if both are given). Without it, the cap defaults to
20 bits (a one-in-a-million pool).

Correlation correction: bits summed across constraints assume they filter
independently. When the datastore can cheaply measure the JOINT count of
all verified filters ANDed together (often the same query that sizes the
survivor set), pass it via --joint-count, a top-level
"verified_joint_count" key (requires pool_size), or a top-level
"verified_joint_survival_rate" key. The verified side of the score then
uses the exact measured joint information, -log2(joint/pool), instead of
the independence approximation; per-constraint bits remain as attribution
and the report shows the adjustment. A joint count of 0 (the verified
filters are jointly unsatisfiable) is valid and scores at the log2(pool)
cap. Inferred constraints cannot be jointly counted and stay summed —
merge heavily overlapping inferred constraints rather than listing them
separately.

Output: a formatted fidelity report (--json for machine-readable, --brief
for a compact conversational summary with no box art or decimal bits).
Input is never mutated; sums are computed on unrounded bits and rounded
only for display.
"""

import argparse
import json
import math
import os
import sys

# Default per-constraint cap; 20 bits ~= a one-in-a-million survival rate.
DEFAULT_MAX_CONSTRAINT_BITS = 20.0
BAR_WIDTH = 20
VALID_TYPES = ("verified", "inferred", "injected")
VALID_RATE_SOURCES = ("measured", "approximated", "estimated")


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


def analyze(constraints: list[dict], pool_size: float | None = None,
            verified_joint_survival_rate: float | None = None) -> dict:
    constraints = [dict(c) for c in constraints]  # never mutate the caller's

    unknown = [c for c in constraints if c.get("type") not in VALID_TYPES]
    if unknown:
        names = ", ".join(repr(c.get("description", "?")) for c in unknown)
        raise ValueError(
            f'constraint "type" must be one of {VALID_TYPES}: {names}')

    verified = [c for c in constraints if c["type"] == "verified"]
    inferred = [c for c in constraints if c["type"] == "inferred"]
    injected = [c for c in constraints if c["type"] == "injected"]

    missing = [c for c in verified + inferred if "estimated_survival_rate" not in c]
    if missing:
        names = ", ".join(repr(c.get("description", "?")) for c in missing)
        raise ValueError(
            f"estimated_survival_rate is required for verified/inferred "
            f"constraints: {names}")

    bad_sources = [c for c in constraints
                   if c.get("rate_source") not in (None, *VALID_RATE_SOURCES)]
    if bad_sources:
        names = ", ".join(repr(c.get("description", "?")) for c in bad_sources)
        raise ValueError(
            f'rate_source must be one of {VALID_RATE_SOURCES}: {names}')

    # Raw (unrounded) bits drive every sum; the per-constraint "bits" field
    # is rounded for display only.
    max_bits = max_bits_for_pool(pool_size)
    raw = {}
    for c in constraints:
        if "estimated_survival_rate" in c:
            raw[id(c)] = bits(float(c["estimated_survival_rate"]), max_bits)
            c["bits"] = round(raw[id(c)], 2)

    verified_bits_summed = sum(raw[id(c)] for c in verified)

    joint_bits = None
    if verified_joint_survival_rate is not None:
        if not verified:
            raise ValueError(
                "verified_joint_survival_rate given but there are no "
                "verified constraints")
        rate = float(verified_joint_survival_rate)
        if not 0 <= rate <= 1:
            raise ValueError(
                f"verified_joint_survival_rate must be in [0, 1], got {rate}")
        joint_bits = bits(rate, max_bits)  # rate 0 -> capped at log2(pool)

    verified_bits = joint_bits if joint_bits is not None else verified_bits_summed
    inferred_bits = sum(raw[id(c)] for c in inferred)
    total_bits = verified_bits + inferred_bits
    score = 1.0 if total_bits == 0 else verified_bits / total_bits

    measured = sum(1 for c in verified + inferred
                   if c.get("rate_source") == "measured")
    approximated = sum(1 for c in verified + inferred
                       if c.get("rate_source") == "approximated")
    unlabeled = [c["description"] for c in verified + inferred
                 if not c.get("rate_source")]

    return {
        "fidelity_score": round(score, 3),
        "pool_size": pool_size,
        "max_constraint_bits": round(max_bits, 2),
        "verified_bits": round(verified_bits, 2),
        "verified_bits_summed": round(verified_bits_summed, 2),
        "verified_rate_basis": "joint-measured" if joint_bits is not None else "summed",
        "verified_joint_survival_rate": verified_joint_survival_rate,
        "correlation_adjustment_bits": (
            round(verified_bits - verified_bits_summed, 2)
            if joint_bits is not None else None),
        "inferred_bits": round(inferred_bits, 2),
        "total_bits": round(total_bits, 2),
        "num_verified_constraints": len(verified),
        "num_inferred_constraints": len(inferred),
        "num_injected_constraints": len(injected),
        "num_measured_rates": measured,
        "num_approximated_rates": approximated,
        "unlabeled_rate_constraints": unlabeled,
        "constraints": constraints,
    }


def merge_reports(labeled_reports: list, bridge_constraints: list | None = None) -> dict:
    """Compose sub-question reports into one composite report.

    For a broad question decomposed across multiple (usually disparate)
    datasets: each sub-report keeps its own within-dataset joint
    correction, verified/inferred bits SUM across sub-questions (a joint
    count across different datasets is not measurable — this is the
    federated limitation, stated in the report), and the decomposition/
    synthesis choices enter as explicit bridging constraints, scored with the
    default 20-bit cap since they have no single pool.

    labeled_reports: [(label, report_dict), ...] where report_dict is the
    output of analyze() (or the harness). bridge_constraints: the same
    constraint shape as analyze() takes — the bridging premises that
    connect the sub-questions to the broad question: the decomposition
    mapping, proxy assumptions, synthesis rules.
    """
    verified = inferred = 0.0
    subs, injected_descriptions = [], []
    for label, r in labeled_reports:
        if not isinstance(r, dict) or "verified_bits" not in r or "inferred_bits" not in r:
            raise ValueError(
                f"input for {label!r} is not a fidelity report (expected the "
                f"--json output of this script or the harness — did you pass "
                f"a raw constraints file?)")
        label = r.get("label", label)
        verified += r["verified_bits"]
        inferred += r["inferred_bits"]
        inj = [c["description"] for c in r.get("constraints", [])
               if c.get("type") == "injected"]
        subs.append({
            "label": label,
            "fidelity_score": r["fidelity_score"],
            "verified_bits": r["verified_bits"],
            "inferred_bits": r["inferred_bits"],
            "verified_rate_basis": r.get("verified_rate_basis", "summed"),
            "num_injected_constraints": len(inj),
        })
        injected_descriptions.extend(f"[{label}] {d}" for d in inj)

    bridge = analyze(bridge_constraints) if bridge_constraints else None
    if bridge:
        verified += bridge["verified_bits"]
        inferred += bridge["inferred_bits"]
        injected_descriptions.extend(
            c["description"] for c in bridge["constraints"]
            if c["type"] == "injected")

    total = verified + inferred
    return {
        "composite": True,
        "fidelity_score": round(1.0 if total == 0 else verified / total, 3),
        "verified_bits": round(verified, 2),
        "inferred_bits": round(inferred, 2),
        "total_bits": round(total, 2),
        "sub_questions": subs,
        "bridge_constraints": bridge["constraints"] if bridge else [],
        "injected_descriptions": injected_descriptions,
        "note": ("Verified bits are summed across sub-questions: joint "
                 "correction applies within each dataset but cannot be "
                 "measured across datasets."),
    }


def render_composite(report: dict) -> str:
    score = report["fidelity_score"]
    filled = round(score * BAR_WIDTH)
    bar = "█" * filled + "░" * (BAR_WIDTH - filled)
    rule = "═" * 50
    thin = "─" * 40

    lines = [
        rule,
        f"  COMPOSITE PROMPT FIDELITY: {score * 100:.1f}%",
        rule,
        "",
        f"  [{bar}] {score * 100:.1f}%",
        "",
        "  Sub-questions:",
    ]
    for s in report["sub_questions"]:
        basis = ", joint-measured" if s["verified_rate_basis"] == "joint-measured" else ""
        lines.append(f"  • {s['label']}: {s['fidelity_score'] * 100:.1f}% — "
                     f"verified {s['verified_bits']:.2f} bits{basis}, "
                     f"inferred {s['inferred_bits']:.2f}")

    if report["bridge_constraints"]:
        lines.append("")
        lines.append("  Bridging premises (decomposition & synthesis):")
        for c in report["bridge_constraints"]:
            icon = {"verified": "✓", "inferred": "?", "injected": "!"}[c["type"]]
            lines.append(constraint_line(icon, c))

    lines += [
        "",
        f"  {thin}",
        f"  Verified:  {report['verified_bits']:>7.2f} bits (summed across sub-questions)",
        f"  Inferred:  {report['inferred_bits']:>7.2f} bits",
        f"  Total:     {report['total_bits']:>7.2f} bits",
    ]
    if report["injected_descriptions"]:
        lines.append(f"  Injected filters across sub-questions:")
        lines.extend(f"    ! {d}" for d in report["injected_descriptions"])
    lines.append(f"  Note: {report['note']}")
    lines.append(rule)
    return "\n".join(lines)


def render_composite_brief(report: dict) -> str:
    score = report["fidelity_score"]
    if score >= 0.8:
        band = "almost all of this answer is verifiable"
    elif score >= 0.4:
        band = "a mix of checked facts and judgment"
    else:
        band = "mostly judgment"
    lines = [f"Composite fidelity: {score * 100:.0f}% — {band}."]
    for s in report["sub_questions"]:
        lines.append(f"• {s['label']}: {s['fidelity_score'] * 100:.0f}% verifiable")
    bridging = [c["description"] for c in report["bridge_constraints"]
                if c["type"] == "inferred"]
    if bridging:
        lines.append(f"Judgment calls bridging them: {'; '.join(bridging)}")
    if report["injected_descriptions"]:
        lines.append("System filters (not requested): "
                     + "; ".join(report["injected_descriptions"]))
    return "\n".join(lines)


def constraint_line(icon: str, c: dict) -> str:
    annotations = []
    if "bits" in c:
        annotations.append(f"{c['bits']:.2f} bits")
    if c.get("rate_source"):
        annotations.append(c["rate_source"])
    suffix = f" ({', '.join(annotations)})" if annotations else ""
    return f"  {icon} {c['description']}{suffix}"


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
    injected = [c for c in report["constraints"] if c["type"] == "injected"]

    if verified:
        lines.append("")
        lines.append("  VERIFIED (checkable against a tool or data source):")
        lines.extend(constraint_line("✓", c) for c in verified)
    if inferred:
        lines.append("")
        lines.append("  INFERRED (requires LLM judgment):")
        lines.extend(constraint_line("?", c) for c in inferred)
    if injected:
        lines.append("")
        lines.append("  INJECTED (system-applied, not requested — excluded from score):")
        lines.extend(constraint_line("!", c) for c in injected)

    def plural(n: int) -> str:
        return "constraint" if n == 1 else "constraints"

    n_v = report["num_verified_constraints"]
    n_i = report["num_inferred_constraints"]
    n_j = report["num_injected_constraints"]
    joint = report["verified_rate_basis"] == "joint-measured"
    lines += [
        "",
        f"  {thin}",
        f"  Verified:  {report['verified_bits']:>7.2f} bits ({n_v} {plural(n_v)}"
        + (", joint-measured)" if joint else ")"),
        f"  Inferred:  {report['inferred_bits']:>7.2f} bits ({n_i} {plural(n_i)})",
        f"  Total:     {report['total_bits']:>7.2f} bits",
    ]
    if joint:
        adj = report["correlation_adjustment_bits"]
        lines.append(
            f"  Correlation: summed {report['verified_bits_summed']:.2f} bits "
            f"→ joint {report['verified_bits']:.2f} ({adj:+.2f} adjustment)")
        if report["verified_joint_survival_rate"] == 0:
            lines.append(
                "  Warning: joint count is 0 — the verified filters are "
                "jointly unsatisfiable; no result can match this request")
    if n_j:
        lines.append(f"  Injected:  {n_j} system "
                     f"{'filter narrows' if n_j == 1 else 'filters narrow'} "
                     f"the pool beyond the request")
    if report["pool_size"] is not None:
        lines.append(
            f"  Pool:      {report['pool_size']:,.0f} rows "
            f"(per-constraint cap: {report['max_constraint_bits']:.1f} bits)")
    lines.append(rule)
    return "\n".join(lines)


def render_brief(report: dict) -> str:
    """Compact summary for conversational use: no box art, no decimal bits."""
    score = report["fidelity_score"]
    if score >= 0.8:
        band = "almost all of this answer is verifiable"
    elif score >= 0.4:
        band = "a mix of checked facts and judgment"
    else:
        band = "mostly judgment"
    lines = [f"Fidelity: {score * 100:.0f}% — {band}."]

    def names(kind: str) -> str:
        return "; ".join(c["description"] for c in report["constraints"]
                         if c["type"] == kind)

    if report["num_verified_constraints"]:
        sources = {c.get("rate_source") for c in report["constraints"]
                   if c["type"] == "verified"}
        source_note = (f" (rates {next(iter(sources))})"
                       if len(sources) == 1 and None not in sources else "")
        lines.append(f"Checked against the data{source_note}: {names('verified')}")
    if report["num_inferred_constraints"]:
        lines.append(f"Judgment calls: {names('inferred')}")
    if report["num_injected_constraints"]:
        lines.append(f"System filters (not requested): {names('injected')}")
    if (report["verified_rate_basis"] == "joint-measured"
            and report["verified_joint_survival_rate"] == 0):
        lines.append("Warning: the verified filters match zero rows — "
                     "no result can satisfy this request.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute a prompt fidelity report from constraints JSON")
    parser.add_argument("inputs", nargs="*",
                        help="Path to constraints JSON (default: stdin); "
                             "with --merge, two or more sub-report JSON files")
    parser.add_argument("--merge", action="store_true",
                        help="Compose sub-question reports (each a --json "
                             "output of this script or the harness) into one "
                             "composite report for a broad multi-dataset "
                             "question")
    parser.add_argument("--bridge",
                        help="Merge mode: constraints JSON of bridging "
                             "premises — the decomposition/synthesis "
                             "judgment calls (usually inferred) that "
                             "connect the sub-questions to the broad "
                             "question")
    parser.add_argument("--pool-size", type=float,
                        help="Candidate pool size; caps per-constraint bits "
                             "at log2(pool_size)")
    parser.add_argument("--joint-count", type=float,
                        help="Measured count of rows matching ALL verified "
                             "filters ANDed (requires a pool size)")
    parser.add_argument("--json", action="store_true",
                        help="Print the report as JSON instead of text")
    parser.add_argument("--brief", action="store_true",
                        help="Print a compact summary (no box art or "
                             "decimal bits) for conversational use")
    args = parser.parse_args()

    if args.merge:
        if len(args.inputs) < 2:
            print("error: --merge needs two or more sub-report files",
                  file=sys.stderr)
            return 1
        try:
            labeled = []
            for path in args.inputs:
                stem = os.path.splitext(os.path.basename(path))[0]
                with open(path, encoding="utf-8") as f:
                    labeled.append((stem, json.load(f)))
            bridge = None
            if args.bridge:
                with open(args.bridge, encoding="utf-8") as f:
                    bridge = json.load(f)
            if isinstance(bridge, dict):
                bridge = bridge["constraints"]
            composite = merge_reports(labeled, bridge)
        except (KeyError, ValueError, TypeError, OSError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(composite, indent=2))
        elif args.brief:
            print(render_composite_brief(composite))
        else:
            print(render_composite(composite))
        return 0

    if len(args.inputs) > 1:
        print("error: multiple input files require --merge", file=sys.stderr)
        return 1
    if args.inputs:
        with open(args.inputs[0], encoding="utf-8") as f:
            raw = f.read()
    else:
        raw = sys.stdin.read()
    data = json.loads(raw)
    pool_size, joint_count, joint_rate = args.pool_size, args.joint_count, None
    if isinstance(data, dict):
        constraints = data["constraints"]
        if pool_size is None:
            pool_size = data.get("pool_size")
        if joint_count is None:
            joint_count = data.get("verified_joint_count")
        joint_rate = data.get("verified_joint_survival_rate")
    else:
        constraints = data

    try:
        if joint_count is not None:
            if joint_rate is not None:
                raise ValueError(
                    "give verified_joint_count or "
                    "verified_joint_survival_rate, not both")
            if pool_size is None:
                raise ValueError(
                    "verified_joint_count requires pool_size to derive the "
                    "joint survival rate")
            joint_rate = joint_count / pool_size
        report = analyze(constraints, pool_size=pool_size,
                         verified_joint_survival_rate=joint_rate)
    except (KeyError, ValueError, TypeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if report["unlabeled_rate_constraints"]:
        names = ", ".join(repr(d) for d in report["unlabeled_rate_constraints"])
        print(f"warning: no rate_source on {names} — the report will not "
              f"show whether these rates were measured or guessed",
              file=sys.stderr)

    if args.json:
        print(json.dumps(report, indent=2))
    elif args.brief:
        print(render_brief(report))
    else:
        print(render(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
