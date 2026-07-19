#!/usr/bin/env python3
"""
Run every experiment spec in experiments/specs/ and emit a comparison table.

Acts as a regression suite for the fidelity framework: each spec is
measured live against its Socrata dataset, scored with the
correlation-corrected calculator, and summarized in one markdown table
(paste-ready for the README).

Usage:
    python3 experiments/run_suite.py             # full per-spec logs + table
    python3 experiments/run_suite.py --quiet     # table only
    python3 experiments/run_suite.py --json      # machine-readable results

Exits nonzero if any spec fails, so it can gate CI.
"""

import argparse
import contextlib
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import socrata_fidelity  # noqa: E402

SPECS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "specs")


def run_spec(path: str, quiet: bool) -> dict:
    with open(path) as f:
        spec = json.load(f)
    ctx = (contextlib.redirect_stdout(io.StringIO()) if quiet
           else contextlib.nullcontext())
    with ctx:
        report = socrata_fidelity.run(spec, sample_rows=0)
    return {
        "spec": os.path.splitext(os.path.basename(path))[0],
        "domain": spec["domain"],
        "pool": report["pool_size"],
        "fidelity": report["fidelity_score"],
        "verified_bits": report["verified_bits"],
        "verified_bits_summed": report["verified_bits_summed"],
        "adjustment": report.get("correlation_adjustment_bits"),
        "inferred_bits": report["inferred_bits"],
    }


def markdown_table(rows: list[dict]) -> str:
    lines = [
        "| Spec | Domain | Pool | Fidelity | Verified bits (joint) | Correlation adj. |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r.get("error"):
            lines.append(f"| `{r['spec']}` | {r['domain']} | — | FAILED | — | — |")
            continue
        adjustment = f"{r['adjustment']:+.2f}" if r["adjustment"] is not None else "n/a"
        lines.append(
            f"| `{r['spec']}` | {r['domain']} | {r['pool']:,.0f} "
            f"| {r['fidelity']:.1%} | {r['verified_bits']:.2f} "
            f"(summed {r['verified_bits_summed']:.2f}) | {adjustment} bits |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run all fidelity experiment specs and summarize")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-spec logs; print only the table")
    parser.add_argument("--json", action="store_true",
                        help="Print results as JSON instead of a table")
    parser.add_argument("--specs-dir", default=SPECS_DIR,
                        help="Directory of spec JSON files (default: specs/)")
    args = parser.parse_args()

    paths = sorted(
        os.path.join(args.specs_dir, name)
        for name in os.listdir(args.specs_dir) if name.endswith(".json"))
    if not paths:
        print(f"error: no spec files in {args.specs_dir}", file=sys.stderr)
        return 1

    rows, failures = [], 0
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        if not args.quiet:
            print(f"\n{'=' * 60}\nSPEC: {name}\n{'=' * 60}")
        try:
            rows.append(run_spec(path, args.quiet))
        except Exception as e:
            failures += 1
            print(f"  FAILED {name}: {e}", file=sys.stderr)
            rows.append({"spec": name, "domain": "?", "error": str(e)})

    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print(f"\n{markdown_table(rows)}")
        ok = [r for r in rows if not r.get("error")]
        adjustments = [r["adjustment"] for r in ok if r["adjustment"] is not None]
        if adjustments:
            print(f"\n{len(ok)}/{len(rows)} specs passed; correlation "
                  f"adjustments span {min(adjustments):+.2f} to "
                  f"{max(adjustments):+.2f} bits")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
