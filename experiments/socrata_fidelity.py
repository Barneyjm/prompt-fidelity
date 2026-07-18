#!/usr/bin/env python3
"""
Fidelity validation harness for Socrata-hosted open datasets.

Runs the prompt-fidelity workflow against a real dataset (NYC Open Data,
data.cdc.gov, data.cityofchicago.org, and most data.gov-federated portals
speak the same SODA API):

1. Measures the pool size and each verified constraint's survival rate via
   server-side counts — the "counts the system returns anyway" tier of the
   skill's measurement guidance. No table scans on our side.
2. Scores the request with the bundled fidelity calculator.
3. Validates the independence assumption: compares the survivor count
   predicted by multiplying individual rates against the measured joint
   count, reporting the gap in bits.
4. Optionally fetches a small declared sample of matching rows.

Stdlib only. Usage:

    python3 experiments/socrata_fidelity.py experiments/specs/nyc_311_noise.json
    python3 experiments/socrata_fidelity.py --sample 5 spec.json

Spec format (JSON):

    {
      "description": "the natural-language request being decomposed",
      "domain": "data.cityofnewyork.us",
      "dataset_id": "erm2-nwe9",
      "constraints": [
        {"description": "...", "type": "verified",
         "where": "<SoQL predicate, measured automatically>"},
        {"description": "...", "type": "inferred",
         "estimated_survival_rate": 0.10}
      ],
      "sample_size": 100,        // optional: declared as an injected filter
      "sample_fields": ["a","b"] // optional: columns for --sample output
    }

Set SOCRATA_APP_TOKEN to raise the anonymous rate limit (optional).
"""

import argparse
import importlib.util
import json
import math
import os
import sys
import urllib.parse
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALCULATOR = os.path.join(
    REPO_ROOT, "skills", "prompt-fidelity", "scripts", "compute_fidelity.py")


def load_calculator():
    """Load the skill's fidelity calculator so the math lives in one place."""
    spec = importlib.util.spec_from_file_location("compute_fidelity", CALCULATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def soda_get(domain: str, dataset_id: str, params: dict) -> list:
    """One SODA API request, returning parsed JSON rows."""
    url = (f"https://{domain}/resource/{dataset_id}.json?"
           + urllib.parse.urlencode(params))
    request = urllib.request.Request(url)
    token = os.environ.get("SOCRATA_APP_TOKEN")
    if token:
        request.add_header("X-App-Token", token)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode())


def soda_count(domain: str, dataset_id: str, where: str | None = None) -> int:
    """Server-side row count, optionally filtered. Never scans client-side."""
    params = {"$select": "count(1) as n"}
    if where:
        params["$where"] = where
    rows = soda_get(domain, dataset_id, params)
    return int(rows[0]["n"])


def run(spec: dict, sample_rows: int) -> dict:
    calc = load_calculator()
    domain, dataset_id = spec["domain"], spec["dataset_id"]

    print(f'Request: "{spec["description"]}"')
    print(f"Dataset: https://{domain}/resource/{dataset_id}\n")

    pool = soda_count(domain, dataset_id)
    print(f"Pool size (system-returned count): {pool:,} rows")

    # Measure each verified constraint's survival rate server-side
    constraints = []
    verified_wheres = []
    for c in spec["constraints"]:
        entry = {"description": c["description"], "type": c["type"]}
        if c["type"] == "verified":
            count = soda_count(domain, dataset_id, c["where"])
            entry["estimated_survival_rate"] = count / pool
            entry["rate_source"] = "measured"
            verified_wheres.append(c["where"])
            print(f"  measured: {c['description']}: {count:,} rows "
                  f"({count / pool:.4%})")
        else:
            entry["estimated_survival_rate"] = c["estimated_survival_rate"]
            entry["rate_source"] = c.get("rate_source", "estimated")
        constraints.append(entry)

    # The joint count both validates independence and sizes the survivor set
    joint = None
    if verified_wheres:
        joint = soda_count(domain, dataset_id,
                           " AND ".join(f"({w})" for w in verified_wheres))
        print(f"  measured: all verified filters combined: {joint:,} rows")

    if spec.get("sample_size") and joint is not None and joint > spec["sample_size"]:
        constraints.append({
            "description": (f"Only a {spec['sample_size']}-row sample of the "
                            f"{joint:,} matching rows judged for inferred criteria"),
            "type": "injected",
        })

    joint_rate = joint / pool if joint else None
    report = calc.analyze(constraints, pool_size=pool,
                          verified_joint_survival_rate=joint_rate)
    print()
    print(calc.render(report))

    # Independence validation: predicted survivors vs measured joint count
    verified_rates = [c["estimated_survival_rate"] for c in constraints
                      if c["type"] == "verified"]
    if joint is not None and len(verified_rates) > 1 and joint > 0:
        predicted = pool * math.prod(verified_rates)
        gap_bits = math.log2(predicted / joint)
        print("\n  Independence check:")
        print(f"    predicted survivors (rates multiplied): {predicted:,.0f}")
        print(f"    measured joint count:                   {joint:,}")
        print(f"    correlation gap: {gap_bits:+.2f} bits "
              f"(vs {report['verified_bits']:.2f} verified bits)")
        report["independence_check"] = {
            "predicted_survivors": round(predicted),
            "measured_joint_count": joint,
            "correlation_gap_bits": round(gap_bits, 2),
        }

    if sample_rows and verified_wheres:
        params = {
            "$where": " AND ".join(f"({w})" for w in verified_wheres),
            "$limit": sample_rows,
        }
        if spec.get("sample_fields"):
            params["$select"] = ",".join(spec["sample_fields"])
        rows = soda_get(domain, dataset_id, params)
        print(f"\n  Sample of matching rows ({len(rows)} shown):")
        for row in rows:
            print(f"    {json.dumps(row)}")

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a prompt-fidelity experiment against a Socrata dataset")
    parser.add_argument("spec", help="Path to an experiment spec JSON file")
    parser.add_argument("--sample", type=int, default=0, metavar="N",
                        help="Also fetch N matching rows for inspection")
    parser.add_argument("--json", action="store_true",
                        help="Print the report as JSON instead of text")
    args = parser.parse_args()

    with open(args.spec) as f:
        spec = json.load(f)

    try:
        report = run(spec, args.sample)
    except urllib.error.URLError as e:
        print(f"error: request to {spec['domain']} failed: {e}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
