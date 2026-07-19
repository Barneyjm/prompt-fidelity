#!/usr/bin/env python3
"""
Fidelity validation harness for Socrata-hosted open datasets.

Runs the prompt-fidelity workflow against a real dataset (NYC Open Data,
data.cdc.gov, data.cityofchicago.org, and most data.gov-federated portals
speak the same SODA API):

1. Measures the pool size and each verified constraint's survival rate via
   server-side counts — the "counts the system returns anyway" tier of the
   skill's measurement guidance. No table scans on our side. Independent
   counts are issued concurrently.
2. Scores the request with the bundled fidelity calculator, applying the
   correlation correction (the measured joint count of all verified
   filters replaces the independence sum).
3. Reports the independence check: survivors predicted by multiplying
   individual rates vs the measured joint count, with the correction the
   calculator actually applied.
4. Optionally attributes correlation pairwise (--pairwise) and fetches a
   small declared sample of matching rows (--sample N).

Stdlib only. Usage:

    python3 experiments/socrata_fidelity.py experiments/specs/nyc_311_noise.json
    python3 experiments/socrata_fidelity.py --pairwise --sample 5 spec.json

Spec format (JSON):

    {
      "description": "the natural-language request being decomposed",
      "domain": "data.cityofnewyork.us",
      "dataset_id": "erm2-nwe9",
      "constraints": [
        {"description": "...", "type": "verified",
         "where": "<SoQL predicate, measured automatically>"},
        {"description": "...", "type": "inferred",
         "estimated_survival_rate": 0.10},
        {"description": "...", "type": "injected"}   // rate optional
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
from concurrent.futures import ThreadPoolExecutor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALCULATOR = os.path.join(
    REPO_ROOT, "skills", "prompt-fidelity", "scripts", "compute_fidelity.py")

# Concurrent count queries per run — polite to anonymous rate limits while
# collapsing serial round-trip latency.
MAX_CONCURRENT_REQUESTS = 4

_calculator = None


def load_calculator():
    """Load the skill's fidelity calculator once; the math lives there."""
    global _calculator
    if _calculator is None:
        spec = importlib.util.spec_from_file_location(
            "compute_fidelity", CALCULATOR)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _calculator = module
    return _calculator


def soda_get(domain: str, dataset_id: str, params: dict,
             retries: int = 1) -> list:
    """One SODA API request, returning parsed JSON rows.

    Retries once on a timeout/connection stall — public portals throttle
    anonymous traffic by stalling connections rather than returning 429s.
    """
    url = (f"https://{domain}/resource/{dataset_id}.json?"
           + urllib.parse.urlencode(params))
    request = urllib.request.Request(url)
    token = os.environ.get("SOCRATA_APP_TOKEN")
    if token:
        request.add_header("X-App-Token", token)
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode())
        except (TimeoutError, OSError):
            if attempt == retries:
                raise


def soda_count(domain: str, dataset_id: str, where: str | None = None) -> int:
    """Server-side row count, optionally filtered. Never scans client-side."""
    params = {"$select": "count(1) as n"}
    if where:
        params["$where"] = where
    rows = soda_get(domain, dataset_id, params)
    return int(rows[0]["n"])


def run(spec: dict, sample_rows: int, pairwise: bool = False) -> dict:
    calc = load_calculator()
    domain, dataset_id = spec["domain"], spec["dataset_id"]

    print(f'Request: "{spec["description"]}"')
    print(f"Dataset: https://{domain}/resource/{dataset_id}\n")

    verified_specs = [c for c in spec["constraints"] if c["type"] == "verified"]
    verified_wheres = [c["where"] for c in verified_specs]
    joint_where = (" AND ".join(f"({w})" for w in verified_wheres)
                   if verified_wheres else None)

    # All counts are independent server-side queries — issue them together.
    # The joint count reuses the single filter when only one verified
    # constraint exists (identical predicate, no second request).
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as pool_exec:
        pool_future = pool_exec.submit(soda_count, domain, dataset_id)
        count_futures = [pool_exec.submit(soda_count, domain, dataset_id, w)
                         for w in verified_wheres]
        joint_future = (pool_exec.submit(soda_count, domain, dataset_id, joint_where)
                        if len(verified_wheres) > 1 else None)
        pool = pool_future.result()
        counts = [f.result() for f in count_futures]
        if joint_future is not None:
            joint = joint_future.result()
        else:
            joint = counts[0] if counts else None

    if pool <= 1:
        raise ValueError(
            f"dataset {domain}/{dataset_id} reports {pool} rows — too small "
            f"to score (pool must exceed 1 row)")

    print(f"Pool size (system-returned count): {pool:,} rows")

    constraints = []
    verified_counts = []
    count_iter = iter(zip(verified_specs, counts))
    for c in spec["constraints"]:
        entry = {"description": c["description"], "type": c["type"]}
        if c["type"] == "verified":
            vc, count = next(count_iter)
            entry["estimated_survival_rate"] = count / pool
            entry["rate_source"] = "measured"
            verified_counts.append((c["description"], c["where"], count))
            print(f"  measured: {c['description']}: {count:,} rows "
                  f"({count / pool:.4%})")
        else:
            # Inferred constraints require a rate; injected ones may omit it.
            if "estimated_survival_rate" in c:
                entry["estimated_survival_rate"] = c["estimated_survival_rate"]
                entry["rate_source"] = c.get("rate_source", "estimated")
        constraints.append(entry)

    if joint is not None:
        print(f"  measured: all verified filters combined: {joint:,} rows")

    if spec.get("sample_size") and joint is not None and joint > spec["sample_size"]:
        constraints.append({
            "description": (f"Only a {spec['sample_size']}-row sample of the "
                            f"{joint:,} matching rows judged for inferred criteria"),
            "type": "injected",
        })

    # Clamp guards a live-data race: the joint count runs concurrently with
    # the pool count, and rows ingested in between can push joint/pool
    # infinitesimally past 1 on near-universal filters.
    joint_rate = min(joint / pool, 1.0) if joint is not None else None
    report = calc.analyze(constraints, pool_size=pool,
                          verified_joint_survival_rate=joint_rate)
    print()
    print(calc.render(report))

    # Independence check: predicted survivors vs the measured joint count.
    # The gap in bits IS the calculator's correlation adjustment — cite it
    # rather than recomputing it.
    verified_rates = [c["estimated_survival_rate"] for c in constraints
                      if c["type"] == "verified"]
    if joint is not None and len(verified_rates) > 1:
        predicted = pool * math.prod(verified_rates)
        print("\n  Independence check:")
        print(f"    predicted survivors (rates multiplied): {predicted:,.0f}")
        print(f"    measured joint count:                   {joint:,}")
        print(f"    applied as correlation adjustment: "
              f"{report['correlation_adjustment_bits']:+.2f} bits "
              f"(vs {report['verified_bits_summed']:.2f} summed verified bits)")
        report["independence_check"] = {
            "predicted_survivors": round(predicted),
            "measured_joint_count": joint,
            "correlation_adjustment_bits": report["correlation_adjustment_bits"],
        }

    # Pairwise attribution: which constraint pairs drive the adjustment?
    # Each pair's contribution uses the SAME sign convention as the
    # aggregate adjustment: pair_adjustment = bits_AB - (bits_A + bits_B)
    # = log2((n_A * n_B) / (pool * n_AB)). Negative = overlapping
    # (positively correlated), positive = anti-correlated; the pairwise
    # values sum to approximately the aggregate adjustment (residual =
    # higher-order interactions). k constraints cost k(k-1)/2 extra counts.
    if pairwise and len(verified_counts) > 1:
        print("\n  Pairwise correlation adjustments:")
        pair_inputs = []
        for i in range(len(verified_counts)):
            for j in range(i + 1, len(verified_counts)):
                pair_inputs.append((verified_counts[i], verified_counts[j]))
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as pool_exec:
            pair_futures = [
                pool_exec.submit(soda_count, domain, dataset_id,
                                 f"({a[1]}) AND ({b[1]})")
                for a, b in pair_inputs]
            pair_counts = [f.result() for f in pair_futures]
        pairs = []
        for ((desc_a, _, n_a), (desc_b, _, n_b)), n_ab in zip(pair_inputs, pair_counts):
            if n_ab == 0 or n_a == 0 or n_b == 0:
                print(f"    {desc_a} × {desc_b}: no overlap")
                continue
            adjustment = math.log2(n_a * n_b / (pool * n_ab))
            pairs.append({"a": desc_a, "b": desc_b,
                          "adjustment_bits": round(adjustment, 2)})
            print(f"    {desc_a} × {desc_b}: {adjustment:+.2f} bits")
        report["pairwise_adjustment_bits"] = pairs

    if sample_rows and joint_where:
        params = {"$where": joint_where, "$limit": sample_rows}
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
    parser.add_argument("--pairwise", action="store_true",
                        help="Measure pairwise correlation adjustments between "
                             "verified constraints (k*(k-1)/2 extra counts)")
    parser.add_argument("--json", action="store_true",
                        help="Print the report as JSON instead of text")
    args = parser.parse_args()

    with open(args.spec) as f:
        spec = json.load(f)

    try:
        report = run(spec, args.sample, pairwise=args.pairwise)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        print(f"error: request to {spec['domain']} failed: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
