"""Aggregate Tier A classifications into the RUBRIC.md deliverables."""
import csv
import json
from collections import Counter

CENSUS_DIR = "/private/tmp/claude-501/-Users-jbarney-Documents-code-prompt-fidelity/99788712-8afe-4cd0-8efa-7473e209fb04/scratchpad/census"

servers = {s["name"]: s for s in json.load(open(f"{CENSUS_DIR}/servers.json"))}
data = json.load(open(f"{CENSUS_DIR}/classifications.json"))
cls = data["classifications"]

# join + dedupe (spot-check pass may not be in here; classifications are pass 1)
seen = set()
rows = []
for c in cls:
    if c["name"] in seen:
        continue
    seen.add(c["name"])
    s = servers.get(c["name"])
    if not s:
        continue
    rows.append({
        "name": c["name"],
        "description": s["description"],
        "species": c["species"],
        "secondary": c.get("secondary") or "",
        "confidence": c["confidence"],
    })

n = len(rows)
counts = Counter(r["species"] for r in rows)
unknown = counts.pop("unknown", 0)
classified = n - unknown

print(f"N joined: {n}  (of {len(servers)} servers; {len(cls)} raw classifications)")
print(f"Excluded as unknown: {unknown} ({unknown/n:.1%})\n")

print("## 1. Distribution (classified only)")
for sp in ("actuation", "relevance", "generative", "predicate"):
    print(f"  {sp:<11} {counts.get(sp,0):4d}  {counts.get(sp,0)/classified:.1%}")

rel_gen = counts.get("relevance", 0) + counts.get("generative", 0)
print(f"\n## 2. Transmitted-heavy hypothesis: relevance+generative = {rel_gen} ({rel_gen/classified:.1%})")

# 3. actuation servers mentioning read-back
READBACK = ("get", "read", "list", "status", "state", "fetch", "retrieve", "query", "view", "monitor", "check")
act = [r for r in rows if r["species"] == "actuation"]
def mentions_readback(desc):
    d = desc.lower()
    return any(w in d for w in READBACK)
act_rb = [r for r in act if mentions_readback(r["description"])]
print(f"\n## 3. Actuation share {len(act)/classified:.1%}; of those, {len(act_rb)}/{len(act)} = {len(act_rb)/max(len(act),1):.1%} mention read-back verbs")

# 3b. secondary species distribution
sec = Counter(r["secondary"] for r in rows if r["secondary"])
print(f"\nSecondary species (where recorded, {sum(sec.values())} servers): {dict(sec)}")

# confidence
conf = Counter(r["confidence"] for r in rows)
print(f"Confidence: {dict(conf)}")

# 4. exemplars: high-confidence, description length 60-200
print("\n## 4. Exemplars")
for sp in ("predicate", "relevance", "generative", "actuation"):
    print(f"\n### {sp}")
    ex = [r for r in rows if r["species"] == sp and r["confidence"] == "H" and 60 <= len(r["description"]) <= 220]
    for r in ex[:5]:
        print(f"  - {r['name']}: {r['description'][:180]}")

# 5. CSV
with open(f"{CENSUS_DIR}/census_tier_a.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["name", "description", "species", "secondary", "confidence"])
    w.writeheader()
    w.writerows(rows)
print(f"\nCSV written: census_tier_a.csv ({n} rows)")

ag = data.get("agreement", {})
if ag.get("total"):
    print(f"Spot-check inter-run agreement: {ag['agree']}/{ag['total']} = {ag['agree']/ag['total']:.1%}")
