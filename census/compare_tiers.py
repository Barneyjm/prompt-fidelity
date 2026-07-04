"""Compare Tier A (description-based) vs Tier B (tool-schema ground truth)."""
import json
from collections import Counter

CENSUS_DIR = "/private/tmp/claude-501/-Users-jbarney-Documents-code-prompt-fidelity/99788712-8afe-4cd0-8efa-7473e209fb04/scratchpad/census"

tier_a = {c["name"]: c for c in json.load(open(f"{CENSUS_DIR}/classifications.json"))["classifications"]}
tier_b = json.load(open(f"{CENSUS_DIR}/tier_b_classifications.json"))["classifications"]

match = 0
near = 0  # A's dominant == B's secondary or vice versa
miss = []
confusion = Counter()
readback = Counter()
for b in tier_b:
    a = tier_a.get(b["name"])
    if not a:
        continue
    confusion[(a["species"], b["species"])] += 1
    if b["species"] == "actuation":
        readback[b["readback"]] += 1
    if a["species"] == b["species"]:
        match += 1
    elif a["species"] == (b.get("secondary") or "") or b["species"] == (a.get("secondary") or ""):
        near += 1
        miss.append((b["name"], a["species"], b["species"], "near"))
    else:
        miss.append((b["name"], a["species"], b["species"], "hard"))

n = match + len(miss) + near
print(f"n={n}  exact={match} ({match/n:.0%})  near (dominant<->secondary swap)={near}  hard miss={len(miss)-0 if False else sum(1 for m in miss if m[3]=='hard')}")
print("\nMisses (name, tierA, tierB(ground truth), kind):")
for m in miss:
    print(" ", m)
print("\nConfusion (tierA -> tierB) where different:")
for (a, b), c in sorted(confusion.items(), key=lambda kv: -kv[1]):
    if a != b:
        print(f"  {a:>10} -> {b:<10} {c}")
print(f"\nActuation servers with actual read-back tools: {dict(readback)}")
