---
name: prompt-fidelity
description: Self-check how much of a request is verifiable vs guesswork before answering it. Use when handling search, filtering, recommendation, or data-retrieval requests that mix objective criteria (checkable via an API, database, file, or calculation) with subjective judgment (mood, style, quality, "feels like", "best"), or when the user asks how confident, reliable, or verifiable an answer is. Decomposes the request into constraints, computes a fidelity score, and reports which parts of the answer are verified vs inferred.
---

# Prompt Fidelity Self-Check

Prompt fidelity measures what fraction of a request's information content can
be **verified** against a deterministic source (API query, database, file
contents, calculation) versus what must be **inferred** through LLM judgment.

```
Fidelity = verified_bits / (verified_bits + inferred_bits)
```

A score of 1.0 means every part of the answer is auditable. A score near 0
means the answer is entirely a judgment call. Reporting this score alongside
an answer tells the user exactly which claims they can trust mechanically and
which they should evaluate themselves.

## When to run this check

- The request filters or ranks items by a mix of objective and subjective
  criteria ("Python repos updated this year that are well-designed").
- The user asks how confident or verifiable your answer is.
- You are about to present results where some criteria were actually checked
  and others were your own judgment — the user deserves to know which is which.

## Workflow

### Step 1 — Decompose the request into atomic constraints

Break the request into individual constraints. Rules:

- Each constraint is one independent requirement.
- A range counts as ONE constraint, not two. "From the 90s" is a single
  constraint ("Released 1990–1999"), not separate "after 1990" and
  "before 1999" constraints.
- Ignore filler that carries no selectivity ("some good", "please find me").

### Step 2 — Classify each constraint

- **verified**: You can check it in the current context with a deterministic
  tool — an API parameter, a SQL/search filter, reading a file, running code,
  or arithmetic. The test is: *could a script confirm each result satisfies
  this, with no model judgment involved?*
- **inferred**: Requires subjective judgment — mood, tone, themes, style,
  quality, comparative feel ("like a Coen Brothers film"), suitability
  ("good for a first date"), pacing, emotional arc.

Classification depends on the tools actually available right now. "Rated
above 7.0" is verified if you can query a ratings API, inferred if you
cannot. When in doubt, classify as inferred — overclaiming verification is
worse than underclaiming it.

### Step 3 — Estimate survival rates

For each constraint, estimate the fraction of the relevant candidate pool
that satisfies it (a decimal in (0, 1)). Information content is
`-log2(survival_rate)`. Calibration guide:

| Selectivity | Survival rate | Bits |
|---|---|---|
| Extremely selective (specific person, exact ID) | 0.001–0.01 | 7–10 |
| Very selective (exact year, rare property) | ~0.01 | ~6.6 |
| Moderately selective (a genre, a decade, a language) | 0.05–0.15 | 2.7–4.3 |
| Broad (rating above average, common property) | 0.3–0.5 | 1–1.7 |

Estimate the **pool size** too — the number of candidate rows/items the
request selects from (a movie catalog ≈ 10^6, a reviews table might be 10^9,
a repo might be 10^3 files). No constraint can carry more information than
it takes to identify a single row, so per-constraint bits are capped at
log2(pool_size). Pass the pool size to the script when you know it (even a
rough order of magnitude); otherwise the cap defaults to 20 bits (a
one-in-a-million pool), which undercounts near-unique selectors like exact
IDs on very large datasets.

### Step 4 — Compute the score

Write the constraints as a JSON array and run the bundled script (stdlib
only, no installs). The script lives at `scripts/compute_fidelity.py`
relative to this SKILL.md:

```bash
python3 <skill-dir>/scripts/compute_fidelity.py --pool-size 1000000000 constraints.json
```

Each constraint object needs `description`, `type` ("verified" or
"inferred"), and `estimated_survival_rate`. Pass `--pool-size` when you know
the candidate pool's rough size (or a top-level `"pool_size"` key in the
JSON); omit it to use the default 20-bit cap. Add `--json` for
machine-readable output. The script prints the fidelity report block —
include it verbatim in your response.

### Step 5 — Answer with calibrated framing

Actually verify the verified constraints — run the queries/checks, don't just
claim them. Then frame the answer by score:

- **≥ 80%**: "These results are verifiably correct on the stated criteria."
  Every result should pass an audit of the verified constraints.
- **40–80%**: Separate the two layers explicitly: "Filtered by X and Y
  (verified); ranked by Z (my judgment)."
- **< 40%**: Say plainly that the answer is mostly judgment, present it as a
  best guess, and offer a higher-fidelity reformulation — e.g. suggest
  objective proxies for the subjective criteria ("'feels like a slow burn'
  → runtime > 120 min, drama genre, pre-2010").

## Example

Request: *"Dark psychological thrillers from the 90s"* with a movie API
available.

```json
[
  {"description": "Thriller genre", "type": "verified", "estimated_survival_rate": 0.10},
  {"description": "Released 1990-1999", "type": "verified", "estimated_survival_rate": 0.10},
  {"description": "Dark/psychological tone", "type": "inferred", "estimated_survival_rate": 0.15}
]
```

Script output: fidelity 70.8% — verified 6.64 bits (2 constraints), inferred
2.74 bits (1 constraint). Framing: "Genre and decade are verified via the
API; 'dark psychological' is my judgment — treat the ranking accordingly."
