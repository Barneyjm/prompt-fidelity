---
name: prompt-fidelity
description: Self-check how much of a tool-backed answer is auditable vs guesswork before giving it. Use when handling search, filtering, recommendation, data-retrieval, or state-changing (action) requests that mix objective criteria (checkable via an API, database, file, or calculation) with subjective judgment (mood, style, quality, "feels like", "best"), or when the user asks how confident, reliable, or verifiable an answer is. Classifies each constraint by pointing at the actual tool-call evidence, computes an auditability score, and reports which parts of the answer are verified, delivered-but-unenforced, altered/dropped, or judgment calls.
---

# Prompt Fidelity Self-Check

Prompt fidelity measures what fraction of a request's information content
was **verifiably executed** versus what rode on LLM judgment. This skill is
the *behavioral* tier of that measurement: a disciplined self-report you run
before answering, with an arithmetic script — no dependencies, no
instrumentation.

```
Auditability = (verified_bits + transmitted_bits) / total_bits
```

The headline is **auditability**: of what the user asked, how much left a
mechanical evidence trail an auditor could check? Strict fidelity
(verified only) is also reported — it is deliberately harsher, since for
search and action tools most intent structurally cannot appear verbatim in
enforced parameters no matter how well you did.

> **This is a self-report, and the report says so.** The mechanical version
> of this measurement — booking constraints against recorded tool calls,
> results, and narration with no model judgment in the loop — is the
> `promptfidelity` package in this repo. **If that package is importable,
> use it (`pf.trace()` / `pf.book()`) instead of self-classifying**; this
> skill is the fallback tier for environments where it isn't. The script's
> output block carries `classification: self-reported` so a reader can
> always tell which tier produced the number.

## When to run this check

- The request filters or ranks items by a mix of objective and subjective
  criteria ("Python repos updated this year that are well-designed").
- The request asks you to *change state* — create an event, start a device,
  build a cart, send a message.
- The user asks how confident or verifiable your answer is.
- You are about to present results where some criteria were actually
  checked and others were your own judgment — the user deserves to know
  which is which.

## Workflow

### Step 1 — Decompose the request into atomic constraints

Break the request into individual constraints. Rules:

- Each constraint is one independent requirement.
- A range counts as ONE constraint ("From the 90s" = "Released 1990–1999"),
  not two.
- Cover EVERY piece of intent, including subjective and contextual ones —
  drop nothing. Ignore only filler with no selectivity ("some good",
  "please find me").

### Step 2 — Classify each constraint BY POINTING AT THE EVIDENCE

Do not classify by asking "could this be checked in principle." Classify by
quoting the **actual tool call you made** (or are about to make). For each
constraint, find the argument that carried it:

- **verified** — an ENFORCED parameter carried it with the right value
  (`with_genres=878`, `zipCode=28209`, a SQL WHERE clause). Quote the
  argument in your report. The backend guarantees results satisfy it.
- **transmitted** — it was delivered via a free-text/advisory input (a
  search query, a ranker hint) the backend does not enforce. Rewording is
  fine — "less than five hundred bucks" arriving as `"under $500"` is still
  transmitted — but the *content* must survive. Quote both your words and
  the argument.
- **unhonored** — you can see in the actual call that the value was ALTERED
  (you asked 8.0, the call said 7.0) or the constraint never reached any
  call at all despite being expressible. This is not a judgment call — it
  is a visible gap between request and execution.
- **inferred** — no tool input could express it (mood, tone, "feels like",
  suitability). It rode on your judgment alone.

The test is mechanical: *can you quote the argument?* If you cannot point
at it, it is not verified and not transmitted — whatever a script could
have done in principle. When in doubt, classify downward: overclaiming
verification is worse than underclaiming it.

**If Step 2 finds unhonored constraints, repair before answering**: re-issue
ONE complete call carrying every previously-honored argument plus the
missing/corrected ones, then re-run this step. Only report unhonored for
what you genuinely could not repair — and disclose those plainly.

### Step 3 — For actions, check the effect

Argument-level checks are only half the story for state-changing calls. A
request can carry every argument perfectly and still fail (permission
denied, validation error), or "succeed" with no evidence either way. Before
narrating an action as done:

- **Read the result.** If it is error-shaped, the action FAILED — say so
  explicitly, never paper over it with a silent retry. A failed call the
  user is never told about is the single worst outcome this skill exists
  to prevent.
- **Read state back when you can.** A follow-up read that reflects your
  write ("zone status: WATERING") is the strongest confirmation there is —
  prefer it over trusting a silent acknowledgment.
- **If you can't confirm, say "unaudited."** "The zone should be running —
  the controller acknowledged the start but I couldn't confirm state" is
  honest; "Done!" is not.

### Step 4 — Estimate survival rates

For each constraint, estimate the fraction of the relevant candidate pool
that satisfies it (a decimal in (0, 1)). Information content is
`-log2(survival_rate)`. Calibration guide:

| Selectivity | Survival rate | Bits |
|---|---|---|
| Extremely selective (specific person, exact ID) | 0.001–0.01 | 7–10 |
| Very selective (exact year, rare property) | ~0.01 | ~6.6 |
| Moderately selective (a genre, a decade, a language) | 0.05–0.15 | 2.7–4.3 |
| Broad (rating above average, common property) | 0.3–0.5 | 1–1.7 |

### Step 5 — Compute the score

Write the constraints as a JSON array and run the bundled script (stdlib
only, no installs). The script lives at `scripts/compute_fidelity.py`
relative to this SKILL.md:

```bash
python3 <skill-dir>/scripts/compute_fidelity.py constraints.json
```

Each constraint object needs `description`, `type` ("verified",
"transmitted", "unhonored", or "inferred"), `estimated_survival_rate`, and —
for verified/transmitted/unhonored — `evidence`: the quoted argument from
Step 2. Add `--json` for machine-readable output. Include the printed
report block verbatim in your response.

### Step 6 — Answer with calibrated framing

Frame by **auditability**, and disclose unhonored constraints regardless of
the score:

- **≥ 80%**: "These results are checkable on the stated criteria." Verified
  parts are guaranteed; transmitted parts were delivered but compliance
  isn't enforced — say which is which.
- **40–80%**: Separate the layers explicitly: "Filtered by X (enforced);
  searched for Y (delivered to the ranker); ranked by Z (my judgment)."
- **< 40%**: Say plainly that the answer is mostly judgment, present it as
  a best guess, and offer a higher-fidelity reformulation — objective
  proxies for the subjective criteria ("'feels like a slow burn'" →
  "runtime > 120 min, drama genre, pre-2010").
- **Any unhonored constraint**: name it and what actually happened ("you
  asked for 8.0+; the search ran at 7.0"). Never let the narration promise
  what the execution didn't deliver — that gap, not a low score, is the
  failure mode.

## Example

Request: *"Dark psychological thrillers from the 90s, rated above 8"* with
a movie API available. The discover call actually made:
`discover(with_genres=53, date.gte=1990-01-01, date.lte=1999-12-31, rating.gte=7.0)`.

```json
[
  {"description": "Thriller genre", "type": "verified",
   "evidence": "with_genres=53", "estimated_survival_rate": 0.10},
  {"description": "Released 1990-1999", "type": "verified",
   "evidence": "date.gte=1990-01-01, date.lte=1999-12-31", "estimated_survival_rate": 0.10},
  {"description": "Rated above 8", "type": "unhonored",
   "evidence": "asked 8.0, call carried rating.gte=7.0", "estimated_survival_rate": 0.05},
  {"description": "Dark/psychological tone", "type": "inferred", "estimated_survival_rate": 0.15}
]
```

Framing: "Genre and decade are enforced by the API. **The rating floor was
relaxed to 7.0 en route — you did not get the 8.0 cutoff you asked for.**
'Dark psychological' is my judgment — treat the ranking accordingly."
