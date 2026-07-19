# Fidelity as a Product Signal

The original motivating use case: LLM-prompted playlists, built on the fly
from a catalog API plus model-derived vibe. The product question was never
"is this answer true?" — it was **"how should the UX present this result,
and how much should we spend computing it?"** Fidelity is the number a
product manager injects into that decision.

This document is the operating contract for that use: what the number can
carry in a product, what it must never be asked to carry, and the loop
that keeps it honest.

## Why the number works here when it wouldn't as a universal metric

The [skill](../skills/prompt-fidelity/SKILL.md) and README present
fidelity with caveats: the score depends on pool choice, decomposition
granularity, and estimated rates. Those caveats are about *portability* —
comparing scores across contexts. A product pins every degree of freedom
that makes the number soft:

| Soft spot (in general)              | Pinned in a product                                    |
|-------------------------------------|--------------------------------------------------------|
| Pool choice is arbitrary            | The pool is *the* catalog — fixed                      |
| Decomposition varies by phrasing    | One versioned decomposer prompt, one model, one temp   |
| Inferred rates are freestyle guesses| A versioned rate table (see `schema/tmdb_fields.json`) |
| Report isn't bound to real queries  | Score computed in-pipeline from the constraints that built the actual API call |
| Calibration claim untested          | Engagement telemetry by band tests it continuously     |

Constant biases don't hurt an **ordinal** signal. Within one product
version, "this request is more verifiable than that one" is reliable even
though neither absolute number means anything on its own.

## The contract

1. **Ordinal, within one product version.** Any change to the decomposer
   prompt, model, rate table, or schema shifts the score distribution.
   Version the scorer; re-tune thresholds on every bump; never compare
   raw scores across versions or products.
2. **Routes framing and compute — never quality.** A 100%-fidelity
   playlist can still be a bad playlist. Do not rank results by fidelity,
   do not present it as confidence-of-goodness, do not surface the raw
   number to users.
3. **Bands, not decimals.** Consume it as 2–4 bands with empirically
   tuned cutoffs. The metric is robust at ±10 points; wide bands absorb
   that; decimals imply precision the inputs don't have.
4. **The disclosure rules still bind.** Injected filters (quality floors,
   candidate caps, sampling) must reach the user in plain language in
   every band. The score is optional UX input; the honesty is not.

## Band recipes (playlist example)

**High (≈ ≥ 0.8) — the request is mostly filters.**
Assertive framing: "Every track here matches: 80s · Synthpop · under 4
min." Show the applied constraints as removable chips. Skip or minimize
vibe language. This band is also the **cheap path**: with little or
nothing inferred, skip the LLM re-rank entirely and serve the query
result (the reference pipeline's `should_rerank` branch is the degenerate
version of this).

**Mid (≈ 0.4–0.8) — filters plus vibe.**
Split presentation: "Filtered by your criteria · ordered by our read of
the vibe." Distinguish the two visually. Offer one-tap refinement chips
generated from the skill's reformulation step — objective proxies for the
subjective part ("pin an era?", "instrumental only?"). This is the
mushiest region of the score (verified↔inferred correlation biases it
most), which is fine: it's also the band whose hybrid UX is most
forgiving of misrouting.

**Low (≈ < 0.4) — mostly vibe.**
Exploratory framing: "Here's our take on that mood — tell us what's off."
Per-track feedback affordances, easy regenerate, prominent refinement.
This band justifies **spending more**, not less: bigger candidate pool,
stronger model for the vibe match, since the LLM is doing nearly all the
work. Low fidelity = high model responsibility = where quality budget
belongs.

Cost routing falls out of the same thresholds: fidelity allocates compute
inversely to verifiability, and a misroute costs cents, not user trust.

## The validation loop

Fidelity makes one falsifiable claim: **verifiability should predict
outcome quality.** In a product this is a telemetry query, not a research
project:

- Log the fidelity score (and band) on every generated result.
- Segment engagement by band: skip rate, saves, completion, regeneration,
  refinement-chip usage.
- If bands separate on engagement, the signal is earning its keep. If
  high- and low-fidelity results perform identically, the score is
  decoration for your product — retire it or fix the rate table.
- Tune band cutoffs by experiment, not by aesthetics; re-run after every
  scorer version bump.

This loop is the calibration experiment the framework otherwise lacks,
running continuously and for free.

## Implementation sketch

```python
from agent import compute_fidelity

# constraints: the SAME objects that built the catalog query — the score
# is attested by construction, not self-reported by a model.
report = compute_fidelity(
    constraints,
    pool_size=CATALOG_SIZE,                      # fixed per product
    verified_joint_survival_rate=joint / CATALOG_SIZE,  # from the query's own result count
)

band = ("high" if report.fidelity_score >= HIGH_CUTOFF
        else "mid" if report.fidelity_score >= MID_CUTOFF
        else "low")
result.ux_treatment = BAND_TREATMENTS[band]
result.injected_disclosures = [c.description for c in report.injected_constraints]
```

Inferred rates come from a versioned table keyed by constraint category
(the shape of `schema/tmdb_fields.json`'s `inferred_attributes`), not
from per-request model estimates — that removes the self-grading degree
of freedom and makes scores reproducible and diffable across releases.

## Known soft spots

- **Boundary flapping**: decomposition nondeterminism can band the same
  request differently on retry. Persist the score per session, or add
  hysteresis around cutoffs.
- **Mid-band bias**: unmodeled correlation between verified filters and
  inferred vibe deflates mid-range scores unpredictably. Keep the mid
  band wide.
- **The score is not a satisfaction predictor.** It measures where the
  answer's information came from. Treating it as a quality score will
  fail the telemetry test above — which is the test working.
