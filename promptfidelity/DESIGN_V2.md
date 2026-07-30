# Prompt Fidelity v2 — Architecture

**The original problem:** measure how much of a prompt's intent an agent actually
executes — trustworthily (the measurement must never be the model grading itself),
cheaply (measurement cannot cost more than the work), and at every scale from one
run to a fleet.

v1 established the accounting core: constraints carrying bits, booked mechanically
into a conserved partition (verified / transmitted / substituted / inferred /
dropped, with agent-side imposed segregated). This document is the v2 design, and
every change in it traces to something measured this session, not something argued.

## What the evidence broke

| v1 assumption | What real data showed | v2 answer |
|---|---|---|
| Hop-1 booking tells the story | 70% of chat-conversation referents booked `dropped` — but most were *answered from model knowledge*, not ignored. Hop 1 cannot distinguish the agent's biggest failure mode from its most common legitimate behavior. | **Hop-2 attribution** (§3) |
| Extraction can be deterministic | Regex found 116 referents where LLM extraction found 322 in identical text (2.8× undercapture), and regex's better delivery rate was selection bias — it only sees intent that already looks like tool arguments. | **Tiered capture** (§1) |
| Exact matching suffices | Models paraphrase user intent into their own query vocabulary; literal tokens vanish and book as dropped. 38 referents recovered by a disclosed-threshold fuzzy matcher. | **Paraphrase tier in matching** (§2) |
| A run is one intent set | Multi-turn conversations revise intent ("actually, 5 minutes"); late-turn referents also deliver at lower rates than openers (53% → 36% when full conversations were audited). | **Intent lifecycle** (§5, partially open) |
| Best-per-constraint merging is enough | A "complete repair" pattern read as incoherent under first-wins tie-breaks; separate calls can each honor pieces while no result set honors the conjunction. | **conjunction_honored + later-call-wins** (shipped) |

## The five layers

### 1. Capture — where constraints come from
Priority order, each tagged in `constraints_source` so every ledger discloses what
its measurement rests on:

1. **Source instrumentation** (`declared`): tap the agent's own decomposition/
   planning step. Every tool-using agent already converts prose to structure; the
   instrumentation taps that conversion instead of reimplementing it. Zero
   marginal extraction cost, highest fidelity. This is the deployment
   recommendation whenever you own the agent.
2. **LLM extraction** (`llm`): for traffic you don't control (audits, archaeology).
   Cheap model, pluggable callable, entries only. Extraction quality is the
   measurement's noise floor — measured, not assumed (spot-check protocol: 10%
   double-extraction, report agreement).
3. **Deterministic floor** (`rules`): regex/vocab for the high-precision subset
   where exact semantics are correct anyway — URLs, IDs, quantities-with-units.
   Never the primary capture layer in deployment.

**Invariant (unchanged, load-bearing): LLMs may propose entries, never accounts.**

### 2. Hop 1 — intent → tool call (the existing ledger)
Mechanical diff of declared params against actual call arguments. Conservation:
the five user-side accounts partition I_total exactly; imposed is segregated
agent-side. Matching is tiered like capture: exact param match → unit-normalized
match (10 minutes ≡ 600 seconds) → paraphrase match for advisory params only
(disclosed threshold, counted separately). Books balance at every rollup level;
a rollup that doesn't balance is detected data loss.

### 3. Hop 2 — tool result → narration (new)
For each booked constraint: did the response **address** it, and was supporting
material **present in the tool results the agent received**?

    status ∈ { unaddressed, addressed_grounded, addressed_ungrounded }

The two-dimensional **fate matrix** (hop-1 account × hop-2 status) is the complete
life of a constraint. The cells that matter most:

- `dropped × addressed_ungrounded` — **answered from model knowledge.** The
  population hop-1 alone mislabels as failure. In chat traffic this is the
  dominant cell, and it is exactly the inference-vs-verification tradeoff the
  original article named.
- `dropped × unaddressed` — **truly ignored.** The real failure.
- `substituted|dropped × addressed_*` — **narration risk**: spoken about without
  verified execution. The per-constraint, fully mechanical form of the honesty gap.
- `verified × addressed_grounded` — the gold path.

Epistemics, stated plainly: hop 2 measures **provenance, not truth**. Grounded
means supporting material existed in results; ungrounded may be perfectly sound
model knowledge. The point is that the user currently can't tell — and now the
ledger says which is which. End-to-end fidelity is the product of the hops.

### 4. Control — the ledger as a loop, not a report
`rec.check()` mid-run; `render("model")` injection; one-complete-call repair;
bounded iterations; `conjunction_honored` as the coherence backstop; fidelity
gates as deploy/CI policy ("no new imposed params" is the highest-value alert).
The model can't argue with the ledger; it can only act until the ledger changes.

### 5. Aggregation — receipt → statement → close → audit → industry
Ledgers are additive because the accounts partition the total: fleet fidelity is
computable exactly from shards, and the trial balance detects loss at every rung.
Emission rides existing telemetry (OTel attributes). Above ~10k agents, privacy
forces aggregation to (account, bits, param, tool, species) tuples — constraint
descriptions are user data and stay home. The census measured the supply side of
the ecosystem; fleet ledgers measure the demand side; the gap between them is the
industry story.

## What stays mechanical vs. pluggable

| Function | Mechanical (always) | Pluggable LLM (tagged, optional) |
|---|---|---|
| Account assignment | book()'s diff — the only assignment site | never |
| Hop-2 status | string/token-overlap vs disclosed thresholds | claim decomposition for long prose (extension point) |
| Capture | rules floor; source instrumentation | extraction for uncontrolled traffic |
| Matching | exact, unit-normalized, threshold paraphrase | semantic similarity (future tier; deterministic given a pinned embedding model) |
| Aggregation keys | params, tools, accounts, species | cluster labels for readability only |

## Open problems (named, not hidden)

1. **Supersession**: "actually, make it 5" books as an independent constraint;
   the superseded ask inflates dropped. Candidate: same-param/same-unit revision
   detection at the audit layer, flagged not deleted (conservation preserved).
2. **Threshold calibration**: 0.5/0.6 word-overlap fractions are disclosed but
   uncalibrated. Protocol: human-label a sample, publish precision/recall per
   threshold — an instrument gets a datasheet.
3. **Semantic matching**: paraphrase recall beyond token overlap needs embeddings —
   deterministic given a pinned model, but introduces a dependency; belongs in an
   extra, never core.
4. **Cross-agent handoffs**: nested ledgers for generative tools exist in the
   spec; wiring provenance through A2A boundaries (the laundering fix) is
   unimplemented.

## One instrument, four altitudes (unchanged)

model → act on unhonored intent · engineer → evidence and call indices ·
product → dropped/substituted intent is a ranked demand backlog · executive →
fidelity headline, undisclosed-filter count, overclaim-risk verdict. Hop 2 adds
per-audience: provenance-risk counts (how much of what we told users came from
nowhere we can point to).
