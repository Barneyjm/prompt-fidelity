# MCP Registry Census — Tool Species Distribution

**Date:** 2026-07-04 · **Source:** registry.modelcontextprotocol.io/v0.1/servers · **Method:** census/RUBRIC.md

## Sample

Full cursor-paginated crawl: 2,500 raw version entries → deduped to latest version per name → **999 active unique servers** (this was the whole registry at crawl time, not a sample). 899/999 publish a remote endpoint. Classification: name + description only (Tier A), 40 batches × 25, one Sonnet classifier per batch at temperature-equivalent settings, schema-forced JSON. Spot-check: 4 batches (100 servers) independently re-classified — **inter-run agreement 92%**.

## 1. Distribution (n = 999; 15 unknown/excluded, 1.5%)

| Species | n | % of classified |
|---|---|---|
| predicate | 349 | 35.5% |
| actuation | 307 | 31.2% |
| generative | 187 | 19.0% |
| relevance | 141 | 14.3% |

Secondary species recorded for 479 servers; predicate is by far the most common secondary (260) — most actuation and relevance servers also expose structured lookups.

## 2. Transmitted-heavy hypothesis: **not supported**

Relevance + generative combined = 328 servers = **33.3%** of classified — almost exactly one third, statistically tied with predicate (35.5%) and actuation (31.2%). The ecosystem being *built* is not dominated by advisory-parameter tools; it splits into rough thirds across verification regimes:

- ~⅓ **verifiable by construction** (predicate: enforced params),
- ~⅓ **verifiable by effect** (actuation: read the world back),
- ~⅓ **transmitted at best** (relevance + generative: advisory params or model output).

Reframed for the Four Promises doc: two-thirds of the registry *can* make a hard promise; one-third structurally cannot, and that third is where inference laundering lives (generative alone is 19% — every one of those servers returns model output in a tool envelope).

## 3. Actuation and read-back

Actuation share: 31.2%. Of actuation servers, only **19.5%** (60/307) *mention* any read-back verb in their description. But Tier B (below) shows descriptions dramatically undersell this: **8 of 9** ground-truth actuation servers actually ship read-back tools (get/status/list alongside create/update). The verified-by-effect assurance tier is architecturally available far more often than registry descriptions advertise — the capacity exists; the framing doesn't.

## 4. Tier B: live tools/list ground truth (error rate of Tier A)

Probed registry remotes with a real MCP streamable-http handshake (initialize → tools/list) until 40 servers yielded live tool inventories. Outcomes across 107 attempts: **40 ok (37%), 48 auth-required (45%), 19 error/timeout (18%)**.

Ground-truth classification from actual tool schemas vs. Tier A description-based labels (n = 40):

| Agreement | n | % |
|---|---|---|
| Exact (dominant matches) | 25 | 62.5% |
| Within dominant↔secondary swap | 37 | 92.5% |
| Hard miss | 3 | 7.5% |

Confusion is symmetric (no single dominant error direction); the largest cell is relevance→predicate (3), i.e. Tier A slightly over-reads "search" language on servers whose actual tools are structured lookups. Implication: headline distribution is trustworthy to roughly ±5 points per species; the thirds framing survives the error rate.

## 5. Exemplars (high-confidence)

**predicate** — "Query Meta Ads performance data — accounts, campaigns, ad sets, ads, metrics & settings" (ai.adadvisor); "Validate HMDA LAR files against CFPB edit checks" (ai.clarid/hmda); "Loan & mortgage calculator, compound interest, ROI, crypto prices, FX conversion" (ai.bankee).

**relevance** — "Fast, intelligent web search and web crawling" (ai.exa/exa); "Search 7,000+ local service businesses across America by category, location, or keyword" (ai.bezal); "AI-powered news intelligence — personalized monitoring, briefings, and semantic search" (ai.agentic-news).

**generative** — "Run 150+ AI apps — image, video, audio, LLMs, 3D and more" (ac.inference.sh); "Turn your app idea into IA, wireframes, PRD, style guides, and dev specs" (ai.bunzee); "Create professional visuals from text, URLs, or PDFs" (ai.canvora).

**actuation** — "Google Ads MCP server — manage campaigns, keywords, and metrics" (ai.adramp); "Meta Ads MCP server with 47 tools for campaigns, creatives, audiences, and insights" (ai.adweave); "MCP server for Anki flashcards: adaptive review, notes, media, and deck management" (ai.ankimcp).

## 6. Methodology notes & caveats

- The registry skews to recently published remote servers — this measures **what's being built**, not what's used. Weighting by usage (PulseMCP) would likely shift mass toward relevance (web search) and predicate (databases).
- Confidence self-ratings: H 33.5%, M 55.6%, L 10.9%. "Connector-to-X" thin descriptions were classified from what X is, confidence L, per rubric.
- Registry descriptions are marketing copy written by publishers; Tier B shows they're directionally honest about species (92.5% within-swap agreement) but systematically silent about read-back capability.
- Tier B sample is biased toward unauthenticated servers (45% of remotes refused without auth) — plausibly toward simpler/demo servers.
- Files: `census_tier_a.csv` (name, description, species, secondary, confidence — 999 rows), `servers.json` (crawl), `tier_b_raw.json` (probe transcripts), `tier_b_classifications.json` (ground truth), scripts (`fetch_registry.py`, `tier_b_probe.py`, `aggregate.py`, `compare_tiers.py`).
