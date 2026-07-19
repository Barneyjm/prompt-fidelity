# Prompt Fidelity

A [Claude Agent Skill](https://code.claude.com/docs/en/skills) that makes Claude self-check how much of an answer is *verified* against real data versus *inferred* through its own judgment — and say so.

```
Fidelity = verified_bits / (verified_bits + inferred_bits)
```

Any time a request mixes objective criteria (checkable via a query, count, or calculation) with subjective judgment (mood, style, quality, "feels like", "best"), the skill decomposes the request into constraints, classifies each as verified or inferred, computes a fidelity score with a bundled dependency-free script, and lets that score shape how confidently Claude states its answer. It works on any dataset Claude has tool access to — an API, a SQL database, a directory of files, a spreadsheet — not just the movie example used throughout this README.

## Install

### Claude Code (plugin marketplace)

```
/plugin marketplace add barneyjm/prompt-fidelity
/plugin install prompt-fidelity@prompt-fidelity
```

### Claude Code (manual)

```bash
# Personal (all projects)
cp -r skills/prompt-fidelity ~/.claude/skills/

# Or per-project
cp -r skills/prompt-fidelity your-project/.claude/skills/
```

### claude.ai

Zip the skill folder and upload it under **Settings → Capabilities → Skills**:

```bash
cd skills && zip -r prompt-fidelity.zip prompt-fidelity
```

Claude invokes the skill automatically when a request mixes objective and subjective criteria, or you can invoke it explicitly with `/prompt-fidelity` in Claude Code.

## How it presents the score

In normal conversation the score shapes the answer rather than appearing as a chart: Claude states in prose which parts came straight from the data and which are its judgment ("the counts and dates are from the database; which ones sound serious is my reading"), citing at most a round percentage. The full report block — the bar, per-constraint bits, correlation adjustment — appears when you ask for the score or audit, invoke the skill explicitly, or the output is going into a file or eval; the script's `--brief` flag covers the middle ground.

Framing follows the score band:

- **≥ 80%**: results should pass an audit of the stated criteria — "These results are verifiably correct."
- **40–80%**: the two layers are named explicitly — "Filtered by X and Y (verified); ranked by Z (my judgment)."
- **< 40%**: the answer is presented as a best guess, with a higher-fidelity reformulation offered.

Injected filters (quality floors, top-N truncation, sampling, default sort order) and coverage limits are always disclosed, regardless of score.

## How the score is calculated

Each constraint has a **survival rate** — the fraction of the candidate pool that satisfies it — measured from the system's own counts wherever possible (an API's `total_results`, planner/catalog statistics, a sampled count) and only estimated as a fallback. The **information content** in bits is `-log2(survival_rate)`. Fidelity is the ratio of verified bits to total bits.

Per-constraint bits are capped at `log2(pool_size)` — a constraint can't carry more information than it takes to identify a single row in the candidate pool. The default cap is 20 bits (a one-in-a-million pool); pass the real pool size via `--pool-size` for larger datasets so near-unique selectors (exact IDs) are weighted correctly.

Three mechanisms keep the framework honest beyond toy examples:

- **Measured vs. estimated survival rates.** Constraints carry a `rate_source` of `"measured"`, `"approximated"`, or `"estimated"` so the report shows whether the constraint *weights* are themselves verified, not just the constraints.
- **Injected filters.** Any filter the system applies that the user never requested is declared with `"type": "injected"`. These are excluded from the score but always listed in the report — silently narrowing the pool is the main way a "100% fidelity" claim becomes dishonest.
- **Correlation correction.** Summing bits assumes constraints filter independently, which real data violates. When the datastore can cheaply measure the joint count of all verified filters ANDed together, pass it (`verified_joint_count` / `--joint-count`) and the verified side of the score uses the exact measured joint information instead of the independence sum. For a large adjustment, pairwise counts (`--pairwise`, or by hand via `log2((n_A × n_B) / (pool × n_AB))`) or an indicator correlation matrix over a judged sample can attribute *where* the correlation comes from.

One more subtlety: "verified" means the *query* is mechanically checkable, not that it faithfully captures intent. A crowd-sourced `melancholy` keyword tag is a verified filter but a noisy proxy for a melancholy tone — the skill's guidance is to split such constraints into a verified query plus an inferred semantic gap.

Full mechanics, worked examples, and the constraint-decomposition workflow live in [`skills/prompt-fidelity/SKILL.md`](skills/prompt-fidelity/SKILL.md).

### Running the script directly

```bash
python3 skills/prompt-fidelity/scripts/compute_fidelity.py constraints.json
```

```python
from agent import compute_fidelity

constraints = [
    {"description": "Thriller genre", "type": "verified", "estimated_survival_rate": 0.10, "rate_source": "measured"},
    {"description": "Dark tone", "type": "inferred", "estimated_survival_rate": 0.15, "rate_source": "estimated"},
]

report = compute_fidelity(constraints, pool_size=1_000_000_000)
print(f"Fidelity: {report.fidelity_score:.1%}")
```

## Validation on real open data

`experiments/socrata_fidelity.py` runs the whole workflow against live Socrata-hosted datasets (NYC Open Data, Chicago, CDC, and most data.gov-federated portals speak the same SODA API) using only server-side counts — no scans, no downloads:

```bash
python3 experiments/socrata_fidelity.py experiments/specs/nyc_311_noise.json
python3 experiments/socrata_fidelity.py --sample 5 experiments/specs/chicago_theft.json
python3 experiments/run_suite.py --quiet   # run every spec, emit the table below
```

Each run measures the pool size and every verified constraint's survival rate from the system's own counts, scores the request with the correlation correction applied, and declares the judging sample as an injected filter. `run_suite.py` runs every spec as a regression suite and exits nonzero on failure, so it can gate CI.

Results across five cities and five data shapes (suite output):

| Spec | Domain | Pool | Fidelity | Verified bits (joint) | Correlation adj. |
|---|---|---|---|---|---|
| `austin_pitbull_adoptions` | data.austintexas.gov | 173,775 | 82.5% | 8.17 (summed 8.40) | -0.23 bits |
| `chicago_theft` | data.cityofchicago.org | 8,595,766 | 76.9% | 9.11 (summed 9.33) | -0.22 bits |
| `moco_speeding` | data.montgomerycountymd.gov | 2,137,572 | 67.7% | 6.96 (summed 6.68) | +0.28 bits |
| `nyc_311_noise` | data.cityofnewyork.us | 21,848,232 | 72.4% | 8.70 (summed 8.57) | +0.12 bits |
| `seattle_aid_calls` | data.seattle.gov | 2,187,508 | 69.0% | 5.16 (summed 5.06) | +0.10 bits |

Every adjustment lands within ±0.3 bits against 5–9 verified bits, so the independence sum is a good approximation on real civic data — but with the joint count measured, the verified side no longer needs the approximation at all. A negative adjustment (Chicago, Austin) means the constraints are positively correlated — the joint pool is larger than independence predicts, so the sum *overstated* the verified information; a positive adjustment (NYC, Seattle, Montgomery County) means mildly negatively correlated constraints, where the sum understated it. Write a new spec JSON to test any other Socrata dataset.

### Case studies: answering real questions with the skill

Two end-to-end runs, each answering a natural question a resident might actually ask. Total API footprint per run: a handful of server-side counts and one small fetch of the matching rows.

**"Were there a lot of illegal fireworks complaints in Williamsburg around July 4th? Which sound like full shows vs stray firecrackers?"** (NYC 311, 21.8M rows)

```
PROMPT FIDELITY: 91.6%
  ✓ Illegal Fireworks complaints     (7.48 bits, measured)
  ✓ Williamsburg zips 11211/11249    (6.31 bits, measured)
  ✓ June 28 – July 6, 2026           (7.63 bits, measured)
  ? "full show vs firecrackers"      (1.74 bits, estimated)
  Correlation: summed 21.43 bits → joint 18.89 (-2.54 adjustment)
```

The verified half answered richly: 45 complaints, peaking at 19 on July 4th, with a repeat-complaint hot spot on South 2nd Street. The inferred half hit a wall — all 45 records had descriptor "N/A" and boilerplate resolutions — so the answer said plainly that the data cannot distinguish shows from firecrackers, and offered the one verifiable proxy (repeat complaints at one address in one night) clearly labeled as inference. A high fidelity score means the *verified part dominates the request*, not that every part is answerable. The −2.54 bit adjustment reflects a real seasonal correlation: fireworks complaints barely exist outside that week, so complaint type and date range heavily overlap.

**"How bad have car break-ins been in Logan Square this summer? Do they look targeted or random?"** (Chicago crimes, 8.6M rows)

```
PROMPT FIDELITY: 94.5%
  ✓ Vehicle break-in (theft/burglary from vehicle)  (8.76 bits, measured)
  ✓ Logan Square (community area 22)                (5.71 bits, measured)
  ✓ Jun 1 – Jul 18, 2026                            (8.40 bits, measured)
  ? "targeted vs random"                            (1.00 bits, estimated)
  Correlation: summed 22.86 bits → joint 17.23 (-5.63 adjustment)
```

Verified: 56 break-ins, up 33% from 42 in the same window last year; 41 of 56 street parking; zero arrests; 17 of the 56 in a single June 2–4 burst. The inferred judgment ("systematic about the area, random about the victim") was grounded in checkable patterns — burst days and no block hit more than twice — with the line drawn explicitly between database facts and interpretation. Two mechanisms earned their keep here: a cheap group-by probe *before* decomposing revealed that "car break-ins" spans two encodings (guessing would have silently dropped 57% of the answer), and the −5.63 bit correlation adjustment absorbed a data-quality landmine — those description labels barely exist before ~2024 because Chicago changed its coding taxonomy, making the all-time per-constraint rate meaningless. Independence predicted ~1 matching row; the measured joint count of 56 kept the score correct despite 25 years of label drift.

## Reference implementation: the movie agent

This repo also ships a small LangGraph movie-recommendation agent (`agent/`) that queries TMDb. It's the worked example the skill's docs and calibration numbers are drawn from, and it's a convenient way to see the fidelity report end to end without wiring up your own dataset — but it's not required to use the skill, which works with whatever tools Claude already has in a session.

### Run it

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env: TMDB_API_KEY (required), ANTHROPIC_API_KEY (required for the Claude provider)

python -m agent.main "Dark psychological thrillers from the 90s"
python -m agent.main --interactive
python -m agent.main --json "Sci-fi movies with a melancholy tone"
python -m agent.main --provider openai "Horror movies from Japan"
```

### Sample output

```
$ python -m agent.main "Action movies from the 1980s rated above 7.0"

══════════════════════════════════════════════════
  PROMPT FIDELITY: 100.0%
══════════════════════════════════════════════════

  [████████████████████] 100.0%

  Constraint Breakdown:
  ────────────────────────────────────────

  VERIFIED (queryable via API):
  ✓ Action genre (3.06 bits)
  ✓ Released in the 1980s (3.32 bits)
  ✓ Rating above 7.0 (2.00 bits)

  ────────────────────────────────────────
  Verified:    8.38 bits (3 constraints)
  Inferred:    0.00 bits (0 constraints)
  Total:       8.38 bits
══════════════════════════════════════════════════

1. The Empire Strikes Back (1980) - 8.4/10
2. Scarface (1983) - 8.2/10
3. Aliens (1986) - 8.0/10
4. Raiders of the Lost Ark (1981) - 7.9/10
5. Indiana Jones and the Last Crusade (1989) - 7.8/10
```

Every result here is verifiably correct: action film, from the 80s, rated above 7.0 — all auditable against TMDb, no LLM judgment involved. Compare a fully-subjective query like `"Movies that feel like a rainy Sunday afternoon"`, which scores 0% fidelity and returns best-guess matches with decaying confidence instead. Both queries carry the same information content (8.38 bits); the difference is entirely in whether that information came from a query or from LLM inference.

### Architecture

```
User prompt
    ↓
┌─────────────────────────────────────┐
│  LLM Decomposition                  │
│  Break prompt into constraints      │
│  Classify as VERIFIED or INFERRED   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  TMDb API Query                     │
│  Build query from verified          │
│  constraints only                   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  LLM Re-ranking                     │
│  Score candidates against           │
│  inferred constraints               │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Fidelity Report                    │
│  Compute and display fidelity       │
│  breakdown to user                  │
└─────────────────────────────────────┘
```

### TMDb verified fields

| Field | TMDb Parameter | Typical Survival Rate |
|-------|---------------|----------------------|
| Genre | `with_genres` | 5-15% per genre |
| Release Year | `primary_release_year` | ~1% |
| Release Decade | `primary_release_date.gte/lte` | ~10% |
| Rating (min) | `vote_average.gte` | 5-50% |
| Runtime | `with_runtime.gte/lte` | 20-50% |
| Language | `with_original_language` | 1-50% |
| Cast | `with_cast` (ID lookup) | 0.1-1% |
| Director | `with_crew` (ID lookup) | 0.01-0.1% |
| Keywords | `with_keywords` (ID lookup) | 1-10% |

Inferred attributes (mood/tone, cinematography, themes, comparative feel, pacing, emotional arc) can't be queried through TMDb and are always classified as inferred by the agent.

### Python API

```python
from agent import MovieRecommendationAgent

agent = MovieRecommendationAgent(provider="anthropic")
result = agent.recommend("90s thrillers with a dark tone")

print(f"Fidelity: {result['fidelity']['fidelity_score']:.1%}")
for movie in result['movies'][:5]:
    print(f"- {movie['title']} ({movie['year']})")
```

## Repository layout

```
prompt-fidelity/
├── .claude-plugin/
│   ├── plugin.json       # Claude Code plugin manifest
│   └── marketplace.json  # Plugin marketplace manifest
├── skills/
│   └── prompt-fidelity/
│       ├── SKILL.md      # The installable skill
│       └── scripts/
│           └── compute_fidelity.py  # Stdlib-only fidelity calculator
├── agent/                # Reference implementation: movie recommendation agent
│   ├── main.py           # LangGraph workflow and CLI
│   ├── decompose.py      # Constraint decomposition (LLM)
│   ├── fidelity.py       # Fidelity calculation
│   ├── tmdb.py           # TMDb API wrapper
│   └── rerank.py         # LLM re-ranking
├── schema/
│   └── tmdb_fields.json  # Verified field definitions
├── examples/
│   └── sample_prompts.json  # Categorized test prompts
├── experiments/
│   ├── socrata_fidelity.py  # Live validation against Socrata open datasets
│   ├── run_suite.py         # Run all specs, emit comparison table
│   └── specs/               # Experiment specs (NYC, Chicago, Seattle, ...)
├── requirements.txt
└── .env.example
```

## Why this matters

1. **Transparency**: users understand which parts of a request were reliably satisfied vs. LLM guesses.
2. **Calibration**: lower fidelity means more uncertainty, so the user knows to check the results more carefully.
3. **Design guidance**: surfaces what queryable fields or metadata would raise fidelity for a given kind of request.
4. **Research**: enables empirical study of the gap between user intent and tool capability.

## License

MIT

Use of TMDB in the reference implementation is for non-commercial purposes only. This product is not endorsed or certified by TMDB. Find out more about TMDB here: https://www.themoviedb.org/
