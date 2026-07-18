# Prompt Fidelity

A movie recommendation agent that demonstrates the **prompt fidelity** framework—measuring how much of a user's intent can be reliably satisfied through structured API queries versus requiring LLM inference.

## The Concept

When you ask an AI agent for recommendations, your request contains different types of constraints:

- **Verified constraints**: Can be directly queried through an API (genre, year, rating, runtime, language, cast, director)
- **Inferred constraints**: Require subjective LLM judgment (mood, tone, themes, "feels like...", pacing, emotional arc)

**Prompt fidelity** is the fraction of your request's information content that can be verified:

```
Fidelity = verified_bits / (verified_bits + inferred_bits)
```

A fidelity score of 1.0 means your request can be fully satisfied through database queries. A score near 0 means almost everything depends on LLM judgment.

## The Theory in Action: 100% vs 0% Fidelity

Same system. Same architecture. Same LLM. Completely different reliability guarantees.

### High Fidelity (100%) — Provably Correct

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

**Every single result is verifiably correct.** Action film? Check. From the 80s? Check. Rated above 7.0? Check. You could audit each one against TMDb and nothing would be wrong. The LLM made zero judgment calls.

### Low Fidelity (0%) — Best Guess

```
$ python -m agent.main "Movies that feel like a rainy Sunday afternoon"

══════════════════════════════════════════════════
  PROMPT FIDELITY: 0.0%
══════════════════════════════════════════════════

  [░░░░░░░░░░░░░░░░░░░░] 0.0%

  Constraint Breakdown:
  ────────────────────────────────────────

  INFERRED (requires LLM judgment):
  ? Cozy, contemplative atmosphere (2.32 bits)
  ? Slower pacing matching Sunday mood (2.00 bits)
  ? Intimate, character-driven storytelling (1.74 bits)
  ? Nostalgic, melancholic, or reflective tone (2.32 bits)

  ────────────────────────────────────────
  Verified:    0.00 bits (0 constraints)
  Inferred:    8.38 bits (4 constraints)
  Total:       8.38 bits
══════════════════════════════════════════════════

1. The Shawshank Redemption (1994) - Match: 92/100
2. The Godfather (1972) - Match: 80/100
3. My First Client (2019) - Match: 72/100
   ...
10. La bicicleta de los Huanca (2007) - Match: 55/100
```

**The results are defensible but unverifiable.** Shawshank and The Godfather? Reasonable picks. But notice the match scores decay from 92 to 55—the LLM is running out of conviction. With no verified constraints to anchor the search, the candidate pool was the entire TMDb catalog, and the LLM had to do all the work. Some picks are great; others are the model reaching for obscure films it isn't confident about.

### The Insight

Both queries have **identical total information content** (8.38 bits). The difference is entirely in *where that information comes from*:

| Query | Verified | Inferred | Fidelity |
|-------|----------|----------|----------|
| "Action movies from the 1980s rated above 7.0" | 8.38 bits | 0 bits | **100%** |
| "Movies that feel like a rainy Sunday afternoon" | 0 bits | 8.38 bits | **0%** |

This is the fidelity frontier in action. The first prompt sits at the maximum—every bit of specificity maps to a queryable field. The second prompt sits at the minimum—the entire request requires LLM inference.

## Use It as a Claude Skill

The framework is also packaged as an installable [Agent Skill](https://code.claude.com/docs/en/skills) in [`skills/prompt-fidelity/`](skills/prompt-fidelity/). Once installed, Claude self-checks its own answers: before responding to a search, filtering, or recommendation request, it decomposes the request into verified vs. inferred constraints, computes the fidelity score with a bundled dependency-free script, and reports which parts of its answer are auditable and which are judgment calls. It generalizes beyond movies — "verified" means checkable with whatever deterministic tools are available in the session (APIs, databases, files, code).

### Install in Claude Code (plugin marketplace)

```
/plugin marketplace add barneyjm/prompt-fidelity
/plugin install prompt-fidelity@prompt-fidelity
```

### Install manually (Claude Code)

```bash
# Personal (all projects)
cp -r skills/prompt-fidelity ~/.claude/skills/

# Or per-project
cp -r skills/prompt-fidelity your-project/.claude/skills/
```

### Install on claude.ai

Zip the skill folder and upload it under **Settings → Capabilities → Skills**:

```bash
cd skills && zip -r prompt-fidelity.zip prompt-fidelity
```

Claude invokes the skill automatically when a request mixes objective and subjective criteria, or you can invoke it explicitly with `/prompt-fidelity` in Claude Code.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up API keys

```bash
cp .env.example .env
# Edit .env and add your API keys:
# - TMDB_API_KEY (required): https://www.themoviedb.org/settings/api
# - ANTHROPIC_API_KEY (required if using Claude, the default provider): https://console.anthropic.com/
```

### 3. Run the agent

```bash
# Single query
python -m agent.main "Dark psychological thrillers from the 90s"

# Interactive mode
python -m agent.main --interactive

# JSON output
python -m agent.main --json "Sci-fi movies with a melancholy tone"

# Use OpenAI instead of Claude
python -m agent.main --provider openai "Horror movies from Japan"
```

## Sample Output

```
╔══════════════════════════════════════════════════════════╗
║  MOVIE RECOMMENDATIONS                                    ║
╚══════════════════════════════════════════════════════════╝

Query: "Dark psychological thrillers from the 90s that feel like a David Fincher film"

══════════════════════════════════════════════════
  PROMPT FIDELITY: 48.2%
══════════════════════════════════════════════════

  [████████████░░░░░░░░] 48.2%

  Constraint Breakdown:
  ────────────────────────────────────────

  VERIFIED (queryable via API):
  ✓ Genre: Thriller (2.74 bits)
  ✓ Released 1990-1999 (3.32 bits)

  INFERRED (requires LLM judgment):
  ? Dark/psychological tone (2.74 bits)
  ? Fincher-like aesthetic (4.32 bits)

  ────────────────────────────────────────
  Verified:    6.06 bits (2 constraints)
  Inferred:    7.06 bits (2 constraints)
  Total:      13.12 bits
══════════════════════════════════════════════════

RECOMMENDATIONS:
──────────────────────────────────────────────────
1. Primal Fear (1996) - 7.7/10
   Director: Gregory Hoblit
   Cast: Richard Gere, Laura Linney, Edward Norton
   Match Score: 92/100
   → Courtroom thriller with dark psychological elements...
```

## Architecture

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

## Project Structure

```
prompt-fidelity/
├── .claude-plugin/
│   ├── plugin.json       # Claude Code plugin manifest
│   └── marketplace.json  # Plugin marketplace manifest
├── skills/
│   └── prompt-fidelity/
│       ├── SKILL.md      # Installable self-check skill for Claude
│       └── scripts/
│           └── compute_fidelity.py  # Stdlib-only fidelity calculator
├── agent/
│   ├── __init__.py       # Package exports
│   ├── main.py           # LangGraph workflow and CLI
│   ├── decompose.py      # Constraint decomposition (LLM)
│   ├── fidelity.py       # Fidelity calculation
│   ├── tmdb.py           # TMDb API wrapper
│   ├── rerank.py         # LLM re-ranking
│   └── display.py        # Output formatting
├── schema/
│   └── tmdb_fields.json  # Verified field definitions
├── examples/
│   └── sample_prompts.json  # Categorized test prompts
├── experiments/
│   ├── socrata_fidelity.py  # Live validation against Socrata open datasets
│   ├── run_suite.py         # Run all specs, emit comparison table
│   └── specs/               # Experiment specs (NYC, Chicago, Seattle, ...)
├── requirements.txt
├── .env.example
└── README.md
```

## How Fidelity is Calculated

Each constraint has an **estimated survival rate**—the fraction of all movies that satisfy it:

- "Released in 1999" → ~1% of movies (survival rate: 0.01)
- "Thriller genre" → ~10% of movies (survival rate: 0.10)
- "Melancholy tone" → ~15% of movies (survival rate: 0.15)

The **information content** (in bits) is `-log2(survival_rate)`:

- 1% survival → 6.64 bits
- 10% survival → 3.32 bits
- 15% survival → 2.74 bits

Fidelity is the ratio of verified bits to total bits.

Per-constraint bits are capped at `log2(pool_size)` — a constraint cannot carry more information than it takes to identify a single row in the candidate pool. The default cap is 20 bits (a one-in-a-million pool, roughly the TMDb catalog); for larger datasets pass `pool_size` to `compute_fidelity()` (or `--pool-size` to the skill's script) so near-unique selectors like exact IDs are counted at their true weight — ~30 bits on a billion-row table:

```python
report = compute_fidelity(constraints, pool_size=1_000_000_000)
```

### Generalizing beyond TMDb

TMDb is just the working example. Three mechanisms keep the framework honest on arbitrary datasets:

- **Measured vs. estimated survival rates.** On a queryable datastore, survival rates shouldn't be guessed — but they shouldn't be bought with expensive scans either. Bits are logarithmic, so order-of-magnitude accuracy suffices; prefer counts the system already returns (`total_results`, search hit counts), catalog/optimizer statistics, or sampled counts over exact `COUNT(*)`, following the target system's own best practices. Constraints carry an optional `rate_source` of `"measured"` (exact or system-returned counts), `"approximated"` (system-derived but inexact — planner statistics, sampled counts, possibly stale), or `"estimated"` (a guess), so the report shows whether the constraint *weights* are themselves verified. The calibration numbers in this repo (a decade ≈ 10%, a genre ≈ 5–15%) are movie-catalog priors; real distributions skew, so measure cheaply when you can.
- **Injected filters.** Any filter the system applies that the user never requested — quality floors, top-N truncation, sampling, default sort order — is declared with `"type": "injected"`. Injected filters are excluded from the fidelity score (they aren't part of the request) but always listed in the report, because silently narrowing the pool is the main way a "100% fidelity, provably correct" claim becomes dishonest. This repo's own pipeline declares its filters: a 50-vote minimum, a top-30 candidate cap, and a top-10 re-rank cutoff.
- **Correlation correction.** Bits summed across constraints assume they filter independently, which real data violates. When the datastore can cheaply measure the joint count of all verified filters ANDed together — often the same query that sizes the survivor set — pass it (`verified_joint_count` in the skill script, `verified_joint_survival_rate` to `compute_fidelity()`), and the verified side of the score uses the exact measured joint information `-log2(joint/pool)` instead of the sum. Per-constraint bits remain as attribution and the report shows the adjustment. Inferred constraints can't be jointly counted, so they stay summed — merge overlapping inferred constraints by hand.

  The aggregate adjustment says *how much* correlation there is, not *where*. To attribute it, compute actual correlations: pairwise `A AND B` counts give each pair's shared information as `log2(pool × n_AB / (n_A × n_B))` — the harness does this with `--pairwise` — or, with rows in hand, build boolean indicator columns per constraint and run a standard correlation matrix (`df.corr()` in pandas / `np.corrcoef`). The indicator route is also the only window into correlation among *inferred* constraints, via a judged sample.

One more subtlety: "verified" means the *query* is mechanically checkable, not that it faithfully captures intent. A crowd-sourced `melancholy` keyword tag is a verified filter but a noisy proxy for a melancholy tone — the skill's guidance is to split such constraints into a verified query plus an inferred semantic gap.

### Validation on real open data

`experiments/socrata_fidelity.py` runs the whole workflow against live Socrata-hosted datasets (NYC Open Data, Chicago, CDC, and most data.gov-federated portals speak the same SODA API) using only server-side counts — no scans, no downloads:

```bash
python3 experiments/socrata_fidelity.py experiments/specs/nyc_311_noise.json
python3 experiments/socrata_fidelity.py --sample 5 experiments/specs/chicago_theft.json
python3 experiments/run_suite.py --quiet   # run every spec, emit the table below
```

Each run measures the pool size and every verified constraint's survival rate from the system's own counts, scores the request with the correlation correction applied (the measured joint count replaces the independence sum on the verified side), and declares the judging sample as an injected filter. `run_suite.py` runs every spec as a regression suite and exits nonzero on failure, so it can gate CI.

Results across five cities and five data shapes (suite output):

| Spec | Domain | Pool | Fidelity | Verified bits (joint) | Correlation adj. |
|---|---|---|---|---|---|
| `austin_pitbull_adoptions` | data.austintexas.gov | 173,775 | 82.4% | 8.17 (summed 8.40) | -0.23 bits |
| `chicago_theft` | data.cityofchicago.org | 8,595,766 | 76.9% | 9.11 (summed 9.33) | -0.22 bits |
| `moco_speeding` | data.montgomerycountymd.gov | 2,137,572 | 67.7% | 6.96 (summed 6.68) | +0.28 bits |
| `nyc_311_noise` | data.cityofnewyork.us | 21,848,232 | 72.4% | 8.70 (summed 8.57) | +0.13 bits |
| `seattle_aid_calls` | data.seattle.gov | 2,187,508 | 69.0% | 5.16 (summed 5.06) | +0.10 bits |

Every adjustment lands within ±0.3 bits against 5–9 verified bits, so the independence sum is a good approximation on real civic data — but with the joint count measured, the verified side no longer needs the approximation at all. Reading the sign: a negative adjustment (Chicago, Austin) means the constraints are positively correlated — the joint pool is larger than independence predicts, so the sum *overstated* the verified information; a positive adjustment (NYC, Seattle, Montgomery County) means mildly negatively correlated constraints, where the sum understated it. Write a new spec JSON to test any other Socrata dataset.

### Case studies: answering real questions with the skill

Two end-to-end runs, each answering a natural question a resident might actually ask. Total API footprint per run: a handful of server-side counts and one small fetch of the matching rows.

**"Were there a lot of illegal fireworks complaints in Williamsburg around July 4th? Which sound like full shows vs stray firecrackers?"** (NYC 311, 21.8M rows)

```
PROMPT FIDELITY: 91.6%
  ✓ Illegal Fireworks complaints     (7.48 bits, measured)
  ✓ Williamsburg zips 11211/11249    (6.31 bits, measured)
  ✓ June 28 – July 6, 2026           (7.63 bits, measured)
  ? "full show vs firecrackers"      (1.74 bits, estimated)
  Correlation: summed 21.42 bits → joint 18.89 (-2.53 adjustment)
```

The verified half answered richly: 45 complaints, peaking at 19 on July 4th, with a repeat-complaint hot spot on South 2nd Street. The inferred half hit a wall — all 45 records had descriptor "N/A" and boilerplate resolutions — so the answer said plainly that the data cannot distinguish shows from firecrackers, and offered the one verifiable proxy (repeat complaints at one address in one night) clearly labeled as inference. A high fidelity score means the *verified part dominates the request*, not that every part is answerable. The −2.53 bit adjustment reflects a real seasonal correlation: fireworks complaints barely exist outside that week, so complaint type and date range heavily overlap.

**"How bad have car break-ins been in Logan Square this summer? Do they look targeted or random?"** (Chicago crimes, 8.6M rows)

```
PROMPT FIDELITY: 94.5%
  ✓ Vehicle break-in (theft/burglary from vehicle)  (8.76 bits, measured)
  ✓ Logan Square (community area 22)                (5.71 bits, measured)
  ✓ Jun 1 – Jul 18, 2026                            (8.40 bits, measured)
  ? "targeted vs random"                            (1.00 bits, estimated)
  Correlation: summed 22.87 bits → joint 17.23 (-5.64 adjustment)
```

Verified: 56 break-ins, up 33% from 42 in the same window last year; 41 of 56 street parking; zero arrests; 17 of the 56 in a single June 2–4 burst. The inferred judgment ("systematic about the area, random about the victim") was grounded in checkable patterns — burst days and no block hit more than twice — with the line drawn explicitly between database facts and interpretation. Two mechanisms earned their keep here: a cheap group-by probe *before* decomposing revealed that "car break-ins" spans two encodings (guessing would have silently dropped 57% of the answer), and the −5.64 bit correlation adjustment absorbed a data-quality landmine — those description labels barely exist before ~2024 because Chicago changed its coding taxonomy, making the all-time per-constraint rate meaningless. Independence predicted ~1 matching row; the measured joint count of 56 kept the score correct despite 25 years of label drift.

## TMDb Verified Fields

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

## Inferred Attributes (LLM Only)

These cannot be queried through TMDb:

- **Mood/tone**: "dark," "uplifting," "meditative"
- **Cinematography**: "handheld," "long takes," "neon-lit"
- **Themes**: "explores grief," "questions identity"
- **Comparative feel**: "feels like a Coen Brothers film"
- **Context**: "good for a first date"
- **Pacing**: "slow burn," "tight and fast"
- **Emotional arc**: "starts hopeful, ends devastating"

## API Reference

### Python API

```python
from agent import MovieRecommendationAgent

agent = MovieRecommendationAgent(provider="anthropic")
result = agent.recommend("90s thrillers with a dark tone")

print(f"Fidelity: {result['fidelity']['fidelity_score']:.1%}")
for movie in result['movies'][:5]:
    print(f"- {movie['title']} ({movie['year']})")
```

### Fidelity Calculation

```python
from agent import compute_fidelity

constraints = [
    {"description": "Thriller genre", "type": "verified", "estimated_survival_rate": 0.10},
    {"description": "Dark tone", "type": "inferred", "estimated_survival_rate": 0.15},
]

report = compute_fidelity(constraints)
print(f"Fidelity: {report.fidelity_score:.1%}")
print(f"Verified: {report.verified_bits:.2f} bits")
print(f"Inferred: {report.inferred_bits:.2f} bits")
```

## Why This Matters

1. **Transparency**: Users understand which parts of their request are reliably satisfied vs. LLM guesses
2. **Calibration**: Lower fidelity = more uncertainty = user should evaluate results more carefully
3. **Design guidance**: Helps API designers understand what features to add to increase fidelity
4. **Research**: Enables empirical study of the gap between user intent and tool capability

## License

MIT

Use of TMDB in this repo is for non-commercial purposes only. this product is not endorsed or certified by TMDB. find out more about TMDB here https://www.themoviedb.org/
