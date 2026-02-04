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

## Examples

| Prompt | Fidelity | Why |
|--------|----------|-----|
| "Action movies from the 80s rated above 7" | ~0.95 | Genre, year, rating are all queryable |
| "90s thrillers that feel like a Coen Brothers film" | ~0.50 | Genre + decade verified; aesthetic is inferred |
| "Movies that feel like a rainy Sunday afternoon" | ~0.05 | Pure mood, almost nothing queryable |

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
# - ANTHROPIC_API_KEY (required): https://console.anthropic.com/
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
├── experiments/          # Validation experiments (TODO)
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
