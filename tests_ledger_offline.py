"""Offline demonstration of the intent ledger -- no API keys needed.

Simulates: "Obscure sci-fi from the 1970s directed by Tarkovsky,
with a slow meditative pace" where the Tarkovsky lookup FAILS
and the agent silently imposes a popularity floor.

Run: python tests_ledger_offline.py
"""
import json
import math

try:
    from agent.ledger import book_ledger
except ModuleNotFoundError:
    # agent/__init__.py imports optional deps (anthropic/openai);
    # load ledger.py directly so this demo runs with zero dependencies.
    import importlib.util
    spec = importlib.util.spec_from_file_location("ledger", "agent/ledger.py")
    _mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_mod)
    book_ledger = _mod.book_ledger

constraints = [
    {"description": "Science fiction genre", "type": "verified",
     "api_param": "with_genres", "api_value": "878",
     "estimated_survival_rate": 0.08},
    {"description": "Released in the 1970s", "type": "verified",
     "api_params": [{"param": "primary_release_date.gte", "value": "1970-01-01"},
                    {"param": "primary_release_date.lte", "value": "1979-12-31"}],
     "estimated_survival_rate": 0.06},
    {"description": "Directed by Tarkovsky", "type": "verified",
     "api_param": "with_crew", "estimated_survival_rate": 0.0005,
     "resolution_note": "Could not find person: Tarkovsky"},   # demoted!
    {"description": "Obscure / little-known", "type": "verified",
     "api_param": "vote_count.lte", "api_value": "200",
     "estimated_survival_rate": 0.5},                          # never sent!
    {"description": "Slow, meditative pacing", "type": "inferred",
     "estimated_survival_rate": 0.1},
]

# What the agent ACTUALLY sent (note: no with_crew, no vote_count.lte,
# and an imposed vote_count.gte the user never asked for)
actual_params = {
    "with_genres": "878",
    "primary_release_date.gte": "1970-01-01",
    "primary_release_date.lte": "1979-12-31",
}

imposed = [
    {"description": "Popularity floor (min 50 votes)", "survival_rate": 0.35,
     "evidence": "discover_movies(params, min_votes=50)"},
    {"description": "Top-30 candidate truncation", "survival_rate": 0.9,
     "evidence": "movies[:30] in query_tmdb_node"},
]

ledger = book_ledger(constraints, actual_params,
                     rerank_criteria_descriptions=["Slow, meditative pacing"],
                     imposed_operations=imposed)

print(json.dumps(ledger.to_dict(), indent=2))

# Honesty gap: the current display would narrate all four
# "verified"-classified constraints as grounded.
narrated = sum(-math.log2(c["estimated_survival_rate"])
               for c in constraints if c["type"] == "verified")
print(f"\nNarrated-verified bits: {narrated:.2f}")
print(f"Actually-verified bits:  {ledger.to_dict()['accounts']['verified']:.2f}")
print(f"Honesty gap: {ledger.honesty_gap(narrated):.2f} bits overstated")
