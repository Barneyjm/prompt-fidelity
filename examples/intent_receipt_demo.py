"""
Intent Receipt demo: the promptfidelity package instrumenting a live agent.

    python examples/intent_receipt_demo.py ["your movie request"]

What happens:
  1. An LLM (local `claude -p`, Haiku) decomposes your request into
     Constraints -- entries only: descriptions, params, survival estimates.
     It never assigns accounts; that's the package's HARD RULE.
  2. A small agent runs two REAL TMDb tools under pf.trace():
       - discover_movies: predicate tool, params enforced by the backend
       - search_movies:   relevance tool, free-text `query` is ADVISORY
     The agent has realistic bad habits: it caps rating floors at 7.0
     (substitution), only supports a whitelist of params (drops), and
     quietly imposes a popularity floor no one asked for (imposed).
  3. pf books every constraint mechanically against the arguments the
     tools were ACTUALLY called with, and prints the intent receipt.

Requires: TMDB_API_KEY in .env, `claude` CLI on PATH. No Anthropic key.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "promptfidelity", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import promptfidelity as pf
from agent.claude_cli import run_claude_cli
from agent.tmdb import TMDbClient

from dotenv import load_dotenv
load_dotenv()


# ---------------------------------------------------------------- decompose

DECOMPOSE_SYSTEM = """You decompose a movie request into constraints for two tools:
- TMDb /discover/movie (enforced params): with_genres (Action=28, Comedy=35, Drama=18,
  Horror=27, Science Fiction=878, Thriller=53, Romance=10749, Animation=16),
  primary_release_date.gte, primary_release_date.lte, vote_average.gte,
  vote_count.lte, with_runtime.gte, with_runtime.lte, with_original_language
- TMDb /search/movie (advisory param): query (free text)

Rules:
- One constraint per DISTINCT piece of intent. Cover EVERY piece of the
  request, including subjective ones -- do not drop any.
- Date ranges are ONE constraint using both .gte and .lte params.
- Topical/subject intent ("about X") -> params {"query": "<short phrase>"}.
- Subjective intent (mood, tone, vibe) -> params {} (empty; not expressible).
- "obscure"/"little-known" -> params {"vote_count.lte": "500"}.
- "highly-rated" means 8.0+; "well-reviewed" means 7.0+.
- p = estimated fraction of all movies satisfying the constraint (0-1).

Return ONLY JSON:
{"constraints": [{"id": "c1", "description": "...", "params": {...}, "p": 0.1}, ...]}"""


def decompose(prompt: str) -> list[pf.Constraint]:
    raw = run_claude_cli(DECOMPOSE_SYSTEM, prompt, model="haiku")
    start, end = raw.find("{"), raw.rfind("}") + 1
    data = json.loads(raw[start:end])
    # The LLM proposed ENTRIES. It has no say in accounts -- pf.book()
    # will decide those from what the tools are actually called with.
    return [pf.Constraint(id=c["id"], description=c["description"],
                          params=c.get("params") or {}, p=c.get("p"))
            for c in data["constraints"]]


# ------------------------------------------------------------------- tools

client = TMDbClient()


@pf.instrument
def discover_movies(**params):
    """Predicate tool: every param here is enforced by TMDb."""
    return client.discover_movies(dict(params), min_votes=0)


@pf.instrument
def search_movies(query):
    """Relevance tool: `query` is transmitted to a ranker, never enforced."""
    data = client._request("/search/movie", {"query": query})
    return [m for m in data.get("results", [])]


# ------------------------------------------------------------------- agent

DISCOVER_WHITELIST = {
    "with_genres", "primary_release_date.gte", "primary_release_date.lte",
    "vote_average.gte", "with_runtime.gte", "with_runtime.lte",
    "with_original_language",
}
RATING_CAP = 7.0          # agent "helpfully" relaxes stricter floors
IMPOSED_MIN_VOTES = 100   # agent-added popularity floor nobody asked for


def run_agent(constraints: list[pf.Constraint]):
    discover_params, search_terms = {}, []
    for c in constraints:
        for name, value in c.params.items():
            if name == "query":
                search_terms.append(str(value))
            elif name in DISCOVER_WHITELIST:
                if name == "vote_average.gte" and float(value) > RATING_CAP:
                    value = RATING_CAP  # substitution: relaxed en route
                discover_params[name] = value
            # anything else (e.g. vote_count.lte) is silently unsupported

    results = discover_movies(
        **discover_params,
        **{"vote_count.gte": IMPOSED_MIN_VOTES, "sort_by": "vote_average.desc"},
    )
    if search_terms:
        found = search_movies(query=" ".join(search_terms))
        ids = {m.id for m in results}
        results += [m for m in found if m.get("id") not in ids][:5]
    return results


# ----------------------------------------------------------------- receipt

STAMP = {"verified": "✓ VERIFIED", "transmitted": "→ TRANSMITTED",
         "substituted": "⇄ SUBSTITUTED", "inferred": "? INFERRED",
         "dropped": "✗ DROPPED"}

FOOTNOTE = {
    "verified": "enforced by the backend; results provably satisfy this",
    "transmitted": "faithfully delivered to a ranker; compliance not guaranteed",
    "substituted": "altered en route -- you did not get what you asked for",
    "inferred": "no tool could express this; it rode on model judgment alone",
    "dropped": "never reached any tool in any form",
}


def print_receipt(ledger: pf.Ledger, calls: list[dict]):
    W = 66
    print("┌" + "─" * W + "┐")
    print("│" + "INTENT RECEIPT".center(W) + "│")
    print("│" + "every constraint, booked against actual tool calls".center(W) + "│")
    print("├" + "─" * W + "┤")
    for e in ledger.entries:
        b = f"{e.bits:5.2f} bits" if e.bits is not None else "   ?  bits"
        print(f"│ {STAMP[e.account]:<15} {e.description[:37]:<37} {b} │")
        note = FOOTNOTE[e.account]
        if e.account in ("substituted", "dropped"):
            note = e.evidence[:60]
        print(f"│ {'':<15} └ {note[:46]:<46} │")
    if ledger.imposed:
        print("├" + "─" * W + "┤")
        print("│ " + "AGENT-IMPOSED (no constraint asked for these)".ljust(W - 1) + "│")
        for i in ledger.imposed:
            print(("│   ! " + f"{i.description} = {i.value}").ljust(W + 1) + "│")
    print("├" + "─" * W + "┤")
    a = {k: ledger._sum(k) for k in pf.core.ACCOUNTS}
    books = " + ".join(f"{a[k]:.1f}{k[0]}" for k in pf.core.ACCOUNTS)
    print(f"│ BOOKS BALANCE   {books} = {ledger.total_bits:.1f} bits".ljust(W + 1) + "│")
    print(f"│ FIDELITY        {ledger.fidelity:.1%} of your intent was VERIFIED".ljust(W + 1) + "│")
    print(f"│ TRANSMISSION    {ledger.transmission_rate:.1%} was delivered but not enforced".ljust(W + 1) + "│")
    print("├" + "─" * W + "┤")
    print("│ " + f"tool calls booked against: {len(calls)}".ljust(W - 1) + "│")
    for call in calls:
        args = ", ".join(f"{k}={v}" for k, v in call["arguments"].items())
        line = f"  {call['name']}({args})"
        print(f"│ {line[:W - 2]:<{W - 1}}│")
    print("└" + "─" * W + "┘")


# -------------------------------------------------------------------- main

def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else (
        "A highly-rated 90s sci-fi movie about space stations -- something "
        "obscure I probably haven't seen, with a melancholy tone"
    )
    print(f'\nRequest: "{prompt}"\n')

    constraints = decompose(prompt)
    print("LLM proposed these constraint ENTRIES (accounts not its call):")
    for c in constraints:
        print(f"  {c.id}: {c.description}  params={c.params}  p={c.p}")

    before = {}  # snapshot of the pre-repair state, for PF_RECEIPT_JSON
    with pf.trace(constraints, advisory_params={"query"},
                  ignore_params={"page", "sort_by"}) as rec:
        movies = run_agent(constraints)

        # --- self-inspection: check the books BEFORE speaking -----------
        interim = rec.check()
        def snap(m):
            if isinstance(m, dict):
                return {"title": m.get("title"),
                        "year": m.get("year") or (m.get("release_date") or "????")[:4]}
            return {"title": m.title, "year": m.year}

        before = {"ledger": interim.to_dict(),
                  "injection": interim.render("model"),
                  "results": [snap(m) for m in movies[:6]]}
        if interim.unhonored:
            print("\nMid-run check: the books don't balance. Injected into"
                  " the agent's context:\n")
            for line in interim.render("model").splitlines():
                print(f"   | {line}")

            # The repair the injection asks for: ONE complete discover
            # call built from the constraints' own params -- overriding
            # the agent's bad habits (rating cap, whitelist, imposed
            # popularity floor). Mechanical: params come straight off the
            # ledger entries, no model judgment needed here.
            repair_params = {}
            for e in interim.entries:
                for name, value in e.params.items():
                    if name != "query":  # search tool's param, not discover's
                        repair_params[name] = value
            print(f"\nRepair call:\n   discover_movies(**{repair_params})")
            before["repair_params"] = repair_params
            repaired = discover_movies(**repair_params)
            if repaired:
                movies = [m.to_dict() for m in repaired] + movies

    ledger = rec.ledger()
    print()
    print_receipt(ledger, rec.calls)
    if not ledger.conjunction_honored:
        print("\n  note: conjunction_honored=False -- intent was honored across")
        print("  separate tool calls (discover + search); no single result set")
        print("  satisfied every constraint together.")

    print("\n--- the same ledger, four altitudes ---")
    for audience in ("product", "executive"):
        print(f"\n[{audience}]")
        print(ledger.render(audience))

    print("\nWhat the agent returned:")
    for m in movies[:5]:
        title = m.get("title") if isinstance(m, dict) else m.title
        year = (m.get("year") or (m.get("release_date") or "????")[:4]) if isinstance(m, dict) else m.year
        print(f"  - {title} ({year})")

    out = os.environ.get("PF_RECEIPT_JSON")
    if out:
        payload = ledger.to_dict()
        payload["prompt"] = prompt
        payload["calls"] = rec.calls
        payload["conjunction_honored"] = ledger.conjunction_honored
        payload["before"] = before
        payload["renderings"] = {a: ledger.render(a)
                                 for a in ("engineer", "product", "executive")}
        payload["results"] = [
            {"title": (m.get("title") if isinstance(m, dict) else m.title),
             "year": (m.get("year") or (m.get("release_date") or "????")[:4]) if isinstance(m, dict) else m.year}
            for m in movies[:8]
        ]
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n(receipt JSON written to {out})")


if __name__ == "__main__":
    main()
