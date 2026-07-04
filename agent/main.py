"""
Main entry point for the Movie Recommendation Agent.

Implements the LangGraph-based workflow:
1. Decompose user prompt into constraints
2. Classify constraints as verified/inferred
3. Query TMDb API with verified constraints
4. Re-rank results using LLM for inferred constraints
5. Book the intent ledger against the params ACTUALLY sent and report
   fidelity, demotions, and imposed filters
"""

import argparse
import json
import os
import sys
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from .decompose import (
    decompose_prompt,
    get_verified_constraints,
    get_inferred_constraints,
    constraints_requiring_lookup,
)
from .claude_cli import claude_cli_available
from .ledger import book_ledger, bits
from .tmdb import TMDbClient, build_discover_params, MovieDetails
from .rerank import rerank_movies, merge_rerank_results
from .display import (
    print_recommendations,
    format_json_output,
    format_ledger_colored,
)


# Load environment variables
load_dotenv()


class AgentState(TypedDict):
    """State object passed through the LangGraph workflow."""

    # Input
    user_prompt: str
    provider: str
    model: str | None

    # After decomposition
    decomposition: dict | None
    verified_constraints: list[dict]
    inferred_constraints: list[dict]

    # After ID resolution
    resolved_constraints: list[dict]

    # After TMDb query
    candidate_movies: list[dict]
    actual_params: dict | None      # params ACTUALLY sent to /discover/movie
    imposed_operations: list[dict]  # agent-added filters no constraint asked for

    # After re-ranking
    ranked_movies: list[dict]
    ranking_notes: str | None
    rerank_criteria: list[str]      # descriptions actually handed to the reranker

    # Intent ledger
    intent_ledger: dict | None

    # Errors
    error: str | None


def decompose_node(state: AgentState) -> AgentState:
    """Decompose the user prompt into constraints."""
    try:
        decomposition = decompose_prompt(
            state["user_prompt"],
            provider=state.get("provider", "anthropic"),
            model=state.get("model")
        )

        return {
            **state,
            "decomposition": decomposition,
            "verified_constraints": get_verified_constraints(decomposition),
            "inferred_constraints": get_inferred_constraints(decomposition),
        }
    except Exception as e:
        return {**state, "error": f"Decomposition failed: {str(e)}"}


def resolve_ids_node(state: AgentState) -> AgentState:
    """Resolve person/keyword names to TMDb IDs."""
    if state.get("error"):
        return state

    try:
        client = TMDbClient()
        resolved = []

        for constraint in state["verified_constraints"]:
            if constraint.get("requires_id_lookup"):
                # Handle person lookup (actor/director)
                if constraint.get("person_name"):
                    person_id = client.get_person_id(constraint["person_name"])
                    if person_id:
                        constraint = constraint.copy()
                        constraint["api_value"] = str(person_id)
                        constraint["resolved"] = True
                    else:
                        # Couldn't find person, convert to inferred
                        constraint = constraint.copy()
                        constraint["type"] = "inferred"
                        constraint["resolution_note"] = f"Could not find person: {constraint['person_name']}"

                # Handle keyword lookup
                elif constraint.get("keyword_name"):
                    keyword_id = client.get_keyword_id(constraint["keyword_name"])
                    if keyword_id:
                        constraint = constraint.copy()
                        constraint["api_value"] = str(keyword_id)
                        constraint["resolved"] = True
                    else:
                        constraint = constraint.copy()
                        constraint["type"] = "inferred"
                        constraint["resolution_note"] = f"Could not find keyword: {constraint['keyword_name']}"

            resolved.append(constraint)

        client.close()

        # Re-separate verified/inferred after resolution
        verified = [c for c in resolved if c.get("type") == "verified"]
        newly_inferred = [c for c in resolved if c.get("type") == "inferred"]
        inferred = state["inferred_constraints"] + newly_inferred

        return {
            **state,
            "resolved_constraints": resolved,
            "verified_constraints": verified,
            "inferred_constraints": inferred,
        }
    except Exception as e:
        return {**state, "error": f"ID resolution failed: {str(e)}"}


def query_tmdb_node(state: AgentState) -> AgentState:
    """Query TMDb API with verified constraints."""
    if state.get("error"):
        return state

    try:
        client = TMDbClient()

        # Build API parameters from verified constraints
        params = build_discover_params(state["verified_constraints"])

        # Query TMDb discover endpoint
        movies = client.discover_movies(params, min_votes=50)

        # Capture what was ACTUALLY sent (before detail fetches overwrite it)
        # -- the intent ledger books against this, not against intentions.
        actual_params = dict(client.last_request_params)

        # Agent-added filters that map to no user constraint
        imposed = []
        if "vote_count.gte" not in params and "vote_count.gte" in actual_params:
            imposed.append({
                "description": f"Popularity floor (min {actual_params['vote_count.gte']} votes)",
                "survival_rate": 0.35,  # est. fraction of TMDb titles clearing 50 votes
                "evidence": f"discover_movies(min_votes={actual_params['vote_count.gte']})",
            })

        # Get detailed info for top candidates (for re-ranking context)
        detailed_movies = []
        for movie in movies[:30]:  # Limit to top 30 for efficiency
            try:
                details = client.get_movie_details(movie.id)
                movie_dict = movie.to_dict()
                movie_dict["director"] = details.director
                movie_dict["cast"] = details.cast_names
                movie_dict["runtime"] = details.runtime
                movie_dict["genres"] = [g["name"] for g in (details.genres or [])]
                detailed_movies.append(movie_dict)
            except Exception:
                # Fall back to basic info if details fetch fails
                detailed_movies.append(movie.to_dict())

        client.close()

        if len(movies) > len(detailed_movies):
            imposed.append({
                "description": f"Top-{len(detailed_movies)} candidate truncation",
                "survival_rate": len(detailed_movies) / len(movies),
                "evidence": "movies[:30] in query_tmdb_node",
            })

        return {
            **state,
            "candidate_movies": detailed_movies,
            "actual_params": actual_params,
            "imposed_operations": state.get("imposed_operations", []) + imposed,
        }
    except Exception as e:
        return {**state, "error": f"TMDb query failed: {str(e)}"}


def rerank_node(state: AgentState) -> AgentState:
    """Re-rank candidates using LLM based on inferred constraints."""
    if state.get("error"):
        return state

    if not state["candidate_movies"]:
        return {
            **state,
            "ranked_movies": [],
            "ranking_notes": "No candidates found from TMDb query."
        }

    try:
        rerank_result = rerank_movies(
            state["candidate_movies"],
            state["inferred_constraints"],
            state["user_prompt"],
            provider=state.get("provider", "anthropic"),
            model=state.get("model"),
            max_results=10
        )

        ranked = merge_rerank_results(
            state["candidate_movies"],
            rerank_result
        )

        # The reranker silently discards candidates (score>30 cutoff, top-10).
        # Book the actual survival fraction as an imposed filter.
        imposed = []
        if len(ranked) < len(state["candidate_movies"]):
            imposed.append({
                "description": f"Rerank cutoff (kept {len(ranked)} of {len(state['candidate_movies'])} candidates)",
                "survival_rate": len(ranked) / len(state["candidate_movies"]),
                "evidence": "rerank score>30 cutoff + max_results=10 in rerank_node",
            })

        return {
            **state,
            "ranked_movies": ranked,
            "ranking_notes": rerank_result.get("ranking_notes"),
            "rerank_criteria": [
                c.get("description", "?") for c in state["inferred_constraints"]
            ],
            "imposed_operations": state.get("imposed_operations", []) + imposed,
        }
    except Exception as e:
        return {**state, "error": f"Re-ranking failed: {str(e)}"}


def book_ledger_node(state: AgentState) -> AgentState:
    """Book the intent ledger against the params actually sent.

    Replaces the old compute_fidelity_node, which scored the LLM's own
    classification of its constraints (circular). The ledger instead diffs
    the decomposition against the ACTUAL tool-call params -- no LLM
    self-reporting in the fidelity loop.
    """
    if state.get("error"):
        return state

    try:
        all_constraints = state["verified_constraints"] + state["inferred_constraints"]
        ledger = book_ledger(
            all_constraints,
            state.get("actual_params") or {},
            rerank_criteria_descriptions=state.get("rerank_criteria") or [],
            imposed_operations=state.get("imposed_operations") or [],
        )
        ledger_dict = ledger.to_dict()

        # Honesty gap: what a naive narration would claim as verified
        # (everything the decomposition CLASSIFIED verified) vs what the
        # ledger actually verified against the tool call.
        decomposition = state.get("decomposition") or {}
        narrated_verified = sum(
            bits(c.get("estimated_survival_rate", 1.0))
            for c in decomposition.get("constraints", [])
            if c.get("type") == "verified"
        )
        ledger_dict["narrated_verified_bits"] = round(narrated_verified, 2)
        ledger_dict["honesty_gap_bits"] = round(ledger.honesty_gap(narrated_verified), 2)
        ledger_dict["naive_fidelity"] = round(
            narrated_verified / ledger.user_total, 3) if ledger.user_total else 1.0

        return {
            **state,
            "intent_ledger": ledger_dict,
        }
    except Exception as e:
        return {**state, "error": f"Ledger booking failed: {str(e)}"}


def should_rerank(state: AgentState) -> str:
    """Determine if re-ranking is needed."""
    if state.get("error"):
        return "end"
    if state["inferred_constraints"]:
        return "rerank"
    return "skip_rerank"


def build_graph() -> StateGraph:
    """Build the LangGraph workflow."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("decompose", decompose_node)
    workflow.add_node("resolve_ids", resolve_ids_node)
    workflow.add_node("query_tmdb", query_tmdb_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("book_ledger", book_ledger_node)

    # Add edges
    workflow.set_entry_point("decompose")
    workflow.add_edge("decompose", "resolve_ids")
    workflow.add_edge("resolve_ids", "query_tmdb")

    # Conditional edge for re-ranking
    workflow.add_conditional_edges(
        "query_tmdb",
        should_rerank,
        {
            "rerank": "rerank",
            "skip_rerank": "book_ledger",
            "end": END,
        }
    )

    workflow.add_edge("rerank", "book_ledger")
    workflow.add_edge("book_ledger", END)

    return workflow.compile()


class MovieRecommendationAgent:
    """High-level interface for the movie recommendation agent."""

    def __init__(
        self,
        provider: Literal["anthropic", "claude-cli", "openai"] = "anthropic",
        model: str | None = None
    ):
        """
        Initialize the agent.

        Args:
            provider: LLM provider ("anthropic", "claude-cli", or "openai").
                "claude-cli" runs LLM calls through the local `claude -p`
                binary (Claude Code auth, no API key; defaults to Haiku)
            model: Specific model to use (defaults to provider's default)
        """
        self.provider = provider
        self.model = model
        self.graph = build_graph()

    def recommend(self, prompt: str) -> dict:
        """
        Get movie recommendations for a prompt.

        Args:
            prompt: Natural language movie request

        Returns:
            Dictionary with recommendations, fidelity report, and metadata
        """
        initial_state: AgentState = {
            "user_prompt": prompt,
            "provider": self.provider,
            "model": self.model,
            "decomposition": None,
            "verified_constraints": [],
            "inferred_constraints": [],
            "resolved_constraints": [],
            "candidate_movies": [],
            "actual_params": None,
            "imposed_operations": [],
            "ranked_movies": [],
            "ranking_notes": None,
            "rerank_criteria": [],
            "intent_ledger": None,
            "error": None,
        }

        # Run the graph
        final_state = self.graph.invoke(initial_state)

        if final_state.get("error"):
            return {"error": final_state["error"]}

        # If no re-ranking happened, use candidate movies directly
        movies = final_state.get("ranked_movies") or final_state.get("candidate_movies", [])

        return {
            "prompt": prompt,
            "movies": movies,
            "ledger": final_state.get("intent_ledger"),
            "decomposition": final_state.get("decomposition"),
            "ranking_notes": final_state.get("ranking_notes"),
        }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Movie Recommendation Agent with Prompt Fidelity Analysis"
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Movie recommendation request (or use --interactive)"
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "claude-cli", "openai"],
        default="anthropic",
        help="LLM provider to use (default: anthropic; claude-cli runs "
             "through the local `claude -p` binary, no API key needed)"
    )
    parser.add_argument(
        "--model",
        help="Specific model to use"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of formatted text"
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--color",
        action="store_true",
        help="Use colored output"
    )

    args = parser.parse_args()

    # No Anthropic key but a local Claude Code install: fall back to
    # `claude -p` (Haiku) instead of failing.
    if (args.provider == "anthropic"
            and not os.getenv("ANTHROPIC_API_KEY")
            and claude_cli_available()):
        print("No ANTHROPIC_API_KEY found; using local `claude -p` (haiku) instead.",
              file=sys.stderr)
        args.provider = "claude-cli"

    # Initialize agent
    agent = MovieRecommendationAgent(
        provider=args.provider,
        model=args.model
    )

    def process_prompt(prompt: str):
        """Process a single prompt and display results."""
        result = agent.recommend(prompt)

        if result.get("error"):
            print(f"Error: {result['error']}", file=sys.stderr)
            return

        ledger = result.get("ledger") or {}

        if args.json:
            print(json.dumps(
                format_json_output(
                    prompt,
                    result.get("movies", []),
                    ledger,
                    result.get("decomposition")
                ),
                indent=2
            ))
        elif args.color:
            print(format_ledger_colored(ledger))
        else:
            print_recommendations(
                prompt,
                result.get("movies", []),
                ledger
            )

    if args.interactive:
        print("Movie Recommendation Agent (type 'quit' to exit)")
        print("=" * 50)
        while True:
            try:
                prompt = input("\nEnter your movie request: ").strip()
                if prompt.lower() in ("quit", "exit", "q"):
                    break
                if prompt:
                    process_prompt(prompt)
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break
    elif args.prompt:
        process_prompt(args.prompt)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
