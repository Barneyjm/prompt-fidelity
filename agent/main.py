"""
Main entry point for the Movie Recommendation Agent.

Implements the LangGraph-based workflow:
1. Decompose user prompt into constraints
2. Classify constraints as verified/inferred
3. Query TMDb API with verified constraints
4. Re-rank results using LLM for inferred constraints
5. Compute and report fidelity score
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
from .fidelity import compute_fidelity, FidelityReport
from .tmdb import TMDbClient, build_discover_params, MovieDetails, DEFAULT_MIN_VOTES
from .rerank import (
    rerank_movies,
    merge_rerank_results,
    RERANK_MAX_RESULTS,
    RERANK_SCORE_THRESHOLD,
)
from .display import (
    print_recommendations,
    format_json_output,
    format_fidelity_colored,
)


# Load environment variables
load_dotenv()

# Candidates passed to re-ranking after the TMDb query. Declared in the
# fidelity report via pipeline_injected_constraints.
CANDIDATE_CAP = 30


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

    # After re-ranking
    ranked_movies: list[dict]
    ranking_notes: str | None

    # Fidelity analysis
    fidelity_report: dict | None

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
        movies = client.discover_movies(params, min_votes=DEFAULT_MIN_VOTES)

        # Get detailed info for top candidates (for re-ranking context)
        detailed_movies = []
        for movie in movies[:CANDIDATE_CAP]:
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

        return {
            **state,
            "candidate_movies": detailed_movies,
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
            max_results=RERANK_MAX_RESULTS
        )

        ranked = merge_rerank_results(
            state["candidate_movies"],
            rerank_result
        )

        return {
            **state,
            "ranked_movies": ranked,
            "ranking_notes": rerank_result.get("ranking_notes"),
        }
    except Exception as e:
        return {**state, "error": f"Re-ranking failed: {str(e)}"}


def pipeline_injected_constraints(state: AgentState) -> list[dict]:
    """
    Filters this pipeline applies that the user never requested.

    Declared so they appear in the fidelity report instead of silently
    narrowing the pool. Excluded from the fidelity score.
    """
    injected = [
        {
            "description": f"Minimum {DEFAULT_MIN_VOTES} votes "
                           f"(quality floor applied to every query)",
            "type": "injected",
            "estimated_survival_rate": 0.10,
        },
        {
            "description": f"Candidates sorted by rating and capped at top "
                           f"{CANDIDATE_CAP} before ranking",
            "type": "injected",
        },
    ]
    if state["inferred_constraints"]:
        injected.append({
            "description": f"Re-ranked results limited to {RERANK_MAX_RESULTS} "
                           f"with match score above {RERANK_SCORE_THRESHOLD}",
            "type": "injected",
        })
    return injected


def compute_fidelity_node(state: AgentState) -> AgentState:
    """Compute fidelity score from constraints."""
    if state.get("error"):
        return state

    try:
        all_constraints = (
            state["verified_constraints"]
            + state["inferred_constraints"]
            + pipeline_injected_constraints(state)
        )
        fidelity_report = compute_fidelity(all_constraints)

        return {
            **state,
            "fidelity_report": fidelity_report.to_dict(),
        }
    except Exception as e:
        return {**state, "error": f"Fidelity computation failed: {str(e)}"}


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
    workflow.add_node("compute_fidelity", compute_fidelity_node)

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
            "skip_rerank": "compute_fidelity",
            "end": END,
        }
    )

    workflow.add_edge("rerank", "compute_fidelity")
    workflow.add_edge("compute_fidelity", END)

    return workflow.compile()


class MovieRecommendationAgent:
    """High-level interface for the movie recommendation agent."""

    def __init__(
        self,
        provider: Literal["anthropic", "openai"] = "anthropic",
        model: str | None = None
    ):
        """
        Initialize the agent.

        Args:
            provider: LLM provider ("anthropic" or "openai")
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
            "ranked_movies": [],
            "ranking_notes": None,
            "fidelity_report": None,
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
            "fidelity": final_state.get("fidelity_report"),
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
        choices=["anthropic", "openai"],
        default="anthropic",
        help="LLM provider to use (default: anthropic)"
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

    # Initialize agent
    agent = MovieRecommendationAgent(
        provider=args.provider,
        model=args.model
    )

    def report_from_dict(fidelity_dict: dict) -> FidelityReport:
        """Rebuild the pipeline's FidelityReport from its to_dict() output.

        Passes through every report-level field (pool size, joint rate) so
        the displayed report matches the one the pipeline computed.
        """
        return compute_fidelity(
            fidelity_dict.get("constraints", []),
            pool_size=fidelity_dict.get("pool_size"),
            verified_joint_survival_rate=fidelity_dict.get(
                "verified_joint_survival_rate"),
        )

    def process_prompt(prompt: str):
        """Process a single prompt and display results."""
        result = agent.recommend(prompt)

        if result.get("error"):
            print(f"Error: {result['error']}", file=sys.stderr)
            return

        if args.json:
            # Create FidelityReport from dict for JSON output
            fidelity_dict = result.get("fidelity", {})
            print(json.dumps(
                format_json_output(
                    prompt,
                    result.get("movies", []),
                    report_from_dict(fidelity_dict),
                    result.get("decomposition")
                ),
                indent=2
            ))
        else:
            # Formatted text output
            fidelity_dict = result.get("fidelity", {})
            fidelity_report = report_from_dict(fidelity_dict)

            if args.color:
                print(format_fidelity_colored(fidelity_report))
            else:
                print_recommendations(
                    prompt,
                    result.get("movies", []),
                    fidelity_report
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
