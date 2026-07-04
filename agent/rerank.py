"""
LLM re-ranking module for movie recommendations.

Takes a list of candidate movies (from TMDb API) and re-ranks them
based on inferred constraints that couldn't be verified through the API.
"""

import json
import os
from typing import Literal

from anthropic import Anthropic
from openai import OpenAI

from .claude_cli import run_claude_cli


RERANK_SYSTEM_PROMPT = """You are a movie expert assistant helping to rank movies based on subjective criteria.

You will be given:
1. A list of candidate movies with their details
2. A set of inferred criteria that these movies should match

Your task is to evaluate each movie against the inferred criteria and return a ranked list of the best matches.

For each movie, provide:
- A relevance score from 0-100 for each criterion
- An overall match score
- A brief explanation of why this movie matches (or doesn't match) the criteria

Respond with a JSON object in this exact format:
{
  "ranked_movies": [
    {
      "id": 12345,
      "title": "Movie Title",
      "overall_score": 85,
      "criterion_scores": {
        "criterion_description_1": 90,
        "criterion_description_2": 80
      },
      "explanation": "Brief explanation of why this movie matches the criteria"
    }
  ],
  "ranking_notes": "Any overall observations about the ranking process"
}

Rank movies by overall_score descending. Only include movies that have a reasonable match (overall_score > 30).
Only output valid JSON, no other text."""


def format_movies_for_reranking(movies: list[dict]) -> str:
    """Format movie list for the LLM prompt."""
    formatted = []
    for m in movies:
        movie_str = f"""
Movie ID: {m.get('id')}
Title: {m.get('title')} ({m.get('year', 'N/A')})
Overview: {m.get('overview', 'No overview available')}
Rating: {m.get('vote_average', 'N/A')}/10 ({m.get('vote_count', 0)} votes)
Language: {m.get('original_language', 'N/A')}
"""
        # Add director and cast if available
        if m.get('director'):
            movie_str += f"Director: {m.get('director')}\n"
        if m.get('cast'):
            movie_str += f"Cast: {', '.join(m.get('cast', [])[:5])}\n"
        if m.get('runtime'):
            movie_str += f"Runtime: {m.get('runtime')} minutes\n"
        if m.get('genres'):
            movie_str += f"Genres: {', '.join(m.get('genres', []))}\n"

        formatted.append(movie_str.strip())

    return "\n---\n".join(formatted)


def format_inferred_criteria(constraints: list[dict]) -> str:
    """Format inferred constraints for the LLM prompt."""
    criteria = []
    for c in constraints:
        criteria.append(f"- {c.get('description', 'Unknown criterion')}")
    return "\n".join(criteria)


def rerank_with_claude(
    movies: list[dict],
    inferred_constraints: list[dict],
    original_prompt: str,
    api_key: str | None = None,
    model: str = "claude-sonnet-4-5",
    max_results: int = 10
) -> dict:
    """
    Re-rank movies using Claude based on inferred constraints.

    Args:
        movies: List of movie dictionaries from TMDb
        inferred_constraints: List of inferred constraint dicts
        original_prompt: The user's original request for context
        api_key: Anthropic API key
        model: Claude model to use
        max_results: Maximum number of movies to return

    Returns:
        Dictionary with ranked movies and notes
    """
    client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    movies_text = format_movies_for_reranking(movies)
    criteria_text = format_inferred_criteria(inferred_constraints)

    user_message = f"""Original user request: "{original_prompt}"

Candidate movies to evaluate:
{movies_text}

Inferred criteria to match (these couldn't be verified through the database):
{criteria_text}

Please rank these movies based on how well they match the inferred criteria. Return the top {max_results} matches."""

    message = client.messages.create(
        model=model,
        max_tokens=4096,
        system=RERANK_SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": user_message}
        ]
    )

    response_text = message.content[0].text

    # Parse JSON response
    try:
        result = json.loads(response_text)
    except json.JSONDecodeError:
        # Try to extract JSON from the response
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        if start >= 0 and end > start:
            result = json.loads(response_text[start:end])
        else:
            raise ValueError(f"Could not parse LLM response as JSON: {response_text}")

    return result


def rerank_with_claude_cli(
    movies: list[dict],
    inferred_constraints: list[dict],
    original_prompt: str,
    model: str = "haiku",
    max_results: int = 10
) -> dict:
    """
    Re-rank movies using the local `claude -p` CLI.

    Uses Claude Code's own authentication -- no ANTHROPIC_API_KEY needed.
    Defaults to Haiku: reranking a fixed candidate list is a cheap task.
    """
    movies_text = format_movies_for_reranking(movies)
    criteria_text = format_inferred_criteria(inferred_constraints)

    user_message = f"""Original user request: "{original_prompt}"

Candidate movies to evaluate:
{movies_text}

Inferred criteria to match (these couldn't be verified through the database):
{criteria_text}

Please rank these movies based on how well they match the inferred criteria. Return the top {max_results} matches."""

    response_text = run_claude_cli(RERANK_SYSTEM_PROMPT, user_message, model=model)

    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        if start >= 0 and end > start:
            return json.loads(response_text[start:end])
        raise ValueError(f"Could not parse LLM response as JSON: {response_text}")


def rerank_with_openai(
    movies: list[dict],
    inferred_constraints: list[dict],
    original_prompt: str,
    api_key: str | None = None,
    model: str = "gpt-4o",
    max_results: int = 10
) -> dict:
    """
    Re-rank movies using OpenAI based on inferred constraints.

    Args:
        movies: List of movie dictionaries from TMDb
        inferred_constraints: List of inferred constraint dicts
        original_prompt: The user's original request for context
        api_key: OpenAI API key
        model: OpenAI model to use
        max_results: Maximum number of movies to return

    Returns:
        Dictionary with ranked movies and notes
    """
    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    movies_text = format_movies_for_reranking(movies)
    criteria_text = format_inferred_criteria(inferred_constraints)

    user_message = f"""Original user request: "{original_prompt}"

Candidate movies to evaluate:
{movies_text}

Inferred criteria to match (these couldn't be verified through the database):
{criteria_text}

Please rank these movies based on how well they match the inferred criteria. Return the top {max_results} matches."""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": RERANK_SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        temperature=0.3,
        response_format={"type": "json_object"}
    )

    response_text = response.choices[0].message.content

    return json.loads(response_text)


def rerank_movies(
    movies: list[dict],
    inferred_constraints: list[dict],
    original_prompt: str,
    provider: Literal["anthropic", "claude-cli", "openai"] = "anthropic",
    api_key: str | None = None,
    model: str | None = None,
    max_results: int = 10
) -> dict:
    """
    Re-rank movies based on inferred constraints.

    Args:
        movies: List of movie dictionaries from TMDb
        inferred_constraints: List of inferred constraint dicts
        original_prompt: The user's original request
        provider: LLM provider ("anthropic", "claude-cli", or "openai")
        api_key: API key (or uses env var)
        model: Model to use
        max_results: Maximum number of movies to return

    Returns:
        Dictionary with ranked movies and notes
    """
    # If no inferred constraints, just return movies as-is (sorted by rating)
    if not inferred_constraints:
        sorted_movies = sorted(
            movies,
            key=lambda m: m.get('vote_average', 0),
            reverse=True
        )[:max_results]
        return {
            "ranked_movies": [
                {
                    "id": m.get("id"),
                    "title": m.get("title"),
                    "overall_score": int(m.get("vote_average", 0) * 10),
                    "criterion_scores": {},
                    "explanation": "No inferred criteria to evaluate; ranked by TMDb rating."
                }
                for m in sorted_movies
            ],
            "ranking_notes": "No inferred constraints; movies ranked by verified criteria only."
        }

    if provider == "anthropic":
        return rerank_with_claude(
            movies,
            inferred_constraints,
            original_prompt,
            api_key=api_key,
            model=model or "claude-sonnet-4-5",
            max_results=max_results
        )
    elif provider == "claude-cli":
        return rerank_with_claude_cli(
            movies,
            inferred_constraints,
            original_prompt,
            model=model or "haiku",
            max_results=max_results
        )
    elif provider == "openai":
        return rerank_with_openai(
            movies,
            inferred_constraints,
            original_prompt,
            api_key=api_key,
            model=model or "gpt-4o",
            max_results=max_results
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


def merge_rerank_results(
    original_movies: list[dict],
    rerank_result: dict
) -> list[dict]:
    """
    Merge re-ranking results back with original movie data.

    Args:
        original_movies: Original movie dicts from TMDb
        rerank_result: Result from rerank_movies()

    Returns:
        List of movie dicts enriched with ranking info
    """
    # Create lookup by ID
    movie_lookup = {m["id"]: m for m in original_movies}

    enriched = []
    for ranked in rerank_result.get("ranked_movies", []):
        movie_id = ranked.get("id")
        if movie_id in movie_lookup:
            movie = movie_lookup[movie_id].copy()
            movie["rerank_score"] = ranked.get("overall_score", 0)
            movie["criterion_scores"] = ranked.get("criterion_scores", {})
            movie["rerank_explanation"] = ranked.get("explanation", "")
            enriched.append(movie)

    return enriched
