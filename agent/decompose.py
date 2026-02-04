"""
Constraint decomposition module using LLM.

Takes a natural language movie request and decomposes it into:
- VERIFIED constraints: can be queried via TMDb API
- INFERRED constraints: require LLM knowledge to evaluate
"""

import json
import os
from typing import Literal

from anthropic import Anthropic
from openai import OpenAI


DECOMPOSITION_SYSTEM_PROMPT = """You are a movie recommendation assistant that decomposes user requests into structured constraints.

Your task is to analyze the user's movie request and break it down into individual constraints, classifying each as either:
- VERIFIED: Can be directly queried through TMDb API (genre, year, rating, runtime, language, cast, director, keywords)
- INFERRED: Requires subjective judgment or knowledge not in TMDb (mood, tone, themes, style, comparative feel, pacing)

For each constraint, estimate the "survival rate" - the fraction of all movies that would satisfy this constraint. This should be a decimal between 0 and 1. Be thoughtful about these estimates:
- Very selective constraints (specific director): 0.001-0.01
- Moderately selective (specific genre): 0.05-0.15
- Broad constraints (rated above 6): 0.3-0.5

For VERIFIED constraints, also provide the TMDb API parameter and value.

IMPORTANT: Date ranges count as ONE constraint, not two. If a user says "from the 90s" or "between 1990 and 1999", that is a SINGLE constraint called "Released in the 1990s" with a combined survival rate for the decade (~0.10). Do NOT split this into separate "after X" and "before Y" constraints. Use the api_params field (plural) to list both parameters needed.

TMDb API parameters you can use:
- primary_release_year: exact year (e.g., "1999")
- primary_release_date.gte: earliest date (e.g., "1990-01-01")
- primary_release_date.lte: latest date (e.g., "1999-12-31")
- For date RANGES (decades, spans): use api_params array with both gte and lte
- with_genres: genre ID(s) - use these IDs:
  Action=28, Adventure=12, Animation=16, Comedy=35, Crime=80, Documentary=99,
  Drama=18, Family=10751, Fantasy=14, History=36, Horror=27, Music=10402,
  Mystery=9648, Romance=10749, Science Fiction=878, Thriller=53, War=10752, Western=37
- with_runtime.gte: minimum runtime in minutes
- with_runtime.lte: maximum runtime in minutes
- vote_average.gte: minimum rating (0-10)
- vote_average.lte: maximum rating (0-10)
- vote_count.gte: minimum number of votes
- with_original_language: ISO 639-1 language code (en, fr, ja, ko, etc.)
- with_cast: person ID (requires lookup)
- with_crew: person ID for director/crew (requires lookup)
- with_keywords: keyword ID (requires lookup)

IMPORTANT: If a constraint mentions a specific person (actor or director), set requires_id_lookup to true and include the person's name in the `person_name` field. Similarly, for keywords requiring lookup, set requires_id_lookup to true and include the keyword in the `keyword_name` field.

Respond with a JSON object in this exact format:
{
  "original_prompt": "the user's original request",
  "constraints": [
    {
      "description": "Human-readable description of the constraint",
      "type": "verified",
      "api_param": "the_tmdb_parameter",
      "api_value": "the_value",
      "estimated_survival_rate": 0.10,
      "requires_id_lookup": false
    },
    {
      "description": "Released in the 1990s",
      "type": "verified",
      "api_params": [
        {"param": "primary_release_date.gte", "value": "1990-01-01"},
        {"param": "primary_release_date.lte", "value": "1999-12-31"}
      ],
      "estimated_survival_rate": 0.10
    },
    {
      "description": "Description of subjective constraint",
      "type": "inferred",
      "estimated_survival_rate": 0.15
    }
  ],
  "interpretation_notes": "Brief explanation of how you interpreted ambiguous parts of the request"
}

Use api_param/api_value for single-parameter constraints. Use api_params array for date ranges that need both gte and lte.

Only output valid JSON, no other text."""


def decompose_with_claude(
    prompt: str,
    api_key: str | None = None,
    model: str = "claude-sonnet-4-5"
) -> dict:
    """
    Decompose a movie request using Claude.

    Args:
        prompt: The user's natural language movie request
        api_key: Anthropic API key (or uses ANTHROPIC_API_KEY env var)
        model: Claude model to use

    Returns:
        Dictionary with constraints and metadata
    """
    client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    message = client.messages.create(
        model=model,
        max_tokens=2048,
        system=DECOMPOSITION_SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": prompt}
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


def decompose_with_openai(
    prompt: str,
    api_key: str | None = None,
    model: str = "gpt-4o"
) -> dict:
    """
    Decompose a movie request using OpenAI.

    Args:
        prompt: The user's natural language movie request
        api_key: OpenAI API key (or uses OPENAI_API_KEY env var)
        model: OpenAI model to use

    Returns:
        Dictionary with constraints and metadata
    """
    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": DECOMPOSITION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        response_format={"type": "json_object"}
    )

    response_text = response.choices[0].message.content

    return json.loads(response_text)


def decompose_prompt(
    prompt: str,
    provider: Literal["anthropic", "openai"] = "anthropic",
    api_key: str | None = None,
    model: str | None = None
) -> dict:
    """
    Decompose a movie request into constraints.

    Args:
        prompt: The user's natural language movie request
        provider: LLM provider to use ("anthropic" or "openai")
        api_key: API key (or uses env var for the provider)
        model: Model to use (defaults to provider's default)

    Returns:
        Dictionary with:
        - original_prompt: str
        - constraints: list of constraint dicts
        - interpretation_notes: str
    """
    if provider == "anthropic":
        return decompose_with_claude(
            prompt,
            api_key=api_key,
            model=model or "claude-sonnet-4-5"
        )
    elif provider == "openai":
        return decompose_with_openai(
            prompt,
            api_key=api_key,
            model=model or "gpt-4o"
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


def get_verified_constraints(decomposition: dict) -> list[dict]:
    """Extract verified constraints from decomposition."""
    return [
        c for c in decomposition.get("constraints", [])
        if c.get("type") == "verified"
    ]


def get_inferred_constraints(decomposition: dict) -> list[dict]:
    """Extract inferred constraints from decomposition."""
    return [
        c for c in decomposition.get("constraints", [])
        if c.get("type") == "inferred"
    ]


def constraints_requiring_lookup(decomposition: dict) -> list[dict]:
    """Get constraints that need ID lookup (persons, keywords)."""
    return [
        c for c in decomposition.get("constraints", [])
        if c.get("requires_id_lookup", False)
    ]
