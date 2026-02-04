"""
TMDb API wrapper for movie discovery and search.

Handles all interactions with The Movie Database API including:
- Movie discovery with filters
- Person (actor/director) ID lookup
- Keyword ID lookup
- Movie details retrieval
"""

import os
from dataclasses import dataclass
from typing import Any

import httpx


TMDB_BASE_URL = "https://api.themoviedb.org/3"


@dataclass
class Movie:
    """Represents a movie from TMDb."""

    id: int
    title: str
    overview: str
    release_date: str | None
    vote_average: float
    vote_count: int
    popularity: float
    genre_ids: list[int]
    original_language: str
    poster_path: str | None
    backdrop_path: str | None

    @property
    def year(self) -> int | None:
        """Extract release year from date."""
        if self.release_date and len(self.release_date) >= 4:
            try:
                return int(self.release_date[:4])
            except ValueError:
                return None
        return None

    @property
    def poster_url(self) -> str | None:
        """Full URL to movie poster."""
        if self.poster_path:
            return f"https://image.tmdb.org/t/p/w500{self.poster_path}"
        return None

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "title": self.title,
            "year": self.year,
            "overview": self.overview,
            "vote_average": self.vote_average,
            "vote_count": self.vote_count,
            "popularity": self.popularity,
            "original_language": self.original_language,
            "poster_url": self.poster_url
        }

    @classmethod
    def from_api_response(cls, data: dict) -> "Movie":
        """Create Movie from TMDb API response."""
        return cls(
            id=data["id"],
            title=data.get("title", "Unknown"),
            overview=data.get("overview", ""),
            release_date=data.get("release_date"),
            vote_average=data.get("vote_average", 0.0),
            vote_count=data.get("vote_count", 0),
            popularity=data.get("popularity", 0.0),
            genre_ids=data.get("genre_ids", []),
            original_language=data.get("original_language", ""),
            poster_path=data.get("poster_path"),
            backdrop_path=data.get("backdrop_path")
        )


@dataclass
class MovieDetails(Movie):
    """Extended movie details including runtime, budget, etc."""

    runtime: int | None = None
    budget: int | None = None
    revenue: int | None = None
    tagline: str | None = None
    genres: list[dict] | None = None
    production_companies: list[dict] | None = None
    credits: dict | None = None

    @property
    def director(self) -> str | None:
        """Extract director name from credits."""
        if self.credits and "crew" in self.credits:
            for person in self.credits["crew"]:
                if person.get("job") == "Director":
                    return person.get("name")
        return None

    @property
    def cast_names(self) -> list[str]:
        """Extract top cast names."""
        if self.credits and "cast" in self.credits:
            return [p.get("name", "") for p in self.credits["cast"][:10]]
        return []

    @classmethod
    def from_api_response(cls, data: dict) -> "MovieDetails":
        """Create MovieDetails from TMDb API response."""
        return cls(
            id=data["id"],
            title=data.get("title", "Unknown"),
            overview=data.get("overview", ""),
            release_date=data.get("release_date"),
            vote_average=data.get("vote_average", 0.0),
            vote_count=data.get("vote_count", 0),
            popularity=data.get("popularity", 0.0),
            genre_ids=[g["id"] for g in data.get("genres", [])],
            original_language=data.get("original_language", ""),
            poster_path=data.get("poster_path"),
            backdrop_path=data.get("backdrop_path"),
            runtime=data.get("runtime"),
            budget=data.get("budget"),
            revenue=data.get("revenue"),
            tagline=data.get("tagline"),
            genres=data.get("genres"),
            production_companies=data.get("production_companies"),
            credits=data.get("credits")
        )


class TMDbClient:
    """Client for TMDb API interactions."""

    def __init__(self, api_key: str | None = None):
        """
        Initialize TMDb client.

        Args:
            api_key: TMDb API key. If not provided, reads from TMDB_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("TMDB_API_KEY")
        if not self.api_key:
            raise ValueError(
                "TMDb API key required. Set TMDB_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self.client = httpx.Client(timeout=30.0)

    def _request(self, endpoint: str, params: dict | None = None) -> dict:
        """Make authenticated request to TMDb API."""
        url = f"{TMDB_BASE_URL}{endpoint}"
        request_params = {"api_key": self.api_key}
        if params:
            request_params.update(params)

        response = self.client.get(url, params=request_params)
        response.raise_for_status()
        return response.json()

    async def _request_async(self, endpoint: str, params: dict | None = None) -> dict:
        """Make async authenticated request to TMDb API."""
        url = f"{TMDB_BASE_URL}{endpoint}"
        request_params = {"api_key": self.api_key}
        if params:
            request_params.update(params)

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, params=request_params)
            response.raise_for_status()
            return response.json()

    def discover_movies(
        self,
        params: dict[str, Any] | None = None,
        page: int = 1,
        min_votes: int = 50
    ) -> list[Movie]:
        """
        Discover movies with optional filters.

        Args:
            params: Dictionary of TMDb discover API parameters
            page: Page number for pagination
            min_votes: Minimum vote count filter (for quality)

        Returns:
            List of Movie objects matching the criteria
        """
        request_params = {
            "page": page,
            "vote_count.gte": min_votes,
            "sort_by": "vote_average.desc"
        }

        if params:
            request_params.update(params)

        data = self._request("/discover/movie", request_params)
        return [Movie.from_api_response(m) for m in data.get("results", [])]

    def get_movie_details(self, movie_id: int, include_credits: bool = True) -> MovieDetails:
        """
        Get detailed information about a specific movie.

        Args:
            movie_id: TMDb movie ID
            include_credits: Whether to include cast and crew

        Returns:
            MovieDetails object with extended information
        """
        params = {}
        if include_credits:
            params["append_to_response"] = "credits"

        data = self._request(f"/movie/{movie_id}", params)
        return MovieDetails.from_api_response(data)

    def search_person(self, name: str) -> list[dict]:
        """
        Search for a person (actor, director, etc.) by name.

        Args:
            name: Person name to search for

        Returns:
            List of matching person records with id, name, known_for
        """
        data = self._request("/search/person", {"query": name})
        return data.get("results", [])

    def get_person_id(self, name: str) -> int | None:
        """
        Get the TMDb ID for a person by name.

        Returns the ID of the most relevant match, or None if not found.
        """
        results = self.search_person(name)
        if results:
            return results[0]["id"]
        return None

    def search_keyword(self, keyword: str) -> list[dict]:
        """
        Search for a keyword by name.

        Args:
            keyword: Keyword to search for

        Returns:
            List of matching keyword records with id and name
        """
        data = self._request("/search/keyword", {"query": keyword})
        return data.get("results", [])

    def get_keyword_id(self, keyword: str) -> int | None:
        """
        Get the TMDb ID for a keyword.

        Returns the ID of the most relevant match, or None if not found.
        """
        results = self.search_keyword(keyword)
        if results:
            return results[0]["id"]
        return None

    def get_genres(self) -> dict[int, str]:
        """
        Get mapping of genre IDs to names.

        Returns:
            Dictionary mapping genre ID to genre name
        """
        data = self._request("/genre/movie/list")
        return {g["id"]: g["name"] for g in data.get("genres", [])}

    def get_genre_id(self, genre_name: str) -> int | None:
        """
        Get genre ID from genre name (case-insensitive).

        Args:
            genre_name: Genre name like "Action" or "Science Fiction"

        Returns:
            Genre ID or None if not found
        """
        genres = self.get_genres()
        genre_lower = genre_name.lower()
        for gid, name in genres.items():
            if name.lower() == genre_lower:
                return gid
        return None

    def close(self):
        """Close the HTTP client."""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def build_discover_params(verified_constraints: list[dict]) -> dict[str, Any]:
    """
    Build TMDb discover API parameters from verified constraints.

    Args:
        verified_constraints: List of verified constraint dicts with either:
            - api_param/api_value for single parameters
            - api_params array for multi-parameter constraints (e.g., date ranges)

    Returns:
        Dictionary of parameters for the discover/movie endpoint
    """
    params = {}

    for constraint in verified_constraints:
        # Handle new api_params array format (for date ranges, etc.)
        if "api_params" in constraint:
            for param_pair in constraint["api_params"]:
                api_param = param_pair.get("param")
                api_value = param_pair.get("value")
                if api_param and api_value is not None:
                    params[api_param] = api_value
        else:
            # Handle original api_param/api_value format
            api_param = constraint.get("api_param")
            api_value = constraint.get("api_value")

            if api_param and api_value is not None:
                # Handle comma-separated values for multi-value params
                if api_param in params:
                    # Append to existing value
                    params[api_param] = f"{params[api_param]},{api_value}"
                else:
                    params[api_param] = api_value

    return params


# Genre ID mapping for convenience
GENRE_IDS = {
    "action": 28,
    "adventure": 12,
    "animation": 16,
    "comedy": 35,
    "crime": 80,
    "documentary": 99,
    "drama": 18,
    "family": 10751,
    "fantasy": 14,
    "history": 36,
    "horror": 27,
    "music": 10402,
    "mystery": 9648,
    "romance": 10749,
    "science fiction": 878,
    "sci-fi": 878,
    "tv movie": 10770,
    "thriller": 53,
    "war": 10752,
    "western": 37
}


def get_genre_id_by_name(genre_name: str) -> int | None:
    """Get genre ID from name without API call."""
    return GENRE_IDS.get(genre_name.lower())
