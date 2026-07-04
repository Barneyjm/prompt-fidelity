"""
Prompt Fidelity Movie Recommendation Agent.

A movie recommendation agent that demonstrates the prompt fidelity framework
by computing and reporting the fraction of user intent that can be reliably
satisfied through structured API queries.
"""

from .fidelity import (
    Constraint,
    FidelityReport,
    compute_fidelity,
    compute_fidelity_frontier,
    compute_max_verified_bits,
)
from .decompose import (
    decompose_prompt,
    get_verified_constraints,
    get_inferred_constraints,
)
from .ledger import (
    IntentLedger,
    LedgerEntry,
    book_ledger,
)
from .tmdb import (
    TMDbClient,
    Movie,
    MovieDetails,
    build_discover_params,
)
from .rerank import (
    rerank_movies,
    merge_rerank_results,
)
from .display import (
    format_recommendations,
    format_fidelity_report,
    format_ledger_report,
    print_recommendations,
    format_json_output,
)
from .main import (
    MovieRecommendationAgent,
    build_graph,
)


__all__ = [
    # Fidelity
    "Constraint",
    "FidelityReport",
    "compute_fidelity",
    "compute_fidelity_frontier",
    "compute_max_verified_bits",
    # Decomposition
    "decompose_prompt",
    "get_verified_constraints",
    "get_inferred_constraints",
    # Intent ledger
    "IntentLedger",
    "LedgerEntry",
    "book_ledger",
    # TMDb
    "TMDbClient",
    "Movie",
    "MovieDetails",
    "build_discover_params",
    # Re-ranking
    "rerank_movies",
    "merge_rerank_results",
    # Display
    "format_recommendations",
    "format_fidelity_report",
    "format_ledger_report",
    "print_recommendations",
    "format_json_output",
    # Main
    "MovieRecommendationAgent",
    "build_graph",
]

__version__ = "0.1.0"
