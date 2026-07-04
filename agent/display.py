"""
Display and output formatting module.

Formats recommendation results with intent-ledger reports for CLI and other
outputs. The ledger narration is explicit about demotions (constraints that
silently lost verified status en route) and imposed filters (constraints the
agent added that no user asked for) -- closing the honesty gap between what
the agent did and what it tells the user it did.
"""

from typing import TextIO
import sys

from .fidelity import FidelityReport, Constraint


ACCOUNT_ICONS = {
    "verified": "✓",
    "transmitted": "→",
    "substituted": "⇄",
    "inferred": "?",
    "dropped": "✗",
    "imposed": "!",
}

ACCOUNT_LABELS = {
    "verified": "VERIFIED (enforced by the API call)",
    "transmitted": "TRANSMITTED (delivered via advisory param; not enforced)",
    "substituted": "SUBSTITUTED (demoted or altered en route)",
    "inferred": "INFERRED (handed to LLM reranker)",
    "dropped": "DROPPED (never queried, never reranked)",
}


def format_ledger_report(ledger: dict) -> str:
    """Format an intent ledger for display, narrating demotions and
    imposed filters instead of hiding them."""
    if not ledger:
        return "\n  (no intent ledger available)\n"

    lines = []
    accounts = ledger.get("accounts", {})
    entries = ledger.get("entries", [])
    user_entries = [e for e in entries if e["account"] != "imposed"]
    imposed_entries = [e for e in entries if e["account"] == "imposed"]

    lines.append(f"\n{'═' * 50}")
    lines.append(f"  PROMPT FIDELITY (ledger): {ledger.get('fidelity', 0):.1%}")
    naive = ledger.get("naive_fidelity")
    if naive is not None:
        lines.append(f"  naive (self-classified):  {naive:.1%}")
    lines.append(f"{'═' * 50}")
    lines.append(f"\n  {format_fidelity_bar(ledger.get('fidelity', 0))}")

    # User constraints grouped by account
    for account, label in ACCOUNT_LABELS.items():
        group = [e for e in user_entries if e["account"] == account]
        if not group:
            continue
        lines.append(f"\n  {label}:")
        for e in group:
            lines.append(f"  {ACCOUNT_ICONS[account]} {e['description']} ({e['bits']:.2f} bits)")

    # Narrate demotions explicitly -- this is the honesty-gap fix
    demoted = [e for e in user_entries if e["account"] == "substituted"]
    if demoted:
        lines.append(f"\n  DEMOTIONS (silently lost verified status):")
        for e in demoted:
            lines.append(f"  ⇄ {e['description']}: {e['evidence']}")

    # Imposed filters the user never asked for
    if imposed_entries:
        lines.append(f"\n  IMPOSED FILTERS (agent-added, undisclosed to user):")
        for e in imposed_entries:
            lines.append(f"  ! {e['description']} ({e['bits']:.2f} bits) -- {e['evidence']}")

    # Books-balance line: conservation across user accounts
    total = sum(accounts.get(a, 0.0) for a in ACCOUNT_LABELS)
    lines.append(f"\n  {'─' * 40}")
    lines.append(
        "  BOOKS: "
        + " + ".join(f"{accounts.get(a, 0.0):.1f}{a[0]}" for a in ACCOUNT_LABELS)
        + f" = {total:.1f} bits"
    )

    gap = ledger.get("honesty_gap_bits")
    if gap is not None:
        lines.append(f"  Honesty gap: {gap:+.2f} bits "
                     f"(narrated {ledger.get('narrated_verified_bits', 0):.2f} vs "
                     f"verified {accounts.get('verified', 0):.2f})")
    lines.append(f"{'═' * 50}\n")

    return "\n".join(lines)


def format_fidelity_bar(score: float, width: int = 20) -> str:
    """Create a visual bar representation of fidelity score."""
    filled = int(score * width)
    empty = width - filled
    bar = "█" * filled + "░" * empty
    return f"[{bar}] {score:.1%}"


def format_constraint_line(constraint: Constraint, show_bits: bool = True) -> str:
    """Format a single constraint for display."""
    icon = "✓" if constraint.constraint_type == "verified" else "?"
    bits_str = f" ({constraint.bits:.2f} bits)" if show_bits else ""
    return f"  {icon} {constraint.description}{bits_str}"


def format_fidelity_report(report: FidelityReport) -> str:
    """Format a complete fidelity report for display."""
    lines = []

    # Header with score
    lines.append(f"\n{'═' * 50}")
    lines.append(f"  PROMPT FIDELITY: {report.fidelity_score:.1%}")
    lines.append(f"{'═' * 50}")

    # Visual bar
    lines.append(f"\n  {format_fidelity_bar(report.fidelity_score)}")

    # Constraint breakdown
    lines.append(f"\n  Constraint Breakdown:")
    lines.append(f"  {'─' * 40}")

    # Verified constraints
    if report.verified_constraints:
        lines.append(f"\n  VERIFIED (queryable via API):")
        for c in report.verified_constraints:
            lines.append(format_constraint_line(c))

    # Inferred constraints
    if report.inferred_constraints:
        lines.append(f"\n  INFERRED (requires LLM judgment):")
        for c in report.inferred_constraints:
            lines.append(format_constraint_line(c))

    # Summary statistics
    lines.append(f"\n  {'─' * 40}")
    lines.append(f"  Verified:  {report.verified_bits:6.2f} bits ({len(report.verified_constraints)} constraints)")
    lines.append(f"  Inferred:  {report.inferred_bits:6.2f} bits ({len(report.inferred_constraints)} constraints)")
    lines.append(f"  Total:     {report.total_bits:6.2f} bits")
    lines.append(f"{'═' * 50}\n")

    return "\n".join(lines)


def format_movie_result(
    movie: dict,
    rank: int,
    show_explanation: bool = True,
    show_scores: bool = False
) -> str:
    """Format a single movie result for display."""
    lines = []

    # Title line
    title = movie.get("title", "Unknown")
    year = movie.get("year", "N/A")
    rating = movie.get("vote_average", 0)
    lines.append(f"{rank}. {title} ({year}) - {rating:.1f}/10")

    # Director and cast if available
    if movie.get("director"):
        lines.append(f"   Director: {movie['director']}")
    if movie.get("cast"):
        cast_str = ", ".join(movie["cast"][:3])
        lines.append(f"   Cast: {cast_str}")

    # Re-rank score if available
    if show_scores and "rerank_score" in movie:
        lines.append(f"   Match Score: {movie['rerank_score']}/100")

    # Explanation if available
    if show_explanation and movie.get("rerank_explanation"):
        # Wrap explanation to reasonable width
        explanation = movie["rerank_explanation"]
        if len(explanation) > 80:
            explanation = explanation[:77] + "..."
        lines.append(f"   → {explanation}")

    return "\n".join(lines)


def format_recommendations(
    prompt: str,
    movies: list[dict],
    ledger: dict,
    show_explanations: bool = True,
    max_display: int = 10
) -> str:
    """Format complete recommendation output."""
    lines = []

    # Header
    lines.append(f"\n{'╔' + '═' * 58 + '╗'}")
    lines.append(f"║  MOVIE RECOMMENDATIONS{' ' * 36}║")
    lines.append(f"{'╚' + '═' * 58 + '╝'}")

    # Original prompt
    if len(prompt) > 55:
        prompt_display = prompt[:52] + "..."
    else:
        prompt_display = prompt
    lines.append(f'\nQuery: "{prompt_display}"')

    # Intent ledger report
    lines.append(format_ledger_report(ledger))

    # Movies
    lines.append("RECOMMENDATIONS:")
    lines.append("─" * 50)

    for i, movie in enumerate(movies[:max_display], 1):
        lines.append(format_movie_result(
            movie,
            rank=i,
            show_explanation=show_explanations,
            show_scores=True
        ))
        if i < len(movies[:max_display]):
            lines.append("")  # Blank line between movies

    if len(movies) > max_display:
        lines.append(f"\n... and {len(movies) - max_display} more")

    return "\n".join(lines)


def print_recommendations(
    prompt: str,
    movies: list[dict],
    ledger: dict,
    file: TextIO = sys.stdout,
    **kwargs
) -> None:
    """Print formatted recommendations to output."""
    output = format_recommendations(prompt, movies, ledger, **kwargs)
    print(output, file=file)


def format_fidelity_summary(report: FidelityReport) -> str:
    """Format a compact fidelity summary for inline display."""
    return (
        f"Fidelity: {report.fidelity_score:.1%} "
        f"({report.verified_bits:.1f}v + {report.inferred_bits:.1f}i = {report.total_bits:.1f} bits)"
    )


def format_json_output(
    prompt: str,
    movies: list[dict],
    ledger: dict,
    decomposition: dict | None = None
) -> dict:
    """Format complete output as JSON-serializable dict."""
    return {
        "query": prompt,
        "ledger": ledger,
        "recommendations": [
            {
                "rank": i + 1,
                "id": m.get("id"),
                "title": m.get("title"),
                "year": m.get("year"),
                "rating": m.get("vote_average"),
                "director": m.get("director"),
                "cast": m.get("cast", [])[:5],
                "match_score": m.get("rerank_score"),
                "explanation": m.get("rerank_explanation")
            }
            for i, m in enumerate(movies)
        ],
        "decomposition": decomposition
    }


# Color support for terminals that support ANSI codes
class Colors:
    """ANSI color codes for terminal output."""

    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def colorize(text: str, color: str) -> str:
    """Add ANSI color codes to text."""
    return f"{color}{text}{Colors.END}"


def format_ledger_colored(ledger: dict) -> str:
    """Format an intent ledger with ANSI colors."""
    if not ledger:
        return "(no intent ledger available)"

    lines = []
    score = ledger.get("fidelity", 0)
    if score >= 0.8:
        score_color = Colors.GREEN
    elif score >= 0.5:
        score_color = Colors.YELLOW
    else:
        score_color = Colors.RED

    lines.append(f"\n{Colors.BOLD}PROMPT FIDELITY (ledger):{Colors.END} "
                 f"{colorize(f'{score:.1%}', score_color)}")
    naive = ledger.get("naive_fidelity")
    if naive is not None:
        lines.append(f"{Colors.BOLD}naive (self-classified):{Colors.END} {naive:.1%}")

    account_colors = {
        "verified": Colors.GREEN,
        "transmitted": Colors.CYAN,
        "substituted": Colors.YELLOW,
        "inferred": Colors.YELLOW,
        "dropped": Colors.RED,
        "imposed": Colors.RED,
    }
    for e in ledger.get("entries", []):
        color = account_colors.get(e["account"], "")
        icon = ACCOUNT_ICONS.get(e["account"], "·")
        lines.append(f"  {colorize(icon, color)} "
                     f"[{colorize(e['account'], color)}] "
                     f"{e['description']} ({e['bits']:.2f} bits)")
        if e["account"] in ("substituted", "imposed"):
            lines.append(f"      {colorize(e['evidence'], Colors.YELLOW)}")

    gap = ledger.get("honesty_gap_bits")
    if gap is not None:
        gap_color = Colors.RED if gap > 0 else Colors.GREEN
        lines.append(f"\n{Colors.BOLD}Honesty gap:{Colors.END} "
                     f"{colorize(f'{gap:+.2f} bits', gap_color)}")

    return "\n".join(lines)


def format_fidelity_colored(report: FidelityReport) -> str:
    """Format fidelity report with colors."""
    lines = []

    # Score with color based on value
    score = report.fidelity_score
    if score >= 0.8:
        score_color = Colors.GREEN
    elif score >= 0.5:
        score_color = Colors.YELLOW
    else:
        score_color = Colors.RED

    lines.append(f"\n{Colors.BOLD}PROMPT FIDELITY:{Colors.END} {colorize(f'{score:.1%}', score_color)}")

    # Constraints
    lines.append(f"\n{Colors.CYAN}Verified:{Colors.END}")
    for c in report.verified_constraints:
        lines.append(f"  {Colors.GREEN}✓{Colors.END} {c.description} ({c.bits:.2f} bits)")

    lines.append(f"\n{Colors.YELLOW}Inferred:{Colors.END}")
    for c in report.inferred_constraints:
        lines.append(f"  {Colors.YELLOW}?{Colors.END} {c.description} ({c.bits:.2f} bits)")

    # Summary
    lines.append(f"\n{Colors.BOLD}Total:{Colors.END} {report.total_bits:.2f} bits "
                 f"({report.verified_bits:.2f} verified + {report.inferred_bits:.2f} inferred)")

    return "\n".join(lines)
