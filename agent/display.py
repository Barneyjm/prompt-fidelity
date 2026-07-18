"""
Display and output formatting module.

Formats recommendation results with fidelity reports for CLI and other outputs.
"""

from typing import TextIO
import sys

from .fidelity import FidelityReport, Constraint


def format_fidelity_bar(score: float, width: int = 20) -> str:
    """Create a visual bar representation of fidelity score."""
    filled = int(score * width)
    empty = width - filled
    bar = "█" * filled + "░" * empty
    return f"[{bar}] {score:.1%}"


CONSTRAINT_ICONS = {"verified": "✓", "inferred": "?", "injected": "!"}


def format_constraint_line(constraint: Constraint, show_bits: bool = True) -> str:
    """Format a single constraint for display."""
    icon = CONSTRAINT_ICONS.get(constraint.constraint_type, "?")
    annotations = []
    if show_bits and constraint.estimated_survival_rate is not None:
        annotations.append(f"{constraint.bits:.2f} bits")
    if constraint.rate_source:
        annotations.append(constraint.rate_source)
    suffix = f" ({', '.join(annotations)})" if annotations else ""
    return f"  {icon} {constraint.description}{suffix}"


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
        lines.append(f"\n  VERIFIED (checkable against the data source):")
        for c in report.verified_constraints:
            lines.append(format_constraint_line(c))

    # Inferred constraints
    if report.inferred_constraints:
        lines.append(f"\n  INFERRED (requires LLM judgment):")
        for c in report.inferred_constraints:
            lines.append(format_constraint_line(c))

    # Injected filters (system-applied, excluded from the score)
    if report.injected_constraints:
        lines.append(f"\n  INJECTED (system-applied, not requested — excluded from score):")
        for c in report.injected_constraints:
            lines.append(format_constraint_line(c))

    # Summary statistics
    joint = report.verified_joint_survival_rate is not None
    lines.append(f"\n  {'─' * 40}")
    lines.append(f"  Verified:  {report.verified_bits:6.2f} bits "
                 f"({len(report.verified_constraints)} constraints"
                 f"{', joint-measured' if joint else ''})")
    lines.append(f"  Inferred:  {report.inferred_bits:6.2f} bits ({len(report.inferred_constraints)} constraints)")
    lines.append(f"  Total:     {report.total_bits:6.2f} bits")
    if joint:
        adjustment = report.verified_bits - report.verified_bits_summed
        lines.append(f"  Correlation: summed {report.verified_bits_summed:.2f} bits "
                     f"→ joint {report.verified_bits:.2f} ({adjustment:+.2f} adjustment)")
    if report.injected_constraints:
        n = len(report.injected_constraints)
        lines.append(f"  Injected:  {n} system "
                     f"{'filter narrows' if n == 1 else 'filters narrow'} "
                     f"the pool beyond the request")
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
    fidelity_report: FidelityReport,
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

    # Fidelity report
    lines.append(format_fidelity_report(fidelity_report))

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
    fidelity_report: FidelityReport,
    file: TextIO = sys.stdout,
    **kwargs
) -> None:
    """Print formatted recommendations to output."""
    output = format_recommendations(prompt, movies, fidelity_report, **kwargs)
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
    fidelity_report: FidelityReport,
    decomposition: dict | None = None
) -> dict:
    """Format complete output as JSON-serializable dict."""
    return {
        "query": prompt,
        "fidelity": fidelity_report.to_dict(),
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

    if report.injected_constraints:
        lines.append(f"\n{Colors.RED}Injected (not requested):{Colors.END}")
        for c in report.injected_constraints:
            lines.append(f"  {Colors.RED}!{Colors.END} {c.description}")

    # Summary
    lines.append(f"\n{Colors.BOLD}Total:{Colors.END} {report.total_bits:.2f} bits "
                 f"({report.verified_bits:.2f} verified + {report.inferred_bits:.2f} inferred)")

    return "\n".join(lines)
