"""
promptfidelity.extraction -- deterministic Tier-1 constraint extraction.

extract_constraints() is *transcription*, never *judgment*: it turns prompt
text into falsifiable {param: value} pairs using regexes and a caller-supplied
vocabulary -- it never scores, ranks, or grades anything, and it never
estimates a survival probability (every Constraint it emits has `p=None`, so
bits book as None and only constraint *counts* are reliable -- this matches
core.py's existing "no estimate" semantics, it isn't a new rule).

This is Tier 1 of a two-tier extraction story:
    Tier 1 (this module): stdlib regex/vocab rules. Deterministic, free,
        auditable -- the same prompt always produces the same constraints.
    Tier 2 (pluggable, NOT implemented here): an LLM-backed extractor,
        passed as the `extractor` callable to recorder.trace() or wrap.wrap().
        Tier 2 constraints are tagged source="llm" so a reader can always
        tell how much of a measurement rests on probabilistic transcription
        vs. these deterministic rules vs. a human declaring intent directly
        (source="declared"). See core.Constraint.source and
        Ledger.to_dict()["summary"]["constraints_source"].

Every Constraint this module produces has `source="rules"` and `p=None`.

Documented limits (kept simple on purpose -- this is Tier 1, not an NLP
pipeline):
    - English only.
    - No negation handling: "not from the 90s" still extracts a 1990s date
      constraint. Callers who need negation should route through an
      `extractor` (Tier 2) instead.
    - Comparator + number extraction requires a nearby vocabulary keyword
      (see `vocab["comparators"]` below) to know which param a bare number
      refers to; a comparator/number with no resolvable concept nearby is
      left untouched and folds into the residue constraint instead of being
      silently dropped.
    - Vocabulary term matching is exact (case-insensitive, word-boundary)
      surface-form matching, not fuzzy/stemmed matching.

## Vocab shape

`vocab` is a plain, JSON-serializable dict with up to three keys, all
optional:

    vocab = {
        "terms": {
            "sci-fi": {"with_genres": "878"},
            "comedy": {"with_genres": "35"},
        },
        "comparators": {
            # concept keyword -> base param name; ".gte"/".lte" is appended
            # depending on which comparator word was used. The keys
            # "rating" and "runtime" are recognized specially (see below);
            # any other key is matched against its own literal text (and a
            # naive singular/plural variant) as the concept keyword.
            "rating": "vote_average",
            "runtime": "with_runtime",
            "votes": "vote_count",
        },
        "date_param": "primary_release_date",
    }

    - vocab["terms"]: surface phrase -> params dict to emit verbatim.
    - vocab["comparators"]: concept keyword -> base param name for
      comparator+number extraction (see extract_constraints docstring).
      "rating" additionally matches the synonyms rate/rated/score/scored;
      "runtime" additionally matches duration units (hours/hrs/minutes/
      mins) directly, with no keyword required, and converts hours to
      minutes.
    - vocab["date_param"]: base param name for year/decade extraction.
      Defaults to None, meaning "recognize the pattern structurally but
      emit nothing" -- there is no param to express it against.

`vocab=None` (or an empty dict) means: only the purely structural date
pattern-matching runs, and since date_param defaults to None in that case,
even that emits nothing. Terms and comparators need a vocab to produce
anything, since there is no param name to attach without one.
"""

import re
from typing import Any

from .core import Constraint

_STOPWORDS = frozenset("""
a an the and or but of to in on at for with by from into onto is are was
were be been being this that these those it its as than then so not no nor
if else when while about above below under over between through during
before after up down out off again further once here there all any both
each few more most other some such only own same too very s t can will just
should now i you he she we they them his her their my your our what which
who whom
""".split())

_GTE_PHRASES = ["at least", "greater than", "higher than", "more than", "longer than", "above", "over"]
_LTE_PHRASES = ["at most", "less than", "no more than", "shorter than", "below", "under"]

_RATING_SYNONYMS = {"rating", "rated", "rate", "score", "scored"}
_RUNTIME_SYNONYMS = {"runtime", "duration", "length"}
_DURATION_UNITS = {"hour", "hours", "hr", "hrs", "minute", "minutes", "min", "mins"}


def _alt(phrases: list[str]) -> str:
    """Build a regex alternation, longest phrase first so e.g. 'at least'
    is tried before a hypothetical shorter overlapping phrase."""
    escaped = sorted((re.escape(p) for p in phrases), key=len, reverse=True)
    return "|".join(p.replace(r"\ ", r"\s+") for p in escaped)


_COMPARATOR_RE = re.compile(
    rf"\b(?P<cmp>{_alt(_GTE_PHRASES + _LTE_PHRASES)})\s+"
    rf"(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>hours?|hrs?|minutes?|mins?)?\b",
    re.IGNORECASE,
)
_COMPARATOR_PLUS_RE = re.compile(r"\b(?P<num>\d+(?:\.\d+)?)\+")

_BETWEEN_RE = re.compile(
    r"\bbetween\s+(?P<y1>(?:19|20)\d{2})\s+and\s+(?P<y2>(?:19|20)\d{2})\b", re.IGNORECASE
)
_DECADE_WORD_RE = re.compile(r"\b(?:the\s+)?['’]?(?P<dec>\d0)s\b", re.IGNORECASE)
_DECADE_DIGIT_RE = re.compile(r"\b(?P<cent>19|20)(?P<dec>\d)0s\b", re.IGNORECASE)
_RELATIVE_YEAR_RE = re.compile(
    r"\b(?P<word>from|since|after|before)\s+(?P<year>(?:19|20)\d{2})\b", re.IGNORECASE
)
_BARE_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


class _Span:
    """One accepted (start, end) extraction, plus what to build a
    Constraint from. Kept internal -- callers only see the final list of
    Constraints from extract_constraints()."""

    __slots__ = ("start", "end", "description", "params")

    def __init__(self, start: int, end: int, description: str, params: dict[str, Any]):
        self.start = start
        self.end = end
        self.description = description
        self.params = params


def _mask(text: str, mask: list[bool], start: int, end: int) -> None:
    for i in range(start, end):
        mask[i] = True


def _free(mask: list[bool], start: int, end: int) -> bool:
    return not any(mask[start:end])


def _extract_dates(text: str, date_param: str | None, mask: list[bool]) -> list[_Span]:
    spans: list[_Span] = []
    if date_param is None:
        # Structural recognition still runs so matched spans don't leak
        # into the residue, but no param exists to express the date
        # against, so nothing is emitted (see module docstring).
        date_param = None

    def emit(m: re.Match, params: dict[str, Any]) -> None:
        if not _free(mask, m.start(), m.end()):
            return
        _mask(text, mask, m.start(), m.end())
        if date_param is not None:
            spans.append(_Span(m.start(), m.end(), text[m.start():m.end()], params))

    for m in _BETWEEN_RE.finditer(text):
        y1, y2 = m.group("y1"), m.group("y2")
        emit(m, {f"{date_param}.gte": f"{y1}-01-01", f"{date_param}.lte": f"{y2}-12-31"}
             if date_param else {})

    for m in _DECADE_WORD_RE.finditer(text):
        # "the 90s" / "the '90s" -- two-digit decade shorthand assumes the
        # 20th century (documented limitation; "1990s" full-form below is
        # unambiguous and always used literally).
        year_start = 1900 + int(m.group("dec"))
        emit(m, {f"{date_param}.gte": f"{year_start}-01-01",
                 f"{date_param}.lte": f"{year_start + 9}-12-31"} if date_param else {})

    for m in _DECADE_DIGIT_RE.finditer(text):
        year_start = int(m.group("cent") + m.group("dec") + "0")
        emit(m, {f"{date_param}.gte": f"{year_start}-01-01",
                 f"{date_param}.lte": f"{year_start + 9}-12-31"} if date_param else {})

    for m in _RELATIVE_YEAR_RE.finditer(text):
        year = int(m.group("year"))
        word = m.group("word").lower()
        if word == "before":
            emit(m, {f"{date_param}.lte": f"{year - 1}-12-31"} if date_param else {})
        else:  # from | since | after
            emit(m, {f"{date_param}.gte": f"{year}-01-01"} if date_param else {})

    for m in _BARE_YEAR_RE.finditer(text):
        year = m.group(0)
        emit(m, {f"{date_param}.gte": f"{year}-01-01", f"{date_param}.lte": f"{year}-12-31"}
             if date_param else {})

    return spans


def _extract_terms(text: str, terms: dict[str, dict[str, Any]], mask: list[bool]) -> list[_Span]:
    spans: list[_Span] = []
    lowered = text.lower()
    for phrase, params in terms.items():
        pattern = re.compile(rf"\b{re.escape(phrase.lower())}\b")
        m = pattern.search(lowered)
        if m and _free(mask, m.start(), m.end()):
            _mask(text, mask, m.start(), m.end())
            spans.append(_Span(m.start(), m.end(), text[m.start():m.end()], dict(params)))
    return spans


def _resolve_concept(text: str, start: int, end: int, comparators: dict[str, str],
                      unit: str | None) -> str | None:
    """Find which vocab['comparators'] base param a bare comparator+number
    match refers to: a nearby concept keyword, or (for 'runtime') a
    duration unit on the number itself. Returns the base param name, or
    None if no concept could be resolved -- callers must leave the match
    untouched (and un-consumed) in that case."""
    if unit and unit.lower() in _DURATION_UNITS and "runtime" in comparators:
        return comparators["runtime"]

    window = text[max(0, start - 25):min(len(text), end + 25)].lower()
    for keyword, base_param in comparators.items():
        if keyword == "rating":
            synonyms = _RATING_SYNONYMS
        elif keyword == "runtime":
            synonyms = _RUNTIME_SYNONYMS
        else:
            synonyms = {keyword.lower(), keyword.lower().rstrip("s")}
        for syn in synonyms:
            if re.search(rf"\b{re.escape(syn)}\w*\b", window):
                return base_param
    return None


def _extract_comparators(text: str, comparators: dict[str, str], mask: list[bool]) -> list[_Span]:
    spans: list[_Span] = []
    if not comparators:
        return spans

    for m in _COMPARATOR_RE.finditer(text):
        if not _free(mask, m.start(), m.end()):
            continue
        cmp_word = m.group("cmp").lower()
        num = m.group("num")
        unit = m.group("unit")
        base_param = _resolve_concept(text, m.start(), m.end(), comparators, unit)
        if base_param is None:
            continue  # no resolvable concept -- leave for residue

        # cmp_word may contain irregular whitespace ("at  least"); compare
        # against the normalized phrase lists instead of exact membership.
        normalized = re.sub(r"\s+", " ", cmp_word)
        is_gte = normalized in _GTE_PHRASES

        if unit and unit.lower().startswith(("hour", "hr")):
            # Convert to minutes -- the canonical unit for a runtime param.
            minutes = float(num) * 60
            value_str = str(int(minutes)) if minutes == int(minutes) else str(minutes)
        else:
            value_str = num  # preserve the number exactly as the prompt wrote it

        suffix = "gte" if is_gte else "lte"
        _mask(text, mask, m.start(), m.end())
        spans.append(_Span(m.start(), m.end(), text[m.start():m.end()],
                            {f"{base_param}.{suffix}": value_str}))

    for m in _COMPARATOR_PLUS_RE.finditer(text):
        if not _free(mask, m.start(), m.end()):
            continue
        base_param = _resolve_concept(text, m.start(), m.end(), comparators, None)
        if base_param is None:
            continue
        num = m.group("num")
        _mask(text, mask, m.start(), m.end())
        spans.append(_Span(m.start(), m.end(), text[m.start():m.end()],
                            {f"{base_param}.gte": num}))

    return spans


def _residue(text: str, mask: list[bool]) -> Constraint | None:
    remaining_chars = [c if not masked else " " for c, masked in zip(text, mask)]
    remaining = "".join(remaining_chars)
    tokens = re.findall(r"[A-Za-z']+", remaining)
    meaningful = [t for t in tokens if len(t) > 1 and t.lower() not in _STOPWORDS]
    if len(meaningful) < 3:
        return None
    description = re.sub(r"\s+", " ", remaining).strip()
    return Constraint(id="residue", description=description, params={}, p=None, source="rules")


def extract_constraints(prompt: str, vocab: dict[str, Any] | None = None) -> list[Constraint]:
    """Deterministic Tier-1 extraction: prompt text -> falsifiable Constraints.

    Pure function -- same (prompt, vocab) always produces the same list, in
    the same order (by position of the matched text in `prompt`, with the
    residue constraint, if any, last). No LLM call, no randomness, no
    network access.

    Every constraint returned has `source="rules"` and `p=None` (rules
    don't estimate survival probability; see this module's docstring).

    See the module docstring for the `vocab` shape and this function's
    documented limits (English-only, no negation handling, exact
    vocabulary-term matching, comparators need a nearby concept keyword).
    """
    vocab = vocab or {}
    terms: dict[str, dict[str, Any]] = vocab.get("terms") or {}
    comparators: dict[str, str] = vocab.get("comparators") or {}
    date_param: str | None = vocab.get("date_param")

    mask = [False] * len(prompt)

    date_spans = _extract_dates(prompt, date_param, mask)
    term_spans = _extract_terms(prompt, terms, mask)
    comparator_spans = _extract_comparators(prompt, comparators, mask)

    all_spans = sorted(date_spans + term_spans + comparator_spans, key=lambda s: s.start)

    constraints = [
        Constraint(
            id=f"c{i}",
            description=span.description.strip(),
            params=span.params,
            p=None,
            source="rules",
        )
        for i, span in enumerate(all_spans)
    ]

    residue = _residue(prompt, mask)
    if residue is not None:
        constraints.append(residue)

    return constraints
