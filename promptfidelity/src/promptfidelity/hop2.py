"""
promptfidelity.hop2 -- attributing a constraint's booked fate to what the
assistant actually SAID, and where the material behind that came from.

Hop 1 (core.book()) answers "did the tool call carry this constraint's
params". It cannot tell two very different situations apart, because both
book identically as `dropped`:
    - the model silently ignored a constraint it had no business ignoring
    - the model answered the constraint correctly, from its own training
      knowledge, without ever needing a tool call to do it (e.g. "is this
      movie rated R?" answered from memory, no rating filter ever sent)

Hop 2 closes that gap. For every constraint already booked by hop 1, it asks
two further, still-mechanical questions against the assistant's own
response text and the tool-result blobs the agent actually received in
this run:

    1. ADDRESSED -- did the response text talk about this constraint at
       all (its param values, or enough of its descriptive words)?
    2. GROUNDED (only asked if addressed) -- was material supporting that
       claim present in at least one tool result the agent actually saw,
       or was it addressed with nothing behind it in evidence?

CRITICAL EPISTEMIC CAVEAT -- read this before trusting any output of this
module: hop 2 measures PROVENANCE, never TRUTH. "Grounded" means matching
material was PRESENT in a tool result the agent received; it is not a
verification that the response's claim is accurate, nor that the model
actually read or used that material to form its answer -- a coincidental
string match is scored identically to a load-bearing one. "Ungrounded"
does NOT mean wrong or fabricated: it may be perfectly sound, well-known
information the model already knew and stated correctly with no tool call
required at all. The entire point of this measurement is that, from the
outside, a user reading the response CANNOT TELL an ungrounded-but-correct
answer from a fabrication -- both look identical in the response text. Hop
2 doesn't resolve that ambiguity (nothing mechanical can, without a truth
oracle); it makes the ambiguity itself visible and countable, per
constraint, instead of leaving it invisible inside a `dropped` account that
already looked the same either way.

All matching in this module is deterministic string containment and
token-overlap arithmetic against disclosed, fixed thresholds (see
ADDRESSED_WORD_FRACTION / GROUNDED_WORD_FRACTION below) -- there is no LLM
anywhere in attribute(). A pluggable LLM-backed claim-decomposition tier
(splitting a response's prose into discrete factual claims before matching
each one, the way extraction.py's `extractor` callable adds an LLM-backed
Tier 2 on top of Tier-1 rules extraction) is a natural extension point here
too, but it is NOT implemented in this module -- see the module docstring
convention set by extraction.py and recorder.py for the same pattern.

The two-hop product. End-to-end fidelity for one constraint is the product
of what each hop measures: whether it reached the tool call honestly (hop
1) times whether, having reached (or not reached) that call, the response
talked about it honestly and with visible support (hop 2). `Hop2Report`'s
`fate_matrix` (hop1_account x hop2_status) is the complete, two-dimensional
account of one constraint's life across a run -- not just "was it in the
call" and not just "was it mentioned", but both together.
"""

from dataclasses import dataclass, field

from .core import Ledger, LedgerEntry
from .extraction import _content_words

# Disclosed calibration constants, not measurements of truth. These
# thresholds are the whole of hop 2's judgment calls, spelled out here so a
# reader can see exactly what "addressed" and "grounded" mean instead of
# trusting an opaque score. Changing them changes what counts as addressed/
# grounded; it never changes what the underlying text actually contained.
MIN_VALUE_LEN = 3
"""A param value shorter than this (as a string) is too generic to trust
as a substring match (e.g. "1", "PG") -- it is never used, on its own, to
call a constraint addressed or grounded."""

MIN_WORDS_FOR_OVERLAP = 2
"""An entry with fewer than this many content words has no reliable
fraction to compute (1 word matching is a 100% "overlap" by construction) --
such entries are addressed/grounded via param-value matching only."""

ADDRESSED_WORD_FRACTION = 0.5
"""Fraction of an entry's content words (description + param values) that
must appear in the response text for word-overlap alone to call it
addressed. Calibration, not a claim that 50% overlap proves the model was
"talking about" this constraint -- it is the disclosed line this module
draws."""

GROUNDED_WORD_FRACTION = 0.6
"""Fraction of an entry's content words that must appear WITHIN A SINGLE
tool-result blob for word-overlap alone to call it grounded. Deliberately
per-blob, not pooled across every blob the agent saw: pooling would let
words scattered across unrelated results launder into apparent support for
a claim no single result actually backs. Higher than
ADDRESSED_WORD_FRACTION because grounding is a stronger claim (material
was actually there) than addressing (the response merely mentioned it)."""


class Hop2Status:
    """Namespace for the three hop-2 status strings -- see this module's
    docstring for what each one means and, critically, what it does NOT
    mean (grounded != true; ungrounded != false)."""

    UNADDRESSED = "unaddressed"
    ADDRESSED_GROUNDED = "addressed_grounded"
    ADDRESSED_UNGROUNDED = "addressed_ungrounded"


STATUSES = (Hop2Status.UNADDRESSED, Hop2Status.ADDRESSED_GROUNDED, Hop2Status.ADDRESSED_UNGROUNDED)


def _matching_words(entry: LedgerEntry) -> set[str]:
    """Content words drawn from an entry's description plus all of its
    param VALUES (not names) -- the vocabulary a response would have to
    echo, at least partly, to count as "talking about" this constraint."""
    words = set(_content_words(entry.description))
    for v in (entry.params or {}).values():
        words |= _content_words(str(v))
    return words


def _param_value_strings(entry: LedgerEntry) -> list[str]:
    """All of an entry's declared param values, coerced to str, in a fixed
    (insertion) order -- what substring matching checks against."""
    return [str(v) for v in (entry.params or {}).values()]


def _addressed(entry_words: set[str], values: list[str], response_text: str) -> tuple[bool, str]:
    """Was this entry talked about in response_text at all?

    True if any param value (len >= MIN_VALUE_LEN) appears in
    response_text, OR (for entries with >= MIN_WORDS_FOR_OVERLAP content
    words) at least ADDRESSED_WORD_FRACTION of those words appear in
    response_text. Entries with too few content words to trust a fraction
    are judged on param-value matching alone -- see MIN_WORDS_FOR_OVERLAP.
    """
    lowered_response = response_text.lower()
    for v in values:
        if len(v) >= MIN_VALUE_LEN and v.lower() in lowered_response:
            return True, f"param value {v!r} found in response text"

    if len(entry_words) < MIN_WORDS_FOR_OVERLAP:
        return False, (
            f"no param value found in response text; only {len(entry_words)} "
            f"content word(s) (< {MIN_WORDS_FOR_OVERLAP}) -- word-overlap not evaluated"
        )

    response_words = _content_words(response_text)
    overlap = entry_words & response_words
    fraction = len(overlap) / len(entry_words)
    if fraction >= ADDRESSED_WORD_FRACTION:
        return True, (
            f"{len(overlap)}/{len(entry_words)} content words {sorted(overlap)} "
            f"found in response text"
        )
    return False, (
        f"no param value found in response text; only {len(overlap)}/{len(entry_words)} "
        f"content words overlap (< {ADDRESSED_WORD_FRACTION:.0%})"
    )


def _grounded(entry_words: set[str], values: list[str], tool_results: list[str]) -> tuple[bool, str]:
    """Was material supporting this (already-addressed) entry present in
    at least one tool result the agent actually received?

    True if any param value (len >= MIN_VALUE_LEN) appears in at least one
    tool_results blob, OR at least GROUNDED_WORD_FRACTION of the entry's
    content words appear within a SINGLE blob (never pooled across blobs --
    see GROUNDED_WORD_FRACTION's docstring on why pooling would launder
    scattered fragments into false support).
    """
    for v in values:
        if len(v) < MIN_VALUE_LEN:
            continue
        lowered_value = v.lower()
        for i, blob in enumerate(tool_results):
            if lowered_value in blob.lower():
                return True, f"param value {v!r} found in tool_results[{i}]"

    if entry_words:
        for i, blob in enumerate(tool_results):
            blob_words = _content_words(blob)
            overlap = entry_words & blob_words
            fraction = len(overlap) / len(entry_words)
            if fraction >= GROUNDED_WORD_FRACTION:
                return True, (
                    f"{len(overlap)}/{len(entry_words)} content words {sorted(overlap)} "
                    f"found in tool_results[{i}] alone"
                )

    return False, (
        "no param value or single-blob word overlap "
        f"(>= {GROUNDED_WORD_FRACTION:.0%}) found in any tool_results entry"
    )


@dataclass
class Hop2Entry:
    """One hop-1 entry's hop-2 attribution: was it addressed in the
    response text, and if so, was it grounded in a tool result the agent
    actually received.

    id/description mirror the source core.LedgerEntry 1:1 (same id -- this
    is an attribution ON TOP of a hop-1 booking, not a new constraint).
    hop1_account is copied through so a reader never has to re-join against
    the original Ledger to see the two-hop picture. evidence is a short,
    mechanical string naming what matched and where (response text vs.
    which tool_results index) -- never a natural-language explanation.
    """

    id: str
    description: str
    hop1_account: str
    status: str
    evidence: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "hop1_account": self.hop1_account,
            "status": self.status,
            "evidence": self.evidence,
        }


@dataclass
class Hop2Report:
    """The hop-2 attribution of every entry in one hop-1 Ledger.

    Construct via attribute() -- never by hand-assigning statuses; see this
    module's docstring for why (mirrors core.py's HARD RULE for accounts:
    no function here accepts a pre-assigned status from model output,
    attribute() computes every one from a mechanical text/blob match).
    """

    entries: list[Hop2Entry] = field(default_factory=list)

    @property
    def fate_matrix(self) -> dict[tuple[str, str], int]:
        """(hop1_account, hop2_status) -> count of entries with that exact
        combination -- the complete, two-dimensional fate of every
        constraint this run booked: not just where hop 1 left it, and not
        just whether hop 2 saw it addressed, but both together."""
        matrix: dict[tuple[str, str], int] = {}
        for e in self.entries:
            key = (e.hop1_account, e.status)
            matrix[key] = matrix.get(key, 0) + 1
        return matrix

    @property
    def narration_risk(self) -> list[Hop2Entry]:
        """Entries addressed (grounded or not) whose hop1_account is
        `dropped` or `substituted` -- the response spoke about a
        constraint whose execution was never fully verified. This is the
        per-constraint, mechanical form of the honesty gap this whole
        package exists to surface: talk without matching action."""
        return [
            e for e in self.entries
            if e.hop1_account in ("dropped", "substituted")
            and e.status in (Hop2Status.ADDRESSED_GROUNDED, Hop2Status.ADDRESSED_UNGROUNDED)
        ]

    @property
    def knowledge_answered(self) -> list[Hop2Entry]:
        """Entries booked `dropped` at hop 1 but addressed_ungrounded at
        hop 2 -- the "answered from the model's own knowledge, with no
        tool call and no visible supporting material" population. Not
        necessarily wrong (see module docstring) -- it is the population a
        reader most needs disclosed, since hop 1 alone renders these
        indistinguishable from constraints that were simply ignored."""
        return [
            e for e in self.entries
            if e.hop1_account == "dropped" and e.status == Hop2Status.ADDRESSED_UNGROUNDED
        ]

    @property
    def ignored(self) -> list[Hop2Entry]:
        """Entries booked `dropped` at hop 1 AND never addressed at hop 2
        -- genuinely silent drops, the population `knowledge_answered`
        exists to be distinguished from."""
        return [
            e for e in self.entries
            if e.hop1_account == "dropped" and e.status == Hop2Status.UNADDRESSED
        ]

    def to_dict(self) -> dict:
        return {
            "entries": [e.to_dict() for e in self.entries],
            "fate_matrix": {f"{account}/{status}": count
                            for (account, status), count in self.fate_matrix.items()},
            "narration_risk": [e.id for e in self.narration_risk],
            "knowledge_answered": [e.id for e in self.knowledge_answered],
            "ignored": [e.id for e in self.ignored],
        }

    def render(self, audience: str) -> str:
        """Delegates to render() in this module -- see that function for
        the two supported audiences ("engineer", "executive")."""
        return render(self, audience)


def attribute(ledger: Ledger, response_text: str, tool_results: list[str]) -> Hop2Report:
    """Attribute every entry in an already-booked hop-1 `ledger` to a hop-2
    status, given the assistant's own `response_text` and the serialized
    `tool_results` blobs the agent actually received during this run.

    Pure function; deterministic string/token matching only (see this
    module's docstring and the ADDRESSED_WORD_FRACTION / GROUNDED_WORD_FRACTION
    constants for the exact rules) -- no LLM call, no network access, no
    randomness. Calling this twice on the same inputs always returns the
    same Hop2Report.

    response_text: the assistant's user-facing narration for this turn (a
    plain string; pass "" if there was none).
    tool_results: serialized tool-result contents the agent actually
    received (e.g. the JSON/text bodies returned to it), one string per
    result. Pass [] if the run made no tool calls at all -- every addressed
    entry then books addressed_ungrounded, since there is nothing to have
    grounded it.
    """
    entries: list[Hop2Entry] = []
    for e in ledger.entries:
        words = _matching_words(e)
        values = _param_value_strings(e)

        addressed, addressed_evidence = _addressed(words, values, response_text)
        if not addressed:
            entries.append(Hop2Entry(
                e.id, e.description, e.account, Hop2Status.UNADDRESSED, addressed_evidence,
            ))
            continue

        grounded, grounded_evidence = _grounded(words, values, tool_results)
        status = Hop2Status.ADDRESSED_GROUNDED if grounded else Hop2Status.ADDRESSED_UNGROUNDED
        entries.append(Hop2Entry(
            e.id, e.description, e.account, status,
            f"{addressed_evidence}; {grounded_evidence}",
        ))

    return Hop2Report(entries=entries)


HOP2_AUDIENCES = ("engineer", "executive")


def _render_engineer(report: Hop2Report) -> str:
    """Full detail: every entry's hop1 account, hop2 status, and evidence,
    plus the fate matrix and the three named risk populations."""
    lines = [
        f"fate_matrix: { {f'{a}/{s}': c for (a, s), c in report.fate_matrix.items()} }",
        "",
        f"entries ({len(report.entries)}):",
    ]
    for e in report.entries:
        lines.append(
            f"  [{e.hop1_account}/{e.status}] {e.id} \"{e.description}\" evidence={e.evidence!r}"
        )
    lines.append("")
    lines.append(f"narration_risk ({len(report.narration_risk)}): "
                 f"{[e.id for e in report.narration_risk]}")
    lines.append(f"knowledge_answered ({len(report.knowledge_answered)}): "
                 f"{[e.id for e in report.knowledge_answered]}")
    lines.append(f"ignored ({len(report.ignored)}): {[e.id for e in report.ignored]}")
    return "\n".join(lines)


def _render_executive(report: Hop2Report) -> str:
    """At most ~5 lines: how much was addressed with no visible support
    (provenance risk), how much was narrated without verified execution,
    how much was simply ignored, one closing risk line."""
    addressed_ungrounded = sum(
        1 for e in report.entries if e.status == Hop2Status.ADDRESSED_UNGROUNDED
    )
    lines = [
        f"{addressed_ungrounded} addressed-ungrounded [provenance risk].",
        f"{len(report.narration_risk)} narration-risk (talked about, execution unverified).",
        f"{len(report.ignored)} ignored (dropped and never addressed).",
    ]
    if addressed_ungrounded or report.narration_risk:
        lines.append(
            "Some claims in the response have no visible supporting material in this "
            "run's tool results -- provenance unknown, not necessarily false."
        )
    else:
        lines.append("Every addressed claim traces to material the agent actually received.")
    return "\n".join(lines)


_HOP2_RENDERERS = {
    "engineer": _render_engineer,
    "executive": _render_executive,
}


def render(report: Hop2Report, audience: str) -> str:
    """Render a Hop2Report as plain text for one of two audiences:
    "engineer" (full per-entry detail) or "executive" (<=5 lines). Raises
    ValueError for anything else.

    Deterministic string templates over Hop2Report/Hop2Entry data only --
    no LLM call, no randomness. Calling this twice on the same report
    always returns the same string. See report.render() for the
    equivalent four-audience instrument over hop-1 Ledgers; this is the
    two-audience hop-2 counterpart, kept in this module (rather than
    report.py) so report.py never has to import hop2.py -- imports stay
    one-directional, hop2 -> core/extraction, report -> core, nothing
    circular.
    """
    try:
        renderer = _HOP2_RENDERERS[audience]
    except KeyError:
        raise ValueError(f"unknown audience {audience!r}; choose one of {HOP2_AUDIENCES}") from None
    return renderer(report)
