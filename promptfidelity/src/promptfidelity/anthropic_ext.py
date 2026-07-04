"""
promptfidelity.anthropic_ext -- book a ledger from an Anthropic Message.

Extra: [anthropic]. The `anthropic` package itself is never imported here --
record() works against ANY duck-typed object shaped like an
anthropic.types.Message (a `.content` list of blocks, each exposing `.type`
and, for tool_use blocks, `.id`, `.name`, `.input`). That means callers can
unit-test this module against a plain dataclass/namedtuple, with no network
call and no anthropic install required.
"""

from dataclasses import dataclass, field
from typing import Any

from .core import Constraint, Ledger, book, merge_ledgers


@dataclass
class ToolCallLedger:
    """The ledger booked for one tool_use block, plus enough identifying
    info to correlate it back to the response that produced it."""

    tool_use_id: str
    tool_name: str
    ledger: Ledger


@dataclass
class RecordResult:
    """Per-call ledgers plus the merged ledger for one response."""

    calls: list[ToolCallLedger] = field(default_factory=list)
    merged: Ledger = field(default_factory=Ledger)


def record(
    response: Any,
    constraints: list[Constraint],
    advisory_params: dict[str, set] | None = None,
) -> RecordResult:
    """Book every tool_use block in `response.content` against `constraints`.

    Non-tool_use blocks (text, thinking, ...) are ignored -- they carry no
    call arguments to diff.

    advisory_params: optionally per-tool-name advisory param sets, e.g.
    {"web_search": {"query"}, "*": {"q"}}. "*" applies to every tool name.

    Returns a RecordResult: one Ledger per tool_use block (`.calls`) plus a
    merged Ledger (`.merged`) using the best booking per constraint across
    every call in the response (see core.merge_ledgers) -- e.g. a
    Sonnet turn that calls discover_movies twice, refining a param the
    second time, should credit the constraint that only the second call
    satisfies.
    """
    advisory_params = advisory_params or {}

    def advisory_for(tool_name: str) -> set:
        return set(advisory_params.get(tool_name, set())) | set(advisory_params.get("*", set()))

    calls: list[ToolCallLedger] = []
    for block in getattr(response, "content", None) or []:
        if getattr(block, "type", None) != "tool_use":
            continue
        arguments = getattr(block, "input", None) or {}
        tool_name = getattr(block, "name", "") or ""
        ledger = book(constraints, arguments, advisory_params=advisory_for(tool_name))
        calls.append(ToolCallLedger(
            tool_use_id=getattr(block, "id", "") or "",
            tool_name=tool_name,
            ledger=ledger,
        ))

    merged = merge_ledgers(constraints, [c.ledger for c in calls])
    return RecordResult(calls=calls, merged=merged)
