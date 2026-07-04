"""Tests for promptfidelity.anthropic_ext.record().

Uses fake, duck-typed objects shaped like an anthropic.types.Message and its
content blocks -- no `anthropic` install and no network call, per the
package's requirement that extras be unit-testable without their optional
dependency.
"""

from dataclasses import dataclass, field
from typing import Any

from promptfidelity.anthropic_ext import record
from promptfidelity.core import Constraint


@dataclass
class FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class FakeToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]
    type: str = "tool_use"


@dataclass
class FakeMessage:
    content: list = field(default_factory=list)


def test_ignores_non_tool_use_blocks():
    constraints = [Constraint(id="c1", description="genre", params={"with_genres": "878"}, p=0.08)]
    response = FakeMessage(content=[
        FakeTextBlock(text="Let me look that up for you."),
    ])
    result = record(response, constraints)
    assert result.calls == []
    assert result.merged.entries[0].account == "dropped"


def test_books_single_tool_use_block():
    constraints = [Constraint(id="c1", description="genre", params={"with_genres": "878"}, p=0.08)]
    response = FakeMessage(content=[
        FakeTextBlock(text="Searching..."),
        FakeToolUseBlock(id="toolu_1", name="discover_movies", input={"with_genres": "878"}),
    ])
    result = record(response, constraints)
    assert len(result.calls) == 1
    call = result.calls[0]
    assert call.tool_use_id == "toolu_1"
    assert call.tool_name == "discover_movies"
    assert call.ledger.entries[0].account == "verified"
    assert result.merged.entries[0].account == "verified"


def test_advisory_params_per_tool_name():
    constraints = [Constraint(id="c1", description="mood", params={"query": "cozy"}, p=0.2)]
    response = FakeMessage(content=[
        FakeToolUseBlock(id="toolu_1", name="web_search", input={"query": "cozy"}),
    ])
    result = record(response, constraints, advisory_params={"web_search": {"query"}})
    assert result.calls[0].ledger.entries[0].account == "transmitted"


def test_advisory_params_wildcard_applies_to_all_tools():
    constraints = [Constraint(id="c1", description="mood", params={"q": "cozy"}, p=0.2)]
    response = FakeMessage(content=[
        FakeToolUseBlock(id="toolu_1", name="any_tool", input={"q": "cozy"}),
    ])
    result = record(response, constraints, advisory_params={"*": {"q"}})
    assert result.calls[0].ledger.entries[0].account == "transmitted"


def test_merged_ledger_credits_best_booking_across_multiple_tool_use_blocks():
    constraints = [
        Constraint(id="c1", description="genre", params={"with_genres": "878"}, p=0.08),
        Constraint(id="c2", description="year", params={"primary_release_year": "1999"}, p=0.1),
    ]
    response = FakeMessage(content=[
        FakeToolUseBlock(id="toolu_1", name="discover_movies", input={"with_genres": "878"}),
        FakeToolUseBlock(id="toolu_2", name="discover_movies",
                          input={"with_genres": "878", "primary_release_year": "1999"}),
    ])
    result = record(response, constraints)
    assert len(result.calls) == 2
    by_id = {e.id: e.account for e in result.merged.entries}
    assert by_id["c1"] == "verified"
    assert by_id["c2"] == "verified"
    # First call alone would have dropped c2 -- confirm it's booked that way
    # per-call, even though the merge credits it as verified overall.
    assert result.calls[0].ledger.entries[1].account == "dropped"


def test_no_tool_use_blocks_at_all_yields_empty_calls_and_dropped_merge():
    constraints = [Constraint(id="c1", description="genre", params={"with_genres": "878"}, p=0.08)]
    response = FakeMessage(content=[])
    result = record(response, constraints)
    assert result.calls == []
    assert result.merged.entries[0].account == "dropped"


def test_response_with_no_content_attribute_at_all():
    class BareResponse:
        pass

    constraints = [Constraint(id="c1", description="genre", params={"with_genres": "878"}, p=0.08)]
    result = record(BareResponse(), constraints)
    assert result.calls == []
    assert result.merged.entries[0].account == "dropped"
