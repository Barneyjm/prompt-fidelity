"""Tests for promptfidelity.wrap.wrap() -- the drop-in client proxy.

Fake, duck-typed Anthropic- and OpenAI-shaped clients only: no network call,
no SDK install, per this package's convention for every extra/integration.
"""

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

import promptfidelity as pf
from promptfidelity.core import Constraint
from promptfidelity.wrap import wrap

# ---------------------------------------------------------------------------
# Fake Anthropic-shaped client
# ---------------------------------------------------------------------------


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


class FakeMessagesNamespace:
    def __init__(self, response: FakeMessage):
        self._response = response
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class FakeAnthropicClient:
    def __init__(self, response: FakeMessage):
        self.messages = FakeMessagesNamespace(response)
        self.some_other_attr = "passthrough-value"


# ---------------------------------------------------------------------------
# Fake OpenAI-shaped client
# ---------------------------------------------------------------------------


@dataclass
class FakeFunctionCall:
    name: str
    arguments: str  # JSON-encoded string, as the real SDK sends it


@dataclass
class FakeToolCall:
    function: FakeFunctionCall
    id: str = "call_1"


@dataclass
class FakeChoiceMessage:
    tool_calls: list = field(default_factory=list)
    content: str | None = None


@dataclass
class FakeChoice:
    message: FakeChoiceMessage


@dataclass
class FakeChatCompletion:
    choices: list = field(default_factory=list)


class FakeCompletionsNamespace:
    def __init__(self, response: FakeChatCompletion):
        self._response = response
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class FakeChatNamespace:
    def __init__(self, completions: FakeCompletionsNamespace):
        self.completions = completions


class FakeOpenAIClient:
    def __init__(self, response: FakeChatCompletion):
        self.chat = FakeChatNamespace(FakeCompletionsNamespace(response))
        self.some_other_attr = "passthrough-value"


VOCAB = {
    "terms": {"sci-fi": {"with_genres": "878"}},
    "comparators": {"rating": "vote_average"},
}


# ---------------------------------------------------------------------------
# Anthropic-shaped tests
# ---------------------------------------------------------------------------


def test_wrap_extracts_constraints_from_first_user_message_anthropic():
    response = FakeMessage(content=[
        FakeToolUseBlock(id="t1", name="discover_movies", input={"with_genres": "878"}),
    ])
    client = wrap(FakeAnthropicClient(response), vocab=VOCAB)

    client.messages.create(messages=[
        {"role": "user", "content": "a sci-fi movie rated above 7"},
    ])

    assert client.fidelity.constraints
    assert all(c.source == "rules" for c in client.fidelity.constraints)
    assert any(c.params == {"with_genres": "878"} for c in client.fidelity.constraints)


def test_wrap_extracts_from_content_block_list_anthropic():
    response = FakeMessage(content=[])
    client = wrap(FakeAnthropicClient(response), vocab=VOCAB)

    client.messages.create(messages=[
        {"role": "user", "content": [{"type": "text", "text": "a sci-fi movie"}]},
    ])

    assert any(c.params == {"with_genres": "878"} for c in client.fidelity.constraints)


def test_wrap_records_tool_use_blocks_and_books_ledger_anthropic():
    response = FakeMessage(content=[
        FakeTextBlock(text="Searching..."),
        FakeToolUseBlock(id="t1", name="discover_movies", input={"with_genres": "878"}),
    ])
    client = wrap(FakeAnthropicClient(response), vocab=VOCAB)

    client.messages.create(messages=[{"role": "user", "content": "a sci-fi movie"}])

    ledger = client.fidelity.ledger()
    assert ledger.entries[0].account == "verified"


def test_wrap_anthropic_passthrough_of_unrelated_attributes():
    response = FakeMessage(content=[])
    raw_client = FakeAnthropicClient(response)
    client = wrap(raw_client)
    assert client.some_other_attr == "passthrough-value"


def test_wrap_anthropic_real_create_still_called_and_returns_response():
    response = FakeMessage(content=[])
    raw_client = FakeAnthropicClient(response)
    client = wrap(raw_client)
    result = client.messages.create(messages=[{"role": "user", "content": "hi"}])
    assert result is response
    assert raw_client.messages.calls == [{"messages": [{"role": "user", "content": "hi"}]}]


def test_wrap_explicit_constraints_skip_extraction_anthropic():
    constraints = [Constraint(id="c1", description="x", params={"with_genres": "878"}, p=0.5)]
    response = FakeMessage(content=[
        FakeToolUseBlock(id="t1", name="discover_movies", input={"with_genres": "878"}),
    ])
    client = wrap(FakeAnthropicClient(response), constraints=constraints)

    # No user message at all -- extraction would find nothing anyway, but
    # explicit constraints should be used regardless.
    client.messages.create(messages=[])

    assert client.fidelity.constraints == constraints
    assert client.fidelity.constraints[0].source == "declared"


def test_wrap_extraction_runs_only_once_per_wrapper():
    calls = []

    def spy_extractor(prompt):
        calls.append(prompt)
        return []

    response = FakeMessage(content=[])
    client = wrap(FakeAnthropicClient(response), extractor=spy_extractor, vocab=VOCAB)

    client.messages.create(messages=[{"role": "user", "content": "first prompt"}])
    client.messages.create(messages=[{"role": "user", "content": "second prompt"}])

    assert len(calls) == 1
    assert calls[0] == "first prompt"


def test_wrap_fidelity_reset_clears_calls_and_re_extracts():
    response = FakeMessage(content=[
        FakeToolUseBlock(id="t1", name="discover_movies", input={"with_genres": "878"}),
    ])
    client = wrap(FakeAnthropicClient(response), vocab=VOCAB)
    client.messages.create(messages=[{"role": "user", "content": "a sci-fi movie"}])
    assert client.fidelity.calls

    client.fidelity_reset()
    assert client.fidelity.calls == []
    assert client.fidelity.constraints == []

    client.messages.create(messages=[{"role": "user", "content": "comedy from 1994"}])
    assert client.fidelity.calls
    assert not any(c.params == {"with_genres": "878"} for c in client.fidelity.constraints)


def test_wrap_merges_llm_extractor_as_llm_source():
    def fake_extractor(prompt):
        return [Constraint(id="mood", description="cozy", params={"query": "cozy"}, p=0.3)]

    response = FakeMessage(content=[])
    client = wrap(FakeAnthropicClient(response), extractor=fake_extractor, vocab=VOCAB)
    client.messages.create(messages=[{"role": "user", "content": "a sci-fi movie, cozy vibe"}])

    llm_constraints = [c for c in client.fidelity.constraints if c.source == "llm"]
    assert llm_constraints
    assert llm_constraints[0].params == {"query": "cozy"}


def test_wrap_prompt_pins_extraction_bypassing_message_scan():
    # The actual message text ("a comedy movie") differs from the pinned
    # prompt -- prompt= should win, not the message content.
    vocab = {"terms": {"sci-fi": {"with_genres": "878"}, "comedy": {"with_genres": "35"}}}
    response = FakeMessage(content=[])
    client = wrap(FakeAnthropicClient(response), prompt="a sci-fi movie", vocab=vocab)

    client.messages.create(messages=[{"role": "user", "content": "a comedy movie"}])

    assert any(c.params == {"with_genres": "878"} for c in client.fidelity.constraints)
    assert not any(c.params == {"with_genres": "35"} for c in client.fidelity.constraints)


def test_wrap_composes_with_open_trace_anthropic():
    response = FakeMessage(content=[
        FakeToolUseBlock(id="t1", name="discover_movies", input={"with_genres": "878"}),
    ])
    client = wrap(FakeAnthropicClient(response))
    outer_constraints = [Constraint(id="c1", description="genre",
                                    params={"with_genres": "878"}, p=0.5)]

    with pf.trace(outer_constraints) as outer_rec:
        client.messages.create(messages=[{"role": "user", "content": "anything"}])

    outer_ledger = outer_rec.ledger()
    assert outer_ledger.entries[0].account == "verified"


# ---------------------------------------------------------------------------
# OpenAI-shaped tests
# ---------------------------------------------------------------------------


def test_wrap_extracts_constraints_from_first_user_message_openai():
    response = FakeChatCompletion(choices=[
        FakeChoice(message=FakeChoiceMessage(tool_calls=[
            FakeToolCall(function=FakeFunctionCall(
                name="discover_movies", arguments=json.dumps({"with_genres": "878"})
            )),
        ])),
    ])
    client = wrap(FakeOpenAIClient(response), vocab=VOCAB)

    client.chat.completions.create(messages=[
        {"role": "user", "content": "a sci-fi movie rated above 7"},
    ])

    assert any(c.params == {"with_genres": "878"} for c in client.fidelity.constraints)


def test_wrap_records_tool_calls_and_books_ledger_openai():
    response = FakeChatCompletion(choices=[
        FakeChoice(message=FakeChoiceMessage(tool_calls=[
            FakeToolCall(function=FakeFunctionCall(
                name="discover_movies", arguments=json.dumps({"with_genres": "878"})
            )),
        ])),
    ])
    client = wrap(FakeOpenAIClient(response), vocab=VOCAB)

    client.chat.completions.create(messages=[{"role": "user", "content": "a sci-fi movie"}])

    ledger = client.fidelity.ledger()
    assert ledger.entries[0].account == "verified"


def test_wrap_openai_passthrough_of_unrelated_attributes():
    response = FakeChatCompletion(choices=[])
    client = wrap(FakeOpenAIClient(response))
    assert client.some_other_attr == "passthrough-value"


def test_wrap_openai_no_tool_calls_records_nothing():
    response = FakeChatCompletion(choices=[
        FakeChoice(message=FakeChoiceMessage(tool_calls=[], content="just chatting")),
    ])
    client = wrap(FakeOpenAIClient(response), vocab=VOCAB)
    client.chat.completions.create(messages=[{"role": "user", "content": "hello"}])
    assert client.fidelity.calls == []


def test_wrap_ignore_params_passthrough():
    constraints = [Constraint(id="c1", description="genre", params={"with_genres": "878"}, p=0.5)]
    response = FakeMessage(content=[
        FakeToolUseBlock(id="t1", name="discover_movies",
                         input={"with_genres": "878", "page": "2"}),
    ])
    client = wrap(FakeAnthropicClient(response), constraints=constraints,
                  ignore_params={"page"})

    client.messages.create(messages=[{"role": "user", "content": "anything"}])

    ledger = client.fidelity.ledger()
    assert ledger.imposed == []
