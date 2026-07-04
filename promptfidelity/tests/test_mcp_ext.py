"""Tests for promptfidelity.mcp_ext.

FidelityMiddleware needs a real fastmcp.server.middleware.Middleware to
subclass, so its tests are skipped (not errored) when fastmcp isn't
installed, via pytest.importorskip. FidelityClientSession and FidelityProxy
need nothing beyond stdlib + core, so those run unconditionally.
"""

import asyncio

import pytest

import promptfidelity as pf
from promptfidelity.core import Constraint
from promptfidelity.mcp_ext import (
    CONSTRAINTS_KEY,
    LEDGER_KEY,
    FidelityClientSession,
    FidelityProxy,
    infer_advisory_params,
)


def test_fidelity_proxy_is_a_documented_stub():
    with pytest.raises(NotImplementedError):
        FidelityProxy()


class _FakeResult:
    def __init__(self, meta=None):
        self.meta = meta or {}


class _FakeSessionNoLedger:
    """Simulates an upstream server with no FidelityMiddleware: it never
    returns a dev.promptfidelity/ledger in the result _meta."""

    def __init__(self):
        self.last_call = None

    async def call_tool(self, name, arguments, meta=None):
        self.last_call = {"name": name, "arguments": arguments, "meta": meta}
        return _FakeResult(meta={})


class _FakeSessionWithLedger:
    """Simulates a server running FidelityMiddleware: it books server-side
    and returns the ledger in the result _meta."""

    async def call_tool(self, name, arguments, meta=None):
        server_ledger = {
            "v": 1, "entries": [], "imposed": [],
            "summary": {"fidelity": 1.0, "verified_bits": 0, "total_bits": 0, "accounts": {}},
            "booked_by": "server",
        }
        return _FakeResult(meta={LEDGER_KEY: server_ledger})


def test_client_session_injects_constraints_into_request_meta():
    session = _FakeSessionNoLedger()
    client = FidelityClientSession(session)
    constraints = [Constraint(id="c1", description="genre", params={"with_genres": "878"}, p=0.08)]

    asyncio.run(client.call_tool_with_intent(
        "discover_movies", {"with_genres": "878"}, constraints, prompt_id="p1"
    ))

    sent_meta = session.last_call["meta"][CONSTRAINTS_KEY]
    assert sent_meta["v"] == 1
    assert sent_meta["prompt_id"] == "p1"
    assert sent_meta["constraints"][0]["id"] == "c1"
    assert sent_meta["constraints"][0]["params"] == {"with_genres": "878"}


def test_client_session_falls_back_to_client_side_booking_when_no_ledger_returned():
    session = _FakeSessionNoLedger()
    client = FidelityClientSession(session)
    constraints = [Constraint(id="c1", description="genre", params={"with_genres": "878"}, p=0.08)]

    _, ledger_dict = asyncio.run(client.call_tool_with_intent(
        "discover_movies", {"with_genres": "878"}, constraints
    ))

    assert ledger_dict["booked_by"] == "client"
    assert ledger_dict["entries"][0]["account"] == "verified"


def test_client_session_uses_server_ledger_when_present():
    session = _FakeSessionWithLedger()
    client = FidelityClientSession(session)
    constraints = [Constraint(id="c1", description="genre", params={"with_genres": "878"}, p=0.08)]

    _, ledger_dict = asyncio.run(client.call_tool_with_intent(
        "discover_movies", {"with_genres": "878"}, constraints
    ))

    assert ledger_dict["booked_by"] == "server"


def test_fidelity_middleware_requires_fastmcp_extra_to_construct():
    pytest.importorskip("fastmcp")
    from promptfidelity.mcp_ext import FidelityMiddleware

    middleware = FidelityMiddleware(advisory_params={"search": {"query"}})
    assert middleware._advisory_for("search") == {"query"}
    assert middleware._advisory_for("other") == set()


def test_fidelity_middleware_auto_advisory_params_infers_from_fastmcp_tool_schema():
    fastmcp = pytest.importorskip("fastmcp")
    from fastmcp import FastMCP

    from promptfidelity.mcp_ext import FidelityMiddleware

    mcp = FastMCP("test-server")

    @mcp.tool
    def search_movies(query: str, genre: str = "") -> str:
        """Search for movies."""
        return "ok"

    class _FakeFastMCPContext:
        def __init__(self, server):
            self.fastmcp = server

    class _FakeContext:
        def __init__(self, server):
            self.fastmcp_context = _FakeFastMCPContext(server)

    middleware = FidelityMiddleware(advisory_params="auto")
    context = _FakeContext(mcp)

    advisory = asyncio.run(middleware._resolve_advisory("search_movies", context))
    assert advisory == {"query"}


def test_fidelity_middleware_auto_advisory_params_degrades_to_empty_without_fastmcp_context():
    pytest.importorskip("fastmcp")
    from promptfidelity.mcp_ext import FidelityMiddleware

    middleware = FidelityMiddleware(advisory_params="auto")

    class _BareContext:
        pass

    advisory = asyncio.run(middleware._resolve_advisory("search_movies", _BareContext()))
    assert advisory == set()


# ---------------------------------------------------------------------------
# infer_advisory_params
# ---------------------------------------------------------------------------


def test_infer_advisory_params_matches_query_named_string_field():
    schema = {"properties": {"query": {"type": "string"}}}
    assert infer_advisory_params(schema) == {"query"}


def test_infer_advisory_params_matches_by_description_when_name_is_generic():
    schema = {"properties": {"q": {"type": "string", "description": "Free-text search terms"}}}
    assert infer_advisory_params(schema) == {"q"}


def test_infer_advisory_params_ignores_enum_constrained_string():
    schema = {"properties": {"query": {"type": "string", "enum": ["a", "b"]}}}
    assert infer_advisory_params(schema) == set()


def test_infer_advisory_params_ignores_pattern_constrained_string():
    schema = {"properties": {"search_text": {"type": "string", "pattern": r"^\w+$"}}}
    assert infer_advisory_params(schema) == set()


def test_infer_advisory_params_ignores_format_constrained_string():
    schema = {"properties": {"question": {"type": "string", "format": "email"}}}
    assert infer_advisory_params(schema) == set()


def test_infer_advisory_params_ignores_non_string_types():
    schema = {"properties": {"search": {"type": "integer"}}}
    assert infer_advisory_params(schema) == set()


def test_infer_advisory_params_ignores_unrelated_string_fields():
    schema = {"properties": {"with_genres": {"type": "string"}}}
    assert infer_advisory_params(schema) == set()


def test_infer_advisory_params_matches_multiple_fields():
    schema = {
        "properties": {
            "query": {"type": "string"},
            "with_genres": {"type": "string", "enum": ["878", "35"]},
            "keywords": {"type": "string"},
        }
    }
    assert infer_advisory_params(schema) == {"query", "keywords"}


def test_infer_advisory_params_handles_missing_or_empty_schema():
    assert infer_advisory_params({}) == set()
    assert infer_advisory_params(None) == set()
    assert infer_advisory_params({"properties": {}}) == set()


def test_infer_advisory_params_case_insensitive_name_match():
    schema = {"properties": {"Search_Query": {"type": "string"}}}
    assert infer_advisory_params(schema) == {"Search_Query"}


# ---------------------------------------------------------------------------
# FidelityClientSession auto mode (plain call_tool)
# ---------------------------------------------------------------------------


class _FakeSessionWithToolsList:
    """Simulates a bare mcp ClientSession: plain call_tool + list_tools,
    no dev.promptfidelity/ledger _meta plumbing at all."""

    def __init__(self, tools):
        self._tools = tools
        self.calls = []

    async def call_tool(self, name, arguments):
        self.calls.append({"name": name, "arguments": arguments})
        return _FakeResult(meta={})

    async def list_tools(self):
        class _Result:
            pass
        result = _Result()
        result.tools = self._tools
        return result


def test_call_tool_auto_mode_records_into_active_trace():
    session = _FakeSessionWithToolsList(tools=[])
    client = FidelityClientSession(session)
    constraints = [Constraint(id="c1", description="genre", params={"with_genres": "878"}, p=0.08)]

    async def run():
        with pf.trace(constraints) as rec:
            await client.call_tool("discover_movies", {"with_genres": "878"})
        return rec

    rec = asyncio.run(run())
    ledger = rec.ledger()
    assert ledger.entries[0].account == "verified"


def test_call_tool_auto_mode_is_noop_recording_outside_trace():
    session = _FakeSessionWithToolsList(tools=[])
    client = FidelityClientSession(session)

    result = asyncio.run(client.call_tool("discover_movies", {"with_genres": "878"}))
    assert session.calls == [{"name": "discover_movies", "arguments": {"with_genres": "878"}}]
    assert result.meta == {}


def test_call_tool_auto_mode_infers_advisory_params_from_cached_schema():
    tools = [{"name": "search_movies", "inputSchema": {
        "properties": {"query": {"type": "string"}}
    }}]
    session = _FakeSessionWithToolsList(tools=tools)
    client = FidelityClientSession(session)
    constraints = [Constraint(id="c1", description="mood", params={"query": "cozy"}, p=0.2)]

    async def run():
        with pf.trace(constraints) as rec:
            await client.call_tool("search_movies", {"query": "cozy"})
        return rec

    rec = asyncio.run(run())
    ledger = rec.ledger()
    assert ledger.entries[0].account == "transmitted"


def test_call_tool_auto_mode_degrades_gracefully_without_list_tools():
    class _NoListToolsSession:
        async def call_tool(self, name, arguments):
            return _FakeResult(meta={})

    client = FidelityClientSession(_NoListToolsSession())
    constraints = [Constraint(id="c1", description="genre", params={"with_genres": "878"}, p=0.08)]

    async def run():
        with pf.trace(constraints) as rec:
            await client.call_tool("discover_movies", {"with_genres": "878"})
        return rec

    rec = asyncio.run(run())
    ledger = rec.ledger()
    assert ledger.entries[0].account == "verified"


def test_call_tool_auto_mode_caches_tools_list_across_calls():
    call_count = {"n": 0}

    class _CountingSession:
        async def call_tool(self, name, arguments):
            return _FakeResult(meta={})

        async def list_tools(self):
            call_count["n"] += 1

            class _Result:
                tools = []
            return _Result()

    client = FidelityClientSession(_CountingSession())

    async def run():
        with pf.trace([]) as rec:
            await client.call_tool("a", {})
            await client.call_tool("b", {})
        return rec

    asyncio.run(run())
    assert call_count["n"] == 1
