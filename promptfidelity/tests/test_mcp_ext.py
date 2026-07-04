"""Tests for promptfidelity.mcp_ext.

FidelityMiddleware needs a real fastmcp.server.middleware.Middleware to
subclass, so its tests are skipped (not errored) when fastmcp isn't
installed, via pytest.importorskip. FidelityClientSession and FidelityProxy
need nothing beyond stdlib + core, so those run unconditionally.
"""

import asyncio

import pytest

from promptfidelity.core import Constraint
from promptfidelity.mcp_ext import (
    CONSTRAINTS_KEY,
    LEDGER_KEY,
    FidelityClientSession,
    FidelityProxy,
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
