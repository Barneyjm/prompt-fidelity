"""Tests for promptfidelity.langchain_ext.FidelityCallbackHandler.

Importing the module must never fail without langchain-core installed; only
instantiating the handler should raise. The handler-behavior tests below
need a real BaseCallbackHandler to subclass, so they're skipped (not
errored) when langchain-core isn't present.
"""

import uuid

import pytest

from promptfidelity.core import Constraint

pytest.importorskip("langchain_core")

from promptfidelity.langchain_ext import FidelityCallbackHandler  # noqa: E402


def test_on_tool_start_records_structured_inputs_and_books():
    constraints = [Constraint(id="c1", description="genre", params={"with_genres": "878"}, p=0.08)]
    handler = FidelityCallbackHandler(constraints)
    handler.on_tool_start(
        {"name": "discover_movies"}, "", run_id=uuid.uuid4(),
        inputs={"with_genres": "878"},
    )
    ledger = handler.ledger()
    assert ledger.entries[0].account == "verified"


def test_falls_back_to_input_str_when_no_structured_inputs():
    constraints = [Constraint(id="c1", description="freeform", params={"input": "cozy movie"}, p=0.2)]
    handler = FidelityCallbackHandler(constraints)
    handler.on_tool_start({"name": "search"}, "cozy movie", run_id=uuid.uuid4())
    ledger = handler.ledger()
    assert ledger.entries[0].account == "verified"


def test_no_calls_books_dropped():
    constraints = [Constraint(id="c1", description="genre", params={"with_genres": "878"}, p=0.08)]
    handler = FidelityCallbackHandler(constraints)
    ledger = handler.ledger()
    assert ledger.entries[0].account == "dropped"


def test_import_alone_does_not_require_langchain():
    # Simulate langchain-core being absent and confirm the module still
    # imports; only construction should raise.
    import promptfidelity.langchain_ext as mod

    original_base = mod._Base
    try:
        mod._Base = object
        with pytest.raises(ImportError):
            mod.FidelityCallbackHandler([])
    finally:
        mod._Base = original_base
