"""
promptfidelity -- measuring how much of a prompt's intent an agent actually
executes.

Every account (verified/transmitted/substituted/inferred/dropped/imposed) is
computed by book() from a mechanical diff against actual tool-call
arguments. See core.py for the full account rules.

HARD RULE: LLMs may propose ledger ENTRIES -- Constraints, i.e. a
description, a params dict, and a survival probability -- but never
ACCOUNTS. No function anywhere in this package accepts a pre-assigned
account from model output; book() is the only place an account is ever set.

Core (stdlib only): Constraint, Ledger, LedgerEntry, bits, book.
Recording (stdlib only): Recorder, trace, instrument.
Extras (each needs its own optional dependency, imported lazily on first
use so importing `promptfidelity` itself never requires any of them):
    record                    -- promptfidelity.anthropic_ext  [anthropic]
    FidelityCallbackHandler   -- promptfidelity.langchain_ext  [langchain]
    FidelityMiddleware        -- promptfidelity.mcp_ext        [mcp]
    FidelityClientSession     -- promptfidelity.mcp_ext        [mcp]
    FidelityProxy             -- promptfidelity.mcp_ext        [mcp] (design stub)
"""

from .core import Constraint, Ledger, LedgerEntry, bits, book
from .recorder import Recorder, instrument, trace

__version__ = "0.1.0"

__all__ = [
    "Constraint",
    "Ledger",
    "LedgerEntry",
    "bits",
    "book",
    "Recorder",
    "trace",
    "instrument",
    "record",
    "FidelityCallbackHandler",
    "FidelityMiddleware",
    "FidelityClientSession",
    "FidelityProxy",
]

_LAZY_EXTRAS = {
    "record": ("anthropic_ext", "record"),
    "FidelityCallbackHandler": ("langchain_ext", "FidelityCallbackHandler"),
    "FidelityMiddleware": ("mcp_ext", "FidelityMiddleware"),
    "FidelityClientSession": ("mcp_ext", "FidelityClientSession"),
    "FidelityProxy": ("mcp_ext", "FidelityProxy"),
}


def __getattr__(name):
    """Lazily resolve extras so `import promptfidelity` alone never requires
    anthropic, langchain, or mcp/fastmcp to be installed -- the underlying
    optional dependency is only imported once one of these names is
    actually accessed."""
    target = _LAZY_EXTRAS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    import importlib

    module = importlib.import_module(f".{module_name}", __name__)
    value = getattr(module, attr_name)
    globals()[name] = value  # cache: subsequent access skips __getattr__
    return value
