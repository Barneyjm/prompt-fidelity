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
Extraction (stdlib only, Tier 1 -- deterministic regex/vocab rules, never
an LLM): extract_constraints.
Reporting (stdlib only, no LLM): render -- deterministic text at four
altitudes ("model" / "engineer" / "product" / "executive") from an
already-booked Ledger. Ledger.unhonored + Ledger.conjunction_honored +
Recorder.check() are the pieces that turn this into a mid-run repair loop:
book the calls made so far, render "model" for whatever's unhonored, feed
it back to the agent as an instruction to repair. See recorder.py's module
docstring and report.py for the details.
Hop-2 attribution (stdlib only, no LLM): attribute -- given an already-
booked Ledger, the assistant's response text, and the tool-result blobs it
actually received, attribute each entry as unaddressed / addressed_grounded
/ addressed_ungrounded. This distinguishes a constraint hop 1 booked
`dropped` because it was ignored from one answered correctly from the
model's own knowledge with no tool call at all -- both book identically at
hop 1. Measures PROVENANCE, never truth: see hop2.py's module docstring.
Wrapping (stdlib only, SDK-shape duck-typed): wrap.
Extras (each needs its own optional dependency, imported lazily on first
use so importing `promptfidelity` itself never requires any of them):
    record                    -- promptfidelity.anthropic_ext  [anthropic]
    FidelityCallbackHandler   -- promptfidelity.langchain_ext  [langchain]
    FidelityMiddleware        -- promptfidelity.mcp_ext        [mcp]
    FidelityClientSession     -- promptfidelity.mcp_ext        [mcp]
    FidelityProxy             -- promptfidelity.mcp_ext        [mcp] (design stub)
"""

from .core import Constraint, Ledger, LedgerEntry, bits, book
from .effects import EffectReport, verify_effects
from .extraction import extract_constraints
from .hop2 import Hop2Report, attribute
from .recorder import Recorder, instrument, trace
from .report import render
from .wrap import wrap

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
    "extract_constraints",
    "render",
    "wrap",
    "attribute",
    "Hop2Report",
    "verify_effects",
    "EffectReport",
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
