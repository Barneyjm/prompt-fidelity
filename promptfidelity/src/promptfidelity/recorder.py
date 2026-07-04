"""
promptfidelity.recorder -- capture tool calls made during a run, then book
them against declared constraints without threading a recorder object
through every function signature.

Usage:
    import promptfidelity as pf

    @pf.instrument
    def discover_movies(**kwargs):
        return tmdb.discover_movies(**kwargs)

    with pf.trace(constraints) as rec:
        discover_movies(with_genres="878", vote_average_gte=7.0)
        ...
    ledger = rec.ledger()

Outside an active trace(), @instrument is a transparent no-op: instrumented
functions can stay instrumented in production code paths that never open a
trace() context, with zero recording overhead beyond one contextvar lookup.

Stdlib only: uses contextvars so recording is correct across concurrent
asyncio tasks/threads, each with (or without) its own active trace.

## Auto-extracting trace()

trace() can also derive its constraints from a raw prompt string instead of
requiring the caller to hand-build a list of Constraint objects:

    with pf.trace(prompt="a slow sci-fi movie from the 90s, rated above 7") as rec:
        ...

This runs promptfidelity.extraction.extract_constraints() (Tier 1:
deterministic regex/vocab rules, source="rules") and, if an `extractor`
callable is also supplied, merges in its output (Tier 2: pluggable,
typically LLM-backed, source="llm" -- see _merge_constraints below for the
exact merge rule). Passing both `constraints` and `prompt` is redundant, not
additive: `constraints` wins and `prompt`/`extractor`/`vocab` are ignored.
Passing neither is an error -- trace() needs to know what intent to book
against.
"""

import contextvars
import functools
from contextlib import contextmanager
from typing import Any, Callable

from .core import Constraint, Ledger, book, merge_ledgers
from .extraction import extract_constraints

_active_recorder: contextvars.ContextVar["Recorder | None"] = contextvars.ContextVar(
    "promptfidelity_active_recorder", default=None
)


def _merge_constraints(
    prompt: str,
    extractor: Callable[[str], list[Constraint]] | None,
    vocab: dict | None,
) -> list[Constraint]:
    """Build the constraint list for trace(prompt=...): rules (Tier 1)
    extraction, optionally merged with one pluggable extractor's output
    (Tier 2, typically LLM-backed).

    Merge rule (documented here since it's the one subtle piece of this
    module):
      - Rules-extracted constraints (source="rules") are always kept.
      - Every constraint the extractor callable returns is force-tagged
        source="llm" here -- callers don't have to trust the extractor to
        set it correctly, and can't accidentally launder an LLM guess as
        source="declared".
      - On a param-name collision between a rules constraint and an llm
        constraint, the rules constraint wins and the llm one is dropped:
        deterministic transcription beats probabilistic transcription: if
        the regex/vocab layer already produced a falsifiable param for
        that name, the LLM's opinion adds nothing but a second, less
        certain claim on the same param.
      - llm constraints whose params don't collide with any rules
        constraint are kept -- they add coverage the rules layer missed.
      - If the extractor produced any constraint with empty params (its
        own "residue"/unmapped-intent marker), the rules layer's own
        "residue" entry is dropped: the llm's broader-than-regex read of
        the leftover intent subsumes the narrower structural one, and
        keeping both would double-count un-mapped intent in I_total.
    """
    rules_constraints = extract_constraints(prompt, vocab)
    if extractor is None:
        return rules_constraints

    llm_constraints = list(extractor(prompt))
    for c in llm_constraints:
        c.source = "llm"

    rules_param_names: set[str] = set()
    for c in rules_constraints:
        rules_param_names |= set(c.params)

    llm_has_residue = any(not c.params for c in llm_constraints)

    kept_rules = [
        c for c in rules_constraints
        if not (llm_has_residue and c.id == "residue")
    ]
    kept_llm = [
        c for c in llm_constraints
        if not (set(c.params) & rules_param_names)
    ]

    return kept_rules + kept_llm


class Recorder:
    """Accumulates tool calls made under one trace() context and books them
    against the constraints declared for that context.

    Not constructed directly in normal use -- trace() creates and yields
    one. Exposed as a name so tests and advanced callers can drive it
    without a `with` block (e.g. record_call() calls made from a callback
    that doesn't have access to the `with`-bound variable).
    """

    def __init__(self, constraints: list[Constraint], advisory_params: set | None = None):
        self.constraints = list(constraints)
        self.advisory_params = advisory_params
        self.calls: list[dict[str, Any]] = []

    def record_call(self, name: str, arguments: dict[str, Any]) -> None:
        """Record one tool call's keyword arguments. Called by @instrument;
        may also be called directly for tool calls that can't be wrapped
        (e.g. calls made through a third-party client object)."""
        self.calls.append({"name": name, "arguments": dict(arguments)})

    def ledger(self) -> Ledger:
        """Book every recorded call against self.constraints and merge into
        one Ledger: the best booking per constraint across all calls (see
        core.merge_ledgers)."""
        if not self.calls:
            return book(self.constraints, {}, self.advisory_params)
        per_call = [
            book(self.constraints, call["arguments"], self.advisory_params)
            for call in self.calls
        ]
        return merge_ledgers(self.constraints, per_call)


@contextmanager
def trace(
    constraints: list[Constraint] | None = None,
    *,
    prompt: str | None = None,
    extractor: Callable[[str], list[Constraint]] | None = None,
    vocab: dict | None = None,
    advisory_params: set | None = None,
):
    """Open a recording context for one prompt/run.

    Tool calls made by @instrument-decorated functions anywhere in the
    dynamic extent of this `with` block (including across `await` points
    and nested function calls) are recorded against the constraints for
    this context. Yields the active Recorder; call `.ledger()` on it,
    typically after the `with` block closes, to book everything recorded.

        with pf.trace(constraints) as rec:
            run_agent(...)
        ledger = rec.ledger()

    Constraints can be supplied two ways:
      - `constraints`: a pre-built list of Constraint -- exactly the
        original behavior. Every constraint keeps whatever `.source` it
        already carries (default "declared").
      - `prompt`: a raw prompt string. Constraints are derived via
        promptfidelity.extraction.extract_constraints(prompt, vocab)
        (Tier 1, source="rules"). If `extractor` is also given, its output
        is merged in as Tier 2 (source="llm", force-set) -- see
        _merge_constraints for the exact merge rule.

    Passing both wins with `constraints` (prompt/extractor/vocab are then
    ignored); passing neither raises ValueError -- trace() must be told
    what intent to book against, one way or another.

    Nested trace() contexts each get their own Recorder (contextvars are
    scoped, not global mutable state), and the outer context's Recorder is
    restored on exit.
    """
    if constraints is None and prompt is None:
        raise ValueError("pf.trace() needs either constraints= or prompt=")
    if constraints is None:
        constraints = _merge_constraints(prompt, extractor, vocab)

    rec = Recorder(constraints, advisory_params=advisory_params)
    token = _active_recorder.set(rec)
    try:
        yield rec
    finally:
        _active_recorder.reset(token)


def instrument(fn):
    """Decorator: record this function's keyword arguments as a tool call
    in the active trace(), if any.

    Transparent no-op when there is no active trace() -- safe to leave on a
    function that also runs outside any trace() context (e.g. in
    production request paths that don't want ledger bookkeeping overhead).

    Only keyword arguments are recorded: positional arguments have no
    parameter name to diff against a constraint's declared `params`, so
    instrumented tools should be called with kwargs. Positional args are
    still passed through to the wrapped function untouched.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        rec = _active_recorder.get()
        if rec is not None:
            rec.record_call(fn.__name__, kwargs)
        return fn(*args, **kwargs)

    return wrapper
