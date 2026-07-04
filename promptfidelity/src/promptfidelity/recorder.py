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
"""

import contextvars
import functools
from contextlib import contextmanager
from typing import Any

from .core import Constraint, Ledger, book, merge_ledgers

_active_recorder: contextvars.ContextVar["Recorder | None"] = contextvars.ContextVar(
    "promptfidelity_active_recorder", default=None
)


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
def trace(constraints: list[Constraint], advisory_params: set | None = None):
    """Open a recording context for one prompt/run.

    Tool calls made by @instrument-decorated functions anywhere in the
    dynamic extent of this `with` block (including across `await` points
    and nested function calls) are recorded against `constraints`. Yields
    the active Recorder; call `.ledger()` on it, typically after the `with`
    block closes, to book everything recorded.

        with pf.trace(constraints) as rec:
            run_agent(...)
        ledger = rec.ledger()

    Nested trace() contexts each get their own Recorder (contextvars are
    scoped, not global mutable state), and the outer context's Recorder is
    restored on exit.
    """
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
