"""
promptfidelity.mcp_ext -- MCP / FastMCP integration implementing the
`dev.promptfidelity/v1` interchange format (Level 1: booking).

Extra: [mcp]. Two independent optional dependencies:
    fastmcp -- for FidelityMiddleware (server-side booking)
    mcp     -- for FidelityClientSession (client-side injection / fallback)
Importing this module never fails without either installed; only
instantiating the class that needs the missing package raises, with a
message naming the extra to install.

Reads intent constraints from CallToolRequest params._meta
(`dev.promptfidelity/constraints`), books every constraint against the
ACTUAL call arguments via promptfidelity.core.book() (pure diff, no LLM in
the loop), and attaches the resulting ledger to CallToolResult._meta
(`dev.promptfidelity/ledger`). Account-assignment logic lives in exactly one
place -- core.book() -- so server-side and client-side booking here always
agree.
"""

from typing import Any

from .core import Constraint, Ledger, book

NAMESPACE = "dev.promptfidelity"
CONSTRAINTS_KEY = f"{NAMESPACE}/constraints"
LEDGER_KEY = f"{NAMESPACE}/ledger"


def _constraints_to_wire(constraints: list[Constraint]) -> list[dict]:
    """Serialize Constraints to the wire shape from META_SPEC.md section 1."""
    return [
        {"id": c.id, "description": c.description, "params": c.params, "p": c.p}
        for c in constraints
    ]


def _constraints_from_wire(dicts: list[dict]) -> list[Constraint]:
    return [
        Constraint(
            id=d.get("id", "?"),
            description=d.get("description", ""),
            params=d.get("params") or {},
            p=d.get("p"),
        )
        for d in dicts
    ]


try:
    from fastmcp.server.middleware import Middleware, MiddlewareContext
except ImportError:
    Middleware = object
    MiddlewareContext = Any


class FidelityMiddleware(Middleware):
    """Books an intent ledger for every tool call carrying
    dev.promptfidelity/constraints in its request _meta.

    Args:
        advisory_params: per-tool advisory argument names, e.g.
            {"search": {"query"}, "*": {"q"}}. "*" applies to every tool.

    Requires fastmcp >= 2.9 (middleware) and mcp >= 1.19 (_meta plumbing) --
    raises ImportError on construction, not on module import, if fastmcp
    isn't installed.

    Usage:
        from fastmcp import FastMCP
        from promptfidelity.mcp_ext import FidelityMiddleware

        mcp = FastMCP("my-server")
        mcp.add_middleware(FidelityMiddleware())
    """

    def __init__(self, advisory_params: dict[str, set] | None = None):
        if Middleware is object:
            raise ImportError(
                "promptfidelity.mcp_ext.FidelityMiddleware requires the 'mcp' "
                "extra: pip install promptfidelity[mcp]"
            )
        self.advisory_params = advisory_params or {}

    def _advisory_for(self, tool_name: str) -> set:
        return set(self.advisory_params.get(tool_name, set())) | \
            set(self.advisory_params.get("*", set()))

    @staticmethod
    def _request_meta(context: Any) -> dict:
        """Locate request _meta across FastMCP versions/transports.

        GOTCHA: client meta arrives at
        `context.fastmcp_context.request_context.meta`, as a pydantic model
        with extra fields allowed (`.model_extra`) -- NOT at
        `context.message.meta`, which is where you'd look first and which
        is typically empty even when the client did send `_meta`. The
        fallback below only matters for raw (non-FastMCP) MCP clients that
        populate `params._meta` directly.
        """
        fc = getattr(context, "fastmcp_context", None)
        rc = getattr(fc, "request_context", None)
        meta = getattr(rc, "meta", None)
        if meta is not None:
            extra = getattr(meta, "model_extra", None)
            if extra:
                return dict(extra)
            if isinstance(meta, dict):
                return meta
        # Fallback: raw MCP clients may populate params._meta directly.
        msg_meta = getattr(context.message, "meta", None)
        if isinstance(msg_meta, dict):
            return msg_meta
        extra = getattr(msg_meta, "model_extra", None)
        return dict(extra) if extra else {}

    async def on_call_tool(self, context: Any, call_next):
        payload = self._request_meta(context).get(CONSTRAINTS_KEY)

        result = await call_next(context)

        if not payload or not isinstance(payload, dict):
            return result  # no constraints declared; transparent pass-through

        tool_name = getattr(context.message, "name", "") or ""
        constraints = _constraints_from_wire(payload.get("constraints", []))
        ledger = book(
            constraints,
            getattr(context.message, "arguments", None) or {},
            advisory_params=self._advisory_for(tool_name),
        )
        ledger_dict = ledger.to_dict()
        ledger_dict["booked_by"] = "server"
        if payload.get("prompt_id"):
            ledger_dict["prompt_id"] = payload["prompt_id"]

        # Attach to result _meta without clobbering existing keys.
        existing = getattr(result, "meta", None) or {}
        try:
            result.meta = {**existing, LEDGER_KEY: ledger_dict}
        except (AttributeError, ValueError, TypeError):
            # Result type doesn't expose meta; degrade gracefully.
            pass
        return result


class FidelityClientSession:
    """Wraps an mcp ClientSession to inject dev.promptfidelity/constraints
    into a tool call's request _meta and extract the resulting
    dev.promptfidelity/ledger from the result's _meta.

    If the server doesn't return a ledger -- it has no FidelityMiddleware,
    or an intermediary stripped _meta -- this falls back to booking
    client-side via core.book() against the arguments the client itself
    sent. That fallback is weaker (it can't see any server-side rewriting
    of the arguments between receipt and execution) but still mechanical,
    never LLM self-reported. The returned ledger dict is tagged
    `"booked_by": "client"` vs `"booked_by": "server"` so callers can tell
    which assurance tier they got.
    """

    def __init__(self, session: Any, advisory_params: dict[str, set] | None = None):
        self._session = session
        self.advisory_params = advisory_params or {}

    def _advisory_for(self, tool_name: str) -> set:
        return set(self.advisory_params.get(tool_name, set())) | \
            set(self.advisory_params.get("*", set()))

    async def call_tool_with_intent(
        self,
        name: str,
        arguments: dict[str, Any],
        constraints: list[Constraint],
        prompt_id: str | None = None,
    ) -> tuple[Any, dict]:
        """Call `name` with `arguments`, declaring `constraints` in request
        _meta, and return `(result, ledger_dict)`.

        `ledger_dict` is the dev.promptfidelity/v1 ledger shape -- from the
        server if it published one in the result _meta, otherwise booked
        client-side as a fallback (see class docstring).
        """
        wire_payload = {"v": 1, "constraints": _constraints_to_wire(constraints)}
        if prompt_id:
            wire_payload["prompt_id"] = prompt_id

        result = await self._session.call_tool(
            name, arguments, meta={CONSTRAINTS_KEY: wire_payload}
        )

        ledger_dict = None
        result_meta = getattr(result, "meta", None) or {}
        if isinstance(result_meta, dict):
            ledger_dict = result_meta.get(LEDGER_KEY)

        if ledger_dict is None:
            ledger = book(constraints, arguments, advisory_params=self._advisory_for(name))
            ledger_dict = ledger.to_dict()
            ledger_dict["booked_by"] = "client"
            if prompt_id:
                ledger_dict["prompt_id"] = prompt_id
        else:
            ledger_dict.setdefault("booked_by", "server")

        return result, ledger_dict


class FidelityProxy:
    """Stub: a transparent MCP proxy that books ledgers on behalf of
    un-instrumented upstream servers.

    NOT IMPLEMENTED. This class only documents the design so the intent is
    recorded without faking a partial implementation -- constructing it
    always raises NotImplementedError.

    Intended design: sit in front of an MCP server that has no
    FidelityMiddleware of its own.
      1. On tools/call, read dev.promptfidelity/constraints from the
         incoming request _meta exactly as FidelityMiddleware.on_call_tool
         does.
      2. Forward the call unmodified to the upstream server.
      3. Book the ledger against the *outgoing* call arguments -- the ones
         the proxy actually forwarded, not the ones it received -- so a
         proxy that itself rewrites arguments (e.g. injecting an API key,
         normalizing a param) is honestly diffed against what really
         reached the backend, not against the client's original intent
         restated.
      4. Attach dev.promptfidelity/ledger to the result _meta before
         returning it to the client, tagged `"booked_by": "proxy"` so
         clients can distinguish this assurance tier from a server's own
         first-party booking.

    Also responsible, per META_SPEC.md section 3, for synthesizing a
    dev.promptfidelity/capacity block on tools/list for upstream tools that
    don't publish one, by probing each tool's declared JSON Schema
    (enum/const/pattern-constrained fields => enforced; free-text string
    fields => advisory). That's a heuristic over a schema, not a mechanical
    diff against a real call, so capacity blocks the proxy synthesizes
    should carry a provenance tag distinguishing them from capacity a
    server self-reports (see META_SPEC.md section 4 on provenance/
    anti-laundering -- the same principle that forbids upgrading an
    inferred tag applies to a proxy's own capacity guesses).
    """

    def __init__(self, *args: Any, **kwargs: Any):
        raise NotImplementedError(
            "FidelityProxy is a design stub, not an implementation -- see "
            "the class docstring for the intended transparent-proxy-booking "
            "design."
        )
