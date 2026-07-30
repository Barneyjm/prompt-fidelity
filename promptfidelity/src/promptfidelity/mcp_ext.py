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

infer_advisory_params() adds a third way to get advisory param names into
that booking, alongside hand-configured per-tool dicts: a structural
heuristic over a tool's JSON Schema, mirroring the registry-census project's
Tier-B method for spotting a free-text search field without asking a model
-- a string param named/described like a search query, with no
enum/pattern/format pinning it down, is assumed advisory (the backend can't
mechanically enforce compliance with free text) unless a human overrides it.
It's a heuristic over a schema, same as core.book()'s advisory_params is
just a set of names -- never a judgment call about a specific call's
arguments, and never anything that touches account assignment itself.
"""

import re
from typing import Any

from .core import Constraint, Ledger, book
from .recorder import _active_recorder

NAMESPACE = "dev.promptfidelity"
CONSTRAINTS_KEY = f"{NAMESPACE}/constraints"
LEDGER_KEY = f"{NAMESPACE}/ledger"

_ADVISORY_NAME_RE = re.compile(
    r"(?:^|_)(q|query|search|keywords?|prompt|text|question)(?:$|_)", re.IGNORECASE
)
_ADVISORY_DESC_MARKERS = ("search", "query", "free-text", "free text", "natural language")


def infer_advisory_params(input_schema: dict) -> set[str]:
    """Heuristic: which string properties of a JSON Schema look like
    free-text fields a backend can't mechanically enforce.

    A property is advisory if all of:
      - `type == "string"`
      - its name matches r"(?:^|_)(q|query|search|keywords?|prompt|text|
        question)(?:$|_)" case-insensitively, OR its `description` contains
        "search", "query", "free-text"/"free text", or "natural language"
      - it has none of `enum`, `pattern`, `format` -- any of those means
        the backend *can* mechanically validate the value, so it's not
        purely advisory.

    This mirrors the registry-census project's Tier-B heuristic: a
    structural read of a schema, never a judgment about a specific call's
    arguments and never a substitute for a human/registry override. Takes
    a bare JSON Schema `properties`-bearing dict (e.g. a tool's
    `inputSchema`/`parameters`); returns the set of advisory property
    names. Missing/malformed schemas degrade to an empty set rather than
    raising.
    """
    properties = (input_schema or {}).get("properties") or {}
    advisory: set[str] = set()
    if not isinstance(properties, dict):
        return advisory
    for name, prop_schema in properties.items():
        if not isinstance(prop_schema, dict):
            continue
        if prop_schema.get("type") != "string":
            continue
        if any(k in prop_schema for k in ("enum", "pattern", "format")):
            continue
        description = str(prop_schema.get("description") or "").lower()
        name_matches = bool(_ADVISORY_NAME_RE.search(name))
        desc_matches = any(marker in description for marker in _ADVISORY_DESC_MARKERS)
        if name_matches or desc_matches:
            advisory.add(name)
    return advisory


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
        advisory_params: either
            - per-tool advisory argument names, e.g.
              {"search": {"query"}, "*": {"q"}} ("*" applies to every
              tool) -- explicit config, unchanged from before; or
            - the literal string "auto": resolve advisory params per call
              by running infer_advisory_params() over the tool's JSON
              Schema, obtained from the FastMCP context (via
              `context.fastmcp_context.fastmcp.get_tool(name)`) when
              possible. Falls back to no advisory params for a given call
              if the schema can't be obtained that way (e.g. a bare/raw
              MCP session with no FastMCP-side tool registry reachable
              from the context) -- it never raises for that reason.

    Requires fastmcp >= 2.9 (middleware) and mcp >= 1.19 (_meta plumbing) --
    raises ImportError on construction, not on module import, if fastmcp
    isn't installed.

    Usage:
        from fastmcp import FastMCP
        from promptfidelity.mcp_ext import FidelityMiddleware

        mcp = FastMCP("my-server")
        mcp.add_middleware(FidelityMiddleware())
        # or, to infer advisory params from each tool's schema instead of
        # hand-listing them:
        mcp.add_middleware(FidelityMiddleware(advisory_params="auto"))
    """

    def __init__(self, advisory_params: dict[str, set] | str | None = None):
        if Middleware is object:
            raise ImportError(
                "promptfidelity.mcp_ext.FidelityMiddleware requires the 'mcp' "
                "extra: pip install promptfidelity[mcp]"
            )
        self.advisory_params = advisory_params if advisory_params is not None else {}

    def _advisory_for(self, tool_name: str) -> set:
        """Explicit-config lookup only -- unchanged behavior from before
        advisory_params="auto" existed. Callers that pass "auto" go
        through _resolve_advisory (async) instead, since schema lookup
        needs an `await`; this method assumes self.advisory_params is a
        plain per-tool dict."""
        return set(self.advisory_params.get(tool_name, set())) | \
            set(self.advisory_params.get("*", set()))

    @staticmethod
    async def _schema_for(context: Any, tool_name: str) -> dict | None:
        """Best-effort: the tool's JSON Schema input definition, read from
        the FastMCP server reachable off this middleware context. Returns
        None (never raises) if fastmcp isn't wired up this way for this
        transport/context, or the tool can't be found."""
        try:
            server = getattr(getattr(context, "fastmcp_context", None), "fastmcp", None)
            if server is None:
                return None
            tool = await server.get_tool(tool_name)
            schema = getattr(tool, "parameters", None)
            return schema if isinstance(schema, dict) else None
        except Exception:
            return None

    async def _resolve_advisory(self, tool_name: str, context: Any) -> set:
        """Advisory param names for one call: "auto" mode infers them from
        the tool's schema (empty set if unobtainable); otherwise this is
        exactly the explicit-config lookup, unchanged."""
        if self.advisory_params == "auto":
            schema = await self._schema_for(context, tool_name)
            return infer_advisory_params(schema) if schema else set()
        return self._advisory_for(tool_name)

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
            advisory_params=await self._resolve_advisory(tool_name, context),
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
        self._tools_cache: dict[str, dict] | None = None

    def _advisory_for(self, tool_name: str) -> set:
        return set(self.advisory_params.get(tool_name, set())) | \
            set(self.advisory_params.get("*", set()))

    async def _cached_input_schema(self, name: str) -> dict | None:
        """Best-effort tools/list cache, built once per session wrapper
        (or after a fresh instance is constructed). Degrades to "no
        schema" -- never raises -- if the underlying session has no
        list_tools(), or listing fails for any reason."""
        if self._tools_cache is None:
            self._tools_cache = {}
            list_tools = getattr(self._session, "list_tools", None)
            if list_tools is not None:
                try:
                    result = await list_tools()
                    tools = getattr(result, "tools", None)
                    if tools is None and isinstance(result, (list, tuple)):
                        tools = result
                    for tool in tools or []:
                        t_name = (
                            tool.get("name") if isinstance(tool, dict)
                            else getattr(tool, "name", None)
                        )
                        schema = (
                            tool.get("inputSchema") if isinstance(tool, dict)
                            else getattr(tool, "inputSchema", None)
                        )
                        if t_name is not None and isinstance(schema, dict):
                            self._tools_cache[t_name] = schema
                except Exception:
                    self._tools_cache = {}
        return self._tools_cache.get(name)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Auto mode: call `name` with `arguments` exactly as the
        underlying session would (no dev.promptfidelity/constraints
        injected in request _meta -- this is NOT call_tool_with_intent),
        and, if a pf.trace() context is currently open, record the call
        into its Recorder so it participates in that trace's ledger.

        Advisory params for the recorded call come from
        infer_advisory_params() over the tool's cached input schema
        (tools/list is fetched and cached once per FidelityClientSession,
        on first use) when the underlying session exposes list_tools().
        Falls back to this session's explicit, constructor-configured
        advisory_params (or none at all) when list_tools isn't available,
        or the tool isn't found in it.

        NOTE: recorder.Recorder keeps one flat advisory_params set for
        the whole trace -- no per-tool-name dict there, unlike
        core.book()'s advisory_params or FidelityMiddleware. This unions
        each tool's inferred advisory names into the active trace's
        Recorder.advisory_params as it discovers them, so an advisory
        name inferred for one tool applies trace-wide from then on.
        """
        result = await self._session.call_tool(name, arguments)

        rec = _active_recorder.get()
        if rec is not None:
            schema = await self._cached_input_schema(name)
            inferred = infer_advisory_params(schema) if schema else self._advisory_for(name)
            if inferred:
                rec.advisory_params = (rec.advisory_params or set()) | inferred
            rec.record_call(name, arguments)
        return result

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
