"""
promptfidelity.wrap -- drop-in instrumentation for an LLM client object.

wrap(client, ...) returns a delegating proxy that behaves exactly like the
wrapped client for everything except the tool-call-issuing create() method,
which it intercepts to auto-extract constraints from the first user message
and auto-record every tool call the model proposes.

Stdlib only, SDK-shape duck-typed: this module never imports `anthropic` or
`openai`. It detects which client shape it's holding by which attribute
path exists (`client.messages.create` for Anthropic, `client.chat.completions
.create` for OpenAI) and wraps only that path -- everything else on the
client (including attributes neither shape has) passes straight through via
`__getattr__`.

IMPORTANT honesty note (see also anthropic_ext.py / mcp_ext.py's own
provenance notes): tool calls recorded here are the model's *proposed*
calls -- the arguments it emitted in a tool_use / tool_calls block. If your
harness rewrites those arguments before actually executing them (e.g.
clamping a limit, normalizing a param, injecting an API key), the ledger
booked from wrap() diverges from what really happened downstream. Combine
wrap() with @promptfidelity.instrument on your *executed* tool functions
for ground truth on what actually ran; use wrap() alone when you only need
"what did the model ask for."

Streaming is out of scope: only the non-streaming `create()` /
`get_final_message()`-style call/response shapes are supported. A
streaming response's tool calls arrive incrementally across chunks, and
reconstructing them correctly is intentionally left to a future extension
rather than faked here.
"""

import json
from typing import Any, Callable

from .core import Constraint
from .recorder import Recorder, _active_recorder, _merge_constraints


def _first_user_text_anthropic(messages: list) -> str | None:
    for msg in messages or []:
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        if role != "user":
            continue
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                b_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
                if b_type == "text":
                    text = block.get("text") if isinstance(block, dict) else getattr(block, "text", "")
                    parts.append(text or "")
            if parts:
                return "\n".join(parts)
    return None


def _first_user_text_openai(messages: list) -> str | None:
    for msg in messages or []:
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        if role != "user":
            continue
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
        if isinstance(content, str):
            return content
    return None


def _record_tool_calls_anthropic(response: Any, recorders: list[Recorder]) -> None:
    for block in getattr(response, "content", None) or []:
        b_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
        if b_type != "tool_use":
            continue
        name = block.get("name") if isinstance(block, dict) else getattr(block, "name", "")
        arguments = block.get("input") if isinstance(block, dict) else getattr(block, "input", None)
        for rec in recorders:
            rec.record_call(name or "", arguments or {})


def _record_tool_calls_openai(response: Any, recorders: list[Recorder]) -> None:
    choices = getattr(response, "choices", None) or []
    for choice in choices:
        message = choice.get("message") if isinstance(choice, dict) else getattr(choice, "message", None)
        tool_calls = (
            message.get("tool_calls") if isinstance(message, dict)
            else getattr(message, "tool_calls", None)
        ) or []
        for tc in tool_calls:
            function = tc.get("function") if isinstance(tc, dict) else getattr(tc, "function", None)
            name = function.get("name") if isinstance(function, dict) else getattr(function, "name", "")
            raw_args = (
                function.get("arguments") if isinstance(function, dict)
                else getattr(function, "arguments", None)
            )
            try:
                arguments = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            except (TypeError, ValueError):
                arguments = {}
            for rec in recorders:
                rec.record_call(name or "", arguments)


class _InterceptedEndpoint:
    """Proxies one `create`-shaped callable (`.messages.create` or
    `.chat.completions.create`), running constraint extraction + tool-call
    recording around every call."""

    def __init__(self, real_create: Callable, wrapped_client: "WrappedClient",
                 first_user_text_fn: Callable, record_fn: Callable):
        self._real_create = real_create
        self._wrapped_client = wrapped_client
        self._first_user_text_fn = first_user_text_fn
        self._record_fn = record_fn

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        wc = self._wrapped_client
        if not wc._constraints_ready:
            text = (
                wc._prompt if wc._prompt is not None
                else self._first_user_text_fn(kwargs.get("messages") or [])
            )
            wc._ensure_constraints(text)

        response = self._real_create(*args, **kwargs)

        recorders = [wc.fidelity]
        active = _active_recorder.get()
        if active is not None and active is not wc.fidelity:
            recorders.append(active)
        self._record_fn(response, recorders)

        return response


class _NestedNamespaceProxy:
    """Delegates attribute access to a wrapped namespace object (e.g.
    `client.messages` or `client.chat`), except for one attribute name that
    is replaced with an intercepted callable (e.g. `.create`)."""

    def __init__(self, real_namespace: Any, intercept_name: str, intercepted: Callable):
        self._real_namespace = real_namespace
        self._intercept_name = intercept_name
        self._intercepted = intercepted

    def __getattr__(self, name: str) -> Any:
        if name == self._intercept_name:
            return self._intercepted
        return getattr(self._real_namespace, name)


class WrappedClient:
    """Delegating proxy around an Anthropic- or OpenAI-shaped client.

    Not constructed directly -- use wrap(). See module docstring for the
    interception and honesty notes.
    """

    def __init__(self, client: Any, constraints: list[Constraint] | None,
                 prompt: str | None, extractor: Callable | None,
                 vocab: dict | None, advisory_params: set | None,
                 ignore_params: set[str] | None = None):
        self._client = client
        self._prompt = prompt
        self._extractor = extractor
        self._vocab = vocab
        self.fidelity = Recorder(
            constraints or [], advisory_params=advisory_params, ignore_params=ignore_params
        )
        self._constraints_ready = constraints is not None

        messages_ns = getattr(client, "messages", None)
        chat_ns = getattr(client, "chat", None)
        completions_ns = getattr(chat_ns, "completions", None) if chat_ns is not None else None

        self._messages_proxy = None
        self._chat_proxy = None

        if messages_ns is not None and hasattr(messages_ns, "create"):
            intercepted = _InterceptedEndpoint(
                messages_ns.create, self, _first_user_text_anthropic, _record_tool_calls_anthropic
            )
            self._messages_proxy = _NestedNamespaceProxy(messages_ns, "create", intercepted)

        if completions_ns is not None and hasattr(completions_ns, "create"):
            intercepted = _InterceptedEndpoint(
                completions_ns.create, self, _first_user_text_openai, _record_tool_calls_openai
            )
            completions_proxy = _NestedNamespaceProxy(completions_ns, "create", intercepted)
            self._chat_proxy = _NestedNamespaceProxy(chat_ns, "completions", completions_proxy)

    def _ensure_constraints(self, text: str | None) -> None:
        if self._constraints_ready:
            return
        if text is not None:
            self.fidelity.constraints = _merge_constraints(text, self._extractor, self._vocab)
        self._constraints_ready = True

    def fidelity_reset(self) -> None:
        """Clear recorded calls and the cached auto-extracted constraints,
        so the next intercepted create() call re-extracts from whatever
        first user message it sees then. Declared (caller-supplied)
        constraints are also cleared -- construct a fresh wrap() if you
        want to keep those across a reset."""
        self.fidelity.calls = []
        self.fidelity.constraints = []
        self._constraints_ready = False

    def __getattr__(self, name: str) -> Any:
        if name == "messages" and self._messages_proxy is not None:
            return self._messages_proxy
        if name == "chat" and self._chat_proxy is not None:
            return self._chat_proxy
        return getattr(self._client, name)


def wrap(
    client: Any,
    *,
    constraints: list[Constraint] | None = None,
    prompt: str | None = None,
    extractor: Callable[[str], list[Constraint]] | None = None,
    vocab: dict | None = None,
    advisory_params: set | None = None,
    ignore_params: set[str] | None = None,
) -> WrappedClient:
    """Wrap an Anthropic- or OpenAI-shaped client for drop-in fidelity
    instrumentation.

        import promptfidelity as pf
        client = pf.wrap(anthropic.Anthropic())
        response = client.messages.create(..., tools=[...])
        client.fidelity.ledger().fidelity

    Constraints can be supplied up front via `constraints=`, or left to be
    extracted (via extraction.extract_constraints, optionally merged with
    an `extractor` -- see _merge_constraints for the merge rule) on the
    first intercepted create() call. The text to extract from is either:
      - `prompt=`, if given -- pins extraction to that literal string,
        bypassing message inspection entirely (useful when the actual
        `messages` kwarg doesn't literally contain the phrasing you want
        extraction to see, e.g. a templated/rewritten prompt); or
      - otherwise, the first `role == "user"` message's text found in that
        call's `messages` kwarg (Anthropic: str or a list of
        `{"type": "text", ...}` blocks; OpenAI: str).
    Either way, extraction runs once per wrapper (or once again after
    fidelity_reset()); the same client handles a whole conversation without
    re-extracting per turn.

    `wrapped.fidelity` is the underlying Recorder: `.ledger()`, `.calls`,
    `.constraints` are all reachable from it exactly as with trace()'s
    yielded Recorder. If an active pf.trace() context is open when a call
    happens, that trace's Recorder also gets fed the same tool calls, so
    wrap() and trace() compose (e.g. a wrap()'d client used inside an
    outer trace() that also records other, non-client tool calls).

    Every other attribute -- everything not `.messages.create` or
    `.chat.completions.create` -- passes straight through to the real
    client untouched.

    `ignore_params` is passed straight through to the internal Recorder
    (`client.fidelity`) -- see Recorder.record_call for what it does and
    why (keeping plumbing args like pagination/auth/sort defaults out of
    the `imposed` account).
    """
    return WrappedClient(
        client, constraints, prompt, extractor, vocab, advisory_params, ignore_params
    )
