"""
promptfidelity.langchain_ext -- book fidelity ledgers from LangChain tool calls.

Extra: [langchain]. Requires `langchain-core` (home of BaseCallbackHandler).
Importing this module never fails when langchain isn't installed -- the
import below is wrapped so FidelityCallbackHandler falls back to a plain
`object` base at import time; only *instantiating* the handler raises, with
a clear message naming the extra to install.
"""

from typing import Any
from uuid import UUID

from .core import Constraint, Ledger, book, merge_ledgers

try:
    from langchain_core.callbacks import BaseCallbackHandler as _Base
except ImportError:
    _Base = object


class FidelityCallbackHandler(_Base):
    """LangChain callback handler that books an intent ledger from every
    tool invocation it observes during a run.

    Usage:
        handler = FidelityCallbackHandler(constraints)
        agent_executor.invoke({"input": ...}, config={"callbacks": [handler]})
        ledger = handler.ledger()

    Raises ImportError on construction if langchain-core isn't installed
    (pip install promptfidelity[langchain]) -- importing this module alone
    never fails, so callers can e.g. `from promptfidelity import langchain_ext`
    for introspection without the dependency present.
    """

    def __init__(self, constraints: list[Constraint], advisory_params: dict[str, set] | None = None):
        if _Base is object:
            raise ImportError(
                "promptfidelity.langchain_ext.FidelityCallbackHandler requires "
                "the 'langchain' extra: pip install promptfidelity[langchain]"
            )
        super().__init__()
        self.constraints = list(constraints)
        self.advisory_params = advisory_params
        self.calls: list[dict[str, Any]] = []

    def _advisory_for(self, tool_name: str) -> set:
        advisory = self.advisory_params or {}
        return set(advisory.get(tool_name, set())) | set(advisory.get("*", set()))

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Record a tool call's structured inputs.

        LangChain >= 0.2 passes the tool's structured `inputs` dict; older
        versions only give the serialized `input_str`, recorded here under
        a single synthetic `input` key as a fallback so this handler still
        produces a (weaker) ledger rather than raising.
        """
        name = (serialized or {}).get("name", "") or ""
        arguments = dict(inputs) if inputs is not None else {"input": input_str}
        self.calls.append({"name": name, "arguments": arguments})

    def ledger(self) -> Ledger:
        """Book the merged ledger across every tool call observed so far
        (best booking per constraint across calls -- see
        core.merge_ledgers). Safe to call mid-run; it only reflects calls
        observed up to that point."""
        if not self.calls:
            return book(self.constraints, {}, self.advisory_params)
        per_call = [
            book(self.constraints, call["arguments"], self._advisory_for(call["name"]))
            for call in self.calls
        ]
        return merge_ledgers(self.constraints, per_call)
