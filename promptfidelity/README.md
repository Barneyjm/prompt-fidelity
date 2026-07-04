# promptfidelity

Measuring how much of a prompt's intent an agent actually executes.

`promptfidelity` decomposes a prompt into **constraints** (small, checkable
pieces of intent: "science fiction genre", "slow, meditative pacing",
"under $50"), then **books** each one, mechanically, against the arguments a
tool call was *actually* made with. No LLM self-report ever assigns the
account — `book()` is a pure diff.

Every constraint lands in exactly one of five user-side accounts:

| Account | Meaning |
|---|---|
| `verified` | every declared param present, matching value, via an enforced field |
| `transmitted` | present and matching, but via an advisory field the backend doesn't enforce (e.g. a free-text search query) |
| `substituted` | a value was altered en route, or only part of a multi-param constraint made it through |
| `inferred` | the constraint declared no params — not expressible against this tool |
| `dropped` | params were declared but none of them appear in the call |

Plus one agent-side account, `imposed`: a call argument that matches no
declared constraint at all — a filter the agent added on its own (a
popularity floor, a truncation, a score cutoff) that no one asked for.

Conservation holds by construction: `I_total = verified + transmitted +
substituted + inferred + dropped` (imposed is excluded — it's not user
intent). `fidelity = verified_bits / I_total`.

### The hard rule

**LLMs may propose ledger *entries* (constraints — a description, the params
they expect to produce, an estimated survival probability), never
*accounts*.** No function in this package accepts a pre-assigned account
from model output. `book()` is the only place an account is ever set, and it
sets one by diffing declared params against the real arguments a tool call
received — nothing else.

## Install

```bash
pip install promptfidelity            # core: stdlib only, zero dependencies
pip install promptfidelity[anthropic] # + book ledgers from Claude tool_use blocks
pip install promptfidelity[langchain] # + a LangChain callback handler
pip install promptfidelity[mcp]       # + MCP/FastMCP middleware and client session
```

## Quickstart: core

```python
from promptfidelity import Constraint, book

constraints = [
    Constraint(id="c1", description="Science fiction genre",
               params={"with_genres": "878"}, p=0.08),
    Constraint(id="c2", description="Slow, meditative pacing", p=0.10),
]

# The params a tool call was ACTUALLY made with:
ledger = book(constraints, {"with_genres": "878"})

print(ledger.fidelity)     # 1.0 -- the only expressible constraint verified
print(ledger.to_dict())    # dev.promptfidelity/v1 ledger shape
```

## Quickstart: recording a run

Booking against one tool call is easy; a real agent turn usually spans
several. `promptfidelity.recorder` uses `contextvars` so you can instrument
tool functions once and have every call inside a `trace()` block recorded,
regardless of how deep the call stack is:

```python
import promptfidelity as pf

@pf.instrument
def discover_movies(**kwargs):
    return tmdb.discover_movies(**kwargs)

with pf.trace(constraints) as rec:
    discover_movies(with_genres="878")

ledger = rec.ledger()
```

Outside an active `trace()`, `@instrument` is a transparent no-op — safe to
leave on tool functions that also run in code paths that never open a
trace.

## Quickstart: `[anthropic]`

```python
from promptfidelity.anthropic_ext import record

response = client.messages.create(..., tools=[...])
result = record(response, constraints)

result.merged.fidelity        # merged across every tool_use block in the turn
result.calls[0].ledger        # per-call ledger, keyed by tool_use_id
```

`record()` works against any duck-typed object shaped like an
`anthropic.types.Message` (a `.content` list of blocks with `.type` and, for
`tool_use` blocks, `.id` / `.name` / `.input`) — the `anthropic` package
itself is never imported, so this extra is unit-testable with no network
call and no SDK install.

## Quickstart: `[langchain]`

```python
from promptfidelity.langchain_ext import FidelityCallbackHandler

handler = FidelityCallbackHandler(constraints)
agent_executor.invoke({"input": ...}, config={"callbacks": [handler]})
ledger = handler.ledger()
```

## Quickstart: `[mcp]`

Server side — attach a ledger to every tool call that declares constraints:

```python
from fastmcp import FastMCP
from promptfidelity.mcp_ext import FidelityMiddleware

mcp = FastMCP("my-server")
mcp.add_middleware(FidelityMiddleware(advisory_params={"search": {"query"}}))
```

Client side — declare constraints on the call and read back whatever ledger
comes back (falling back to client-side booking if the server has no
middleware):

```python
from promptfidelity.mcp_ext import FidelityClientSession

client = FidelityClientSession(session)
result, ledger = await client.call_tool_with_intent(
    "discover_movies", {"with_genres": "878"}, constraints, prompt_id="p1",
)
ledger["booked_by"]  # "server" or "client"
```

`FidelityProxy` is a documented design stub for a transparent MCP proxy that
books ledgers on behalf of un-instrumented upstream servers — see its
docstring in `mcp_ext.py`; it is intentionally not implemented.

## Design notes

- Core (`core.py`) and the recorder (`recorder.py`) are **stdlib only** — no
  dependency, optional or otherwise, is required to book a ledger.
- Each extra imports its optional dependency lazily, and only inside the
  function/class that needs it, so `import promptfidelity` never requires
  `anthropic`, `langchain-core`, `mcp`, or `fastmcp` to be installed.
- `merge_ledgers()` (used by the recorder, the Anthropic extra, and the
  LangChain extra) merges several per-call ledgers booked against the same
  constraints into one, crediting each constraint its *best* booking across
  calls (`verified > transmitted > substituted > dropped`) — a later,
  corrective tool call shouldn't leave an earlier miss on the books.

See `../schema` and the `dev.promptfidelity/v1` interchange format spec for
the wire shapes this package implements.

## Tests

```bash
python -m pytest promptfidelity/tests/ -q
```

Tests for the `anthropic` extra use fake, duck-typed response objects — no
API key or network access required. Tests for the `mcp` extra that need a
real `fastmcp.server.middleware.Middleware` to subclass are skipped (not
failed) via `pytest.importorskip` when `fastmcp` isn't installed; the
client-side tests need nothing beyond stdlib and always run.
