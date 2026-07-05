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

## Drop-in usage

The fastest way in: wrap your LLM client and let `promptfidelity` pull
constraints straight out of the first user message.

```python
import anthropic
import promptfidelity as pf

client = pf.wrap(anthropic.Anthropic())

response = client.messages.create(
    model="claude-...", tools=[...],
    messages=[{"role": "user", "content": "a slow sci-fi movie from the 90s, rated above 7"}],
)

client.fidelity.ledger().fidelity   # how much of that intent actually landed
```

`pf.wrap()` works against both Anthropic-shaped (`.messages.create`) and
OpenAI-shaped (`.chat.completions.create`) clients, detected structurally --
it's a delegating proxy, so everything else about the client passes through
untouched. It extracts constraints once (from the first user message it
sees), then records every tool call the model proposes into
`client.fidelity` (a `Recorder` -- same `.ledger()`/`.calls`/`.constraints`
as `trace()` below).

Prefer to keep using `trace()` directly? Give it a prompt instead of a
hand-built constraint list and it runs the same extraction:

```python
with pf.trace(prompt="a slow sci-fi movie from the 90s, rated above 7") as rec:
    run_agent(...)
ledger = rec.ledger()
```

### Tiered extraction and provenance

Constraints can come from three places, and every one of them is tagged so
a reader can tell which:

| `source` | Where it came from | Trust |
|---|---|---|
| `declared` | a human/caller wrote the Constraint directly | as good as the human |
| `rules` | `promptfidelity.extraction.extract_constraints` -- deterministic regex/vocab transcription (Tier 1), no LLM, no scoring | mechanical, auditable |
| `llm` | a pluggable `extractor` callable, typically model-backed (Tier 2) | probabilistic transcription |

`declared` > `rules` > `llm` isn't a ranking of *virtue* -- it's a ranking
of *how mechanically checkable the transcription step was*. Extraction is
always transcription (prompt text -> falsifiable params), never judgment:
nothing in this layer scores or grades anything, and no extractor --
rules or LLM -- ever assigns an *account*; that's still exclusively
`book()`'s job (see the hard rule below). `Ledger.to_dict()["summary"]
["constraints_source"]` totals up entries per tag, e.g.
`{"declared": 2, "rules": 3, "llm": 1}`, so the receipt discloses how much
of a fidelity number rests on LLM transcription versus deterministic rules
versus a human stating intent directly.

**Proposed vs. executed.** Both `wrap()` and the MCP client's auto
`call_tool()` (below) record what the model *proposed* to call --
arguments straight out of a `tool_use`/`tool_calls` block. If your harness
rewrites those arguments before actually executing them (clamping a limit,
normalizing a param, injecting a key), the ledger diverges from what really
ran downstream. Combine with `@pf.instrument` on your *executed* tool
functions for ground truth on what actually happened.

## Close the loop: self-inspection and repair

`book()`/`merge_ledgers()` are pure and cheap, so an agent loop doesn't have
to wait until the run is over to check the books -- it can check them
**before it speaks**, after every tool call, and let the model repair
itself while the trace is still open:

```python
MAX_REPAIRS = 3

with pf.trace(constraints) as rec:
    discover_movies(with_genres="878")           # the model's first attempt

    for _ in range(MAX_REPAIRS):
        ledger = rec.check()
        if not ledger.unhonored:
            break
        injection = ledger.render("model")       # inject this into the model's context
        # ... let the model issue ONE corrective tool call here ...
    else:
        pass  # exhausted MAX_REPAIRS with entries still unhonored -- disclose, don't spin

ledger = rec.ledger()
```

`rec.check()` is a semantic alias for `rec.ledger()` -- an interim booking
taken mid-trace instead of only at the end. `Ledger.unhonored` is the
subset of entries (`substituted` and `dropped`) a corrective tool call can
actually fix; `dropped`/`substituted` params are named on the entry itself
(`LedgerEntry.params`), so `render("model")` can say exactly what to send.
**Always bound your repair iterations** -- some constraints are
structurally unhonorable against a given tool (the backend just has no
such param), and an unbounded loop that only exits on an empty
`unhonored` list will spin forever against one of those.

**The conjunction caveat.** `Ledger.conjunction_honored` answers a
different question than `unhonored` does. `merge_ledgers()` (what
`rec.ledger()`/`rec.check()` run under the hood) credits each constraint
its *best ever* booking across every call in the trace -- so "everything
verified" can mean everything was verified, just never all by the *same*
call. A repair loop that patches only the missing param in a small
follow-up call can turn a `dropped` entry into `verified` while leaving
`conjunction_honored` `False`: nothing ever received one request carrying
every honored argument together, so no single result set reflects the
whole intent. The honest repair isn't "add the missing param to a minimal
follow-up call" -- it's "re-issue ONE complete call carrying every
previously-honored param plus the missing one," which is exactly what
`render("model")`'s fixed instruction line tells the model to do.

**`report.render(ledger, audience)`** (also `ledger.render(audience)`) is
one instrument at four altitudes -- deterministic string templates over
ledger data, no LLM involved:

| Audience | What it's for |
|---|---|
| `model` | Short and actionable -- meant to be injected into an agent's own context mid-run. Empty string when there's nothing to repair. |
| `engineer` | Full detail: every entry's account, bits, source, call index, and evidence; imposed args; fidelity + basis; the conjunction flag. |
| `product` | Plain language, no jargon, no bits -- what got delivered as asked, delivered approximately, handled by model judgment, or not delivered at all. |
| `executive` | At most ~5 lines: the fidelity headline (with its basis caveat spelled out when it matters), how much was altered/dropped without disclosure, how many filters the agent added unasked, one closing risk line. |

`ignore_params` (on `Recorder`, `trace()`, and `wrap()`) keeps plumbing
arguments -- pagination, auth, sort defaults -- out of the `imposed`
account entirely, so they don't turn the repair signal into noise:

```python
with pf.trace(constraints, ignore_params={"page", "api_key", "sort_by"}) as rec:
    ...
```

See `promptfidelity/examples/repair_loop.py` for a runnable, self-contained
walkthrough (no network, no keys) of the whole loop end to end.

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
# or let it infer advisory params from each tool's JSON Schema instead:
mcp.add_middleware(FidelityMiddleware(advisory_params="auto"))
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

Or skip declaring intent per call and just call tools normally — plain
`call_tool()` auto-records into whatever `pf.trace()` is open (advisory
params inferred from each tool's cached schema via `infer_advisory_params`):

```python
with pf.trace(constraints) as rec:
    await client.call_tool("discover_movies", {"with_genres": "878"})
ledger = rec.ledger()
```

`FidelityProxy` is a documented design stub for a transparent MCP proxy that
books ledgers on behalf of un-instrumented upstream servers — see its
docstring in `mcp_ext.py`; it is intentionally not implemented.

## Design notes

- Core (`core.py`), the extractor (`extraction.py`), the recorder
  (`recorder.py`), the reporting layer (`report.py`), and the client
  wrapper (`wrap.py`) are **stdlib only** — no dependency, optional or
  otherwise, is required to book a ledger, run Tier-1 extraction, render a
  report, or wrap an Anthropic-/OpenAI-shaped client. Reporting has no LLM
  anywhere in it: `render()` is a deterministic string template over
  already-booked `Ledger` data.
- Each extra imports its optional dependency lazily, and only inside the
  function/class that needs it, so `import promptfidelity` never requires
  `anthropic`, `langchain-core`, `mcp`, or `fastmcp` to be installed.
- `merge_ledgers()` (used by the recorder, the Anthropic extra, and the
  LangChain extra) merges several per-call ledgers booked against the same
  constraints into one, crediting each constraint its *best* booking across
  calls (`verified > transmitted > substituted > dropped`) — a later,
  corrective tool call shouldn't leave an earlier miss on the books.
- Every `Constraint` carries a `source` (`"declared"` | `"rules"` | `"llm"`),
  which flows through to each `LedgerEntry` and into
  `Ledger.to_dict()["summary"]["constraints_source"]` — a per-source entry
  count disclosing how much of a given fidelity number rests on LLM
  transcription vs. deterministic rules vs. a human declaring intent
  directly. `source` is provenance metadata only; it never influences
  `book()`'s account assignment (see the hard rule).

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
