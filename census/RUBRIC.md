# Registry Census — Classification Rubric (Tier A)

Unit: one registry entry (latest version only, dedupe by name).
Input: name + description (+ publisher metadata when present).
Output per server: dominant species, secondary species (optional), confidence (H/M/L).

## Species definitions (assign DOMINANT function)

- **predicate** — queries/filters structured data with enforced parameters;
  results provably satisfy the constraint. Signals: database, SQL, filter,
  lookup by ID, catalog/discover APIs, geocoding a specific address.
- **relevance** — free-text search/ranking; core param is advisory.
  Signals: "search", semantic search, web search, vector/RAG retrieval,
  recommendations, "find relevant".
- **generative** — result is model/agent output. Signals: "AI-powered",
  summarize, deep research, generate (image/video/music/text), translate,
  sentiment analysis, "answers questions about".
- **actuation** — changes external state. Signals: create/update/delete/send/
  post/deploy/pay/book/manage/control; device or account control.

## Tie-breaks

- Mixed servers: dominant = the species of the tools a user would name the
  server for; record secondary. (e.g. Shopify admin = actuation, secondary
  predicate.)
- Computational tools (calculators, code execution, converters) = predicate
  (deterministic, verifiable by re-execution). Note them for the footnote.
- Pure "connector to X" descriptions with no verbs: classify from what X is;
  confidence L.
- Unclassifiable/spam/empty description: bucket "unknown", exclude from
  distribution, report the exclusion rate.

## Sonnet delegation prompt (per batch of ~25)

System: return ONLY JSON array [{"name":..., "species":..., "secondary":...,
"confidence":...}] using the rubric above (embed rubric verbatim).
Temperature 0. Spot-check 10% by a second pass; report inter-run agreement.

## Deliverables

1. Distribution table (n, % per species + unknown/excluded).
2. Transmitted-heavy hypothesis verdict: relevance + generative combined share.
3. Actuation share, and of those, % whose descriptions mention any read-back tool.
4. 5 quotable exemplars per species.
5. CSV: name, description, species, secondary, confidence.
6. Methodology note: registry skews to recently-published remote servers —
   frame as "what's being built."
