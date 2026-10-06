# Fieldnote — Deep Research Agent

A working, local-first deep-research prototype for an ML/Research Engineering portfolio. Fieldnote takes a complex question through explicit planning, retrieval, reading, evidence extraction, gap analysis, and cited report-writing stages. Its intermediate state remains visible in both the API and UI.

This is intentionally a prototype: it favors legible architecture and useful extension points over a large agent framework or production infrastructure.

## Architecture

```text
┌───────────────┐       POST / GET polling       ┌────────────────────┐
│ React + Vite  │ ◄────────────────────────────► │ FastAPI API        │
│ research UI   │                                │ background task    │
└───────────────┘                                └─────────┬──────────┘
                                                          │
                              ┌───────────────────────────▼──────────────────────────┐
                              │ ResearchOrchestrator + central ResearchState         │
                              └──────┬──────────┬──────────┬──────────┬──────────────┘
                                     │          │          │          │
                                ┌────▼───┐ ┌────▼───┐ ┌────▼────┐ ┌──▼───────────┐
                                │Planner │ │ Search │ │ Fetcher │ │ Evidence /   │
                                │Pydantic│ │Provider│ │HTML→text│ │ Gaps / Writer│
                                └────────┘ └────┬───┘ └─────────┘ └──────┬───────┘
                                               │                         │
                                      Demo or Tavily          Demo or OpenAI API
```

No LangChain is used. The orchestration loop is visible in `backend/app/services/orchestrator.py`; all major stages share the `LLMClient` interface, while retrieval uses the replaceable `SearchProvider` interface. In-memory storage implements `ResearchRepository`, which can later be backed by PostgreSQL or Redis.

## Workflow

1. Clarify the objective and produce 3–6 structured subquestions and queries.
2. Search queries concurrently through the configured provider.
3. Fetch each unique page and extract readable text using Trafilatura, with BeautifulSoup fallback.
4. Extract source-grounded evidence for each subquestion.
5. Assess coverage and create targeted follow-up queries when gaps remain.
6. Run no more than two research rounds.
7. Synthesize a Markdown report and validate that numbered citations map to collected sources.

The API exposes the complete `ResearchState`, including search results and evidence, to support future retrieval and faithfulness evaluation.

## Project structure

```text
backend/app/
  agents/       planner, evidence extractor, gap analyzer, report writer
  models/       typed research state and stage outputs
  services/     LLM abstraction and explicit orchestrator
  storage/      repository interface and in-memory implementation
  tools/        search provider and resilient page fetcher
  main.py       FastAPI application and dependency wiring
backend/tests/  schema, interface, citation, and orchestration tests
frontend/src/   React UI, typed API client, progress and artifact components
```

## Quick start

Requirements: Python 3.11+ and Node.js 20+.

### 1. Backend

```bash
cd deep-research-agent/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
uvicorn app.main:app --reload --port 8000
```

### 2. Frontend

In a second terminal:

```bash
cd deep-research-agent/frontend
npm install
npm run dev -- --port 5173
```

Open:

- UI: http://localhost:5173
- API health: http://localhost:8000/api/health
- FastAPI docs: http://localhost:8000/docs

The frontend dev server proxies `/api` to port 8000.

## Configuration

Copy `.env.example` to `.env`. The app starts without it in demo mode.

| Variable | Purpose | Default |
|---|---|---|
| `OPENAI_API_KEY` | Enables model-backed planning, extraction, gap analysis, and writing | unset (demo LLM) |
| `LLM_MODEL` | One model name shared across all stages | `gpt-4.1-mini` |
| `SEARCH_PROVIDER` | `demo` or `tavily` | `demo` |
| `TAVILY_API_KEY` | Enables live Tavily web search when provider is `tavily` | unset |
| `BACKEND_PORT` | Documented backend port | `8000` |
| `FRONTEND_PORT` | CORS and frontend port | `5173` |
| `MAX_RESEARCH_ROUNDS` | Hard research-loop limit | `2` |
| `MAX_RESULTS_PER_QUERY` | Search breadth per query | `3` |
| `MAX_SOURCES` | Fetch/cost guardrail | `12` |

`SERPER_API_KEY` and `BRAVE_API_KEY` are reserved configuration fields for future provider implementations. Never put real keys in `.env.example` or commit `.env`.

For live research:

```env
OPENAI_API_KEY=your-key
LLM_MODEL=gpt-4.1-mini
SEARCH_PROVIDER=tavily
TAVILY_API_KEY=your-key
```

If only an OpenAI key is present, model stages are live while search remains demo-backed. If Tavily is selected without its key, search safely falls back to demo mode and the API/UI reports that state.

## API

```bash
curl -X POST http://localhost:8000/api/research \
  -H 'Content-Type: application/json' \
  -d '{"question":"When does hybrid retrieval outperform dense retrieval in RAG systems?"}'

curl http://localhost:8000/api/research/<task_id>
```

Tasks execute in-process and the UI polls for updated state. Individual search, fetch, and extraction failures are isolated; a task only fails when a required stage cannot produce a valid result.

## Tests

Tests are deterministic and require no paid API calls:

```bash
cd deep-research-agent/backend
source .venv/bin/activate
pytest -q
```

The suite covers planner schemas, state defaults, the search contract, end-to-end orchestration with mock components, and citation/source mapping.

## Example question

> Compare dense retrieval, hybrid retrieval, and cross-encoder reranking for modern RAG systems. When does reranking provide the largest benefit?

## Current limitations

- Demo mode uses a small embedded set of real paper URLs and deterministic synthesis; it is for product validation, not trustworthy open-domain research.
- Tasks and reports disappear when the backend restarts.
- Polling is used instead of streaming events.
- Page extraction is best-effort and does not handle paywalls, JavaScript-only pages, PDFs, or robots restrictions specially.
- Citation validation checks mappings, but not yet whether each cited claim is entailed by the source.
- Concurrency and token/cost accounting are basic local guardrails rather than production controls.

## Roadmap

Future versions will add:

1. hybrid RAG,
2. embeddings and vector search,
3. cross-encoder reranking,
4. evaluation benchmarks for Recall@K, MRR, nDCG, faithfulness, completeness, latency, and cost,
5. persistent PostgreSQL/Redis storage,
6. authentication,
7. tracing and production observability,
8. cloud deployment.

The existing `SearchProvider`, `LLMClient`, `ResearchRepository`, typed search/evidence records, and explicit loop are the intended seams for those additions.

