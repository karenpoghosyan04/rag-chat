# Little LLM Chat

A compact hybrid RAG application for learning how a browser, Python backend, structured PDF retrieval, on-demand vision, and a hosted LLM API fit together. It supports Gemini or OpenAI, uses PyMuPDF4LLM with its layout engine, and safely renders model-generated Markdown and LaTeX.

## Mental model

```text
Browser UI                    Python server                         Gemini / OpenAI
----------                    -------------                         ---------------
upload a PDF ───────────────► PyMuPDF4LLM layout extraction
                              headings + paragraphs + tables
                              tunable sentence windows and overlap
                              SQLite PDF/chunk catalogue
                              Qdrant vector index on disk

type a question
      │
      └── POST /api/chat ───► skip retrieval for greetings
                              hybrid BM25 + semantic retrieval
                              Reciprocal Rank Fusion
                              range + semantic duplicate collapse
                              retrieve only relevant chunks
                              if figure-related: render 1–2 pages
                                    │
                                    └── text + optional images ─────► model
                                    ◄── cited response ──────────────┘
      ◄── reply + sources ────┘
```

The browser never receives your API key. That is the most important reason for having a backend instead of calling the model directly from JavaScript.

Embeddings run locally with `google/embeddinggemma-2` through Sentence Transformers by default. Queries use the official `SearchQuery` prompt and chunks use `Document`; both are L2-normalized. The text-only configuration omits the independent vision and audio encoders, uses float32 on CPU (bfloat16 on supported CUDA devices), and does not consume Gemini/OpenAI embedding quota. Gemini or OpenAI is still used for final answer generation.

Uploaded PDFs are content-addressed with SHA-256 and persisted under `data/` by default. SQLite stores each unique PDF once together with its chunk text and metadata; Qdrant stores and searches the embedding vectors. Re-uploading identical bytes with the same chunking and embedding configuration reuses the existing index without parsing or embedding again. The most recently used document is restored when the server restarts. Set `RAG_DATA_DIR` to move this storage, and back up that directory if the documents must survive disk or machine loss.

```text
data/documents.sqlite3   original PDF bytes, chunks, metadata, active selection
data/qdrant/             persistent embedding collections and vector indexes
```

## Run it

1. Make a local environment file:

   ```bash
   cd /Users/karenpoghosyan/Desktop/project/basic-chat-app
   cp .env.example .env
   ```

2. Create a modern Python environment and install the PDF pipeline. PyMuPDF4LLM's current layout engine requires Python 3.10 or newer:

   ```bash
   python3.13 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   ```

3. Open `.env` and configure the Gemini free tier:

   ```env
   LLM_PROVIDER=gemini
   GEMINI_API_KEY=your_key_here
   GEMINI_MODEL=gemini-3.6-flash
   GEMINI_EMBEDDING_MODEL=gemini-embedding-001
   EMBEDDING_PROVIDER=local
   LOCAL_EMBEDDING_MODEL=google/embeddinggemma-2
   LOCAL_EMBEDDING_BATCH_SIZE=1
   APP_PORT=8000
   ```

4. Start the Python server:

   ```bash
   .venv/bin/python app.py
   ```

5. Open http://localhost:8000, click **Add PDF**, and ask a question about it.

Stop it with `Ctrl+C`.

## RAG tuning menu

Open **Settings** in the top-right corner. Values are validated by both the browser and Python server and persist in browser `localStorage`.

- **Chunking:** sentences per chunk and overlapping sentences. Changing either automatically re-parses and re-embeds the active PDF.
- **Retrieval mode:** hybrid, embeddings only, or BM25 only.
- **Two-stage retrieval:** BM25 and dense search first produce 25 broad RRF candidates without an aggressive cutoff. A local ONNX cross-encoder reranks those query/passage pairs, and `min_score` is applied to its normalized per-passage relevance score.
- **Relevance gates:** `embedding_min_similarity` is a query-level abstention gate over the strongest dense match (and a candidate floor in embeddings-only mode). `min_score` is a separate passage-level cross-encoder gate. The calibrated defaults are `0.63` and `0.20`, respectively; see `embedding-eval/comparison.md` for the split and distributions.
- **Hybrid fusion:** candidate pool, RRF rank constant, BM25 weight, and embedding weight.
- **Context:** number of retrieved PDF chunks and conversation messages sent to the model.
- **Vision:** enable/disable page rendering and choose the maximum attached pages.
- **Generation:** model temperature and maximum output tokens.

Citation enforcement is intentionally not optional. When PDF evidence is supplied, responses must expose a valid `[Source N]` or `[Visual page N]` label.

The defaults are a starting point, not universally optimal. Tune them against a labeled question set and measure retrieval recall, citation accuracy, unsupported-answer rate, latency, and API cost.

## Measure retrieval quality

Do this before changing chunking, embedding models, or retrieval thresholds. Create a JSONL file with questions and the evidence each question should retrieve. A label may use a page, a short exact passage, or both; an empty `relevant` list marks a question the PDF cannot answer.

```json
{"question":"Why use multi-head attention?","relevant":[{"page":5,"contains":"Multi-head attention allows the model"}]}
{"question":"Does the paper discuss speech recognition?","relevant":[]}
```

Run the evaluator against the same index and settings as the app:

```bash
cd /Users/karenpoghosyan/Desktop/project/basic-chat-app
.venv/bin/python rag_eval.py \
  --pdf /path/to/document.pdf \
  --questions eval_questions.jsonl \
  --k 1,3,5 \
  --method hybrid \
  --output retrieval-report.json
```

The report includes:

- **Recall@K:** fraction of labeled evidence items found in the first K chunks. This is the main retrieval safety metric.
- **Hit rate@K:** fraction of answerable questions with at least one useful chunk.
- **MRR@K:** rewards placing the first useful chunk near the top.
- **Precision@K:** how much of the limited context window is useful evidence.
- **No-answer accuracy:** whether unrelated questions correctly retrieve nothing.
- **Latency p50/p95:** retrieval time including query embedding.

Use at least 30–50 real questions spanning direct wording, paraphrases, tables, figures, multi-part answers, and unanswerable prompts. Keep a held-out subset so tuning thresholds does not overfit the same questions. You can pass a UI-compatible settings JSON file with `--settings`, and use `--fail-under-recall 0.85` to make a regression fail in CI.

Generate this repository's verified stress set from the sample Transformer paper:

```bash
.venv/bin/python generate_eval_questions.py \
  --pdf /Users/karenpoghosyan/Downloads/testrag.pdf \
  --output eval_questions.jsonl
```

The generator currently emits 35 paraphrased answerable questions and 6 unrelated questions. It validates every exact evidence phrase and page against chunks produced by the application and rejects questions that copy a three-word evidence phrase.

Run the default evaluation and the 18-configuration sensitivity benchmark together:

```bash
.venv/bin/python rag_eval.py \
  --pdf /Users/karenpoghosyan/Downloads/testrag.pdf \
  --questions eval_questions.jsonl \
  --k 1,3,5 \
  --method hybrid \
  --output retrieval-report.json \
  --benchmark-output benchmark-report.json \
  --chunk-sizes 256,512,1024 \
  --chunk-overlaps 0,64 \
  --dense-weights 0,0.5,1
```

Token-budget chunking is sentence-aligned and remains inside semantic sections. A dense weight of `0` measures BM25 only, `1` measures embeddings only, and intermediate values use Reciprocal Rank Fusion with `bm25_weight = 1 - dense_weight`. The benchmark report records Recall@K, MRR, precision, latency, no-answer accuracy, and the rate at which unrelated questions still pass configured retrieval gates.

For CI, gate both evidence recall and abstention behavior:

```bash
.venv/bin/python rag_eval.py \
  --pdf /path/to/document.pdf \
  --questions eval_questions.jsonl \
  --method hybrid \
  --fail-under-recall 0.85 \
  --fail-under-no-answer-accuracy 0.85
```

The evaluator refuses to label a BM25 fallback as a hybrid run when embeddings are unavailable.

Qdrant runs in embedded, on-disk mode, so this learning app does not require a separate database server. Dense similarity queries go through Qdrant while the active document keeps a small in-memory BM25 index for hybrid retrieval. For a multi-process or production deployment, move Qdrant to its standalone server and mount its storage on a persistent volume. A vector database improves persistence and search scaling; it does not make weak embeddings, chunking, or labels more accurate.

## What to read, in order

1. `static/index.html` — the page structure.
2. `static/app.js` — collects messages, calls Python with `fetch`, and renders replies.
3. `app.py` → `ChatHandler.do_POST` — receives the browser request.
4. `app.py` → `call_llm` — chooses a provider; `call_gemini` and `call_openai` show their different API shapes.
5. `app.py` → `structure_aware_chunks` — builds four-sentence windows with one sentence of overlap.
6. `app.py` → `DocumentIndex.search` — combines thresholded BM25 and cosine-similarity rankings with Reciprocal Rank Fusion, then removes overlapping and semantically duplicate candidates before Top-K.
7. `app.py` → `DocumentIndex.render_pages` — renders only selected figure pages for multimodal questions.
8. `static/styles.css` — presentation only; changing it cannot expose the API key.

## A PDF to try

Download [Attention Is All You Need from arXiv](https://arxiv.org/pdf/1706.03762), upload it, and try:

> Why did the authors use multi-head attention, and how is it different from a single attention function?

Also try: “What does Figure 1 show?” The app retrieves the caption, renders the relevant page, and sends that page image to the configured multimodal model.

Uploads are limited to 10 MB and one PDF is active at a time, but every unique PDF remains in persistent storage. Uploading a new document embeds every chunk once; uploading the same bytes with the same index configuration reuses the stored chunks and vectors. Each non-greeting PDF question embeds the query once. With the default local provider these operations consume CPU but no embedding API quota. Visual questions can send rendered PDF pages to the configured answer model, which adds latency, token usage, and provider-side data processing. Ordinary text questions do not send page images.

Grounded answers are instructed to cite PDF claims inline as `[Source N]` or `[Visual page N]`. The server validates that at least one valid supplied label appears whenever PDF evidence is used and adds an evidence footer if the provider omits every citation.

## Try the API without the UI

With the server running:

```bash
curl -X POST http://localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Explain recursion in one paragraph."}]}'
```

## Run the tests

The tests mock provider requests, so they do not spend money or require an API key:

```bash
python3 -m unittest discover -s tests -v
```

## Small exercises

1. Change the assistant instructions in `call_openai`.
2. Add a model selector to the UI, but validate allowed models on the server.
3. Save the `messages` array to `localStorage` and restore it on reload.
4. Render Markdown safely.
5. Replace `http.server` with FastAPI after you understand the current request flow.

## Current limitations

- Conversation history only lives in the browser tab.
- One PDF is active in the chat at a time. Previously uploaded documents remain in `RAG_DATA_DIR`, but the UI does not yet include a document-library picker.
- Retrieval builds a recall-oriented set of up to 25 BM25+dense RRF candidates and reranks them with `jinaai/jina-reranker-v1-tiny-en` through FastEmbed. The dense and reranker gates have distinct calibrated roles described above. Post-rerank filtering removes chunks whose document-global source ranges overlap by at least 40%, chunks with at least 70% literal word-sequence containment, chunks sharing an exact normalized sentence of 80 or more characters, and embedding pairs above 0.88 cosine similarity when both chunks exceed 20 words. Sentence matching also works in BM25-only mode. Markdown blockquotes such as contribution footnotes form separate chunk boundaries instead of being mixed into the preceding abstract.
- PyMuPDF4LLM handles layout, columns, headings, tables, and visual-region detection. If no local OCR engine is installed, image-only scanned text is handled only when a document or visual question triggers the vision fallback.
- Figure understanding is probabilistic. Do not rely on a vision model for precise chart values that are not clearly legible.
- Responses are returned all at once rather than streamed token by token.
- There is no login, database, rate limiting, or production deployment configuration.
- The tiny `.env` loader intentionally supports only simple `KEY=value` lines.

These are good next lessons, not bugs in the learning scope.
