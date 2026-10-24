# ContextFlow

ContextFlow is a PDF question-answering application with hybrid retrieval, local reranking, and cited responses from Gemini or OpenAI. It combines a browser interface with a Python backend, persistent document storage, and on-demand page rendering for visual questions.

## Features

- Layout-aware PDF extraction with headings, tables, and sentence-aligned chunks.
- Hybrid BM25 and semantic search with Reciprocal Rank Fusion, local cross-encoder reranking, and duplicate filtering.
- Relevance thresholds for rejecting questions without sufficient document evidence.
- Inline source citations and page-image context for questions about figures and charts.
- Persistent PDF and vector storage, index reuse for identical uploads, and restoration of the most recently used document.
- Fixed, tuned chunking, retrieval, context, vision, and generation defaults.
- Sanitized Markdown and LaTeX rendering.

## Search model choice

The **Document search** selector offers **Gemma** (the default) and **BGE** (faster indexing with lower recorded memory use). These models retrieve PDF evidence; the configured Gemini/OpenAI model still writes the answer.

Changing the search model rebuilds the active PDF index before new questions can run. The choice persists in the local data directory and applies to this single-user app's shared active document. A failed model load keeps the previous selection and index. Existing chat messages remain visible.

Gemma uses the tuned 512-token chunks, 64-token overlap, 10 candidates, 3 context chunks and 0.75/0.25 BM25/dense weights. Its dense gate is 0.63 and its normalized Jina passage cutoff is 0.20. BGE uses the same retrieval budget with its separately recorded 0.555 dense gate and 0.20 passage cutoff. That BGE gate came from the earlier frozen-corpus calibration; it is not a claim of optimal tuning for every PDF.

The recorded Gemma tuning run reached 94.3% Recall@3 and 97.1% Recall@5 on one PDF and the existing 41 questions. The earlier controlled model comparison favored BGE on Recall@5 before the later Gemma tuning. Model, chunking, thresholds and evaluation scope must accompany any comparison. These retrieval results do not establish generated-answer accuracy or universal model superiority.

## Architecture

The browser sends chat and PDF upload requests to a Python HTTP server. API keys are handled exclusively server-side to prevent client exposure; retrieved text and selected page images are sent to the configured model provider for answer generation.

- **PDF processing:** PyMuPDF4LLM and its layout engine extract structured content for sentence-aligned chunking with configurable overlap.
- **Storage:** SQLite stores PDF bytes, chunks, metadata, and the active document selection. Embedded Qdrant persists vectors; BM25 maintains an in-memory index for the active document.
- **Embeddings:** `google/embeddinggemma-2` runs locally through Sentence Transformers by default, using normalized query and document embeddings. FastEmbed supports alternative local embedding models. Gemini and OpenAI embeddings are also configurable.
- **Reranking:** FastEmbed runs the local ONNX cross-encoder `jinaai/jina-reranker-v1-tiny-en`. Dense similarity and passage relevance thresholds filter weak matches before context assembly.
- **Generation:** Gemini or OpenAI APIs generate answers from retrieved context and optional rendered pages. The server checks for a valid supplied citation label and adds an evidence footer when all citations are omitted.

Storage defaults to `data/documents.sqlite3` and `data/qdrant/`. Set `RAG_DATA_DIR` to relocate it and back up the directory for recovery. PDFs are identified by SHA-256; identical uploads reuse the index when chunking and embedding settings match.

## Quickstart

Run these commands from `basic-chat-app/`. Python 3.13 matches the Docker and CI environments.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Configure `.env` with a provider API key and an available model ID. The example file contains the application's default model settings.

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=your_gemini_model_id
EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_MODEL=google/embeddinggemma-2
LOCAL_EMBEDDING_BATCH_SIZE=1
RERANKER_MODEL=jinaai/jina-reranker-v1-tiny-en
RAG_DATA_DIR=./data
APP_HOST=127.0.0.1
APP_PORT=8000
```

For OpenAI, set `LLM_PROVIDER=openai`, `OPENAI_API_KEY`, and `OPENAI_MODEL`. Use a multimodal model for visual questions. Local embeddings require no embedding API calls; set `EMBEDDING_PROVIDER=gemini` or `openai` to use hosted embeddings instead.

```bash
python app.py
```

Open [localhost:8000](http://localhost:8000), select **Add PDF**, and upload a document. Local models download on first use and are cached in `.models/`. Stop the server with `Ctrl+C`.

The app uses the selected search model and its fixed backend profile for every chat and PDF upload. The
retrieval profile uses **512 estimated tokens, 64-token overlap, 3 context chunks,
10 candidates, and BM25/dense weights 0.75/0.25**, with RRF constant 60, dense
similarity threshold `0.63`, and passage reranker threshold `0.20`. Legacy browser-saved
tuning values do not affect the app. The search-model selector is the supported runtime choice. The complete profile is recorded in
`tuned-settings.json`; offline evaluation can still compare other configurations.

## Evaluation & Quality Benchmarks

`rag_eval.py` evaluates retrieval against a PDF and a labeled JSONL question set. Each record specifies a question and relevant evidence by page, exact text, or both; an empty `relevant` array denotes an unanswerable question.

```json
{"question":"Why use multi-head attention?","relevant":[{"page":5,"contains":"Multi-head attention allows the model"}]}
{"question":"Does the paper discuss speech recognition?","relevant":[]}
```

```bash
python rag_eval.py \
  --pdf /path/to/document.pdf \
  --questions /path/to/questions.jsonl \
  --k 1,3,5 \
  --method hybrid \
  --output retrieval-report.json \
  --fail-under-recall 0.85 \
  --fail-under-no-answer-accuracy 0.85
```

Reports include Recall@K, hit rate, MRR, precision, no-answer accuracy, and retrieval latency. `--settings /path/to/settings.json` accepts a retrieval profile such as `tuned-settings.json`. Hybrid evaluation fails if embeddings are unavailable instead of reporting a BM25 fallback as hybrid retrieval.

To compare chunk sizes, overlap, and retrieval weights, add these options to the evaluation command:

```bash
  --benchmark-output benchmark-report.json \
  --chunk-sizes 256,512,1024 \
  --chunk-overlaps 0,64 \
  --dense-weights 0,0.5,1
```

This runs 18 configurations using sentence-aligned token budgets. Dense weights of `0` and `1` select only BM25 or dense candidates, respectively; intermediate weights combine both rankings. The hybrid query-level dense confidence gate and passage reranker remain active in these comparisons.

For a wider search, add `--min-scores 0.1,0.2,0.5`,
`--embedding-thresholds 0.6,0.63,0.65`, `--candidate-counts 10,25,50`,
and `--rrf-values 1,30,60,120`. `--k` also compares context counts when selecting settings.
The selection maximizes Precision@K subject to `--minimum-recall` and
`--minimum-no-answer` (both default to `0.85`). Include unanswerable questions to
measure the latter. Use `--best-settings-output tuned-settings.json` to export a
winner; export fails if no configuration meets both floors. Reranker pairs are
cached during grid search, so benchmark latency is not production latency and
does not influence selection. Evaluation builds an isolated index without
replacing the PDF uploaded in the app.

Settings interact: compare complete configurations instead of combining the
individually best value of each parameter. These are retrieval measurements on
the supplied tuning questions; use new questions and documents before claiming
general accuracy improvements. Generation temperature, answer length, history,
and vision settings require an answer-generation evaluation and are not selected
by this retrieval benchmark.

`replay_quality.py` checks for regressions against the recorded benchmark using frozen document vectors, an isolated in-memory index, and cached local embedding and reranker models:

```bash
python replay_quality.py
```

The replay writes `reports/retrieval-current.json` and fails on worse evidence ranks, lost correct refusals, or decreased retrieval metrics relative to `reports/retrieval-baseline.json`. It also checks source and evaluation-input integrity through `check_quality.py`. It does not access uploaded-document storage or call a hosted LLM. Replace the baseline with `--capture` only after reviewing and measuring an intentional change.

The recorded single-document benchmark contains 35 answerable and 6 unanswerable questions, with **91.4% Recall@5** and **6/6 correct rejections**. These results measure retrieval on the included benchmark; they do not establish answer faithfulness or performance on other PDFs.

The 2026-10-24 tuning sweep evaluated 610 distinct settings in 630 runs, followed
by uncached query-embedding and reranker verification on the same questions.
Against freshly evaluated previous defaults, Precision@3 increased from **27.6%
to 31.4%**, Recall@3 from **82.9% to 94.3%**, and Recall@5 from **91.4% to
97.1%**, retaining all six correct rejections. The selected app context count is
3; its operating-point recall also exceeds the old 4-chunk default's 88.6%.
`reports/tuning-summary.json` records the staged search, model versions, input
hashes, full settings, and metrics. `reports/tuning/verified-baseline.json` and
`verified-tuned.json` contain per-question results. The frozen replay retains
its original recorded configuration so it continues to detect changes to
retrieval behavior independently of the newly tuned defaults.

## Docker & CI

Build and run from `basic-chat-app/` with a configured `.env`:

```bash
docker build -t contextflow .
docker run --rm -p 8000:8000 --env-file .env \
  -e APP_HOST=0.0.0.0 -e RAG_DATA_DIR=/app/data \
  -v pdf-rag-data:/app/data \
  -v pdf-rag-models:/app/.models \
  contextflow
```

Open [localhost:8000](http://localhost:8000). Named volumes persist uploaded documents and model caches. The container runs as a non-root user; local models download on first use. A local image build requires a running Docker daemon.

The GitHub Actions workflow in [`.github/workflows/rag-checks.yml`](../.github/workflows/rag-checks.yml) runs source-integrity checks, unit tests, and offline relevance-model training on matching pushes and pull requests, then uploads the training report. It does not download large model weights or call paid APIs. Full retrieval replay requires locally cached models and runs separately.

Run the same checks locally with:

```bash
python -m pip install -r requirements-experiments.txt
python check_quality.py
python -m unittest discover -s tests -v
python train_relevance.py --output artifacts/ci-relevance
```

## Current Limitations

- PDFs over 10 MB ask for confirmation before uploading. Large books can take several minutes or longer to upload and index; keep the tab open until processing finishes. There is no application file-size cap, and processing remains limited by available server memory and resources. One PDF is active per chat, and stored documents have no library picker in the UI.
- Conversation history exists only in the browser tab, and responses are returned without token streaming.
- The server has no authentication or rate limiting. Embedded Qdrant requires a standalone server configuration for a multi-process deployment.
- Image-only scans require local OCR or the question-triggered vision fallback. Figure interpretation depends on model accuracy and image legibility.
- Citation-label validation does not verify that every generated claim is supported by its cited evidence.
- Retrieval thresholds are calibrated on a single document and require validation for other document collections.
- The `.env` loader supports only simple `KEY=value` lines.
- The Docker image build and container startup health endpoint have been verified locally. GitHub Actions validation runs on pushes and pull requests; hosted deployment has not been verified.
