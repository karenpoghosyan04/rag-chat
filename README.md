# ContextFlow

A Python app for asking questions about PDFs. Upload a document, ask a question, and get an answer with references to the relevant pages. Gemini or OpenAI generates the answers; document search runs locally by default.

## Setup

Use Python 3.13. From the repository root:

```bash
cd basic-chat-app
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and fill in `GEMINI_API_KEY`. Check that `GEMINI_MODEL` is a model available to your account. Then start the server:

```bash
python app.py
```

Open [localhost:8000](http://localhost:8000) and click Add PDF. Local models download on first use, so the first upload takes longer. Keep the tab open while the document is being indexed.

## Environment

The settings are in [basic-chat-app/.env.example](basic-chat-app/.env.example). Copy that file rather than creating an empty `.env`.

- `LLM_PROVIDER`: `gemini` or `openai`. Set the matching API key and model: `GEMINI_API_KEY` / `GEMINI_MODEL` or `OPENAI_API_KEY` / `OPENAI_MODEL`.
- `EMBEDDING_PROVIDER`: `local` by default. Set it to `gemini` or `openai` to use that provider's embedding API instead.
- `LOCAL_EMBEDDING_MODEL`: defaults to `google/embeddinggemma-2`. The app also offers BGE in the Document search selector.
- `RAG_DATA_DIR`: document and vector storage, defaulting to `basic-chat-app/data/` when unset.
- `APP_HOST` and `APP_PORT`: default to `127.0.0.1:8000`.

Use a model that accepts images for questions about figures and charts. API keys stay on the server, but retrieved document text and selected page images are sent to the answer provider. Keep `.env` out of Git.

## How it works

Most of the backend is in [app.py](basic-chat-app/app.py). It serves the browser files in `static/` and handles uploads and chat requests.

1. PyMuPDF4LLM extracts the PDF text and layout. The app splits it into chunks of about 512 tokens with 64 tokens of overlap.
2. SQLite stores the PDF, chunks, and metadata. Embedded Qdrant stores vectors. Uploading the same PDF can reuse its index when the settings match.
3. Each question searches the document using BM25 keyword search and embedding similarity. The results are merged, reranked with a local Jina model, and filtered for duplicates and weak matches.
4. Up to three text chunks, plus page images when needed, go to Gemini or OpenAI. The browser displays the answer with source references.

Switching between Gemma and BGE rebuilds the active document's index. Retrieval defaults are recorded in [tuned-settings.json](basic-chat-app/tuned-settings.json).

## Checks

Run these from `basic-chat-app/` with the virtual environment active:

```bash
pip install -r requirements-experiments.txt
python check_quality.py
python -m unittest discover -s tests -v
python train_relevance.py --output artifacts/ci-relevance
```

GitHub Actions runs the same checks. `check_quality.py` checks protected source and data hashes; it does not measure answer quality. The training script is an offline experiment and does not change the model used by the app.

For retrieval evaluation, use `rag_eval.py --help`. `replay_quality.py` compares retrieval against the saved baseline and requires cached local models. Reports are in `basic-chat-app/reports/`. The recorded tuning results use one PDF and 41 questions, so they are not a general accuracy score.

## Docker

From `basic-chat-app/`, with `.env` configured:

```bash
docker build -t contextflow .
docker run --rm -p 8000:8000 --env-file .env \
  -e APP_HOST=0.0.0.0 -e RAG_DATA_DIR=/app/data \
  -v pdf-rag-data:/app/data \
  -v pdf-rag-models:/app/.models \
  contextflow
```

The volumes keep documents and model downloads between runs.

## Things to know

- This is a single-user app with one shared active document. There is no authentication or rate limiting; keep it local unless you add access controls.
- Back up the data directory to keep uploaded PDFs and indexes. Chat history stays in the browser tab.
- Scanned PDFs need OCR or the vision fallback. Large files can take several minutes and use substantial memory.
- Source references help locate evidence, but they do not guarantee that every statement in an answer is supported.
