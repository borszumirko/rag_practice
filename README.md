# RAG experiments on Natural Questions

A question-answering experiment comparing **recursive vs. semantic chunking** and
**dense vs. hybrid retrieval** on a shared Wikipedia corpus. The pipeline accepts
only a question, retrieves evidence, and generates a short answer. The four final
configurations are evaluated on the same 400 held-out questions, after tuning on
100 development questions.

## Basic overview

### Dataset

The dataset is a filtered subset of the original **Natural Questions (NQ)**,
which includes Wikipedia snapshots and answer/evidence annotations. Preparation
uses two validation shards from a pinned Hugging Face mirror revision, verifies
their checksums, and selects examples deterministically with seed 42.

- **500 answerable questions**, each from a different Wikipedia page, with short
  answers and paragraph evidence supplied by at least two annotators.
- **500 full Wikipedia page snapshots** in one searchable corpus: 18,345,769
  characters and 3,461,637 whitespace-separated tokens.
- **100 development / 400 held-out questions**, split by page identity. All 500
  pages are indexed; held-out questions and labels are excluded from tuning.
- Exact source character offsets preserve answer and evidence locations through
  extraction, chunking, and context clipping. Alternative annotations and
  multi-span answers are retained.

Only `corpus.jsonl` is indexed. Questions contain no reference-page hints; labels
and provenance are used by the evaluator. The corpus uses the annotated historical
snapshots rather than current Wikipedia pages. This is a small, filtered,
corpus-wide NQ-derived task; its scores are **not official NQ benchmark scores**.
See [the dataset plan](docs/dataset-plan.md) for selection rules, provenance,
attribution, licensing, and limitations.

### RAG components

| Component | Implementation and evaluated settings |
| --- | --- |
| Recursive chunking | LangChain `RecursiveCharacterTextSplitter`; selected size **2,000 characters**, target overlap **200**; **11,148 chunks**. Splits by paragraph, newline, space, then character. |
| Semantic chunking | Native LangChain `SemanticChunker`; selected breakpoint **percentile 99**, buffer size **1**; **2,607 chunks**. Embeds buffered sentence windows and splits at large adjacent cosine distances. No size cap or overlap. |
| Embeddings | `qwen/qwen3-embedding-8b` through OpenRouter, full **4,096 dimensions**. Documents use plain text; queries receive the configured retrieval instruction. |
| Vector store | Persistent local **ChromaDB**, cosine search. Model/text embeddings are cached in SQLite. |
| Dense retrieval | Top **5** chunks by vector similarity. |
| Hybrid retrieval | Dense + **BM25Okapi**, combined with weighted reciprocal rank fusion (RRF); **50 candidates** per retriever, **0.75 dense / 0.25 BM25**, rank constant **60**. |
| Context assembly | Final top **5** chunks, at most **10,000 characters** including joining newlines. Clips the last retained chunk and updates its source offsets. No reranker. |
| Generation | `google/gemini-2.5-flash-lite` through OpenRouter; temperature **0**, reasoning disabled, maximum **128 output tokens**. Evidence-only prompt requests the shortest complete answer, or `UNKNOWN`. |

Chunk lengths and overlap are measured in **characters, not tokens**; recursive
overlap is a target and can be smaller at boundaries. Semantic chunks can exceed
the context budget, so fewer than five may fit.


## Running the project

Run all commands from the repository root. Requirements: **Python 3.13+**, **uv**,
and an **OpenRouter API key** for embedding and generation. Chroma runs locally.
### 1. Set up the environment and dataset

```sh
uv sync --locked --group data --group dev
```

Create a `.env` file in the repository root:

```dotenv
OPENROUTER_API_KEY=your_openrouter_api_key
```

Then download and prepare the pinned dataset:

```sh
uv run --group data python scripts/prepare_nq.py --download
```

The download is about 379 MB; prepared artifacts are about 72 MB. Subsequent
runs reuse cached downloads. Omit `--download` to rebuild offline from those
files. Preparation makes no model calls. Generated data and `.env` are ignored
by Git.

| Prepared file under `data/natural_questions/prepared/` | Purpose |
| --- | --- |
| `corpus.jsonl` | Full page text and source metadata; the only indexing input. |
| `development_questions.jsonl`, `held_out_questions.jsonl` | Question IDs and text. |
| `development_labels.jsonl`, `held_out_labels.jsonl` | Reference answers and annotated spans, for evaluation only. |
| `provenance.jsonl` | Original source-token to extracted-character mapping. |
| `manifest.json`, `validation_report.json` | Source/artifact checksums, selection policy, IDs, and validation results. |

### 2. Index and serve the API through Hydra

The main entry point uses `server.yaml`, which inherits the experiment settings
and selects recursive **2,000/200**, Qwen3 embeddings, Gemini Flash-Lite, and dense
retrieval. Index once, then start the server with the same configuration:

```sh
uv run python -m app.main command=index embedding.batch_size=64 api.interval_seconds=3
uv run python -m app.main
```

The installed `rag` command also runs this Hydra entry point. Override settings
on the command line or edit `src/app/conf/server.yaml` and its inherited groups:

```sh
uv run python -m app.main server.port=8080 server.rate_limit=60/minute
uv run python -m app.main retrieval.method=hybrid retrieval.dense_weight=0.75

# Semantic preparation/indexing and serving must use matching chunk settings.
uv run python -m app.main command=index chunking=semantic \
  chunking.breakpoint_threshold_amount=99 embedding.batch_size=64 \
  +embedding.encoding_format=base64 +embedding.max_batch_characters=80000
uv run python -m app.main chunking=semantic chunking.breakpoint_threshold_amount=99

# The main entry point also dispatches the experiment commands.
uv run python -m app.main command=evaluate evaluation.split=development evaluation.limit=100
uv run python -m app.main command=evaluate evaluation.split=held_out evaluation.limit=400
uv run python -m app.main command=ask 'question="What is the capital of France?"'
```

The server opens an existing complete index during startup; it does not download
data, build missing indices, or access evaluation labels. A missing/incomplete
index stops startup. Model calls and SQLite access run on one dedicated worker;
health checks remain responsive while generation runs.

Send a question:

```sh
curl http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"When did Amnesia: The Dark Descent come out?","thread_id":"example"}'
```

Responses include the answer, actual generation model, retained source slices
with character offsets, retrieval timing, API-reported generation token counts,
and cache/security metadata. `thread_id` is a client correlation ID; each question
is independent. Existing security checks, response caching, rate limiting,
`/health`, `/metrics`, and `/cache/stats` remain available. Cache hits reuse answer
and source metadata without calling the models again.

API settings (`server.host`, `server.port`, `server.environment`,
`server.log_level`, `server.rate_limit`, `server.cache_ttl_seconds`, and
`server.tracing_enabled`) are configured through Hydra. The API key remains an
environment secret in `.env`. `PORT` is accepted through Hydra's environment resolver for
hosting. Docker Compose mounts `./data` for the prepared corpus, caches, and
index; provision matching data separately for other hosting environments.

### 3. Run the dense experiment CLI directly

Inspect the default configuration without making API calls:

```sh
uv run python -m app.rag --cfg job
```

The original `app.rag` CLI defaults are recursive **1,000/200**, semantic **percentile 95**, and
**30 development questions**. Use the explicit overrides below for the final
selected configurations.

Build the recursive 2,000/200 index and ask a question:

```sh
uv run python -m app.rag command=chunks \
  chunking.chunk_size=2000 chunking.chunk_overlap=200

uv run python -m app.rag command=index \
  chunking.chunk_size=2000 chunking.chunk_overlap=200 \
  embedding.batch_size=64 api.interval_seconds=3

uv run python -m app.rag command=ask \
  chunking.chunk_size=2000 chunking.chunk_overlap=200 \
  'question="When did Amnesia: The Dark Descent come out?"'
```

Evaluate it on development, then on all held-out questions:

```sh
uv run python -m app.rag command=evaluate \
  chunking.chunk_size=2000 chunking.chunk_overlap=200 \
  evaluation.split=development evaluation.limit=100 evaluation.seed=42

uv run python -m app.rag command=evaluate \
  chunking.chunk_size=2000 chunking.chunk_overlap=200 \
  evaluation.split=held_out evaluation.limit=400 evaluation.seed=42
```

Build and evaluate semantic percentile 99:

```sh
uv run python -m app.rag command=index chunking=semantic \
  chunking.breakpoint_threshold_amount=99 \
  embedding.batch_size=64 +embedding.encoding_format=base64 \
  +embedding.max_batch_characters=80000 api.interval_seconds=3

uv run python -m app.rag command=evaluate chunking=semantic \
  chunking.breakpoint_threshold_amount=99 \
  evaluation.split=development evaluation.limit=100 evaluation.seed=42

uv run python -m app.rag command=evaluate chunking=semantic \
  chunking.breakpoint_threshold_amount=99 \
  evaluation.split=held_out evaluation.limit=400 evaluation.seed=42
```

Semantic indexing also prepares sentence-window embeddings and boundaries on its
first run. Recursive `command=chunks` is offline; semantic preparation, indexing,
query embeddings, and generation can incur API charges. Add
`evaluation.generate=false` to an evaluation for retrieval-only scoring.

Index each chunking/model combination before asking or evaluating. Indexing
resumes missing chunks and reuses cached embeddings; use one index writer at a
time. Direct `app.rag` evaluations create new Hydra runs and **generate answers
again** rather than resuming previous responses.

### Artifacts and checks

| Location | Contents |
| --- | --- |
| `data/rag/chunks/` | Fingerprinted chunk files and preparation manifests. |
| `data/rag/embeddings.sqlite3` | Model/text embedding cache shared across indices and queries. |
| `data/chroma/` | Persistent vector collections. |
| `data/runs/`, `data/sweeps/` | Hydra configs, question manifests, retrieved contexts, predictions, summaries, API usage/error logs. |
| `data/experiments/` | Experiment comparisons, frozen selections, checkpoints, costs, and validation reports. |

Run the offline project checks with:

```sh
uv run --group data --group dev python -m pytest -q
```

## API features and deployment

### Security checks and rate limiting

`/chat` runs [SecurityPipeline](src/app/security.py) before retrieval and after
answer generation:

1. Rejects input matching known prompt-injection patterns, such as requests to
   ignore instructions or reveal a system prompt.
2. Removes repeated delimiter sequences and separates double template braces.
3. Masks recognized email addresses, phone numbers, US SSNs, and card-number
   patterns before passing the question to retrieval and generation.
4. Masks the same PII patterns in the answer and replaces output matching the
   configured harmful-content patterns with a blocked-response message.

Blocked or empty-after-cleaning questions return HTTP 400. Request validation
limits `message` to 1–10,000 characters. Security notes appear in the response.
These are regex-based checks with limited pattern coverage. The generation
prompt also tells the model to treat instructions in retrieved evidence as
quoted content.

SlowAPI applies a per-client-IP limit, default **20 requests/minute**, returning
HTTP 429 when exceeded. Change it through Hydra:

```sh
uv run python -m app.main server.rate_limit=60/minute
```

The current API has no application-level authentication or authorization.

### Response caching and embedding reuse

There are two separate caches:

| Cache | Behavior |
| --- | --- |
| API response cache | In-memory, per server process, default **300-second TTL**. Keys hash the cleaned/masked question after lowercasing and trimming whitespace. Stores the validated answer, sources, model, and timing/token metadata. Cache hits skip retrieval and generation and return `cached=true`. It is cleared on restart and is not shared across instances. |
| Embedding cache | Persistent SQLite at `data/rag/embeddings.sqlite3`, keyed by **exact text and embedding model**. Reuses document, sentence-window, and query vectors across indexing and experiments. It survives restarts when its file is retained. |

Configure the response TTL with `server.cache_ttl_seconds`; setting it to `0`
prevents response-cache hits. `/cache/stats` reports hits, misses, hit rate, and
stored entry count. The response cache is shared across callers in one process;
`thread_id` is not part of its key. Original model token metadata remains in a
cached response, while `/metrics` counts model usage only on cache misses.
Experiment runners separately checkpoint completed answers; direct CLI
`ask`/`evaluate` calls do not use the API response cache.

### LangSmith tracing and local observability

LangSmith tracing is **disabled by default** through Hydra. To enable the
existing trace points, add credentials and a project name to `.env` (or set them
in the hosting environment):

```dotenv
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=natural-questions-rag
```

Then launch with:

```sh
uv run python -m app.main server.tracing_enabled=true
```

The existing `@traceable` functions emit `security_check_input` and
`security_check_output` traces inside the request's tracing context. Input-check
traces receive the original question, before masking. The RAG worker inherits
the request context, but **retrieval and generation currently have no dedicated
LangSmith spans**: model requests use the experiment's plain HTTP client rather
than an automatically traced LangChain model. This setting currently controls
API tracing; it does not instrument the standalone experiment scripts.

Structured application logs go to standard output. Run directories save resolved
Hydra configurations and API call/error logs with usage, provider/model IDs,
cost, and latency. `/metrics` exposes process-local request/error totals, average
latency, cache-hit rate, and API-reported generation token totals; embedding token
usage remains in the API logs. Metrics reset on restart. `/health` checks that the
RAG pipeline, security layer, and response cache were initialized; it does not
make a live provider request.

### Docker and possible Render hosting

For local containers, prepare the dataset and selected index first, then run:

```sh
docker compose up --build
```

[docker-compose.yml](docker-compose.yml) mounts `./data` into the container and
loads `.env`. The [Dockerfile](Dockerfile) launches `app.main` through Hydra and
checks `/health`. The image itself contains application code, not the dataset or
vector index.

[render.yml](render.yml) supplies a Render web-service template with:

- Build: `pip install uv && uv sync --frozen --no-dev`.
- Start: `uv run --no-sync python -m app.main server.environment=production`.
- `OPENROUTER_API_KEY` supplied as a secret; `PORT` resolved by Hydra.
- Health check at `/health`; automatic deployments disabled.

## Results

All four configurations below were evaluated on the **same 400 held-out
questions**. The recursive dense run was completed October 30, 2026; the semantic
and hybrid reports were completed November 1, 2026. Every pipeline uses Qwen3
embeddings, the top-5 / 10,000-character context budget, and the same Gemini
model and evidence-only prompt.

| Chunking | Retrieval | Answer EM | Token F1 | Complete answer spans | Evidence coverage | Reference-page hit |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Recursive 2,000/200 | Dense | 54.00% (216/400) | **72.87%** | **95.25%** | **95.25%** | **100.00%** |
| Recursive 2,000/200 | Hybrid, dense weight 0.75 | **55.75% (223/400)** | 72.78% | 92.00% | 92.00% | **100.00%** |
| Semantic percentile 99 | Dense | 51.00% (204/400) | 66.13% | 82.50% | 82.26% | 92.75% |
| Semantic percentile 99 | Hybrid, dense weight 0.75 | 50.25% (201/400) | 65.53% | 82.75% | 82.51% | 92.50% |

### Metrics and calculation


| Metric | Per-question calculation |
| --- | --- |
| **Answer exact match (EM)** | 1 if the normalized prediction's token sequence equals any normalized reference sequence; otherwise 0. Token order matters. |
| **Token F1** | For each reference, count multiset token overlap `m` (repeated tokens count up to their frequency in both answers). With `p` prediction tokens and `g` reference tokens, precision is `m/p`, recall is `m/g`, and F1 is `2m/(p+g)`. Use the maximum F1 across references. If either token list is empty, score 1 only if both lists are equal, otherwise 0. |
| **Complete answer spans** | 1 if the retained context fully covers **every short-answer span from at least one alternative annotation** on the reference page; otherwise 0. Spans may be covered jointly by multiple chunks. |
| **Evidence coverage** | For each annotated evidence passage, divide covered source characters by passage length; take the maximum over alternative annotations. Merge overlapping retrieved intervals so duplicate characters count once. |
| **Reference-page hit** | 1 if any retained chunk comes from the annotated reference page; otherwise 0. |
