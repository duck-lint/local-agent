# local-agent

`local-agent` is a narrow retrieval runtime for local corpora.

The product center is the callable application layer, not the CLI. `LocalAgentApp` owns corpus ingestion, chunk production, embeddings, retrieval, grounded answering, diagnostics, and durable memory. The CLI is a thin adapter over that core so a later Tauri surface can call the same operations directly.

Quick operator runbook: [`OPERATOR_QUICKREF.md`](OPERATOR_QUICKREF.md)

## What This Repo Is

- A retrieval- and evidence-focused local runtime.
- A library-first application core with thin interfaces.
- A SQLite-backed corpus and embedding store with deterministic contracts.
- A vault-aware markdown ingester with heading-aware chunking and stable document/chunk identity.
- A constrained grounded-answer flow with citation validation and auditable run logs.

## What This Repo Is Not

- Not a general autonomous agent platform.
- Not a framework for arbitrary filesystem exploration.
- Not a UI repo.
- Not a place for duplicated business logic across adapters.

## Runtime Shape

The runtime is built around one canonical corpus contract:

- `DocumentRecord`
  - `doc_key`, `source_name`, `rel_path`, `source_uri`, `source_hash`
  - `frontmatter`, `title`, `folder`, `doc_type`, `sensitivity`
  - `entry_date`, `source_date`
- `ChunkRecord`
  - `chunk_key`, `doc_key`, `chunk_index`, `section_index`
  - `heading_path`, `chunk_anchor`, `chunk_title`
  - `text`, `chunk_hash`, `start_char`, `end_char`, `out_links`

That contract is produced once during ingestion and then reused by embeddings, retrieval, citation checks, diagnostics, memory, and any future UI surface.

## Core API

The main entry point is [`agent/app.py`](agent/app.py).

```python
from agent import LocalAgentApp

app = LocalAgentApp.from_config()
app.ingest_corpus()
app.embed_corpus()
result = app.answer_grounded("What do the notes say about coherence?")
print(result.text)
```

Primary operations:

- `LocalAgentApp.from_config(...)`
- `ingest_corpus(...)`
- `embed_corpus(...)`
- `retrieve(...)`
- `answer_grounded(...)`
- `chat(...)`
- `doctor(...)`
- `add_memory(...)`, `list_memory(...)`, `delete_memory(...)`, `export_memory(...)`

These methods return structured results and do not depend on CLI printing.

## Corpus Ingestion

The corpus ingester ports the vault-aware logic that matters for retrieval quality:

- markdown normalization aligned to vault content
- frontmatter parsing
- canonical heading-path handling
- section-aware chunk construction
- stable document and chunk keys
- wikilink extraction
- date extraction from paths and metadata

The only supported chunking and embedding-preprocess profile is `obsidian_v1`.

## Storage

The runtime keeps separate SQLite stores for:

- corpus metadata and chunks
- embeddings
- durable memory

Derived stores are rebuilt against the current contract when schema versions change. The runtime does not try to preserve outdated derived state.

## Grounded Answering

`ask` and `LocalAgentApp.answer_grounded(...)` use the same grounded flow:

1. Retrieve lexical and vector candidates from the indexed corpus.
2. Build a bounded evidence snapshot.
3. Call the answer model with retrieved evidence only.
4. Validate cited chunk keys and heading/path invariants.
5. Write an auditable `runs/<run_id>/run.json`.

`chat` is intentionally simpler: one prompt, one model response, one run log.

## Configuration

The shipped config lives at [`configs/default.yaml`](configs/default.yaml).

Top-level vocabulary:

- `model`, `model_fast`, `model_big`
- `prefer_fast`, `big_triggers`
- `ollama_base_url`
- `max_tokens`, `max_tokens_big_second`
- `timeout_s`, `timeout_s_big_second`
- `temperature`
- `max_chars_full_read`
- `workroot`
- `security`
- `corpus`
- `embeddings`
- `retrieval`
- `grounding`
- `runs`
- `memory`

Important paths:

- `workroot` points at the external runtime data root.
- `security.allowed_roots` are resolved under the active `security_root`.
- `corpus.db_path`, `embeddings.db_path`, and `memory.db_path` are resolved under the workroot unless given as absolute paths.

Ollama host precedence:

1. `--ollama-base-url`
2. `LOCAL_AGENT_OLLAMA_BASE_URL`
3. `OLLAMA_BASE_URL`
4. config `ollama_base_url`
5. built-in default `http://127.0.0.1:11434`

Workroot precedence:

1. `--workroot`
2. `LOCAL_AGENT_WORKROOT`
3. config `workroot`

## Workroot Layout

Default external layout:

```text
local-agent-workroot/
  allowed/
    corpus/
    scratch/
  runs/
  index/
  embeddings/
  memory/
```

The repo remains code-only. Live corpus data, run logs, and derived stores stay outside the checkout.

## CLI

The CLI remains available through `local-agent` and `python -m agent`, but it is only an adapter over the application layer.

Common commands:

```bash
python -m agent chat "ping"
python -m agent index --json
python -m agent embed --json
python -m agent query "coherence" --limit 5
python -m agent ask "Summarize the indexed notes about coherence."
python -m agent doctor --require-grounding --json
python -m agent memory list --json
```

Command summary:

- `chat`
- `ask`
- `index`
- `query`
- `embed`
- `memory add|list|delete|export`
- `doctor`

## Setup

Requirements:

- Python 3.11+
- Ollama reachable when using `chat`, `ask`, or Ollama-backed embeddings

Install:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .
```

Optional Torch embedding extras:

```bash
pip install -e ".[torch-embed]"
```

Smoke checks:

```bash
python -m agent doctor --no-ollama
python -m agent index --json
python -m agent embed --json
python -m agent ask "Summarize the indexed notes about coherence."
```

## Diagnostics

`doctor` checks:

- security roots
- corpus DB existence and contract match
- embedding completeness and freshness
- retrieval readiness
- memory evidence integrity
- optional Ollama reachability

Use `--require-grounding` when embeddings and retrieval must be ready for success.

## Testing

Run the unit suite:

```bash
python -m unittest discover -s tests -v
```

The suite covers:

- config loading and rejection of obsolete vocabulary
- vault-aware corpus normalization and stable chunk identity
- callable `LocalAgentApp` behavior
- CLI adapter behavior
- doctor/grounding readiness checks
- tool security behavior

## Security Model

Filesystem reads and durable-memory exports remain bounded by the policy in [`agent/tools.py`](agent/tools.py) and [`agent/app.py`](agent/app.py):

- allowlisted roots only
- allowlisted extensions only
- hidden path denial by default
- absolute path denial by default
- lexical and resolved containment checks
- durable-memory export stays under `security_root` and must target a `.json` file

This repo optimizes for inspectability, bounded behavior, and evidence discipline over flexibility.
