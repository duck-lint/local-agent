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

When memory records include `--chunk-key` evidence, those chunk keys are validated against the current corpus index. Memory exports also include corpus-contract provenance and any dangling evidence keys so downstream tooling can distinguish canonical evidence links from stale compatibility data.

## Setup

Requirements:

- Python 3.11+
- Ollama reachable when using `chat`, `ask`, or Ollama-backed embeddings

Install:

```bash
python -m venv .venv
```

Activate the environment:

- Linux/macOS: `source .venv/bin/activate`
- Windows PowerShell: `.\.venv\Scripts\Activate.ps1`

Then install the package:

```bash
python -m pip install -e .
```

Optional Torch embedding extras:

```bash
python -m pip install -e ".[torch-embed]"
```

`requirements.txt` is a pinned CUDA-oriented environment snapshot for release or GPU-specific setups, not the default cross-platform development install.

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

## Packaging

Create a curated release zip from the repo root:

```bash
python scripts/make_release_zip.py --dry-run
python scripts/make_release_zip.py --out dist/local-agent-release.zip
```

When you need to include an external workroot payload, pass it explicitly or rely on `LOCAL_AGENT_WORKROOT`:

```bash
python scripts/make_release_zip.py --include-workroot --workroot ../local-agent-workroot
```

## Security Model

Filesystem reads and durable-memory exports remain bounded by the policy in [`agent/tools.py`](agent/tools.py) and [`agent/app.py`](agent/app.py):

- allowlisted roots only
- allowlisted extensions only
- hidden path denial by default
- absolute path denial by default
- lexical and resolved containment checks
- durable-memory export stays under `security_root` and must target a `.json` file

## Contract Boundaries and Safest Integration Shape

Direct answer: the current repo contract is already library-first and coherent. Future work should treat `LocalAgentApp`, the corpus and embedding contracts, doctor checks, citation validation, and adapter thinness as the stable center. New work should integrate by extending those existing surfaces, not by reintroducing a staged pipeline shape or duplicating behavior in the CLI or a later UI.

### Current Contract Map

- API boundary
  - `LocalAgentApp` in [`agent/app.py`](agent/app.py) is the canonical callable surface.
  - Public operations return structured result objects rather than relying on CLI printing.
- Data boundary
  - `DocumentRecord` and `ChunkRecord` are the canonical corpus model reused by ingestion, embeddings, retrieval, grounding, diagnostics, and memory.
  - The corpus contract includes stable document and chunk identity, heading paths, anchors, titles, hashes, metadata, dates, and outbound links.
- Storage boundary
  - SQLite remains the persistence spine for corpus state, embeddings, and durable memory.
  - Embeddings and other derived state are rebuildable against the current contract rather than preserved for backward compatibility.
- UI and adapter boundary
  - The CLI is an adapter over `LocalAgentApp`; it forwards flags, serializes structured results, and should not own business logic.
  - A future UI surface should call the same application layer directly.
- Tooling and diagnostics boundary
  - `doctor` is the canonical readiness check for security roots, corpus schema and contract alignment, embedding freshness, retrieval readiness, memory evidence integrity, and optional Ollama reachability.
  - `runs/<run_id>/run.json` is part of the operational contract for auditable execution.
- Security boundary
  - Tool reads stay constrained to configured roots and extensions, with hidden paths and unsafe path traversal denied by policy.
- Test boundary
  - The tests protect vocabulary changes, corpus contract stability, callable app behavior, CLI adapter thinness, doctor readiness semantics, and tool security behavior.

### Stable vs Transitional vs Accidental

- Canonical and stable
  - The library-first runtime shape.
  - The `LocalAgentApp` callable API and structured result types.
  - The corpus contract centered on `DocumentRecord` and `ChunkRecord`.
  - Doctor check semantics, including `--require-grounding`.
  - Thin-adapter CLI behavior.
  - Security-root-bounded tool access.
- Transitional but intentional
  - The CLI itself is still an active surface, but its role is explicitly transitional toward shared adapter patterns rather than ownership of logic.
  - The external embedding store is part of the intended architecture, but operational readiness still depends on actually populating it for the active corpus.
- Accidental or underspecified
  - Any change that depends on parsing human-readable CLI output instead of structured results.
  - Any workflow that assumes derived SQLite state must survive contract or schema changes unchanged.
  - Any new adapter that silently forks grounding, retrieval, or citation rules outside the runtime modules.

### Safest Integration Shape

- Add behavior in the application layer first, then expose it through thin adapters.
- Reuse the existing contract objects and runtime helpers instead of inventing parallel payload shapes.
- Treat corpus schema changes, chunk identity changes, embedding fingerprint changes, and citation format changes as contract changes that require synchronized updates to diagnostics and tests.
- Keep grounded answering auditable: retrieved evidence, citations, and `run.json` output should stay aligned.
- Prefer explicit structured outputs for automation, tests, and any later UI work.

### Main Risk Areas

- Corpus contract drift
  - Changes to chunking, heading-path normalization, stable IDs, or metadata fields can invalidate embeddings, citations, memory evidence, and doctor readiness checks.
- Derived-state drift
  - Embedding configuration, preprocess signatures, and corpus contract signatures must remain aligned or be rebuilt together.
- Adapter leakage
  - Reintroducing business logic into the CLI or another surface would weaken the current contract and create behavior skew.
- Security regressions
  - Expanding filesystem access outside the current bounded policy would break an explicit repo invariant.
- Ambiguity around readiness
  - Non-strict doctor success does not mean grounded retrieval is ready; `--require-grounding` remains the explicit readiness contract when embeddings must be usable.

This repo optimizes for inspectability, bounded behavior, and evidence discipline over flexibility.
