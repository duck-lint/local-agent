# local-agent Operator Quick Reference

This is the short runbook for the library-first runtime. For architecture and contract details, see [`README.md`](README.md).

## Fast Start

1. Create a Python environment and install the package.

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .
```

2. Ensure the external workroot exists.

```text
local-agent-workroot/
  allowed/
    corpus/
    scratch/
  runs/
```

3. Check the runtime without network dependencies.

```bash
python -m agent doctor --no-ollama
```

4. Build corpus and embeddings.

```bash
python -m agent index --json
python -m agent embed --json
```

5. Run a grounded answer.

```bash
python -m agent ask "Summarize the indexed notes about coherence."
```

## Most-Used Commands

```bash
python -m agent chat "ping"
python -m agent index --json
python -m agent query "coherence" --limit 5
python -m agent embed --json
python -m agent ask "Summarize the indexed notes about coherence."
python -m agent doctor --json
python -m agent doctor --require-grounding --json
python -m agent memory list --json
local-agent ask "Summarize the indexed notes about coherence."
```

## Important Flags

- `--workroot`
  - override the external runtime data root
- `--ollama-base-url`
  - override the Ollama host
- `doctor --no-ollama`
  - skip network reachability checks
- `doctor --require-grounding`
  - fail unless embeddings and retrieval are ready
- `ask --fast`
  - force the faster answer path
- `ask --big`
  - force the larger answer model
- `embed --rebuild`
  - refresh every embedding row
- `index --rebuild`
  - refresh every corpus document and chunk row

## Config Vocabulary

The shipped config uses:

- `security`
- `corpus`
- `embeddings`
- `retrieval`
- `grounding`
- `runs`
- `memory`

If the config still contains older pipeline keys, the runtime rejects it.

## Runtime Checks

`doctor` verifies:

- security roots are configured
- corpus DB exists and matches the active chunk contract
- embeddings exist and match the active corpus and embedding settings
- retrieval can return candidates when grounding is required
- durable memory evidence still points at current chunk keys

## Validation Order For Changes

Validate contract surfaces in this order so failures stay interpretable:

1. **CLI and config vocabulary**
   - confirm the adapter still exposes the current runtime terms: `corpus`, `embeddings`, `retrieval`, `grounding`, `runs`, `memory`
   - confirm `doctor` still uses `--require-grounding`
2. **Corpus contract**
   - re-run corpus sync and confirm document keys, chunk keys, headings, anchors, and metadata remain stable on a second pass
3. **Grounded answer contract**
   - grounded answers should either emit citations in the exact form `[source: rel_path#heading_path | chunk_key]` or surface citation-validation failures in `runs/<run_id>/run.json`
   - heading normalization is intentionally tolerant of punctuation-only differences; changed path or chunk identity is not
4. **Diagnostics**
   - run `python -m agent doctor --no-ollama` first to isolate local schema and storage issues
   - only treat `python -m agent doctor --require-grounding --json` as a readiness check after corpus and embeddings are current
5. **Tool security**
   - keep traversal, hidden-path, and ambiguous bare-filename denials intact

## Failure Triage

Open the latest run log:

```text
runs/<run_id>/run.json
```

Check these fields first:

1. `ok`
2. `error_code`, `error_message`
3. `resolved_config_path`, `config_root`, `package_root`, `workroot`, `security_root`
4. `assistant_text`
5. `retrieval`
6. `citations`

## Common Fixes

- `CONFIG_ERROR`
  - inspect [`configs/default.yaml`](configs/default.yaml) for invalid or obsolete keys
- `DOCTOR_CORPUS_DB_MISSING`
  - run `python -m agent index --rebuild --json`
- `DOCTOR_EMBEDDINGS_MISSING`
  - run `python -m agent embed --json`
- `DOCTOR_EMBED_PREPROCESS_MISMATCH`
  - run `python -m agent embed --rebuild --json`
- `DOCTOR_RETRIEVAL_NOT_READY`
  - rebuild corpus and embeddings, then rerun `doctor --require-grounding`
- `DOCTOR_MEMORY_DANGLING_EVIDENCE`
  - reset or repair durable memory entries that cite removed chunk keys

## Security Checks

Expected success:

```bash
python -m agent ask "Summarize allowed/corpus/note.md."
```

Expected denial:

```bash
python -m agent ask "Read ../../etc/passwd."
python -m agent ask "Read allowed/corpus/.env."
```

## Adapter Rule

Business logic belongs in `LocalAgentApp` and its supporting modules. If you add a new surface later, call the core runtime directly instead of re-implementing corpus, retrieval, grounding, or memory behavior in the adapter.
