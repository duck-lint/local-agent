# local-agent Operator Quick Reference

This is the short runbook. For full architecture and policy detail, see `README.md`.

## 1) Fast start checklist

1. Ollama is up:
```bash
curl http://127.0.0.1:11434/api/tags
```
2. Python env exists and deps installed (`requests`, `pyyaml`).
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .
```
On Linux/macOS:
```bash
source .venv/bin/activate
pip install -e .
```
3. Repo config exists (always used):
- Runtime config path is fixed to `local-agent/configs/default.yaml`.
- Launch directory does not change config selection.
- No config file is required in `local-agent-workroot`.
- Optional data root override: `--workroot` (or `LOCAL_AGENT_WORKROOT`, or config `workroot`) with precedence `--workroot` > env > config.

4. Allowlisted roots exist (or `auto_create_allowed_roots: true`):
- Keep `security.roots_must_be_within_security_root: true` and set `workroot` to the sibling data root (default: `../local-agent-workroot/`).
```text
allowed/corpus/
allowed/scratch/
runs/
```

## 2) Most-used commands

Windows:
```bash
.\.venv\Scripts\python.exe -m agent chat "ping"
.\.venv\Scripts\python.exe -m agent ask "Summarize the indexed notes about coherence."
.\.venv\Scripts\python.exe -m agent embed --json
.\.venv\Scripts\python.exe -m agent memory list --json
.\.venv\Scripts\python.exe -m agent doctor
```

Cross-platform:
```bash
python -m agent chat "ping"
python -m agent ask "Summarize the indexed notes about coherence."
python -m agent embed --json
python -m agent memory list --json
python -m agent doctor
python -m agent doctor --no-ollama
python -m agent doctor --require-phase3 --json
local-agent chat "ping"
local-agent ask "Summarize the indexed notes about coherence."
local-agent embed --json
local-agent memory list --json
local-agent doctor
local-agent --workroot ../local-agent-workroot ask "Summarize the indexed notes about coherence."
```

Model routing flags:
```bash
python -m agent ask --fast "Read allowed/corpus/secret.md and summarize it."
python -m agent ask --big "Read allowed/corpus/secret.md and give a thorough synthesis."
python -m agent ask --full "Read allowed/corpus/secret.md and summarize it."
local-agent ask --fast "Read allowed/corpus/secret.md and summarize it."
local-agent ask --big "Read allowed/corpus/secret.md and give a thorough synthesis."
local-agent ask --full "Read allowed/corpus/secret.md and summarize it."
```

Clean release zip:
```bash
python scripts/make_release_zip.py
python scripts/make_release_zip.py --dry-run
python scripts/make_release_zip.py --include-workroot
```
`--include-workroot` is curated (top-level boot/docs + sample allowed payload) and excludes `local-agent-workroot/runs/**`.

## 3) File path behavior (important)

- Bare filename (`note.md`):
  - searched across allowlisted roots
  - one hit = success
  - multiple hits = `AMBIGUOUS_PATH`
- Explicit subpath (`allowed/corpus/secret.md`):
  - resolved relative to `security_root` (which equals `workroot` when provided)
  - still must remain inside allowlisted roots
- Absolute paths are denied by default.
- Hidden paths (for example `.env`) are denied by default.

## 4) Read this first when a run fails

Open the latest run log:
```text
runs/<run_id>/run.json
```

Check these fields in order:
1. `ok`
2. `error_code`, `error_message`
3. `resolved_config_path`, `config_root`, `package_root`, `workroot`, `security_root`
4. `raw_first` and `assistant_text`
5. `tool_trace`
6. `evidence_required`, `evidence_status`
7. `raw_second`, `second_pass_retry_reason`

## 5) Common error codes and fixes

- `CONFIG_ERROR`
  - security config invalid (often no valid `allowed_roots`)
  - fix roots or enable `auto_create_allowed_roots`
- `PATH_DENIED`
  - extension/hidden-path/traversal/absolute-path policy denial
  - use allowlisted path and extension
- `FILE_NOT_FOUND`
  - file not found under allowlisted roots
  - verify path relative to `security_root` and allowlisted roots
- `AMBIGUOUS_PATH`
  - duplicate filename across roots
  - use explicit subpath (`allowed/corpus/...`)
- `EVIDENCE_NOT_ACQUIRED`
  - tool call not acquired though required by question
  - re-run with explicit "Read <file>" phrasing
- `EVIDENCE_TRUNCATED`
  - thorough request but evidence remained partial
  - use `--full` or increase `max_chars_full_read`
- `SECOND_PASS_FORMAT_VIOLATION`
  - answer formatting still invalid after retry
  - simplify prompt or reduce requested output scope
- `DOCTOR_INDEX_DB_MISSING`
  - preflight found no index DB at configured path
  - run `python -m agent index --rebuild --json`
- `DOCTOR_CHUNKER_SIG_MISMATCH`
  - preflight found stale chunking fingerprint vs current config
  - run `python -m agent index --scheme obsidian_v1 --rebuild --json` (or your configured scheme)
- `DOCTOR_PHASE3_EMBEDDINGS_DB_MISSING`
  - strict phase3 preflight found no embeddings DB
  - run `python -m agent embed --json`
- `DOCTOR_EMBED_OUTDATED_REQUIRE_PHASE3`
  - strict phase3 preflight found outdated/mismatched embedding rows
  - run `python -m agent embed --json` (or `--rebuild --json`)
- `DOCTOR_MEMORY_DANGLING_EVIDENCE`
  - durable memory points at chunk keys not present in phase2 index
  - delete/repair dangling records

## 6) Security sanity checks

Expected success:
```bash
python -m agent ask "Read allowed/corpus/secret.md and summarize it."
python -m agent ask "Read allowed/corpus/secret.md and list key claims."
```

Expected denial:
```bash
python -m agent ask "Read ../../etc/passwd and summarize it."
python -m agent ask "Read allowed/corpus/.env and summarize it."
```

Expected ambiguity:
```bash
python -m agent ask "Read dupe.md and summarize it."
```

## 7) One-liner troubleshooting flow

1. If startup fails, fix `CONFIG_ERROR`.
2. If tool fails, inspect path policy and `tool_trace`.
3. If evidence fails, check `evidence_status` and truncation fields.
4. If answer fails, inspect second-pass violations/retry metadata.
5. Re-run with `--workroot` (if needed), `--fast`, `--big`, or `--full` as needed.
   For offline preflight, use `python -m agent doctor --no-ollama`.
