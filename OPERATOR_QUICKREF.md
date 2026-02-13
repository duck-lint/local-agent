# local-agent Operator Quick Reference

This is the short runbook. For full architecture and policy detail, see `README.md`.

## 1) Fast start checklist

1. Ollama is up:
```bash
curl http://127.0.0.1:11434/api/tags
```
2. Python env exists and deps installed (`requests`, `pyyaml`).
3. Allowlisted roots exist (or `auto_create_allowed_roots: true`):
```text
corpus/
runs/
scratch/
```

## 2) Most-used commands

Windows:
```bash
.\.venv\Scripts\python.exe -m agent chat "ping"
.\.venv\Scripts\python.exe -m agent ask "Read secret.md and summarize it in 5 bullets."
```

Cross-platform:
```bash
python -m agent chat "ping"
python -m agent ask "Read secret.md and summarize it in 5 bullets."
```

Model routing flags:
```bash
python -m agent ask --fast "Read secret.md and summarize it."
python -m agent ask --big "Read secret.md and give a thorough synthesis."
python -m agent ask --full "Read secret.md and summarize it."
```

## 3) File path behavior (important)

- Bare filename (`secret.md`):
  - searched across allowlisted roots
  - one hit = success
  - multiple hits = `AMBIGUOUS_PATH`
- Explicit subpath (`corpus/secret.md`):
  - resolved relative to configured workspace root
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
3. `raw_first` and `assistant_text`
4. `tool_trace`
5. `evidence_required`, `evidence_status`
6. `raw_second`, `second_pass_retry_reason`

## 5) Common error codes and fixes

- `CONFIG_ERROR`
  - security config invalid (often no valid `allowed_roots`)
  - fix roots or enable `auto_create_allowed_roots`
- `PATH_DENIED`
  - extension/hidden-path/traversal/absolute-path policy denial
  - use allowlisted path and extension
- `FILE_NOT_FOUND`
  - file not found under allowlisted roots
  - verify path relative to workspace/roots
- `AMBIGUOUS_PATH`
  - duplicate filename across roots
  - use explicit subpath (`corpus/...`)
- `EVIDENCE_NOT_ACQUIRED`
  - tool call not acquired though required by question
  - re-run with explicit "Read <file>" phrasing
- `EVIDENCE_TRUNCATED`
  - thorough request but evidence remained partial
  - use `--full` or increase `max_chars_full_read`
- `SECOND_PASS_FORMAT_VIOLATION`
  - answer formatting still invalid after retry
  - simplify prompt or reduce requested output scope

## 6) Security sanity checks

Expected success:
```bash
python -m agent ask "Read secret.md and summarize it."
python -m agent ask "Read corpus/secret.md and summarize it."
```

Expected denial:
```bash
python -m agent ask "Read ../../etc/passwd and summarize it."
python -m agent ask "Read corpus/.env and summarize it."
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
5. Re-run with `--fast`, `--big`, or `--full` as needed.
