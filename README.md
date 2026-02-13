# local-agent

A small, security-focused CLI wrapper around Ollama for one job:

`read local file evidence -> produce a constrained answer -> log everything needed to audit the run`

This project is intentionally narrow. It is not a general autonomous agent framework. It is an evidence-gated runner that tries to make hallucination and unsafe file access expensive or impossible by default.

Quick operator runbook: [`OPERATOR_QUICKREF.md`](OPERATOR_QUICKREF.md)

## Table of contents

- Operator quick reference
- What this project is and is not
- Why this exists
- End-to-end architecture
- Execution flow (`chat` vs `ask`)
- Evidence and admissibility model
- Security model for file reads
- Model routing and token budget behavior
- Output quality controls (format + footer)
- Run logging and auditability
- Setup and quickstart
- CLI usage and recipes
- Configuration reference
- Error codes and troubleshooting
- Testing and verification
- Extending safely
- Practical limitations

## Operator quick reference

For day-to-day commands and failure triage, use:
- [`OPERATOR_QUICKREF.md`](OPERATOR_QUICKREF.md)

## What this project is and is not

What it is:
- A deterministic orchestrator around one model call (`chat`) or two model calls (`ask`).
- A strict tool-call protocol plus evidence validation.
- A sandboxed local file reader with typed failure modes.
- A reproducible run logger (`runs/<run_id>/run.json`).

What it is not:
- Not LangChain, not a planner/executor loop, not an unbounded tool agent.
- Not a retrieval pipeline with embeddings/vector DB.
- Not a "trust the model by default" UX.

## Why this exists

Common failure patterns in LLM + tool systems are well-known:
- The model claims it read a file that it never read.
- Tool-call JSON is malformed or mixed with prose and silently ignored.
- File access is too broad (path traversal, hidden files, absolute path reads).
- Partial reads are treated as full coverage.
- Documents contain prompt injection content and the model follows it.

local-agent turns these into explicit contracts:
- strict tool-call parsing
- fail-closed evidence gates
- sandboxed file access policy
- typed error codes
- auditable run logs with redaction

## End-to-end architecture

Core modules:
- `agent/__main__.py`
  - CLI parsing (`chat`, `ask`)
  - model selection (fast/big/default)
  - ask state machine
  - evidence validation and fail-closed behavior
  - second-pass output checks and retry logic
  - run logging
- `agent/tools.py`
  - `ToolSpec`, `ToolError`, `TOOLS`
  - `read_text_file` implementation
  - sandbox policy initialization and path validation
- `agent/protocol.py`
  - strict + robust tool-call parsing
  - supports prefix JSON tool-call extraction and trailing text capture
- `configs/default.yaml`
  - model defaults, token/time budgets, security policy
- `tests/test_tools_security.py`
  - sandbox and resolution behavior regression tests
- `SECURITY.md`
  - manual security verification checklist

## Execution flow

### `chat` mode

Single model call:
1. Send user prompt.
2. Print model response.
3. Log sanitized raw response and metadata.

No tool use, no evidence gates.

### `ask` mode

Two-pass control flow with one model-requested tool call:

1. Pass 1 (tool-selection prompt):
   - Model must either:
     - answer directly, or
     - emit `{"type":"tool_call","name":"...","args":{...}}`
2. Runner parses tool call:
   - strict parse first
   - prefix JSON parse fallback if response starts with tool-call JSON and contains trailing text
3. If a tool call is emitted:
   - execute tool
   - validate evidence
   - optional auto re-read for full-evidence questions when first read was truncated
4. Pass 2 (answer-only prompt):
   - tools forbidden
   - output quality checks enforced
5. If formatting violations:
   - one retry with stricter prompt
   - if still invalid -> typed failure

Important:
- If question semantics require file evidence and admissible evidence is not acquired, runner returns typed failure and does not ask model to guess.
- The runner may perform one additional `read_text_file` call itself for full-evidence rereads. This is not a model-requested second tool choice; it is runner-side evidence completion logic.

## Evidence and admissibility model

`read_text_file` evidence contract:
- `path` (absolute)
- `sha256` (hash of full text)
- `chars_full` (full length)
- `chars_returned` (returned text length)
- `truncated` (bool)
- `text` (possibly truncated content)

Evidence is rejected when:
- required fields are missing
- field types are wrong
- char counts are inconsistent
- file is empty for summary-style tasks
- tool returned error

If evidence is invalid/missing when required:
- run fails closed
- returns typed JSON failure
- no second-pass "best effort" answer

## Security model for `read_text_file`

Security policy is configured at startup from `configs/default.yaml`.

Controls:
- allowlisted roots (`allowed_roots`)
- allowlisted extensions (`allowed_exts`)
- deny absolute/anchored paths (`deny_absolute_paths`)
- deny hidden path segments (`deny_hidden_paths`)
- optional emergency bypass (`allow_any_path`, default false)
- root validation behavior (`auto_create_allowed_roots`, `roots_must_be_within_security_root`)

Path request styles:

1. Bare filename (no slash/backslash)
- Example: `note.md`
- Searched across allowlisted roots in order
- Exactly one match -> allowed
- Multiple matches -> `AMBIGUOUS_PATH`
- None -> `FILE_NOT_FOUND` (if search path was valid but file missing)

2. Explicit subpath (contains slash/backslash)
- Example: `corpus/project/note.md`
- Treated as workspace-root relative
- Must still fall within an allowlisted root

Additional protections:
- lexical containment checks before existence checks
- strict resolve checks for existing paths (symlink escape defense)
- extension and hidden-path checks before content read

## Model routing and budget behavior

Model selection supports default and split-model operation:
- Legacy/default: only `model` configured -> both passes use `model`
- Split mode:
  - pass 1 defaults to `model_fast` when `prefer_fast` is true
  - pass 2 may upgrade to `model_big` when question matches `big_triggers`

CLI overrides:
- `--fast`: force fast model for both passes
- `--big`: force big model for answer pass
- `--full`: force full evidence read attempt when tool used

Budget controls:
- `max_tokens` and `timeout_s` base values
- `max_tokens_big_second` and `timeout_s_big_second` for large answer pass
- `max_chars_full_read` cap for runner-side rereads

## Output quality controls

Pass 2 includes explicit constraints:
- no tool calls
- no tool-call JSON envelopes
- no markdown tables (bullet/paragraph style preferred)
- no claims beyond provided evidence
- include canonical evidence scope footer

Validation checks:
- table detector heuristic
- tool-call detector on pass 2 output
- exact scope-footer last-line check

Retry behavior:
- one retry for format violations
- fast-path optimization: if only missing scope footer, append locally and skip retry
- if retry still violates format -> `SECOND_PASS_FORMAT_VIOLATION`

Scope footer format:

```text
Scope: full evidence from read_text_file (5159/5159), sha256=14e424b8f1f06f8c2e2f43867f52f37f6ffb95f8434f743f2a94f367a7d2c999
```

## Run logging and auditability

Each invocation writes:

```text
runs/<run_id>/run.json
```

Key logged fields:
- run metadata (`mode`, question/prompt, timings)
- model selection (`raw_first_model`, `raw_second_model`)
- raw model responses with `message.thinking` stripped
- tool trace
- evidence status (`required`, `status`, truncation, char counts)
- retry metadata (if used)
- final assistant text

Redaction rule:
- for file-read results, logs keep metadata + `text_preview` only (first 800 chars)
- full file text is not logged by default

## Setup and quickstart

Requirements:
- Python 3.10+ (3.11 recommended)
- Ollama running locally (default `http://127.0.0.1:11434`)
- repo config available at `configs/default.yaml` (always used; see Config location below)
- dependencies: `requests`, `pyyaml`

Install (editable):

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .
```

On Linux/macOS, use:

```bash
source .venv/bin/activate
pip install -e .
```

Config location (important):
- Runtime always loads config from the repo file: `local-agent/configs/default.yaml`.
- Launch directory does not change which config file is selected.
- Root semantics: `config_root` comes from the loaded config path, `package_root` from installed code location, optional `workroot` comes from `--workroot` / `LOCAL_AGENT_WORKROOT` / config `workroot`, and `security_root` is the path anchor used for tool security and run logs.

Split repo/workroot setup (no workroot config required):
- Keep your single live config in repo: `local-agent/configs/default.yaml`.
- Point `security.allowed_roots` at your sibling workroot data folders (already set in this repo):
  - `../local-agent-workroot/allowed/corpus/`
  - `../local-agent-workroot/allowed/runs/`
  - `../local-agent-workroot/allowed/scratch/`
- Keep `security.roots_must_be_within_security_root: true` and set `workroot` to the sibling data root (default in this repo: `../local-agent-workroot/`).

Ensure allowlisted dirs exist (or keep `auto_create_allowed_roots: true`):

```text
corpus/
runs/
scratch/
```

Smoke test:

```bash
.venv\\Scripts\\python -m agent chat "ping"
.venv\\Scripts\\python -m agent ask "Read secret.md and summarize it."
local-agent ask "Read secret.md and summarize it."
local-agent --workroot ../local-agent-workroot ask "Read allowed/corpus/secret.md and summarize it."
```

## CLI usage and recipes

Basic:

```bash
python -m agent chat "<prompt>"
python -m agent ask "<question>"
local-agent chat "<prompt>"
local-agent ask "<question>"
local-agent --workroot ../local-agent-workroot ask "<question>"
```

`ask` flags:
- `--big`
- `--fast`
- `--full`

Common patterns:

1. Summarize file in `corpus/` by bare filename:

```bash
python -m agent ask "Read test1a.md and summarize it in 5 bullets."
```

2. Disambiguate duplicate names:

```bash
python -m agent ask "Read corpus/test1a.md and summarize it."
```

3. Request high-depth synthesis:

```bash
python -m agent ask --big "Read corpus/test1a.md and give a thorough synthesis."
```

## Configuration reference

Top-level:
- `model`, `model_fast`, `model_big`
- `prefer_fast`
- `big_triggers`
- `max_tokens`, `max_tokens_big_second`
- `timeout_s`, `timeout_s_big_second`
- `read_full_on_thorough`
- `max_chars_full_read`
- `full_evidence_triggers`
- `temperature`
- `ollama_base_url`

Security (`security:`):
- `allowed_roots`
- `allowed_exts`
- `deny_absolute_paths`
- `deny_hidden_paths`
- `allow_any_path`
- `auto_create_allowed_roots`
- `roots_must_be_within_security_root`

Current defaults in this repo are intentionally conservative:
- only `.md`, `.txt`, `.json` reads
- roots limited to `corpus/`, `runs/`, `scratch/`
- absolute/hidden path denial enabled

## Error codes and troubleshooting

Typed failure format:

```json
{"ok": false, "error_code": "...", "error_message": "..."}
```

Frequent codes and first checks:
- `CONFIG_ERROR`
  - verify `security.allowed_roots` resolve to valid directories
- `PATH_DENIED`
  - check extension allowlist, hidden segments, traversal/absolute path use
- `FILE_NOT_FOUND`
  - file not found under allowlisted roots
- `AMBIGUOUS_PATH`
  - duplicate bare filename; use explicit subpath
- `EVIDENCE_NOT_ACQUIRED`
  - model did not produce admissible tool call when evidence required
- `FILE_EMPTY`
  - source file empty for summarize request
- `EVIDENCE_TRUNCATED`
  - full evidence required but read remained partial
- `UNEXPECTED_TOOL_CALL_SECOND_PASS`
  - model violated answer-only phase
- `SECOND_PASS_FORMAT_VIOLATION`
  - output still violated format after one retry

Debug tip:
- open latest `runs/<run_id>/run.json`
- inspect `resolved_config_path`, `config_root`, `package_root`, `workroot`, and `security_root` first
- inspect `tool_trace`, `evidence_status`, `raw_first`, `raw_second`, and retry fields

## Testing and verification

Run unit tests:

```bash
python -m unittest -v tests.test_tools_security
```

Coverage includes:
- allowlisted read success
- explicit subpath success
- explicit subpath workspace-root anchoring (independent of process CWD)
- ambiguous bare filename rejection
- extension and hidden path denial (including `.env`)
- traversal/absolute path denial
- workspace-root file rejection when not allowlisted
- fail-closed misconfiguration behavior
- symlink escape denial (POSIX test)

Manual security checklist:
- see `SECURITY.md`

## Extending safely

If you add tools:
1. Add new `ToolSpec` in `agent/tools.py`.
2. Decide if output is admissible evidence.
3. If admissible, add explicit validator in runner logic.
4. Keep pass boundaries strict:
   - pass 1: tool decision
   - pass 2: answer only from provided tool output
5. Add tests for security and contract behavior.

## Practical limitations

Intentional limits:
- single model-requested tool call per ask run
- bounded read/token budgets
- strict formatting and protocol checks can produce "hard fails" rather than graceful-but-risky answers

Non-goals:
- broad autonomous task execution
- unrestricted filesystem exploration
- hidden-file or arbitrary-extension access by default

## Design philosophy

This runner is built around three constraints:
- Finitude: bounded resources are explicit, not hidden.
- Integrity: only typed evidence is admissible for evidence-required asks.
- Scope discipline: partial coverage must be disclosed mechanically.

Mental model:
- a small "epistemic linter" around local-file Q&A, optimized for correctness and auditability over flexibility.
