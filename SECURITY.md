# Security Verification

The `read_text_file` tool is sandboxed by `security` config in `configs/default.yaml`.

## Policy notes

- `allowed_roots` are relative to the workspace (`Path.cwd()`) unless absolute paths are provided.
- `".corpus"` is a literal hidden directory name. Use `"corpus/"` or `"./corpus/"` unless you intentionally created a folder named `.corpus`.
- `auto_create_allowed_roots`:
  - `true`: missing allowlisted roots are created at startup.
  - `false`: missing roots are ignored.
- `roots_must_be_within_workspace`:
  - `true`: any root outside the workspace is rejected.
- If no valid roots remain after validation, startup fails with:
  - `{"ok": false, "error_code": "CONFIG_ERROR", ...}`

## Manual checks

1. Allowed read (under allowlisted roots):
`python -m agent ask "Read secret.md and summarize it."`
Place the file at `corpus/secret.md` (or another allowlisted root).

2. Workspace-root file denied when root is not allowlisted:
`python -m agent ask "Read secret.md and summarize it."`

3. Traversal/outside-root denial:
`python -m agent ask "Read ../../etc/passwd and summarize it."`

Expected denied responses are typed failures with `error_code` such as `PATH_DENIED`, `FILE_NOT_FOUND`, or `CONFIG_ERROR`.
