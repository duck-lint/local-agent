# Security Verification

The `read_text_file` tool is sandboxed by `security` config in `configs/default.yaml`.

Manual checks:

1. Allowed read inside project root:
`python -m agent ask "Read ./organon.md and summarize it in 5 bullets."`

2. Traversal/outside-root denial:
`python -m agent ask "Read ../../etc/passwd and summarize it."`

Expected result for denied paths: typed failure with `error_code` such as `PATH_DENIED` in `run.json`.
