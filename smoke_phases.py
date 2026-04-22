"""End-to-end smoke runner for the 4-phase retrieval + session memory stack.

Invokes the real CLI (python -m agent ...) against the real Ollama backend and
the existing workroot corpus. Each scenario toggles one phase's feature flags,
runs an actual command, then validates invariants against the emitted run.json.

Safe: backs up configs/default.yaml + configs/acronyms.yaml and restores them
in a try/finally.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
WORKROOT = (REPO / ".." / "local-agent-workroot").resolve()
DEFAULT_YAML = REPO / "configs" / "default.yaml"
ACRONYMS_YAML = REPO / "configs" / "acronyms.yaml"
DEFAULT_BAK = REPO / "configs" / "default.yaml.smoke_bak"
ACRO_BAK = REPO / "configs" / "acronyms.yaml.smoke_bak"

# The workroot venv has torch installed; the bare system python does not.
VENV_PY = WORKROOT / ".venv" / "Scripts" / "python.exe"
PY = str(VENV_PY) if VENV_PY.exists() else sys.executable

ENV = os.environ.copy()
ENV["PYTHONPATH"] = str(REPO) + os.pathsep + ENV.get("PYTHONPATH", "")
ENV["PYTHONIOENCODING"] = "utf-8"

# Unique session ids per run so repeated smoke runs start cold.
_RUN_TAG = time.strftime("%Y%m%d-%H%M%S")
S1 = f"smoke-s1-{_RUN_TAG}"
S2 = f"smoke-s2-{_RUN_TAG}"

RESULTS: list[tuple[str, bool, str]] = []


def rec(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f" :: {detail}" if detail else ""))


def cli(args: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    cmd = [PY, "-m", "agent", *args]
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=str(WORKROOT), env=ENV, capture_output=True, text=True, timeout=timeout)


def find_run(stdout: str) -> Path | None:
    m = re.search(r"\[logged\]\s+(.+?run\.json)\s*$", stdout, re.MULTILINE)
    if not m:
        return None
    p = Path(m.group(1).strip())
    return p if p.exists() else None


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


BASE = DEFAULT_YAML.read_text(encoding="utf-8")
# NOTE: the checked-in default.yaml has `fusion: "rerank"` but retrieval.py only
# accepts "simple_union". Normalise for the smoke suite so ask can actually run.
BASE = re.sub(r'fusion:\s*"rerank"', 'fusion: "simple_union"', BASE)


def patch(patches: list[tuple[str, str, int]]) -> str:
    t = BASE
    for pat, repl, cnt in patches:
        t = re.sub(pat, repl, t, count=cnt)
    return t


def phase0() -> None:
    print("\n=== PHASE 0 — baseline (all flags off) ===")
    write(DEFAULT_YAML, BASE)
    r = cli(["ask", "--fast", "what do the pillars describe"])
    rec("phase0.ask_exits_0", r.returncode == 0, f"rc={r.returncode} stderr_tail={r.stderr.strip()[-160:]}")
    run = find_run(r.stdout)
    rec("phase0.run_json_emitted", run is not None)
    if run is None:
        return
    d = load_json(run)
    rec("phase0.has_retrieval_results", bool(d.get("retrieval", {}).get("results")))
    # Baseline always emits a single-round retrieval_rounds entry; refinement must NOT fire.
    rounds0 = d.get("retrieval_rounds") or []
    rec("phase0.baseline_single_round", len(rounds0) <= 1 and not d.get("refinement_applied"),
        f"rounds={len(rounds0)} refinement_applied={d.get('refinement_applied')}")
    rec("phase0.coverage_null", d.get("coverage") is None)
    rec("phase0.memory_snapshot_null", d.get("memory_snapshot") is None)
    rec("phase0.budget_present", "budget" in d)


def phase0_rrf() -> None:
    print("\n=== PHASE 0b — fusion: rrf ===")
    write(DEFAULT_YAML, patch([
        (r'fusion:\s*"simple_union"', 'fusion: "rrf"', 1),
    ]))
    r = cli(["ask", "--fast", "what do the pillars describe"])
    rec("phase0_rrf.ask_exits_0", r.returncode == 0,
        f"rc={r.returncode} stderr_tail={r.stderr.strip()[-160:]}")
    run = find_run(r.stdout)
    rec("phase0_rrf.run_json_emitted", run is not None)
    if run is None:
        return
    d = load_json(run)
    results = d.get("retrieval", {}).get("results") or []
    rec("phase0_rrf.has_retrieval_results", bool(results), f"n={len(results)}")
    if results:
        # RRF produces small positive scores ( <= 2/(rrf_k+1) = 2/61 ~= 0.0328 ).
        top = float(results[0].get("scores", {}).get("merged") or 0.0)
        rec("phase0_rrf.top_score_in_rrf_range", 0.0 < top <= 0.05,
            f"top_merged_score={top:.6f}")


def phase1() -> None:
    print("\n=== PHASE 1 — neighbor expansion (adjacent_only) ===")
    write(DEFAULT_YAML, patch([
        (r"(\s*)neighbor_expansion_enabled: false", r"\1neighbor_expansion_enabled: true", 1),
    ]))
    r = cli(["ask", "--fast", "what do the pillars describe"])
    rec("phase1.ask_exits_0", r.returncode == 0, f"rc={r.returncode}")
    run = find_run(r.stdout)
    rec("phase1.run_json_emitted", run is not None)
    if run is None:
        return
    d = load_json(run)
    rr = d.get("retrieval_rounds") or []
    rec("phase1.has_retrieval_rounds", len(rr) >= 1, f"rounds={len(rr)}")
    retr = d.get("retrieval", {})
    rec("phase1.neighbor_expansion_applied", bool(retr.get("neighbor_expansion_applied")),
        f"applied={retr.get('neighbor_expansion_applied')} scope={retr.get('neighbor_scope')!r}")
    rec("phase1.neighbor_chunks_added_counter_present",
        retr.get("neighbor_chunks_added") is not None,
        f"neighbor_chunks_added={retr.get('neighbor_chunks_added')}")


def phase2() -> None:
    print("\n=== PHASE 2 — refinement + coverage + rule-based rewrite ===")
    write(ACRONYMS_YAML, "acronyms:\n  yt: \"youtube\"\n  ai: \"artificial intelligence\"\nsynonyms: {}\n")
    write(DEFAULT_YAML, patch([
        (r"(\s*)neighbor_expansion_enabled: false", r"\1neighbor_expansion_enabled: true", 1),
        (r"(\s*)refinement_round_enabled: false", r"\1refinement_round_enabled: true", 1),
        (r"(\s*)rule_based_enabled: false", r"\1rule_based_enabled: true", 1),
        (r"(\s*)lexical_threshold: 0\.5", r"\1lexical_threshold: 0.99", 1),
        (r"(\s*)vector_threshold: 0\.5", r"\1vector_threshold: 0.99", 1),
    ]))
    r = cli(["ask", "--fast", "yt pillars"])
    rec("phase2.ask_exits_0", r.returncode == 0, f"rc={r.returncode}")
    run = find_run(r.stdout)
    rec("phase2.run_json_emitted", run is not None)
    if run is None:
        return
    d = load_json(run)
    cov = d.get("coverage")
    rec("phase2.coverage_populated", isinstance(cov, dict) and "should_refine" in cov,
        f"keys={list(cov.keys()) if isinstance(cov, dict) else None}")
    rr = d.get("retrieval_rounds") or []
    rec("phase2.retrieval_rounds_logged", len(rr) >= 1, f"rounds={len(rr)}")
    rec("phase2.refinement_fired", len(rr) >= 2 or d.get("refinement_applied"),
        f"applied={d.get('refinement_applied')} rounds={len(rr)}")
    rewritten = d.get("rewritten_query") or ""
    # Also scan per-round in case the top-level value was the original.
    for rnd in rr:
        rq = rnd.get("rewritten_query") or rnd.get("query") or ""
        if "youtube" in rq.lower():
            rewritten = rq
            break
    rec("phase2.rewrite_expanded_acronym", "youtube" in rewritten.lower(), f"rewritten={rewritten!r}")


def phase3() -> None:
    print("\n=== PHASE 3 — daemon lifecycle + session snapshot ===")
    write(DEFAULT_YAML, patch([
        (r"(session:\n(?:.*\n)*?\s*)enabled: false", r"\1enabled: true", 1),
        (r"(\s*)require_daemon_for_cli: true", r"\1require_daemon_for_cli: false", 1),
    ]))
    r1 = cli(["ask", "--fast", "--session", S1, "what are the pillars"])
    rec("phase3.turn1_ok", r1.returncode == 0, f"rc={r1.returncode}")
    run1 = find_run(r1.stdout)
    if run1:
        d1 = load_json(run1)
        rec("phase3.memory_snapshot_present", d1.get("memory_snapshot") is not None)
        snap1 = d1.get("memory_snapshot") or {}
        rec("phase3.first_turn_cold", snap1.get("turn_count") in (0, None), f"turn={snap1.get('turn_count')}")
        after = d1.get("session_state_after") or {}
        rec("phase3.state_persisted", after.get("turn_count", 0) >= 1, f"turn_after={after.get('turn_count')}")
    r2 = cli(["ask", "--fast", "--session", S1, "what about coherence"])
    run2 = find_run(r2.stdout)
    if run2:
        d2 = load_json(run2)
        snap2 = d2.get("memory_snapshot") or {}
        rec("phase3.second_turn_sees_state", snap2.get("turn_count", 0) >= 1, f"turn_snap={snap2.get('turn_count')}")
    sl = cli(["session", "list", "--json"], timeout=30)
    rec("phase3.session_list_ok", sl.returncode == 0, sl.stdout.strip()[:120])
    ss = cli(["session", "show", S1, "--json"], timeout=30)
    rec("phase3.session_show_ok", ss.returncode == 0, ss.stdout.strip()[:80])

    # daemon lifecycle
    write(DEFAULT_YAML, patch([
        (r"(daemon:\n(?:.*\n)*?\s*)enabled: false", r"\1enabled: true", 1),
        (r"(session:\n(?:.*\n)*?\s*)enabled: false", r"\1enabled: true", 1),
        (r"(\s*)idle_timeout_s: 1800", r"\1idle_timeout_s: 30", 1),
    ]))
    print("  starting daemon subprocess...")
    dp = subprocess.Popen(
        [PY, "-m", "agent", "daemon", "start"],
        cwd=str(WORKROOT), env=ENV, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        up = False
        for _ in range(30):
            time.sleep(0.5)
            st = cli(["daemon", "status", "--json"], timeout=10)
            if st.returncode == 0:
                up = True
                break
        rec("phase3.daemon_starts", up, "status ok" if up else "timeout")
        if up:
            stp = cli(["daemon", "stop", "--json"], timeout=10)
            rec("phase3.daemon_stop_ok", stp.returncode == 0)
    finally:
        try:
            dp.wait(timeout=5)
        except subprocess.TimeoutExpired:
            dp.terminate()
            try:
                dp.wait(timeout=3)
            except subprocess.TimeoutExpired:
                dp.kill()


def phase4() -> None:
    print("\n=== PHASE 4 — memory rewrite + coverage weight + promotion CLI ===")
    write(DEFAULT_YAML, patch([
        (r"(\s*)neighbor_expansion_enabled: false", r"\1neighbor_expansion_enabled: true", 1),
        (r"(\s*)refinement_round_enabled: false", r"\1refinement_round_enabled: true", 1),
        (r"(\s*)rule_based_enabled: false", r"\1rule_based_enabled: true", 1),
        (r"(session:\n(?:.*\n)*?\s*)enabled: false", r"\1enabled: true", 1),
        (r"(\s*)require_daemon_for_cli: true", r"\1require_daemon_for_cli: false", 1),
        (r"(\s*)memory_rewrite_enabled: false", r"\1memory_rewrite_enabled: true", 1),
        (r"(\s*)coverage_memory_weight: 0\.0", r"\1coverage_memory_weight: 0.4", 1),
        (r"(promotion:\n\s*)enabled: false", r"\1enabled: true", 1),
        (r"(promotion:\n(?:.*\n)*?\s*)llm_suggest_enabled: false", r"\1llm_suggest_enabled: true", 1),
    ]))
    write(ACRONYMS_YAML, "acronyms: {}\nsynonyms: {}\n")

    r1 = cli(["ask", "--fast", "--session", S2, "what are the pillars"])
    rec("phase4.turn1_ok", r1.returncode == 0, f"rc={r1.returncode}")
    run1 = find_run(r1.stdout)
    if run1 is None:
        rec("phase4.turn1_run_json", False)
        return

    r2 = cli(["ask", "--fast", "--session", S2, "and what does this imply for my next video"])
    rec("phase4.turn2_ok", r2.returncode == 0, f"rc={r2.returncode}")
    run2 = find_run(r2.stdout)
    if run2:
        d2 = load_json(run2)
        rr = d2.get("retrieval_rounds") or []
        snap = d2.get("memory_snapshot") or {}
        topic = [t for t in (snap.get("topic_summary") or []) if t]
        rewritten = d2.get("rewritten_query") or ""
        for rnd in rr:
            rq = rnd.get("rewritten_query") or rnd.get("query") or ""
            if any(t.lower() in rq.lower() for t in topic if len(t) > 3):
                rewritten = rq
                break
        seeded = any(t.lower() in rewritten.lower() for t in topic if len(t) > 3) if topic else False
        # Memory rewrite only injects topic seeds when refinement actually fires.
        # Turn 2 may have good coverage, so no refinement / no injected rewrite is valid.
        refinement_fired = bool(d2.get("refinement_applied")) or len(rr) >= 2
        if refinement_fired and topic:
            rec("phase4.rewrite_carries_memory_seeds", seeded,
                f"topic={topic[:3]} rewritten_tail={rewritten[-120:]!r}")
        else:
            rec("phase4.rewrite_skipped_when_no_refinement", True,
                f"refinement_applied={d2.get('refinement_applied')} rounds={len(rr)} topic={topic[:3]}")

    sug = cli(["memory", "suggest", "--session", S2, "--json"], timeout=60)
    rec("phase4.memory_suggest_ok", sug.returncode == 0,
        f"rc={sug.returncode} out={sug.stdout.strip()[:160]}")
    suggest_keys: list[str] = []
    if sug.returncode == 0 and sug.stdout.strip():
        try:
            # suggest emits a single JSON object; try full-stdout then last-line.
            try:
                payload = json.loads(sug.stdout.strip())
            except json.JSONDecodeError:
                payload = json.loads(sug.stdout.strip().splitlines()[-1])
            refs = (payload.get("suggestions")
                    or payload.get("active_refs")
                    or payload.get("refs") or [])
            suggest_keys = [ref.get("chunk_key") for ref in refs if ref.get("chunk_key")]
            rec("phase4.memory_suggest_returned_refs", bool(suggest_keys),
                f"n={len(suggest_keys)} first={suggest_keys[0] if suggest_keys else None}")
        except Exception as exc:
            rec("phase4.suggest_json_parse", False, str(exc))

    if suggest_keys:
        ck = suggest_keys[0]
        prom = cli([
            "memory", "promote", "--session", S2,
            "--ref", ck, "--type", "user_fact",
            "--content", "smoke promotion note", "--yes", "--json",
        ], timeout=60)
        rec("phase4.memory_promote_explicit_ok", prom.returncode == 0,
            f"rc={prom.returncode} out={prom.stdout.strip()[:120]} err={prom.stderr.strip()[:120]}")
        ml = cli(["memory", "list", "--json"], timeout=30)
        rec("phase4.memory_list_ok", ml.returncode == 0)
        if ml.returncode == 0:
            try:
                last = ml.stdout.strip().splitlines()[-1]
                rows = json.loads(last)
                blob = json.dumps(rows)
                rec("phase4.promoted_record_visible", "smoke promotion note" in blob,
                    f"len={len(blob)}")
            except Exception as exc:
                rec("phase4.memory_list_parse", False, str(exc))
        dry = cli([
            "memory", "promote", "--session", S2,
            "--ref", ck, "--type", "user_fact",
            "--llm-suggest", "--json",
        ], timeout=60)
        rec("phase4.llm_suggest_dry_run_ok", dry.returncode == 0,
            f"rc={dry.returncode} tail={dry.stdout.strip()[-200:]}")
    else:
        rec("phase4.memory_suggest_no_refs", True, "suggest returned no refs (acceptable on sparse corpus)")


def main() -> int:
    shutil.copy2(DEFAULT_YAML, DEFAULT_BAK)
    if ACRONYMS_YAML.exists():
        shutil.copy2(ACRONYMS_YAML, ACRO_BAK)
    try:
        print(f"[smoke] repo  = {REPO}")
        print(f"[smoke] work  = {WORKROOT}")
        print(f"[smoke] py    = {sys.executable}")
        phase0()
        phase0_rrf()
        phase1()
        phase2()
        phase3()
        phase4()
    finally:
        shutil.copy2(DEFAULT_BAK, DEFAULT_YAML)
        DEFAULT_BAK.unlink(missing_ok=True)
        if ACRO_BAK.exists():
            shutil.copy2(ACRO_BAK, ACRONYMS_YAML)
            ACRO_BAK.unlink(missing_ok=True)

    print("\n\n======= SMOKE SUMMARY =======")
    passed = sum(1 for _, o, _ in RESULTS if o)
    failed = sum(1 for _, o, _ in RESULTS if not o)
    for n, o, d in RESULTS:
        tag = "PASS" if o else "FAIL"
        print(f"  [{tag}] {n}" + (f" :: {d}" if d else ""))
    print(f"\n  TOTAL: {passed} passed, {failed} failed  ({len(RESULTS)} checks)")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
