from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import yaml

from agent.protocol import ToolCall, try_parse_tool_call
from agent.tools import TOOLS, ToolError


DEFAULT_CONFIG: Dict[str, Any] = {
    "model": "gpt-oss:120b",
    "ollama_base_url": "http://127.0.0.1:11434",
    "max_tokens": 800,
    "temperature": 0.2,
    "timeout_s": 300,
    "prefer_fast": True,
    "big_triggers": ["deep", "long", "essay", "synthesize", "thorough", "in depth"],
}


def load_config() -> Dict[str, Any]:
    cfg_path = Path("configs") / "default.yaml"
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("configs/default.yaml must contain a mapping/object")
    return data


def make_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def make_run_dir() -> Path:
    base = Path("runs")
    run_id = make_run_id()
    run_dir = base / run_id
    if not run_dir.exists():
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    # Rare collision if two runs start in the same second.
    for i in range(1, 1000):
        candidate = base / f"{run_id}_{i:03d}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
    raise RuntimeError("Unable to allocate a unique run directory")


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def now_unix() -> float:
    return time.time()


def print_output(text: str) -> None:
    """Print model output safely on Windows consoles with limited code pages."""
    try:
        print(text)
    except UnicodeEncodeError:
        sys.stdout.buffer.write((text + "\n").encode("utf-8", errors="replace"))
        sys.stdout.flush()


def strip_thinking(resp: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Remove message.thinking from Ollama responses before writing logs."""
    if resp is None:
        return None
    out = dict(resp)
    message = out.get("message")
    if isinstance(message, dict) and "thinking" in message:
        msg = dict(message)
        msg.pop("thinking", None)
        out["message"] = msg
    return out


def redact_tool_result_for_log(tool_result: Dict[str, Any], preview_chars: int = 800) -> Dict[str, Any]:
    """
    For file-read tools, keep only metadata + a small preview in run logs.
    """
    if not isinstance(tool_result, dict):
        return {"error": f"Unexpected non-dict tool result: {type(tool_result).__name__}"}

    text = tool_result.get("text")
    if isinstance(text, str):
        return {
            "path": tool_result.get("path"),
            "sha256": tool_result.get("sha256"),
            "chars_full": tool_result.get("chars_full"),
            "chars_returned": tool_result.get("chars_returned"),
            "truncated": tool_result.get("truncated"),
            "text_preview": text[:preview_chars],
        }
    return dict(tool_result)


def ensure_ollama_up(base_url: str, timeout_s: int) -> None:
    r = requests.get(f"{base_url}/api/tags", timeout=timeout_s)
    r.raise_for_status()


def ollama_chat(
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout_s: int,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": float(temperature),
            "num_predict": int(max_tokens),
        },
    }
    r = requests.post(f"{base_url}/api/chat", json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, dict):
        raise ValueError("Unexpected Ollama response type; expected JSON object")
    return data


def get_assistant_text(resp: Dict[str, Any]) -> str:
    message = resp.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if not isinstance(content, str):
        return ""
    return content.strip()


def _as_model_name(value: Any) -> Optional[str]:
    if isinstance(value, str):
        model = value.strip()
        if model:
            return model
    return None


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "on"}:
            return True
        if v in {"0", "false", "no", "off"}:
            return False
    return default


def get_default_model_for_logs(cfg: Dict[str, Any]) -> str:
    return _as_model_name(cfg.get("model_big")) or _as_model_name(cfg.get("model")) or DEFAULT_CONFIG["model"]


def select_models(
    cfg: Dict[str, Any],
    question: str,
    force_big_second: bool = False,
    force_fast: bool = False,
) -> Tuple[str, str]:
    """
    Select model for ask state-1 (tool selection) and state-2 (final response).
    """
    base_model = _as_model_name(cfg.get("model")) or DEFAULT_CONFIG["model"]
    model_fast = _as_model_name(cfg.get("model_fast"))
    model_big = _as_model_name(cfg.get("model_big"))
    prefer_fast = _as_bool(cfg.get("prefer_fast"), True)

    raw_triggers = cfg.get("big_triggers")
    if isinstance(raw_triggers, list):
        big_triggers = [str(x).strip().lower() for x in raw_triggers if str(x).strip()]
    else:
        big_triggers = [str(x).lower() for x in DEFAULT_CONFIG["big_triggers"]]

    q_lower = question.lower()
    wants_big_second_by_trigger = any(trigger in q_lower for trigger in big_triggers)
    has_split_models = model_fast is not None or model_big is not None

    if not has_split_models:
        first_model = base_model
        second_model = base_model
    elif prefer_fast:
        first_model = model_fast or model_big or base_model
        second_model = model_fast or model_big or base_model
        if wants_big_second_by_trigger and model_big:
            second_model = model_big
    else:
        if model_big:
            first_model = model_big
            second_model = model_big
        elif model_fast:
            first_model = model_fast
            second_model = model_fast
        else:
            first_model = base_model
            second_model = base_model

    if force_fast:
        forced_fast = model_fast or base_model or model_big or DEFAULT_CONFIG["model"]
        first_model = forced_fast
        second_model = forced_fast
    elif force_big_second:
        second_model = model_big or second_model or first_model or base_model

    return first_model, second_model


def evidence_required_by_question(question: str) -> bool:
    q = question.strip().lower()
    if not q:
        return False

    if re.search(r"\.(md|txt|json|yaml|yml)\b", q):
        return True
    if re.search(r"\bread\s+\S+", q):
        return True
    if re.search(r"\bsummar(?:ize|ise|y)\b", q) and re.search(r"\S+\.\w+\b", q):
        return True
    if re.search(r"\bwhat does\s+.+\s+say\b", q) and re.search(r"\S+\.\w+\b", q):
        return True
    return False


def requires_nonempty_file_content(question: str) -> bool:
    q = question.lower()
    return bool(re.search(r"\bsummar(?:ize|ise|y)\b|\bsummary\b", q))


def validate_read_text_file_evidence(
    tool_result: Dict[str, Any],
    require_nonempty: bool,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str], str]:
    """
    Validate read_text_file evidence contract.
    Returns (evidence, error_code, error_message, evidence_status).
    """
    if not isinstance(tool_result, dict):
        return None, "EVIDENCE_INVALID", "Tool returned a non-object result.", "invalid"
    if "error" in tool_result:
        err = tool_result.get("error")
        msg = f"Tool failed: {err}" if isinstance(err, str) else "Tool failed."
        return None, "TOOL_ERROR", msg + " Confirm the file path and permissions.", "missing"

    required_keys = ("path", "sha256", "chars_full", "chars_returned", "truncated", "text")
    missing = [k for k in required_keys if k not in tool_result]
    if missing:
        return (
            None,
            "EVIDENCE_INVALID",
            f"Tool result missing required fields: {', '.join(missing)}.",
            "invalid",
        )

    path = tool_result.get("path")
    sha256 = tool_result.get("sha256")
    chars_full = tool_result.get("chars_full")
    chars_returned = tool_result.get("chars_returned")
    truncated = tool_result.get("truncated")
    text = tool_result.get("text")

    if not isinstance(path, str) or not path:
        return None, "EVIDENCE_INVALID", "Evidence field 'path' must be a non-empty string.", "invalid"
    if not Path(path).is_absolute():
        return None, "EVIDENCE_INVALID", "Evidence field 'path' must be absolute.", "invalid"
    if not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", sha256):
        return None, "EVIDENCE_INVALID", "Evidence field 'sha256' must be a lowercase 64-char hex string.", "invalid"
    if not isinstance(chars_full, int) or chars_full < 0:
        return None, "EVIDENCE_INVALID", "Evidence field 'chars_full' must be an int >= 0.", "invalid"
    if not isinstance(chars_returned, int) or chars_returned < 0:
        return None, "EVIDENCE_INVALID", "Evidence field 'chars_returned' must be an int >= 0.", "invalid"
    if chars_returned > chars_full:
        return None, "EVIDENCE_INVALID", "Evidence field 'chars_returned' cannot exceed 'chars_full'.", "invalid"
    if not isinstance(truncated, bool):
        return None, "EVIDENCE_INVALID", "Evidence field 'truncated' must be a bool.", "invalid"
    if not isinstance(text, str):
        return None, "EVIDENCE_INVALID", "Evidence field 'text' must be a string.", "invalid"
    if len(text) != chars_returned:
        return None, "EVIDENCE_INVALID", "Evidence text length does not match 'chars_returned'.", "invalid"
    if truncated and chars_returned >= chars_full and chars_full > 0:
        return None, "EVIDENCE_INVALID", "Evidence field 'truncated' is inconsistent with char counts.", "invalid"
    if (not truncated) and chars_returned != chars_full:
        return None, "EVIDENCE_INVALID", "Evidence field 'truncated' is inconsistent with char counts.", "invalid"
    if require_nonempty and chars_full == 0:
        return None, "FILE_EMPTY", "The target file is empty; cannot summarize empty content.", "invalid"

    return tool_result, None, None, "valid"


def make_typed_failure(error_code: str, error_message: str) -> str:
    payload = {
        "ok": False,
        "error_code": error_code,
        "error_message": error_message,
    }
    return json.dumps(payload, ensure_ascii=False)


def build_tool_system_prompt() -> str:
    tools_payload = [
        {
            "name": spec.name,
            "description": spec.description,
            "args_schema": spec.args_schema,
        }
        for spec in TOOLS.values()
    ]
    tools_json = json.dumps(tools_payload, ensure_ascii=False, indent=2)
    return (
        "You are a local CLI agent.\n"
        "When you need a tool, respond with exactly one JSON object and nothing else.\n"
        'Tool-call envelope: {"type":"tool_call","name":"<tool_name>","args":{...}}\n'
        "If you output a tool_call JSON, output NOTHING ELSE. No commentary. No markdown.\n"
        "If you violate this, the system will ignore your output.\n"
        "If no tool is needed, answer normally.\n"
        "If the user asks you to read a local file, you must call read_text_file.\n"
        "Use valid, minimal args.\n"
        f"Available tools:\n{tools_json}"
    )


def run_chat(cfg: Dict[str, Any], prompt: str) -> int:
    run_dir = make_run_dir()
    run_id = run_dir.name
    started = now_unix()
    record: Dict[str, Any] = {
        "run_id": run_id,
        "mode": "chat",
        "prompt": prompt,
        "model": cfg["model"],
        "started_unix": started,
    }

    try:
        ensure_ollama_up(cfg["ollama_base_url"], timeout_s=cfg["timeout_s"])
        resp = ollama_chat(
            base_url=cfg["ollama_base_url"],
            model=cfg["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=cfg["temperature"],
            max_tokens=cfg["max_tokens"],
            timeout_s=cfg["timeout_s"],
        )
        assistant_text = get_assistant_text(resp)
        print_output(assistant_text)

        record["assistant_text"] = assistant_text
        record["raw_response"] = strip_thinking(resp)
        record["ok"] = True
        return_code = 0
    except Exception as exc:
        record["ok"] = False
        record["error"] = str(exc)
        print(f"error: {exc}", file=sys.stderr)
        return_code = 1
    finally:
        ended = now_unix()
        record["ended_unix"] = ended
        record["elapsed_s"] = round(ended - started, 3)
        write_json(run_dir / "run.json", record)
        print(f"\n[logged] {run_dir / 'run.json'}")

    return return_code


def run_ask_one_tool(
    cfg: Dict[str, Any],
    question: str,
    force_big_second: bool = False,
    force_fast: bool = False,
) -> int:
    run_dir = make_run_dir()
    run_id = run_dir.name
    started = now_unix()
    first_model, second_model = select_models(
        cfg=cfg,
        question=question,
        force_big_second=force_big_second,
        force_fast=force_fast,
    )
    record: Dict[str, Any] = {
        "run_id": run_id,
        "mode": "ask",
        "question": question,
        "model": get_default_model_for_logs(cfg),
        "raw_first_model": first_model,
        "raw_second_model": second_model,
        "started_unix": started,
        "tool_trace": [],
        "evidence_required": evidence_required_by_question(question),
        "evidence_status": "missing",
    }

    try:
        ensure_ollama_up(cfg["ollama_base_url"], timeout_s=cfg["timeout_s"])

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": build_tool_system_prompt()},
            {"role": "user", "content": question},
        ]

        first = ollama_chat(
            base_url=cfg["ollama_base_url"],
            model=first_model,
            messages=messages,
            temperature=cfg["temperature"],
            max_tokens=cfg["max_tokens"],
            timeout_s=cfg["timeout_s"],
        )
        first_text = get_assistant_text(first)
        record["raw_first"] = strip_thinking(first)

        final_text = first_text
        second: Optional[Dict[str, Any]] = None

        parsed = try_parse_tool_call(first_text)
        if parsed is not None:
            # If model emitted a tool call, evidence is required before completion.
            record["evidence_required"] = True

            tool_call: ToolCall = parsed.tool_call
            tool = TOOLS.get(tool_call.name)
            if tool is None:
                tool_result: Dict[str, Any] = {"error": f"Tool not found: {tool_call.name}"}
            else:
                # Keep tool payload bounded unless the model explicitly requested otherwise.
                if tool_call.name == "read_text_file" and "max_chars" not in tool_call.args:
                    tool_call.args["max_chars"] = 12000
                try:
                    tool_result = tool.func(tool_call.args)
                except ToolError as exc:
                    tool_result = {"error": str(exc)}
                except Exception as exc:  # defensive: tool crash must be treated as tool failure
                    tool_result = {"error": f"Unhandled tool exception: {exc}"}

            trace_item: Dict[str, Any] = {
                "call": {"name": tool_call.name, "args": tool_call.args},
                "result": redact_tool_result_for_log(tool_result),
            }
            if parsed.trailing_text:
                trace_item["trailing_text_ignored_preview"] = parsed.trailing_text[:200]
            record["tool_trace"].append(trace_item)

            require_nonempty = requires_nonempty_file_content(question)
            if tool_call.name == "read_text_file":
                _, error_code, error_message, evidence_status = validate_read_text_file_evidence(
                    tool_result=tool_result,
                    require_nonempty=require_nonempty,
                )
            else:
                if isinstance(tool_result, dict) and "error" in tool_result:
                    error_code = "TOOL_ERROR"
                    err = tool_result.get("error")
                    details = err if isinstance(err, str) else "Tool failed."
                    error_message = f"Tool failed: {details}"
                    evidence_status = "missing"
                else:
                    error_code = "EVIDENCE_INVALID"
                    error_message = (
                        f"Tool '{tool_call.name}' does not provide admissible file evidence for this request."
                    )
                    evidence_status = "invalid"

            if error_code is not None:
                record["evidence_status"] = evidence_status
                record["ok"] = False
                record["error_code"] = error_code
                record["error_message"] = error_message
                final_text = make_typed_failure(error_code=error_code, error_message=error_message or "Unknown error.")
                print_output(final_text)
                record["assistant_text"] = final_text
                record["raw_second"] = None
                return_code = 1
                return return_code

            record["evidence_status"] = "valid"
            canonical_tool_call = json.dumps(
                {"type": "tool_call", "name": tool_call.name, "args": tool_call.args},
                ensure_ascii=False,
            )
            messages.append({"role": "assistant", "content": canonical_tool_call})
            messages.append({"role": "tool", "content": json.dumps(tool_result, ensure_ascii=False)})

            second = ollama_chat(
                base_url=cfg["ollama_base_url"],
                model=second_model,
                messages=messages,
                temperature=cfg["temperature"],
                max_tokens=cfg["max_tokens"],
                timeout_s=cfg["timeout_s"],
            )
            final_text = get_assistant_text(second)
        elif record["evidence_required"]:
            error_code = "EVIDENCE_NOT_ACQUIRED"
            error_message = (
                "Evidence is required for this question, but no admissible read_text_file tool call was acquired. "
                "Re-run and ensure the model calls read_text_file with the target path first."
            )
            record["ok"] = False
            record["error_code"] = error_code
            record["error_message"] = error_message
            record["evidence_status"] = "missing"
            final_text = make_typed_failure(error_code=error_code, error_message=error_message)
            print_output(final_text)
            record["assistant_text"] = final_text
            record["raw_second"] = None
            return_code = 1
            return return_code

        print_output(final_text)

        record["assistant_text"] = final_text
        record["raw_second"] = strip_thinking(second)
        record["ok"] = True
        return_code = 0
    except Exception as exc:
        record["ok"] = False
        record["error_code"] = "INTERNAL_ERROR"
        record["error_message"] = str(exc)
        record["evidence_status"] = "invalid"
        print(f"error: {exc}", file=sys.stderr)
        return_code = 1
    finally:
        ended = now_unix()
        record["ended_unix"] = ended
        record["elapsed_s"] = round(ended - started, 3)
        write_json(run_dir / "run.json", record)
        print(f"\n[logged] {run_dir / 'run.json'}")

    return return_code


def main() -> int:
    parser = argparse.ArgumentParser(prog="agent")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_chat = sub.add_parser("chat", help="Send a single prompt.")
    p_chat.add_argument("prompt", type=str)

    p_ask = sub.add_parser("ask", help="Ask with at most one tool call.")
    p_ask.add_argument("question", type=str)
    ask_speed_group = p_ask.add_mutually_exclusive_group()
    ask_speed_group.add_argument("--big", action="store_true", help="Force big model for the second ask call.")
    ask_speed_group.add_argument("--fast", action="store_true", help="Force fast model for both ask calls.")

    args = parser.parse_args()
    cfg = {**DEFAULT_CONFIG, **load_config()}

    if args.cmd == "chat":
        return run_chat(cfg, args.prompt)
    if args.cmd == "ask":
        return run_ask_one_tool(
            cfg,
            args.question,
            force_big_second=bool(getattr(args, "big", False)),
            force_fast=bool(getattr(args, "fast", False)),
        )
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
