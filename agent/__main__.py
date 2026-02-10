# agent/__main__.py
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import yaml

from agent.protocol import try_parse_tool_call
from agent.tools import TOOLS, ToolError


# -----------------------------
# Config + IO
# -----------------------------

def load_config() -> Dict[str, Any]:
    cfg_path = Path("configs") / "default.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def make_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def now_unix() -> float:
    return time.time()


def strip_thinking(resp: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Ollama responses sometimes include message.thinking. Keep logs lean.
    """
    if resp is None:
        return None
    out = dict(resp)
    msg = out.get("message")
    if isinstance(msg, dict) and "thinking" in msg:
        msg2 = dict(msg)
        msg2.pop("thinking", None)
        out["message"] = msg2
    return out


# -----------------------------
# Ollama client
# -----------------------------

def ensure_ollama_up(base_url: str) -> None:
    r = requests.get(f"{base_url}/api/tags", timeout=5)
    r.raise_for_status()


def ollama_chat(
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout_s: int = 300,
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
    return r.json()


def get_assistant_text(resp: Dict[str, Any]) -> str:
    msg = resp.get("message") or {}
    return (msg.get("content") or "").strip()


# -----------------------------
# Agent behaviors
# -----------------------------

def build_tool_system_prompt() -> str:
    """
    Tool calling protocol (strict + parseable):
    If a tool is needed, respond with EXACTLY ONE LINE of JSON, and nothing else:
    {"type":"tool_call","name":"<tool_name>","args":{...}}
    Otherwise answer normally.
    """
    tools_payload = [
        {
            "name": spec.name,
            "description": spec.description,
            "args_schema": spec.args_schema,
        }
        for spec in TOOLS.values()
    ]

    return (
        "You are a local agent running on the user's computer.\n"
        "If you need to use a tool, you MUST respond with exactly one line of JSON and nothing else:\n"
        '{"type":"tool_call","name":"<tool_name>","args":{...}}\n'
        "If you do NOT need a tool, respond normally.\n"
        "Rules:\n"
        "- Do not invent tool results.\n"
        "- If the user asks you to read a local file, you must call read_text_file.\n"
        "- Keep tool args minimal and valid.\n"
        "Available tools:\n"
        f"{json.dumps(tools_payload, indent=2, ensure_ascii=False)}\n"
    )


def run_chat(cfg: Dict[str, Any], prompt: str) -> Dict[str, Any]:
    base_url = cfg["ollama_base_url"]
    model = cfg["model"]

    ensure_ollama_up(base_url)

    started = now_unix()
    resp = ollama_chat(
        base_url=base_url,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=cfg.get("temperature", 0.2),
        max_tokens=cfg.get("max_tokens", 800),
        timeout_s=cfg.get("timeout_s", 300),
    )
    ended = now_unix()

    assistant_text = get_assistant_text(resp)
    run_id = make_run_id()
    run_dir = Path("runs") / run_id

    record = {
        "run_id": run_id,
        "mode": "chat",
        "model": model,
        "prompt": prompt,
        "started_unix": started,
        "ended_unix": ended,
        "elapsed_s": round(ended - started, 3),
        "assistant_text": assistant_text,
        "raw_response": strip_thinking(resp),
    }
    write_json(run_dir / "run.json", record)

    print(assistant_text)
    print(f"\n[logged] {run_dir / 'run.json'}")
    return record


def run_ask_one_tool(cfg: Dict[str, Any], question: str) -> Dict[str, Any]:
    """
    Agent loop with at most ONE tool call.
    """
    base_url = cfg["ollama_base_url"]
    model = cfg["model"]

    ensure_ollama_up(base_url)

    system_prompt = build_tool_system_prompt()
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    started = now_unix()

    first = ollama_chat(
        base_url=base_url,
        model=model,
        messages=messages,
        temperature=cfg.get("temperature", 0.0),  # tool-calling prefers deterministic
        max_tokens=cfg.get("max_tokens", 800),
        timeout_s=cfg.get("timeout_s", 300),
    )
    first_text = get_assistant_text(first)

    tool_trace: List[Dict[str, Any]] = []
    final_text = first_text
    second: Optional[Dict[str, Any]] = None

    tool_call = try_parse_tool_call(first_text)
    if tool_call is not None:
        spec = TOOLS.get(tool_call.name)
        if spec is None:
            tool_result = {"error": f"Tool not found: {tool_call.name}"}
        else:
            try:
                tool_result = spec.func(tool_call.args)
            except ToolError as e:
                tool_result = {"error": str(e)}

        tool_trace.append(
            {"call": {"name": tool_call.name, "args": tool_call.args}, "result": tool_result}
        )

        # Feed the tool result back into the model for a final answer.
        messages.append({"role": "assistant", "content": first_text})
        messages.append({"role": "tool", "content": json.dumps(tool_result, ensure_ascii=False)})

        second = ollama_chat(
            base_url=base_url,
            model=model,
            messages=messages,
            temperature=cfg.get("temperature", 0.2),
            max_tokens=cfg.get("max_tokens", 800),
            timeout_s=cfg.get("timeout_s", 300),
        )
        final_text = get_assistant_text(second)

    ended = now_unix()
    run_id = make_run_id()
    run_dir = Path("runs") / run_id

    record = {
        "run_id": run_id,
        "mode": "ask",
        "model": model,
        "question": question,
        "started_unix": started,
        "ended_unix": ended,
        "elapsed_s": round(ended - started, 3),
        "tool_trace": tool_trace,
        "assistant_text": final_text,
        "raw_first": strip_thinking(first),
        "raw_second": strip_thinking(second),
    }
    write_json(run_dir / "run.json", record)

    print(final_text)
    print(f"\n[logged] {run_dir / 'run.json'}")
    return record


# -----------------------------
# CLI
# -----------------------------

def main() -> int:
    parser = argparse.ArgumentParser(prog="agent")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_chat = sub.add_parser("chat", help="Send a single prompt and log the run.")
    p_chat.add_argument("prompt", type=str)

    p_ask = sub.add_parser("ask", help="Agent loop with up to 1 tool call.")
    p_ask.add_argument("question", type=str)

    args = parser.parse_args()
    cfg = load_config()

    # Minimal config validation
    for k in ("model", "ollama_base_url"):
        if k not in cfg:
            raise SystemExit(f"Missing config key: {k} in configs/default.yaml")

    if args.cmd == "chat":
        run_chat(cfg, args.prompt)
        return 0

    if args.cmd == "ask":
        run_ask_one_tool(cfg, args.question)
        return 0

    raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
