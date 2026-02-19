"""Persistence CLI: interactive run and single-shot prompt with memory augmentation.

Goals:
- Feel like `ollama run <model>` but with persistent memory baked in.
- Auto-start `ollama serve` locally if it's not already running.
- Store every prompt/response in the configured SQLite memory store.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib import error, request

from edyant.persistence import (
    MemoryAugmentedAdapter,
    SqliteMemoryStore,
    default_data_dir,
)
from edyant.persistence.adapters import OllamaAdapter


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 11434


def _read_prompt_arg(prompt_arg: str | None) -> str:
    if prompt_arg and prompt_arg != "-":
        return prompt_arg
    data = sys.stdin.read()
    if not data:
        raise SystemExit("No prompt provided (pass as argument or pipe via stdin).")
    return data


def _check_ollama(url: str, timeout: float = 2.0) -> bool:
    try:
        req = request.Request(url, method="GET")
        with request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


def _start_ollama(bin_path: str, host: str, port: int, wait_secs: float = 5.0) -> subprocess.Popen:
    proc = subprocess.Popen(  # noqa: S603
        [bin_path, "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    base_url = f"http://{host}:{port}/api/version"
    deadline = time.time() + wait_secs
    while time.time() < deadline:
        if _check_ollama(base_url, timeout=0.5):
            return proc
        time.sleep(0.2)
    proc.terminate()
    raise SystemExit("Failed to start ollama serve within timeout. Is ollama installed?")


def _build_adapter(args: argparse.Namespace) -> MemoryAugmentedAdapter:
    base = OllamaAdapter(
        model=args.model,
        url=args.url,
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_sleep=args.retry_sleep,
    )
    store = SqliteMemoryStore(args.store)
    return MemoryAugmentedAdapter(base, store, context_k=args.context_k)


def _handle_prompt(args: argparse.Namespace) -> None:
    adapter = _build_adapter(args)
    try:
        prompt = _read_prompt_arg(args.prompt)
        output = adapter.generate(prompt)
        sys.stdout.write(output.text)
        if not output.text.endswith("\n"):
            sys.stdout.write("\n")
    finally:
        adapter.close()


def _handle_run(args: argparse.Namespace) -> None:
    base_url = f"http://{args.host}:{args.port}/api/version"
    started_proc: subprocess.Popen | None = None
    if not _check_ollama(base_url, timeout=1.0):
        started_proc = _start_ollama(args.ollama_bin, args.host, args.port, wait_secs=args.serve_timeout)

    # Ensure model name
    if args.model is None:
        raise SystemExit("Model is required (e.g., `edyant run llama3`).")

    # Point adapter at the local ollama HTTP endpoint
    args.url = args.url or f"http://{args.host}:{args.port}/api/generate"
    adapter = _build_adapter(args)

    print(f"[edyant] Connected to ollama at {args.url}, store={args.store}")
    print("[edyant] Type your prompt. Ctrl+C or /exit to quit.")
    try:
        while True:
            try:
                prompt = input(">>> ").strip()
            except KeyboardInterrupt:
                print()  # newline
                break
            if not prompt:
                continue
            if prompt in {"/exit", "/quit"}:
                break
            output = adapter.generate(prompt)
            print(output.text)
    finally:
        adapter.close()
        if started_proc:
            started_proc.terminate()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="edyant.persistence")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common_adapter_args(p: argparse.ArgumentParser) -> None:
        default_store = default_data_dir() / "graph.sqlite"
        p.add_argument("--store", type=Path, default=default_store, help="Path to SQLite store file")
        p.add_argument("--context-k", type=int, default=5, help="Number of memory hits to inject")
        p.add_argument("--timeout", type=float, default=60.0, help="Request timeout (seconds)")
        p.add_argument("--max-retries", type=int, default=3, help="Retry attempts for model calls")
        p.add_argument("--retry-sleep", type=float, default=2.0, help="Seconds between retries")

    p_prompt = sub.add_parser("prompt", help="Single-shot prompt with persistence")
    common_adapter_args(p_prompt)
    p_prompt.add_argument("--model", required=False, default=None, help="Model name (defaults to OLLAMA_MODEL env)")
    p_prompt.add_argument("--url", required=False, default=None, help="Ollama API URL (defaults to OLLAMA_API_URL env)")
    p_prompt.add_argument("prompt", nargs="?", help="Prompt text or '-' for stdin")
    p_prompt.set_defaults(func=_handle_prompt)

    p_run = sub.add_parser("run", help="Interactive REPL that auto-starts ollama serve and persists memory")
    common_adapter_args(p_run)
    p_run.add_argument("model", nargs="?", help="Model name (e.g., llama3)")
    p_run.add_argument("--url", required=False, default=None, help="Override Ollama API URL")
    p_run.add_argument("--host", default=DEFAULT_HOST, help="Ollama host (for auto-serve check)")
    p_run.add_argument("--port", type=int, default=DEFAULT_PORT, help="Ollama port (for auto-serve check)")
    p_run.add_argument("--ollama-bin", default="ollama", help="Path to ollama binary")
    p_run.add_argument("--serve-timeout", type=float, default=8.0, help="Seconds to wait for ollama serve to become healthy")
    p_run.set_defaults(func=_handle_run)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)

    # env fallbacks for prompt command
    if args.cmd == "prompt":
        if args.model is None:
            from os import getenv

            env_model = getenv("OLLAMA_MODEL")
            if not env_model:
                raise SystemExit("Model is required (use --model or set OLLAMA_MODEL).")
            args.model = env_model
        if args.url is None:
            from os import getenv

            env_url = getenv("OLLAMA_API_URL")
            if not env_url:
                raise SystemExit("Ollama URL is required (use --url or set OLLAMA_API_URL).")
            args.url = env_url

    args.func(args)


if __name__ == "__main__":
    main()
