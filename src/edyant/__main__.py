"""Top-level edyant CLI with persistence baked in.

Usage:
    python -m edyant run <model>
    python -m edyant prompt --model <model> [--url ...] "your prompt"

Behavior:
- Auto-starts `ollama serve` if not already running (run command).
- Uses MemoryAugmentedAdapter + SqliteMemoryStore to persist every turn.
- Default store: ~/.edyant/persistence/memory.sqlite
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib import request

from edyant.persistence import MemoryAugmentedAdapter, SqliteMemoryStore
from edyant.persistence.adapters import OllamaAdapter
from edyant.persistence.memorygraph.server import GraphConfig, run_memorygraph_server

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 11434
DEFAULT_STORE = Path.home() / ".edyant" / "persistence" / "memory.sqlite"
DEFAULT_GRAPH_PORT = 8787


def _check_ollama(url: str, timeout: float = 1.0) -> bool:
    try:
        req = request.Request(url, method="GET")
        with request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return 200 <= resp.status < 300
    except Exception:
        return False


def _start_ollama(bin_path: str, host: str, port: int, wait_secs: float = 8.0) -> subprocess.Popen:
    try:
        proc = subprocess.Popen(  # noqa: S603
            [bin_path, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except FileNotFoundError:
        raise SystemExit(f"ollama binary not found at '{bin_path}'. Install ollama or point --ollama-bin.")
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
        prompt = args.prompt or sys.stdin.read()
        if not prompt:
            raise SystemExit("No prompt provided (arg or stdin).")
        output = adapter.generate(prompt)
        sys.stdout.write(output.text)
        if not output.text.endswith("\n"):
            sys.stdout.write("\n")
    finally:
        adapter.close()


def _handle_run(args: argparse.Namespace) -> None:
    base_url = f"http://{args.host}:{args.port}/api/version"
    started_proc = None
    if not _check_ollama(base_url, timeout=1.0):
        started_proc = _start_ollama(args.ollama_bin, args.host, args.port, wait_secs=args.serve_timeout)

    if args.model is None:
        raise SystemExit("Model is required (e.g., `python -m edyant run llama3`).")

    args.url = args.url or f"http://{args.host}:{args.port}/api/generate"
    adapter = _build_adapter(args)

    print(f"[edyant] Using store={args.store}")
    print(f"[edyant] Ollama endpoint={args.url}")
    print("[edyant] Enter prompts (Ctrl+C or /exit to quit)")

    try:
        while True:
            try:
                prompt = input(">>> ").strip()
            except KeyboardInterrupt:
                print()
                break
            if not prompt:
                continue
            if prompt in {"/exit", "/quit"}:
                break
            try:
                output = adapter.generate(prompt)
                print(output.text)
            except Exception as exc:  # noqa: BLE001
                print(f"[edyant] error: {exc}", file=sys.stderr)
    finally:
        adapter.close()
        if started_proc:
            started_proc.terminate()

def _handle_memorygraph(args: argparse.Namespace) -> None:
    cfg = GraphConfig(
        store=args.store,
        host=args.host,
        port=args.port,
        max_edges=args.max_edges,
        open_browser=args.open_browser,
    )
    print(f"[edyant] memorygraph serving {cfg.store} at http://{cfg.host}:{cfg.port}/")
    print(f"[edyant] Press Ctrl+C to stop. ")
    if cfg.open_browser:
        print("[edyant] opening browser...")
    run_memorygraph_server(cfg)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="edyant")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--store", type=Path, default=DEFAULT_STORE, help="Path to SQLite store file")
        p.add_argument("--context-k", type=int, default=5, help="Number of memory hits to inject")
        p.add_argument("--timeout", type=float, default=60.0, help="Request timeout (seconds)")
        p.add_argument("--max-retries", type=int, default=3, help="Retry attempts for model calls")
        p.add_argument("--retry-sleep", type=float, default=2.0, help="Seconds between retries")

    p_run = sub.add_parser("run", help="Interactive REPL with persistence; auto-starts ollama serve")
    common(p_run)
    p_run.add_argument("model", nargs="?", help="Model name (e.g., llama3, qwen2.5:3b)")
    p_run.add_argument("--url", required=False, default=None, help="Override Ollama API URL")
    p_run.add_argument("--host", default=DEFAULT_HOST, help="Ollama host for health check")
    p_run.add_argument("--port", type=int, default=DEFAULT_PORT, help="Ollama port for health check")
    p_run.add_argument("--ollama-bin", default="ollama", help="Path to ollama binary")
    p_run.add_argument("--serve-timeout", type=float, default=8.0, help="Seconds to wait for ollama serve")
    p_run.set_defaults(func=_handle_run)

    p_prompt = sub.add_parser("prompt", help="Single-shot prompt with persistence (no REPL)")
    common(p_prompt)
    p_prompt.add_argument("--model", required=False, default=None, help="Model name (or OLLAMA_MODEL env)")
    p_prompt.add_argument("--url", required=False, default=None, help="Ollama API URL (or OLLAMA_API_URL env)")
    p_prompt.add_argument("prompt", nargs="?", help="Prompt text or stdin")
    p_prompt.set_defaults(func=_handle_prompt)

    p_graph = sub.add_parser("memorygraph", help="Launch a graph viz UI for the persistence store")
    p_graph.add_argument("--store", type=Path, default=DEFAULT_STORE, help="Path to SQLite store file")
    p_graph.add_argument("--host", default="127.0.0.1", help="Bind host for the viewer")
    p_graph.add_argument("--port", type=int, default=DEFAULT_GRAPH_PORT, help="Bind port for the viewer")
    p_graph.add_argument("--max-edges", type=int, default=500, help="Max edges in the initial summary")
    p_graph.add_argument("--open-browser", action="store_true", help="Open browser automatically")
    p_graph.set_defaults(func=_handle_memorygraph)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)

    if args.cmd == "prompt":
        if args.model is None:
            from os import getenv

            args.model = getenv("OLLAMA_MODEL")
            if not args.model:
                raise SystemExit("Model is required (use --model or set OLLAMA_MODEL).")
        if args.url is None:
            from os import getenv

            args.url = getenv("OLLAMA_API_URL")
            if not args.url:
                raise SystemExit("Ollama URL is required (use --url or set OLLAMA_API_URL).")

    args.func(args)


if __name__ == "__main__":
    main()
