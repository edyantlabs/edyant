# edyant Persistence (Implementation Guide)

Semantic memory layer for LLM systems that keeps working context, learns from outcomes, and reconnects relevant history on every interaction. This is the operational companion to `docs/about_persistence.md`.

## Scope
- Turn stateless LLM calls into continuity-aware interactions (workflows, preferences, incidents, successes/failures).
- Provide a portable, pluggable storage layer (SQLite by default; caller controls path/volume).
- Remain independent from benchmarking: no imports from `src/edyant/benchmark/*`.

## Package map
- `src/edyant/persistence/api.py`: public interfaces (`MemoryStore`, `MemoryHit`, `Episode`, `NullMemoryStore`).
- `src/edyant/persistence/types.py`: model IO type used by persistence (`ModelOutput`).
- `src/edyant/persistence/adapters/`: provider adapters + registry (`ModelAdapter`, `register_adapter`, `create_adapter`, `OllamaAdapter`, `OllamaJudgeAdapter`).
- `src/edyant/persistence/memory_adapter.py`: `MemoryAugmentedAdapter` that wraps any `ModelAdapter` with retrieve/store hooks.
- `src/edyant/persistence/storage/sqlite_store.py`: SQLite-backed `MemoryStore` with lightweight spreading activation.
- `src/edyant/persistence/config.py`: `default_data_dir()` resolver (`EDYANT_DATA_DIR` → `XDG_DATA_HOME` → `~/.local/share/edyant/persistence`).
- `src/edyant/persistence/__init__.py`: exports all of the above for consumers.

## Storage responsibility
- The framework never writes inside the repo. Callers choose the path/volume:
  ```python
  from edyant.persistence import SqliteMemoryStore, default_data_dir
  store = SqliteMemoryStore(default_data_dir() / "graph.sqlite")
  ```
- Tests: use tempdirs or `NullMemoryStore`.

## Data model (SQLite backend)
- **nodes**: `id`, `prompt`, `response`, `created_at`, `metadata` (JSON).
- **edges**: `source`, `target`, `weight` (accumulating, primary key on pair).
- **Episode** (in code): prompt/response + metadata; **MemoryHit**: retrieved text + score.

## Retrieval (lightweight spreading activation)
1) Token-overlap similarity between query and recent nodes (bounded candidate set).
2) Edge-weight boosts from the top base hits to their neighbors.
3) Merge, score-sort, return top_k `MemoryHit`s.
   - Default formatter prepends a “Context (from memory)” block before the user prompt.

## Write path
- `record_episode(prompt, output, metadata) -> node_id`
- `update_edges(source_id, related_ids, weight=1.0)` to strengthen associations (called automatically by `MemoryAugmentedAdapter` for retrieved hits).

## CLI usage (ollama-style wrapper)

Interactive REPL that auto-starts `ollama serve` if needed and persists context:
```
python -m edyant.persistence.cli run llama3 \
  --store ~/.edyant/persistence/graph.sqlite
```
- If ollama isn’t running, it launches `ollama serve` locally and waits up to 8s.
- Each turn uses `MemoryAugmentedAdapter` so prompts/responses are stored in the SQLite graph.
- Exit with `/exit`, `/quit`, or Ctrl+C.

Single-shot without a daemon (opens store, runs once, exits):
```
python -m edyant.persistence.cli prompt \
  --model llama3 \
  --url http://localhost:11434/api/generate \
  "Summarize today's meeting notes."
```

Defaults: `--store` falls back to `EDYANT_DATA_DIR` or `~/.local/share/edyant/persistence/graph.sqlite`; model/URL fall back to `OLLAMA_MODEL` and `OLLAMA_API_URL` if flags are omitted.

## Configuration knobs
- `context_k`: number of hits injected into the prompt (default 5).
- `formatter`: custom callable `(prompt, hits) -> enriched_prompt`.
- `SqliteMemoryStore` pragmas: WAL enabled by default; schema auto-created.

## Data format examples
- **Episode metadata**: `{ "adapter": "ollama", "run_id": "...", "judge_score": 0.72 }`
- **MemoryHit metadata**: anything stored with the node (e.g., dataset tags, user id).

## Operational guidance
- Rotate/compact: copy or vacuum the SQLite file offline if needed; edges have ON DELETE CASCADE.
- Export: read `nodes` table to JSONL for audits; design keeps `response_raw` optional via `ModelOutput.raw`.
- Safety: for high-risk domains, wrap the formatter to include safety rails or evaluator outputs before the prompt.

## Roadmap (next milestones)
- Add embedding-aware candidate generation and hybrid scoring.
- Background summarization/decay jobs (`persistence/jobs/` placeholder).
- CLI utilities under `persistence/cli` for inspect/compact/export.

## Relation to the conceptual framework
- **Episodic**: nodes table.
- **Semantic**: emerges via edge topology; future embeddings/rules will strengthen this layer.
- **Procedural**: edge weight updates from successful/failed outcomes.

## Similar works
- Mem0, Memento MCP (see comparisons in `docs/about_persistence.md`); key differentiator here is outcome-driven edge updates plus procedural memory in topology.
