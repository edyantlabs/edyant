"""SQLite-backed MemoryStore with lightweight spreading activation.

The goal is portability: the caller decides where the DB file lives
(e.g., mounted volume in Docker or local temp dir). Retrieval uses a
cheap lexical similarity plus edge-weight boosts to approximate
spreading activation without external dependencies.
"""
from __future__ import annotations

import json
import re
import sqlite3
import uuid
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from ..api import Episode, MemoryHit, MemoryStore
from ..types import ModelOutput

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text)}


def _similarity(query_tokens: set[str], text: str) -> float:
    if not query_tokens:
        return 0.0
    doc_tokens = _tokenize(text)
    if not doc_tokens:
        return 0.0
    overlap = len(query_tokens & doc_tokens)
    return overlap / max(len(query_tokens), len(doc_tokens))


class SqliteMemoryStore(MemoryStore):
    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                created_at TEXT NOT NULL,
                metadata TEXT
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS edges (
                source TEXT NOT NULL,
                target TEXT NOT NULL,
                weight REAL NOT NULL,
                PRIMARY KEY (source, target),
                FOREIGN KEY (source) REFERENCES nodes(id) ON DELETE CASCADE,
                FOREIGN KEY (target) REFERENCES nodes(id) ON DELETE CASCADE
            );
            """
        )
        self._conn.commit()

    def retrieve(self, query: str, top_k: int = 5) -> list[MemoryHit]:
        cur = self._conn.cursor()
        # Limit candidate size to keep things cheap.
        cur.execute("SELECT id, prompt, response, metadata FROM nodes ORDER BY created_at DESC LIMIT 500")
        rows = cur.fetchall()
        query_tokens = _tokenize(query)
        base_hits: list[MemoryHit] = []
        for node_id, prompt, response, metadata_json in rows:
            text = f"{prompt}\n{response}"
            score = _similarity(query_tokens, text)
            if score <= 0.0:
                continue
            metadata = json.loads(metadata_json) if metadata_json else {}
            base_hits.append(MemoryHit(node_id=node_id, text=text, score=score, metadata=metadata))

        # Boost via edge weights (simple spreading activation).
        boosts: defaultdict[str, float] = defaultdict(float)
        for hit in base_hits[: top_k * 2]:
            for neighbor_id, weight in self._edges_from(hit.node_id):
                boosts[neighbor_id] += weight * 0.5

        merged: list[MemoryHit] = []
        seen = set()
        for hit in base_hits:
            hit.score += boosts.get(hit.node_id, 0.0)
            merged.append(hit)
            seen.add(hit.node_id)
        # Add pure neighbors not present in base_hits.
        for neighbor_id, weight in boosts.items():
            if neighbor_id in seen:
                continue
            node = self._fetch_node(neighbor_id)
            if node is None:
                continue
            merged.append(
                MemoryHit(
                    node_id=neighbor_id,
                    text=f"{node.prompt}\n{node.response}",
                    score=weight,
                    metadata=node.metadata,
                )
            )

        merged.sort(key=lambda h: h.score, reverse=True)
        return merged[:top_k]

    def record_episode(
        self,
        prompt: str,
        output: ModelOutput,
        metadata: dict | None = None,
    ) -> str:
        node_id = uuid.uuid4().hex
        created_at = datetime.now(timezone.utc).isoformat()
        merged_meta = dict(metadata or {})
        # Persist provider metadata if present.
        merged_meta.update(output.meta or {})
        payload = (
            node_id,
            prompt,
            output.text,
            created_at,
            json.dumps(merged_meta, ensure_ascii=False),
        )
        self._conn.execute(
            "INSERT INTO nodes (id, prompt, response, created_at, metadata) VALUES (?, ?, ?, ?, ?)",
            payload,
        )
        self._conn.commit()
        return node_id

    def update_edges(self, source_id: str, related_ids: Sequence[str], weight: float = 1.0) -> None:
        cur = self._conn.cursor()
        for target_id in related_ids:
            cur.execute(
                """
                INSERT INTO edges (source, target, weight) VALUES (?, ?, ?)
                ON CONFLICT(source, target) DO UPDATE SET weight = weight + excluded.weight
                """,
                (source_id, target_id, float(weight)),
            )
        self._conn.commit()

    def _edges_from(self, node_id: str) -> Iterable[tuple[str, float]]:
        cur = self._conn.cursor()
        cur.execute("SELECT target, weight FROM edges WHERE source = ?", (node_id,))
        return cur.fetchall()

    def _fetch_node(self, node_id: str) -> Episode | None:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT id, prompt, response, created_at, metadata FROM nodes WHERE id = ?",
            (node_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        metadata = json.loads(row[4]) if row[4] else {}
        return Episode(
            node_id=row[0],
            prompt=row[1],
            response=row[2],
            created_at=datetime.fromisoformat(row[3]),
            metadata=metadata,
        )

    def close(self) -> None:
        self._conn.close()
