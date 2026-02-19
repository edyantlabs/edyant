"""Memory graph visualization service (HTML + JSON API).

Exposes a tiny HTTP server that reads the persistence SQLite store and serves:
- /              : D3-based force layout UI
- /graph/summary : backbone/top-edges view
- /graph/neighbors?node_id=... : localized expansion for a node
- /health        : basic liveness check
"""
from __future__ import annotations

from .server import run_memorygraph_server

__all__ = ["run_memorygraph_server"]
