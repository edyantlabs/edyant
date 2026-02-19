# edyant Memory Graph Viewer

Interactive force-directed view of the persistence SQLite store with dynamic, zoom-aware expansion.

## Usage

```bash
python -m edyant memorygraph \
  --store ~/.edyant/persistence/memory.sqlite \
  --open-browser
```

Options:
- `--store` (default `~/.edyant/persistence/memory.sqlite`): path to the SQLite persistence file.
- `--host` (default `127.0.0.1`): bind host for the viewer.
- `--port` (default `8787`): bind port.
- `--max-edges` (default `500`): maximum edges returned in the initial summary/backbone.
- `--open-browser`: auto-launch your browser to the viewer URL.

## What it does
- Starts a lightweight HTTP server (no extra deps) serving:
  - `/` HTML + D3 visualization
  - `/graph/summary` backbone view (top-weight edges, nodes they touch)
  - `/graph/neighbors?node_id=<id>` on-demand expansion of a node’s neighborhood
  - `/health` basic liveness
- Opens your browser (if `--open-browser`) to a force-directed graph:
  - Zoom/pan with mouse; double-click a node to expand neighbors.
  - Auto-expands a few nearby nodes when zoomed in (>1.8x) to reveal local structure.
  - Slider for `min weight` filters weak edges; reset and refresh controls.

## Data shown
- Nodes come from `nodes` table; label is prompt snippet (≤80 chars).
- Links come from `edges` table; link distance and thickness scale with `weight`.
- Degrees are computed per response payload for sizing.

## Dynamic loading strategy
- Initial load: `/graph/summary` returns a pruned backbone (top `max_edges` by weight) to keep the overview legible.
- Local detail: `/graph/neighbors` is called on node double-click or automatically when zooming in; it returns top neighbors (by weight, default k=50) for that node.
- View-driven density: zooming out leaves the backbone; zooming in incrementally pulls more local nodes/edges without redrawing the entire graph.

## JSON API shape
- `/graph/summary`: `{ "nodes": [{id, label, degree}], "links": [{source, target, weight}] }`
- `/graph/neighbors`: same shape, limited to the seed node + its neighbors.
- All responses are UTF-8 JSON; errors return `{ "error": "..." }` with 4xx/5xx codes.

## Troubleshooting
- If the store path is wrong or locked, the viewer returns a JSON error banner.
- If the port is in use, rerun with `--port <other>`.
- Very large stores: increase `--max-edges` for denser backbones, but keep it moderate for performance.
