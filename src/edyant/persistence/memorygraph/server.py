"""Serve a lightweight force-directed graph view over the persistence SQLite store.

No external deps: stdlib http.server + D3 from CDN.
"""
from __future__ import annotations

import json
import sqlite3
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


@dataclass
class GraphConfig:
    store: Path
    host: str = "127.0.0.1"
    port: int = 8787
    max_edges: int = 500
    open_browser: bool = False


HTML_PAGE = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>edyant Memory Graph</title>
  <style>
    body { margin: 0; font-family: "Helvetica Neue", Arial, sans-serif; background: #eaf2f8; color: #1f2a44; }
    #topbar { padding: 10px 12px; background: #d7e3ef; border-bottom: 1px solid #c3ceda; display: flex; gap: 10px; align-items: center; }
    #graph { width: 100vw; height: calc(100vh - 48px); }
    button, input[type="range"] { cursor: pointer; }
    .node { stroke: #f0f4f8; stroke-width: 0.5px; }
    .link { stroke: rgba(80,115,160,0.35); }
    .label { fill: #3c4a63; font-size: 10px; pointer-events: none; }
    .tooltip { position: fixed; padding: 8px 10px; background: rgba(255,255,255,0.95); color: #1f2a44; border: 1px solid #c3ceda; border-radius: 6px; font-size: 12px; max-width: 360px; z-index: 10; pointer-events: none; display: none; box-shadow: 0 6px 18px rgba(31,42,68,0.12); }
  </style>
</head>
<body>
<div id="topbar">
  <button id="reset">Reset</button>
  <label>Min weight <input id="minWeight" type="range" min="0" max="5" step="0.1" value="0"></label>
  <button id="refresh">Refresh</button>
  <span id="status"></span>
</div>
<div id="graph"></div>
<div id="tooltip" class="tooltip"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<script>
const width = window.innerWidth;
const height = window.innerHeight - 48;
let summaryLoaded = false;
let nodes = new Map();
let links = [];
let expanded = new Set();
let minWeight = 0;

const svg = d3.select('#graph').append('svg').attr('width', width).attr('height', height);
const container = svg.append('g');
const linkLayer = container.append('g');
const nodeLayer = container.append('g');
const labelLayer = container.append('g');
const tooltip = document.getElementById('tooltip');
const statusEl = document.getElementById('status');

let simulation = d3.forceSimulation()
  .force('link', d3.forceLink().id(d => d.id).distance(l => 140 / (1 + (l.weight||1))).strength(0.9))
  .force('charge', d3.forceManyBody().strength(-60))
  .force('center', d3.forceCenter(width/2, height/2))
  .alphaDecay(0.05);

function setStatus(msg) { statusEl.textContent = msg; }

function showTooltip(html, x, y) {
  tooltip.innerHTML = html;
  tooltip.style.left = `${x + 12}px`;
  tooltip.style.top = `${y + 12}px`;
  tooltip.style.display = 'block';
}
function hideTooltip(){ tooltip.style.display='none'; }

function render() {
  const nodeData = Array.from(nodes.values());
  const linkSel = linkLayer.selectAll('line').data(links, d => `${d.source.id||d.source}-${d.target.id||d.target}`);
  linkSel.exit().remove();
  linkSel.enter().append('line').attr('class','link').attr('stroke-width', d=> Math.max(0.5, Math.min(4, d.weight||0.5)));

  const nodeSel = nodeLayer.selectAll('circle').data(nodeData, d=>d.id);
  nodeSel.exit().remove();
  nodeSel.enter().append('circle')
    .attr('class','node')
    .attr('r', d=> 4 + Math.sqrt(d.degree||1))
    .attr('fill', d=> expanded.has(d.id) ? '#7dd3fc' : '#93c5fd')
    .on('mouseenter', (event,d)=> showTooltip(`<strong>${d.label}</strong><br>deg=${d.degree||0}<br>weight≈${(d.weight||0).toFixed(2)}`, event.clientX, event.clientY))
    .on('mouseleave', hideTooltip)
    .on('dblclick', (_,d)=> expandNode(d.id));

  const labelSel = labelLayer.selectAll('text').data(nodeData, d=>d.id);
  labelSel.exit().remove();
  labelSel.enter().append('text').attr('class','label').attr('dy',-8).text(d=>d.label.slice(0,40));

  simulation.nodes(nodeData).on('tick', () => {
    linkLayer.selectAll('line')
      .attr('x1', d=>d.source.x).attr('y1', d=>d.source.y)
      .attr('x2', d=>d.target.x).attr('y2', d=>d.target.y);
    nodeLayer.selectAll('circle')
      .attr('cx', d=>d.x).attr('cy', d=>d.y);
    labelLayer.selectAll('text')
      .attr('x', d=>d.x).attr('y', d=>d.y);
  });
  simulation.force('link').links(links);
  simulation.alpha(0.9).restart();
}

async function fetchJSON(path) {
  const resp = await fetch(path);
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(text || resp.statusText);
  }
  return resp.json();
}

async function loadSummary() {
  setStatus('loading summary...');
  const data = await fetchJSON(`/graph/summary?min_weight=${minWeight}`);
  nodes.clear();
  links = [];
  data.nodes.forEach(n=> nodes.set(n.id, n));
  data.links.forEach(l=> links.push(l));
  summaryLoaded = true;
  expanded.clear();
  render();
  setStatus(`summary: ${nodes.size} nodes / ${links.length} links`);
}

async function expandNode(id){
  if (expanded.has(id)) return;
  setStatus(`expanding ${id}...`);
  const data = await fetchJSON(`/graph/neighbors?node_id=${encodeURIComponent(id)}&min_weight=${minWeight}`);
  data.nodes.forEach(n=> nodes.set(n.id, n));
  data.links.forEach(l=> links.push(l));
  expanded.add(id);
  render();
  setStatus(`expanded ${id}`);
}

function reset(){ loadSummary(); }

const zoom = d3.zoom().scaleExtent([0.5, 8]).on('zoom', (event)=>{
  container.attr('transform', event.transform);
  if(event.transform.k > 1.8){
    // auto expand a few visible nodes
    const nodeArray = Array.from(nodes.values());
    for (let i=0;i<Math.min(3,nodeArray.length);i++){
      const n = nodeArray[(Math.random()*nodeArray.length)|0];
      if (!expanded.has(n.id)) { expandNode(n.id); break; }
    }
  }
});
svg.call(zoom);

// UI wiring
 document.getElementById('reset').onclick = reset;
 document.getElementById('refresh').onclick = ()=> loadSummary();
 document.getElementById('minWeight').oninput = (e)=>{ minWeight = parseFloat(e.target.value); loadSummary(); };

loadSummary();
</script>
</body>
</html>
"""


def _open_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _fetch_summary(conn: sqlite3.Connection, max_edges: int, min_weight: float) -> dict[str, Any]:
    edges = conn.execute(
        "SELECT source, target, weight FROM edges WHERE weight >= ? ORDER BY weight DESC LIMIT ?",
        (min_weight, max_edges),
    ).fetchall()
    node_ids: set[str] = set()
    for row in edges:
        node_ids.add(row[0])
        node_ids.add(row[1])
    if not node_ids:
        nodes = []
    else:
        placeholders = ",".join(["?"] * len(node_ids))
        node_rows = conn.execute(
            f"SELECT id, prompt, response, created_at, metadata FROM nodes WHERE id IN ({placeholders})",
            tuple(node_ids),
        ).fetchall()
        nodes = [
            {
                "id": r[0],
                "label": (r[1] or "")[:80] or r[0],
                "degree": 0,
            }
            for r in node_rows
        ]
    # degree
    degree = {}
    for row in edges:
        degree[row[0]] = degree.get(row[0], 0) + 1
        degree[row[1]] = degree.get(row[1], 0) + 1
    for n in nodes:
        n["degree"] = degree.get(n["id"], 0)
    links = [
        {"source": row[0], "target": row[1], "weight": row[2]}
        for row in edges
    ]
    return {"nodes": nodes, "links": links}


def _fetch_neighbors(conn: sqlite3.Connection, node_id: str, min_weight: float, k: int = 50) -> dict[str, Any]:
    neighbors = conn.execute(
        """
        SELECT target AS nid, weight FROM edges WHERE source = ? AND weight >= ?
        UNION ALL
        SELECT source AS nid, weight FROM edges WHERE target = ? AND weight >= ?
        ORDER BY weight DESC LIMIT ?
        """,
        (node_id, min_weight, node_id, min_weight, k),
    ).fetchall()
    neighbor_ids = {row[0] for row in neighbors}
    node_rows = conn.execute(
        "SELECT id, prompt, response, created_at, metadata FROM nodes WHERE id IN ({seq})".format(
            seq=",".join(["?"] * (len(neighbor_ids) + 1))
        ),
        (node_id, *neighbor_ids) if neighbor_ids else (node_id,),
    ).fetchall()
    nodes = [
        {
            "id": r[0],
            "label": (r[1] or "")[:80] or r[0],
            "degree": None,
        }
        for r in node_rows
    ]
    links = [
        {"source": node_id, "target": row[0], "weight": row[1]}
        for row in neighbors
    ]
    return {"nodes": nodes, "links": links}


class GraphHandler(BaseHTTPRequestHandler):
    cfg: GraphConfig

    def _send_json(self, payload: dict[str, Any], code: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        if path == "/" or path == "/index.html":
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/health":
            self._send_json({"status": "ok"})
            return
        if path == "/graph/summary":
            min_w = float(qs.get("min_weight", [0])[0])
            try:
                conn = _open_db(self.cfg.store)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, code=500)
                return
            with conn:
                data = _fetch_summary(conn, self.cfg.max_edges, min_w)
            self._send_json(data)
            return
        if path == "/graph/neighbors":
            node_id = qs.get("node_id", [None])[0]
            if not node_id:
                self._send_json({"error": "node_id required"}, code=400)
                return
            min_w = float(qs.get("min_weight", [0])[0])
            k = int(qs.get("k", [50])[0])
            try:
                conn = _open_db(self.cfg.store)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, code=500)
                return
            with conn:
                data = _fetch_neighbors(conn, node_id, min_w, k=k)
            self._send_json(data)
            return
        self._send_json({"error": "not found"}, code=404)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return  # quiet


def run_memorygraph_server(cfg: GraphConfig) -> None:
    handler_cls = type("CfgHandler", (GraphHandler,), {"cfg": cfg})
    server = ThreadingHTTPServer((cfg.host, cfg.port), handler_cls)
    url = f"http://{cfg.host}:{cfg.port}/"

    if cfg.open_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
