"""Lightweight HTTP API for multi-source vector search.

Exposes FalkorDB SourceIndex search over HTTP for SearXNG engines.
Supports all indexed sources (producthunt, youtube, telegram, etc.).

Endpoints:
  GET /search?q=...&source=producthunt&n=5  — vector search
  GET /sources                               — list available sources
  GET /health                                — status + total count
"""

import json
import os
from http.server import BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from http.server import HTTPServer
from urllib.parse import urlparse, parse_qs

# Lazy-init to avoid slow startup
_idx = None


def _get_index():
    global _idx
    if _idx is None:
        from codegraph_mcp.vectors.source_index import SourceIndex

        _idx = SourceIndex(backend="st")
        sources = _idx.list_sources()
        total = sum(s.get("count", 0) for s in sources)
        names = [s["source"] for s in sources]
        print(f"SourceIndex ready: {total} items across {len(sources)} sources: {', '.join(names)}")
    return _idx


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class SearchHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/search":
            self._handle_search(parsed)
        elif parsed.path == "/sources":
            self._handle_sources()
        elif parsed.path == "/health":
            self._handle_health()
        else:
            self._json_response({"error": "not found"}, 404)

    def _handle_search(self, parsed):
        params = parse_qs(parsed.query)
        query = params.get("q", [""])[0]
        n = int(params.get("n", ["5"])[0])
        source = params.get("source", [None])[0]

        if not query:
            self._json_response({"error": "missing q param"}, 400)
            return

        idx = _get_index()

        # Validate source if specified
        if source:
            available = [s["source"] for s in idx.list_sources()]
            if source not in available:
                self._json_response({
                    "error": f"unknown source: {source}",
                    "available": available,
                }, 400)
                return

        results = idx.search(query, source=source, n_results=n)
        self._json_response({"results": results, "query": query, "source": source})

    def _handle_sources(self):
        idx = _get_index()
        sources = idx.list_sources()
        self._json_response({"sources": sources})

    def _handle_health(self):
        idx = _get_index()
        count = idx.count()
        sources = idx.list_sources()
        self._json_response({
            "status": "ok",
            "total": count,
            "sources": len(sources),
        })

    def _json_response(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


def main():
    port = int(os.environ.get("SOLOGRAPH_SEARCH_PORT", "8002"))
    print(f"Solograph search API starting on :{port}")
    print(f"  GET /search?q=AI+tool&source=producthunt&n=5")
    print(f"  GET /sources")
    print(f"  GET /health")

    # Pre-warm index
    _get_index()

    server = ThreadingHTTPServer(("0.0.0.0", port), SearchHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down")


if __name__ == "__main__":
    main()
