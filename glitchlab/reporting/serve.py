#!/usr/bin/env python3
"""
GLITCHLAB Dashboard Server

Serves the attestation dashboard and auto-loads audit.jsonl from the repo's
.glitchlab/logs/ directory. No drag-and-drop needed.

Usage:
    python3 glitchlab/reporting/serve.py --repo ~/Desktop/glitchlab-soapbox
    # Opens http://localhost:8080
"""

import argparse
import json
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path


class DashboardHandler(SimpleHTTPRequestHandler):
    """Serves dashboard files + injects audit.jsonl as an API endpoint."""

    repo_path: Path = Path(".")
    dashboard_dir: Path = Path(".")

    def do_GET(self):
        if self.path == "/api/audit":
            self.serve_audit()
        elif self.path == "/" or self.path == "/index.html":
            self.serve_dashboard()
        else:
            # Serve static files from dashboard dir
            os.chdir(self.dashboard_dir)
            super().do_GET()

    def serve_audit(self):
        audit_file = self.repo_path / ".glitchlab" / "logs" / "audit.jsonl"
        if not audit_file.exists():
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(b'{"error": "audit.jsonl not found"}')
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(audit_file.read_bytes())

    def serve_dashboard(self):
        dashboard_file = self.dashboard_dir / "index.html"
        if not dashboard_file.exists():
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Dashboard not found")
            return

        content = dashboard_file.read_text()

        # Inject auto-load script that fetches from /api/audit on page load
        inject = """
<script>
window.addEventListener('load', async () => {
    try {
        const resp = await fetch('/api/audit');
        if (resp.ok) {
            const text = await resp.text();
            const lines = [];
            let buffer = '';
            let depth = 0;
            for (const ch of text) {
                buffer += ch;
                if (ch === '{') depth++;
                if (ch === '}') depth--;
                if (depth === 0 && buffer.trim()) {
                    lines.push(buffer.trim());
                    buffer = '';
                }
            }
            if (buffer.trim()) lines.push(buffer.trim());
            if (lines.length > 0 && window.__setAuditData) {
                window.__setAuditData(lines);
            }
        }
    } catch(e) { console.log('No audit data available:', e); }
});
</script>
"""
        # Inject the auto-load script and a hook in the React app
        content = content.replace(
            "const [rawLines,setRawLines]=useState(null);",
            "const [rawLines,setRawLines]=useState(null);\n"
            "  React.useEffect(()=>{ window.__setAuditData = setRawLines; "
            "return ()=>{ delete window.__setAuditData; }; },[]);"
        )
        content = content.replace("</body>", inject + "</body>")

        encoded = content.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def main():
    parser = argparse.ArgumentParser(description="GLITCHLAB Dashboard Server")
    parser.add_argument("--repo", type=str, default=".", help="Path to the GLITCHLAB repository")
    parser.add_argument("--port", type=int, default=8080, help="Port to serve on")
    args = parser.parse_args()

    repo_path = Path(args.repo).resolve()
    dashboard_dir = Path(__file__).parent.resolve()

    DashboardHandler.repo_path = repo_path
    DashboardHandler.dashboard_dir = dashboard_dir

    audit_file = repo_path / ".glitchlab" / "logs" / "audit.jsonl"
    print(f"⚡ GLITCHLAB Attestation Dashboard")
    print(f"   Repo:      {repo_path}")
    print(f"   Audit log: {audit_file} ({'found' if audit_file.exists() else 'not found'})")
    print(f"   Dashboard: http://localhost:{args.port}")
    print(f"   Refresh the browser to reload latest audit data.\n")

    server = HTTPServer(("0.0.0.0", args.port), DashboardHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()