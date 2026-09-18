"""Loopback-only FlySOC research platform. No telemetry is sent off this machine."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import webbrowser
from urllib.parse import urlsplit

from .brain import BrainLab

WEB = Path(__file__).parent / "web"


def make_handler(lab: BrainLab):
    class Handler(BaseHTTPRequestHandler):
        def send(self, body: bytes, content_type: str, code: int = 200, gzip_encoded: bool = False):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
            if gzip_encoded:
                self.send_header("Content-Encoding", "gzip")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def send_json(self, value, code: int = 200):
            self.send(json.dumps(value, allow_nan=False).encode("utf-8"), "application/json; charset=utf-8", code)

        def valid_host(self) -> bool:
            expected = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            return self.headers.get("Host") in expected

        def do_GET(self):
            if not self.valid_host():
                return self.send_json({"error": "Loopback host required"}, 403)
            path = urlsplit(self.path).path
            if path == "/api/status":
                return self.send_json(lab.status())
            if path == "/api/alerts":
                return self.send_json(lab.alert_catalog())
            if path == "/api/fly/layout":
                return self.send_json(lab.fly_layout())
            if path.startswith("/api/connectome/neuron/"):
                try:
                    index = int(path.rsplit("/", 1)[-1])
                    if lab.connectome_summary is None or not 0 <= index < len(lab.nodes):
                        raise ValueError("Neuron index out of range")
                    row = lab.nodes.iloc[index]
                    return self.send_json({"index": index, "root_id": str(row.root_id),
                                           "class": str(row["class"]), "super_class": str(row.super_class),
                                           "nt_type": str(row.nt_type)})
                except ValueError as error:
                    return self.send_json({"error": str(error)}, 400)
            if path == "/api/connectome/scene":
                scene = lab.connectome_path / "prepared/scene.json.gz"
                if scene.exists():
                    return self.send(scene.read_bytes(), "application/json", gzip_encoded=True)
                return self.send_json({"error": "Connectome unavailable"}, 404)
            relative = "index.html" if path == "/" else path.lstrip("/")
            candidate = (WEB / relative).resolve()
            if not candidate.is_relative_to(WEB.resolve()) or not candidate.is_file():
                return self.send_json({"error": "Not found"}, 404)
            mime = "text/javascript" if candidate.suffix == ".js" else mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
            return self.send(candidate.read_bytes(), mime)

        def do_POST(self):
            origin = self.headers.get("Origin")
            allowed_origins = {f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"}
            if not self.valid_host() or (origin is not None and origin not in allowed_origins):
                return self.send_json({"error": "Same-origin loopback requests required"}, 403)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536 or self.headers.get_content_type() != "application/json":
                    raise ValueError("A JSON body of at most 64 KiB is required")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("Expected a JSON object")
                path = urlsplit(self.path).path
                if path == "/api/fly/trace":
                    return self.send_json(lab.trace_alert(body.get("index", 0), body.get("alert")))
                if path == "/api/connectome/pulse":
                    return self.send_json(lab.pulse_frame(body.get("reset_class"), body.get("advance", True)))
                return self.send_json({"error": "Not found"}, 404)
            except (ValueError, KeyError, TypeError) as error:
                return self.send_json({"error": str(error)}, 400)

        def log_message(self, message, *args):
            if " 200 " not in str(args):
                super().log_message(message, *args)

    return Handler


def main():
    parser = argparse.ArgumentParser(description="FlySOC local research platform")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--run")
    parser.add_argument("--open-browser", action="store_true", help="Open the local interactive viewer")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Port must be between 1024 and 65535")
    lab = BrainLab(args.run)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(lab))
    print(f"FlySOC Research Platform ready: http://127.0.0.1:{args.port}", flush=True)
    if args.open_browser:
        webbrowser.open(f"http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
