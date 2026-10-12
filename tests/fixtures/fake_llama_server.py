"""Stand-in for llama.cpp's llama-server: answers /health and /v1/models on loopback."""

import argparse
import json
import os
import socketserver
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

parser = argparse.ArgumentParser()
parser.add_argument("--alias", default="fake")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, required=True)
args, _ = parser.parse_known_args()

# Simulate "out of GPU memory right after training": fail the first start, then work.
fail_flag = os.environ.get("FAKE_LLAMA_FAIL_ONCE_FILE")
if fail_flag and not os.path.exists(fail_flag):
    open(fail_flag, "w").close()
    sys.exit(1)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 (http.server naming)
        if self.path == "/health":
            body = {"status": "ok"}
        elif self.path == "/v1/models":
            body = {"data": [{"id": args.alias}]}
        else:
            self.send_error(404)
            return
        payload = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_):
        pass


class Server(HTTPServer):
    def server_bind(self):
        # HTTPServer.server_bind resolves the host's FQDN, which takes tens of seconds
        # on some macOS machines. Use the plain TCP bind, which still sets SO_REUSEADDR
        # so a restart on the same port works straight after a stop.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


Server((args.host, args.port), Handler).serve_forever()
