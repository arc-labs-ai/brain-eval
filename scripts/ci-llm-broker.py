#!/usr/bin/env python3
"""A minimal LLM broker, enough to boot a real brain-server in CI.

Brain resolves an LLM credential **per namespace**: the serving key is not
known until a write names its namespace, so a process-wide `[llm] api_key` is
not a substitute and the server refuses to boot without `[llm.broker]`. In
production that broker is the Arc gateway. CI has one shared provider key and
no control plane, so this stands in for it.

Two halves of the contract, both taken from `brain-llm`'s `BrokerResolver`:

  GET  {base}/llm/ns/{namespace}/resolve
        Bearer {service_key}  ->  200 {"provider": ..., "model": ...}
        Carries no key material. 409 means the tenant registered no key.

  POST {base}/llm/ns/{namespace}/{provider}/v1/chat/completions   (openai)
  POST {base}/llm/ns/{namespace}/{provider}/v1/messages           (anthropic)
        Proxied upstream with the real provider credential, which Brain never
        sees. This is why resolve returns no key: completions go *through*
        the broker.

Every namespace resolves to the same provider and model here. That is the
whole point of a CI stand-in — the per-namespace indirection is what is being
satisfied, not exercised.

Env:
    BROKER_SERVICE_KEY   the bearer Brain authenticates to this broker with
    BROKER_PROVIDER      `openai` | `anthropic`  (derived from the model if unset)
    BROKER_MODEL         model id served to every namespace
    BROKER_LLM_API_KEY   the real upstream provider key
    BROKER_LISTEN        host:port (default 0.0.0.0:9099)
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SERVICE_KEY = os.environ.get("BROKER_SERVICE_KEY", "")
MODEL = os.environ.get("BROKER_MODEL", "")
LLM_API_KEY = os.environ.get("BROKER_LLM_API_KEY", "")
UPSTREAM = {
    "openai": "https://api.openai.com",
    "anthropic": "https://api.anthropic.com",
}


def provider() -> str:
    """Explicit if set, else derived from the model id the way Brain does."""
    explicit = os.environ.get("BROKER_PROVIDER", "").strip().lower()
    if explicit:
        return explicit
    return "anthropic" if MODEL.startswith("claude") else "openai"


RESOLVE = re.compile(r"^/llm/ns/([^/]+)/resolve$")
COMPLETE = re.compile(r"^/llm/ns/([^/]+)/(openai|anthropic)(/v1/.+)$")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("broker: " + (fmt % args) + "\n")

    def _json(self, status: int, body: dict) -> None:
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _authorized(self) -> bool:
        got = self.headers.get("authorization", "")
        return got == f"Bearer {SERVICE_KEY}"

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's contract
        m = RESOLVE.match(self.path)
        if not m:
            self._json(404, {"error": "no such route"})
            return
        if not self._authorized():
            self._json(401, {"error": "bad service key"})
            return
        if not LLM_API_KEY:
            # The code Brain reads as "this tenant has no key registered".
            self._json(409, {"error": "no credential for namespace"})
            return
        self._json(200, {"provider": provider(), "model": MODEL})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's contract
        m = COMPLETE.match(self.path)
        if not m:
            self._json(404, {"error": "no such route"})
            return
        if not self._authorized():
            self._json(401, {"error": "bad service key"})
            return
        _ns, prov, tail = m.group(1), m.group(2), m.group(3)
        length = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(length) if length else b""

        req = urllib.request.Request(UPSTREAM[prov] + tail, data=body, method="POST")
        req.add_header("content-type", "application/json")
        if prov == "anthropic":
            req.add_header("x-api-key", LLM_API_KEY)
            req.add_header(
                "anthropic-version",
                self.headers.get("anthropic-version", "2023-06-01"),
            )
        else:
            req.add_header("authorization", f"Bearer {LLM_API_KEY}")

        try:
            with urllib.request.urlopen(req, timeout=120) as up:
                payload, status = up.read(), up.status
        except urllib.error.HTTPError as e:
            payload, status = e.read(), e.code
        except Exception as e:  # noqa: BLE001 - any transport failure is a 502
            self._json(502, {"error": f"upstream unreachable: {e}"})
            return

        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main() -> int:
    if not SERVICE_KEY:
        sys.stderr.write("broker: BROKER_SERVICE_KEY is required\n")
        return 2
    if not MODEL:
        sys.stderr.write("broker: BROKER_MODEL is required\n")
        return 2
    host, _, port = os.environ.get("BROKER_LISTEN", "0.0.0.0:9099").rpartition(":")
    srv = ThreadingHTTPServer((host or "0.0.0.0", int(port)), Handler)
    sys.stderr.write(
        f"broker: listening on {host or '0.0.0.0'}:{port} "
        f"provider={provider()} model={MODEL} "
        f"upstream_key={'set' if LLM_API_KEY else 'UNSET (resolve will 409)'}\n"
    )
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
