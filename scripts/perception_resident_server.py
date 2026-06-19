#!/usr/bin/env python3
"""Resident perception server to reuse a single loaded model process.

Usage:
  python scripts/perception_resident_server.py --host 127.0.0.1 --port 18080

Request:
  POST /predict
  {"image_path": "dataset/multi/WeatherBench/rain/test/input/231.jpg"}
"""

from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perception_module import _get_model_and_processor, predict_degradation


class _PerceptionHandler(BaseHTTPRequestHandler):
    server_version = "PerceptionResident/1.0"

    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
            return
        self._send_json(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/predict":
            self._send_json(404, {"error": "not_found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": "invalid_content_length"})
            return

        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8")) if raw else {}
        except Exception:
            self._send_json(400, {"error": "invalid_json"})
            return

        image_path = str(data.get("image_path", "")).strip()
        if not image_path:
            self._send_json(400, {"error": "missing_image_path"})
            return

        resolved_path = Path(image_path)
        if not resolved_path.is_absolute():
            resolved_path = (Path.cwd() / resolved_path).resolve()

        if not resolved_path.exists():
            self._send_json(404, {"error": "image_not_found", "image_path": str(resolved_path)})
            return

        try:
            result = predict_degradation(str(resolved_path))
            self._send_json(200, result if isinstance(result, dict) else {"result": result})
        except Exception as exc:
            self._send_json(500, {"error": "predict_failed", "detail": f"{type(exc).__name__}: {exc}"})

    def log_message(self, fmt: str, *args: Any) -> None:
        # Keep stdout clean for easier terminal usage.
        return


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resident weather perception server")
    parser.add_argument("--host", default=os.getenv("PERCEPTION_SERVER_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PERCEPTION_SERVER_PORT", "18080")))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _get_model_and_processor()
    server = HTTPServer((args.host, args.port), _PerceptionHandler)
    print(f"resident_perception_server_ready host={args.host} port={args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
