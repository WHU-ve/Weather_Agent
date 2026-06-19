#!/usr/bin/env python3
"""Send requests to resident perception server.

Usage:
  python scripts/perception_resident_client.py --image dataset/.../231.jpg
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resident perception client")
    parser.add_argument("--image", required=True, help="Image path")
    parser.add_argument("--host", default=os.getenv("PERCEPTION_SERVER_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PERCEPTION_SERVER_PORT", "18080")))
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    url = f"http://{args.host}:{args.port}/predict"
    payload = json.dumps({"image_path": args.image}).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            body = resp.read().decode("utf-8")
            print(body)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(json.dumps({"error": f"http_{exc.code}", "detail": detail}, ensure_ascii=False))
        raise SystemExit(1)
    except Exception as exc:
        print(json.dumps({"error": "request_failed", "detail": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
