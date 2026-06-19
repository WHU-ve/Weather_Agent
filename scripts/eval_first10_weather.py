#!/usr/bin/env python3
"""Evaluate the first 10 images in a WeatherBench split and count label hits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from perception_module import predict_degradation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Input directory, e.g. dataset/multi/WeatherBench/haze/test/input")
    parser.add_argument("--label", required=True, help="Target label to count in predictions")
    parser.add_argument("--limit", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    files = sorted([p for p in root.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}])[: args.limit]

    results: List[Dict[str, object]] = []
    hit_count = 0
    target = args.label.lower()

    for path in files:
        result = predict_degradation(str(path))
        degradations = [str(item).lower() for item in result.get("degradations", [])] if isinstance(result, dict) else []
        hit = target in degradations
        hit_count += int(hit)
        results.append(
            {
                "file": path.name,
                "degradations": result.get("degradations", []) if isinstance(result, dict) else [],
                "hit": hit,
                "image_description": result.get("image_description", "") if isinstance(result, dict) else str(result),
            }
        )

    summary = {
        "root": str(root),
        "label": args.label,
        "total": len(results),
        "hit_count": hit_count,
        "hit_rate": (hit_count / len(results)) if results else 0.0,
        "results": results,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
