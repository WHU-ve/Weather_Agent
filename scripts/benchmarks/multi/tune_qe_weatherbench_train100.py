#!/usr/bin/env python3
"""Grid-search QE alpha on WeatherBench train split.

This script:
1) Runs all experts for each task/sample once and caches their outputs.
2) Computes QE task/general scores for each candidate once.
3) Sweeps alpha in [0, 1] and selects the best candidate per alpha.
4) Reports mean PSNR/SSIM and best alpha for each task.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
	from skimage.metrics import peak_signal_noise_ratio as sk_psnr
	from skimage.metrics import structural_similarity as sk_ssim
except Exception as err:  # pragma: no cover
	raise RuntimeError(
		"scikit-image is required for PSNR/SSIM computation. "
		"Please install it in the current environment."
	) from err


THIS_FILE = Path(__file__).resolve()
WEATHER_AGENT_ROOT = THIS_FILE.parents[3]
if str(WEATHER_AGENT_ROOT) not in sys.path:
	sys.path.insert(0, str(WEATHER_AGENT_ROOT))

os.chdir(WEATHER_AGENT_ROOT)

# Default to offline mode to avoid HuggingFace network retries during long benchmark runs.
# Users can still override by explicitly exporting these env vars before launching.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from quality_evaluator import QualityEvaluator  # noqa: E402
from utils.dehazing import dehazing_toolbox  # noqa: E402
from utils.deraining import deraining_toolbox  # noqa: E402
from utils.desnowing import desnowing_toolbox  # noqa: E402


TASK_SPECS = {
	"derain": {
		"wb_key": "rain",
		"qe_task": "derain",
		"toolbox": deraining_toolbox,
	},
	"dehaze": {
		"wb_key": "haze",
		"qe_task": "dehaze",
		"toolbox": dehazing_toolbox,
	},
	"desnow": {
		"wb_key": "snow",
		"qe_task": "desnow",
		"toolbox": desnowing_toolbox,
	},
}


@dataclass
class CandidateInfo:
	name: str
	image_path: Path
	task_score: float
	general_score: float
	task_main_util: float


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--dataset_root",
		type=Path,
		default=WEATHER_AGENT_ROOT / "dataset/multi/WeatherBench",
		help="WeatherBench root containing rain/haze/snow/train/{input,target}",
	)
	parser.add_argument(
		"--tasks",
		nargs="+",
		default=["derain", "dehaze", "desnow"],
		choices=list(TASK_SPECS.keys()),
	)
	parser.add_argument("--split", default="train", choices=["train", "test"])
	parser.add_argument("--max_samples", type=int, default=20)
	parser.add_argument(
		"--sample_mode",
		type=str,
		default="random",
		choices=["random", "head"],
		help="Image sampling mode per task. random=uniform random subset, head=sorted first N.",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=None,
		help="Optional base seed for reproducible random sampling.",
	)
	parser.add_argument(
		"--alphas",
		type=str,
		default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
		help="Comma-separated alpha list.",
	)
	parser.add_argument(
		"--work_dir",
		type=Path,
		default=WEATHER_AGENT_ROOT / "tmp/alpha_grid_weatherbench_train20_random",
	)
	parser.add_argument(
		"--report_json",
		type=Path,
		default=WEATHER_AGENT_ROOT / "update_log/alpha_grid_weatherbench_train20_random.json",
	)
	parser.add_argument(
		"--keep_intermediate",
		action="store_true",
		help="Keep per-expert input temp directories.",
	)
	parser.add_argument(
		"--run_gpu_id",
		type=int,
		default=None,
		help="Optional single GPU id passed to tool invocation.",
	)
	return parser.parse_args()


def parse_alphas(alpha_text: str) -> List[float]:
	alphas: List[float] = []
	for token in alpha_text.split(","):
		token = token.strip()
		if not token:
			continue
		val = float(token)
		val = float(np.clip(val, 0.0, 1.0))
		alphas.append(val)
	if not alphas:
		raise ValueError("No valid alpha values provided")
	return sorted(set(alphas))


def list_images(folder: Path) -> List[Path]:
	exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
	return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts])


def sample_inputs(
	all_inputs: Sequence[Path],
	max_samples: int,
	sample_mode: str,
	rng: random.Random,
) -> List[Path]:
	if max_samples <= 0:
		return []
	if len(all_inputs) <= max_samples:
		return list(all_inputs)
	if sample_mode == "head":
		return list(all_inputs[:max_samples])
	# Uniform random subset without replacement.
	return sorted(rng.sample(list(all_inputs), k=max_samples), key=lambda p: p.name)


def match_target(input_path: Path, target_dir: Path) -> Optional[Path]:
	exact = target_dir / input_path.name
	if exact.exists():
		return exact
	stem = input_path.stem
	for p in list_images(target_dir):
		if p.stem == stem:
			return p
	return None


def load_rgb_float(path: Path) -> np.ndarray:
	bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
	if bgr is None:
		raise RuntimeError(f"Failed to read image: {path}")
	rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
	return rgb.astype(np.float32) / 255.0


def compute_psnr_ssim(pred_path: Path, gt_path: Path) -> Tuple[float, float]:
	pred = load_rgb_float(pred_path)
	gt = load_rgb_float(gt_path)
	if pred.shape != gt.shape:
		h = min(pred.shape[0], gt.shape[0])
		w = min(pred.shape[1], gt.shape[1])
		pred = pred[:h, :w]
		gt = gt[:h, :w]
	psnr = float(sk_psnr(gt, pred, data_range=1.0))
	ssim = float(sk_ssim(gt, pred, channel_axis=2, data_range=1.0))
	return psnr, ssim


def run_experts_once(
	input_img: Path,
	task_key: str,
	toolbox: Sequence,
	sample_cache_dir: Path,
	run_gpu_id: Optional[int],
	keep_intermediate: bool,
) -> Dict[str, Path]:
	outputs: Dict[str, Path] = {}
	for tool in toolbox:
		name = str(getattr(tool, "tool_name", tool.__class__.__name__)).lower()
		tool_dir = sample_cache_dir / name
		output_dir = tool_dir / "output"
		output_img = output_dir / "output.png"
		if output_img.exists():
			outputs[name] = output_img
			continue

		if tool_dir.exists():
			shutil.rmtree(tool_dir)
		input_dir = tool_dir / "input"
		input_dir.mkdir(parents=True, exist_ok=True)
		output_dir.mkdir(parents=True, exist_ok=True)
		shutil.copy(input_img, input_dir / "input.png")

		try:
			tool(input_dir=input_dir, output_dir=output_dir, silent=True, run_gpu_id=run_gpu_id)
		except Exception as err:
			print(f"[WARN][{task_key}] tool={name} failed on {input_img.name}: {err}")
			continue

		if output_img.exists():
			outputs[name] = output_img
		else:
			print(f"[WARN][{task_key}] tool={name} has no output.png on {input_img.name}")

		if not keep_intermediate:
			in_dir = tool_dir / "input"
			if in_dir.exists():
				shutil.rmtree(in_dir)

	return outputs


def build_candidate_infos(
	qe: QualityEvaluator,
	qe_task_name: str,
	expert_outputs: Dict[str, Path],
) -> List[CandidateInfo]:
	infos: List[CandidateInfo] = []
	for name, image_path in expert_outputs.items():
		row = qe._extract_features(str(image_path))
		task_score, task_details = qe._task_score(qe_task_name, row)
		general_score, _ = qe._general_score(row)
		infos.append(
			CandidateInfo(
				name=name,
				image_path=image_path,
				task_score=float(task_score),
				general_score=float(general_score),
				task_main_util=float(task_details.get("main_util", 0.0)),
			)
		)
	return infos


def select_candidate_for_alpha(candidates: Sequence[CandidateInfo], alpha: float) -> CandidateInfo:
	# Match QE logic: final score first, then task score, then task main util.
	best = candidates[0]
	best_final = alpha * best.task_score + (1.0 - alpha) * best.general_score
	for cand in candidates[1:]:
		cand_final = alpha * cand.task_score + (1.0 - alpha) * cand.general_score
		if cand_final > best_final + 1e-12:
			best = cand
			best_final = cand_final
			continue
		if abs(cand_final - best_final) <= 1e-12:
			if cand.task_score > best.task_score + 1e-12:
				best = cand
				best_final = cand_final
				continue
			if abs(cand.task_score - best.task_score) <= 1e-12 and cand.task_main_util > best.task_main_util:
				best = cand
				best_final = cand_final
	return best


def evaluate_task(
	task_key: str,
	dataset_root: Path,
	split: str,
	max_samples: int,
	sample_mode: str,
	base_seed: Optional[int],
	alphas: Sequence[float],
	work_dir: Path,
	run_gpu_id: Optional[int],
	keep_intermediate: bool,
) -> Dict[str, object]:
	spec = TASK_SPECS[task_key]
	wb_key = spec["wb_key"]
	qe_task = spec["qe_task"]
	toolbox = spec["toolbox"]

	input_dir = dataset_root / wb_key / split / "input"
	target_dir = dataset_root / wb_key / split / "target"
	if not input_dir.exists() or not target_dir.exists():
		raise FileNotFoundError(f"Missing WeatherBench split dirs: {input_dir} / {target_dir}")

	all_inputs = list_images(input_dir)
	task_seed = None if base_seed is None else int(base_seed + sum(ord(ch) for ch in task_key))
	rng = random.Random(task_seed)
	inputs = sample_inputs(all_inputs, max_samples=max_samples, sample_mode=sample_mode, rng=rng)
	if not inputs:
		raise RuntimeError(f"No input images found in {input_dir}")

	qe = QualityEvaluator(normalize=False)
	task_cache_root = work_dir / task_key
	task_cache_root.mkdir(parents=True, exist_ok=True)

	per_alpha_psnr: Dict[float, List[float]] = {a: [] for a in alphas}
	per_alpha_ssim: Dict[float, List[float]] = {a: [] for a in alphas}
	per_alpha_picks: Dict[float, Dict[str, int]] = {a: {} for a in alphas}
	skipped = 0

	for idx, inp in enumerate(inputs, start=1):
		gt = match_target(inp, target_dir)
		if gt is None:
			skipped += 1
			print(f"[WARN][{task_key}] no GT for {inp.name}, skipping")
			continue

		sample_dir = task_cache_root / inp.stem
		sample_dir.mkdir(parents=True, exist_ok=True)

		expert_outputs = run_experts_once(
			input_img=inp,
			task_key=task_key,
			toolbox=toolbox,
			sample_cache_dir=sample_dir,
			run_gpu_id=run_gpu_id,
			keep_intermediate=keep_intermediate,
		)
		if not expert_outputs:
			skipped += 1
			print(f"[WARN][{task_key}] no valid expert output on {inp.name}, skipping")
			continue

		candidates = build_candidate_infos(qe, qe_task, expert_outputs)
		if not candidates:
			skipped += 1
			continue

		for alpha in alphas:
			picked = select_candidate_for_alpha(candidates, alpha)
			psnr, ssim = compute_psnr_ssim(picked.image_path, gt)
			per_alpha_psnr[alpha].append(psnr)
			per_alpha_ssim[alpha].append(ssim)
			per_alpha_picks[alpha][picked.name] = per_alpha_picks[alpha].get(picked.name, 0) + 1

		if idx % 10 == 0:
			print(f"[{task_key}] processed {idx}/{len(inputs)}")

	summary_rows = []
	best_alpha = None
	best_score = -1e18
	for alpha in alphas:
		psnr_vals = per_alpha_psnr[alpha]
		ssim_vals = per_alpha_ssim[alpha]
		if not psnr_vals:
			row = {
				"alpha": alpha,
				"count": 0,
				"psnr_mean": None,
				"ssim_mean": None,
				"composite": None,
				"pick_hist": per_alpha_picks[alpha],
			}
			summary_rows.append(row)
			continue

		psnr_mean = float(np.mean(psnr_vals))
		ssim_mean = float(np.mean(ssim_vals))
		composite = psnr_mean + 100.0 * ssim_mean
		row = {
			"alpha": alpha,
			"count": len(psnr_vals),
			"psnr_mean": psnr_mean,
			"ssim_mean": ssim_mean,
			"composite": composite,
			"pick_hist": per_alpha_picks[alpha],
		}
		summary_rows.append(row)

		if composite > best_score:
			best_score = composite
			best_alpha = alpha

	return {
		"task": task_key,
		"wb_key": wb_key,
		"qe_task": qe_task,
		"sample_mode": sample_mode,
		"sample_seed": task_seed,
		"selected_inputs": [p.name for p in inputs],
		"requested_samples": len(inputs),
		"skipped_samples": skipped,
		"evaluated_samples": int(max(0, len(inputs) - skipped)),
		"best_alpha": best_alpha,
		"best_composite": best_score if best_alpha is not None else None,
		"rows": summary_rows,
	}


def main() -> None:
	args = parse_args()
	alphas = parse_alphas(args.alphas)
	args.work_dir.mkdir(parents=True, exist_ok=True)
	args.report_json.parent.mkdir(parents=True, exist_ok=True)

	all_results = {
		"dataset_root": str(args.dataset_root),
		"split": args.split,
		"max_samples": args.max_samples,
		"sample_mode": args.sample_mode,
		"seed": args.seed,
		"alphas": alphas,
		"tasks": {},
	}

	for task in args.tasks:
		print(f"\n=== Grid Search Start: {task} ===")
		task_result = evaluate_task(
			task_key=task,
			dataset_root=args.dataset_root,
			split=args.split,
			max_samples=args.max_samples,
			sample_mode=args.sample_mode,
			base_seed=args.seed,
			alphas=alphas,
			work_dir=args.work_dir,
			run_gpu_id=args.run_gpu_id,
			keep_intermediate=args.keep_intermediate,
		)
		all_results["tasks"][task] = task_result
		print(
			f"=== Grid Search Done: {task} | best_alpha={task_result['best_alpha']} "
			f"| evaluated={task_result['evaluated_samples']} ==="
		)

	args.report_json.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
	print(f"\nReport saved to: {args.report_json}")

	print("\n===== Best Alpha Summary =====")
	for task in args.tasks:
		task_res = all_results["tasks"][task]
		print(
			f"{task}: best_alpha={task_res['best_alpha']}, "
			f"best_composite={task_res['best_composite']}, "
			f"evaluated={task_res['evaluated_samples']}"
		)


if __name__ == "__main__":
	main()
