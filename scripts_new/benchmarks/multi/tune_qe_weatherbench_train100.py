#!/usr/bin/env python3
import argparse
import csv
import itertools
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from PIL import UnidentifiedImageError
import torchvision.transforms as transforms
import pyiqa

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quality_evaluator import QualityEvaluator
from perception_module import predict_degradation_vector, release_model as release_perception_model
from utils.deraining import deraining_toolbox
from utils.dehazing import dehazing_toolbox
from utils.desnowing import desnowing_toolbox


TASK_TO_DATASET = {
    'derain': 'rain',
    'dehaze': 'haze',
    'desnow': 'snow',
}

TASK_TO_TOOLBOX = {
    'derain': deraining_toolbox,
    'dehaze': dehazing_toolbox,
    'desnow': desnowing_toolbox,
}

TASK_TO_PROB_INDEX = {
    'derain': 0,
    'dehaze': 1,
    'desnow': 3,
}


@dataclass
class CandidateRecord:
    sample: str
    candidate: str
    image_path: str
    psnr: float
    ssim: float
    clipiqa: float
    musiq: float
    niqe: float
    residual: float


class MetricSuite:
    def __init__(self, device: str):
        self.device = device
        self.to_tensor = transforms.ToTensor()
        self.psnr = pyiqa.create_metric('psnr', device=device)
        self.ssim = pyiqa.create_metric('ssim', device=device)

    def evaluate(self, pred_path: Path, gt_path: Path) -> Dict[str, float]:
        pred_img = Image.open(pred_path).convert('RGB')
        gt_img = Image.open(gt_path).convert('RGB')
        if pred_img.size != gt_img.size:
            pred_img = pred_img.resize(gt_img.size, Image.BICUBIC)

        pred = self.to_tensor(pred_img).unsqueeze(0).to(self.device)
        gt = self.to_tensor(gt_img).unsqueeze(0).to(self.device)
        return {
            'PSNR': float(self.psnr(pred, gt).item()),
            'SSIM': float(self.ssim(pred, gt).item()),
        }


def parse_args():
    parser = argparse.ArgumentParser(description='Task-wise QE grid search on WeatherBench split with random sampling.')
    parser.add_argument('--split', choices=['train', 'test'], default='test', help='Dataset split to sample from.')
    parser.add_argument('--limit', type=int, default=150, help='Samples per task from the chosen split.')
    parser.add_argument('--seed', type=int, default=20260409, help='Random seed for sampling.')
    parser.add_argument('--output_root', default='outputs_qe_tune_weatherbench_test150', help='Tuning output directory.')
    parser.add_argument('--include_tasks', default='derain,dehaze,desnow', help='Comma-separated tasks.')
    parser.add_argument('--weight_values', default='0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65',
                        help='Discrete values for each weight before enforcing sum-to-one.')
    parser.add_argument('--weight_min', type=float, default=0.05)
    parser.add_argument('--weight_max', type=float, default=0.65)
    parser.add_argument('--residual_grid_derain', default='2,3,4,5,6,7,8')
    parser.add_argument('--residual_grid_desnow', default='1,1.5,2,2.5,3,3.5,4')
    parser.add_argument('--residual_grid_dehaze', default='1,1.5,2,2.5,3,3.5,4')
    parser.add_argument('--step_score_grid', default='0.0,0.005,0.01,0.015,0.02', help='Grid for STEP_SCORE_MAX_DROP.')
    parser.add_argument('--score_guard_extra_drop', type=float, default=0.01)
    parser.add_argument('--resume', action='store_true', default=True)
    return parser.parse_args()


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def _is_readable_image(path: Path) -> bool:
    if not path.exists() or not path.is_file() or not _is_image(path):
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def _parse_float_grid(raw: str) -> List[float]:
    vals = []
    for tok in raw.split(','):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(float(tok))
    return sorted(set(vals))


def weight_grid(vals: List[float], weight_min: float, weight_max: float) -> List[tuple[float, float, float]]:
    combos = []
    for wc in vals:
        for wm in vals:
            wn = round(1.0 - wc - wm, 10)
            if wn < weight_min or wn > weight_max:
                continue
            if wc < weight_min or wc > weight_max:
                continue
            if wm < weight_min or wm > weight_max:
                continue
            combos.append((wc, wm, wn))
    combos = sorted(set(combos))
    return combos


def penalty_grid(task: str, grids: Dict[str, List[float]]) -> List[float]:
    if task in grids:
        return grids[task]
    return [0.0]


def scaled_metrics(clipiqa: float, musiq: float, niqe: float):
    c = float(np.clip(clipiqa / 1.0, 0.0, 1.0))
    m = float(np.clip(musiq / 100.0, 0.0, 1.0))
    n = float(np.clip(niqe / 10.0, 0.0, 1.0))
    return c, m, n


def quality_score(rec: CandidateRecord, wc: float, wm: float, wn: float, residual_lambda: float) -> float:
    c, m, n = scaled_metrics(rec.clipiqa, rec.musiq, rec.niqe)
    return wc * c + wm * m - wn * n - residual_lambda * rec.residual


def objective(psnr: np.ndarray, ssim: np.ndarray) -> float:
    return float(psnr.mean() + 20.0 * ssim.mean())


def gather_pairs(task: str, split: str, limit: int, seed: int):
    dname = TASK_TO_DATASET[task]
    input_dir = PROJECT_ROOT / f'dataset/multi/WeatherBench/{dname}/{split}/input'
    gt_dir = PROJECT_ROOT / f'dataset/multi/WeatherBench/{dname}/{split}/target'
    if not input_dir.exists() or not gt_dir.exists():
        raise FileNotFoundError(f'Missing {split} dirs for {task}: {input_dir} / {gt_dir}')
    inputs = sorted([p for p in input_dir.iterdir() if p.is_file() and _is_image(p)])
    pairs = []
    bad_count = 0
    for inp in inputs:
        gt = gt_dir / inp.name
        if gt.exists() and gt.is_file() and _is_image(gt):
            if not _is_readable_image(inp) or not _is_readable_image(gt):
                bad_count += 1
                continue
            pairs.append((inp, gt))
    if bad_count > 0:
        print(f'[WARN][{task}] skipped unreadable pairs during gather: {bad_count}')
    rng = np.random.default_rng(seed + sum(ord(c) for c in task) + (0 if split == 'train' else 1000))
    if limit > 0 and len(pairs) > limit:
        idx = rng.choice(len(pairs), size=limit, replace=False)
        idx = sorted(idx.tolist())
        pairs = [pairs[i] for i in idx]
    return pairs


def ensure_candidate(
    task: str,
    sample: str,
    inp: Path,
    gt: Path,
    candidates_root: Path,
    metric_suite: MetricSuite,
    evaluator: QualityEvaluator,
) -> List[CandidateRecord]:
    if not _is_readable_image(gt):
        print(f'[WARN][{task}] unreadable gt, skip sample={sample}, gt={gt}')
        return []

    sample_dir = candidates_root / sample
    sample_dir.mkdir(parents=True, exist_ok=True)

    input_png = sample_dir / 'input.png'
    if not input_png.exists():
        shutil.copy(inp, input_png)

    toolbox = TASK_TO_TOOLBOX[task]
    for tool in toolbox:
        tool_dir = sample_dir / f'tool_{tool.tool_name}'
        out_png = tool_dir / 'output.png'
        if out_png.exists():
            continue

        if tool.work_dir is None or not tool.work_dir.exists() or tool.script_path is None or not tool.script_path.exists():
            continue

        in_dir = tool_dir / 'input'
        out_dir = tool_dir / 'output'
        if tool_dir.exists():
            shutil.rmtree(tool_dir)
        in_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(input_png, in_dir / 'input.png')

        try:
            tool(input_dir=in_dir, output_dir=out_dir, silent=True)
            generated = out_dir / 'output.png'
            if generated.exists():
                out_png.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(generated, out_png)
        except Exception:
            continue

    records: List[CandidateRecord] = []
    prob_idx = TASK_TO_PROB_INDEX[task]

    for candidate_name, image_path in [('input', input_png)] + [
        (d.name.replace('tool_', ''), d / 'output.png')
        for d in sorted(sample_dir.glob('tool_*'))
    ]:
        if not image_path.exists():
            continue
        if not _is_readable_image(image_path):
            continue
        try:
            m = metric_suite.evaluate(image_path, gt)
            clipiqa, musiq, niqe = evaluator._metrics(str(image_path))
        except (UnidentifiedImageError, OSError, ValueError):
            continue
        try:
            probs = np.asarray(predict_degradation_vector(str(image_path)), dtype=np.float32).reshape(-1)
            residual = float(probs[prob_idx]) if prob_idx < probs.size else 0.0
        except Exception:
            residual = 0.0
        records.append(
            CandidateRecord(
                sample=sample,
                candidate=candidate_name,
                image_path=str(image_path),
                psnr=float(m['PSNR']),
                ssim=float(m['SSIM']),
                clipiqa=float(clipiqa),
                musiq=float(musiq),
                niqe=float(niqe),
                residual=residual,
            )
        )

    return records


def evaluate_grid(
    records_by_sample: Dict[str, List[CandidateRecord]],
    task: str,
    w_grid: List[tuple[float, float, float]],
    p_grid: List[float],
    step_score_grid: List[float],
    extra_drop: float,
):

    best = None
    leaderboard = []
    for (wc, wm, wn), residual_lambda, step_drop in itertools.product(w_grid, p_grid, step_score_grid):
        chosen_psnr = []
        chosen_ssim = []
        chosen_expert = 0

        for sample, recs in records_by_sample.items():
            if not recs:
                continue

            recs_sorted = sorted(
                recs,
                key=lambda r: quality_score(r, wc, wm, wn, residual_lambda),
                reverse=True,
            )
            best_rec = recs_sorted[0]
            input_rec = next((r for r in recs if r.candidate == 'input'), None)
            if input_rec is None:
                input_rec = best_rec

            if best_rec.candidate != 'input':
                q_in = quality_score(input_rec, wc, wm, wn, residual_lambda)
                q_best = quality_score(best_rec, wc, wm, wn, residual_lambda)
                if q_best < q_in - (step_drop + extra_drop):
                    best_rec = input_rec

            chosen_psnr.append(best_rec.psnr)
            chosen_ssim.append(best_rec.ssim)
            if best_rec.candidate != 'input':
                chosen_expert += 1

        if len(chosen_psnr) == 0:
            continue

        psnr_arr = np.asarray(chosen_psnr, dtype=np.float64)
        ssim_arr = np.asarray(chosen_ssim, dtype=np.float64)
        score = objective(psnr_arr, ssim_arr)
        row = {
            'task': task,
            'weights': [wc, wm, wn],
            'residual_penalty': residual_lambda,
            'step_score_max_drop': step_drop,
            'mean_psnr': float(psnr_arr.mean()),
            'mean_ssim': float(ssim_arr.mean()),
            'objective': float(score),
            'expert_pick_ratio': float(chosen_expert / len(chosen_psnr)),
            'num_samples': int(len(chosen_psnr)),
        }
        leaderboard.append(row)
        if best is None or row['objective'] > best['objective']:
            best = row

    leaderboard.sort(key=lambda x: x['objective'], reverse=True)
    return best, leaderboard[:20]


def main():
    args = parse_args()
    tasks = [t.strip() for t in args.include_tasks.split(',') if t.strip()]
    output_root = (PROJECT_ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    step_grid = _parse_float_grid(args.step_score_grid)
    weight_values = _parse_float_grid(args.weight_values)
    w_grid = weight_grid(weight_values, args.weight_min, args.weight_max)
    residual_grids = {
        'derain': _parse_float_grid(args.residual_grid_derain),
        'desnow': _parse_float_grid(args.residual_grid_desnow),
        'dehaze': _parse_float_grid(args.residual_grid_dehaze),
    }
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    metric_suite = MetricSuite(device)
    evaluator = QualityEvaluator(normalize=False)

    started = time.time()
    global_result = {
        'limit': args.limit,
        'tasks': {},
        'search_space': {
            'weights': w_grid,
            'weight_values': weight_values,
            'weight_min': args.weight_min,
            'weight_max': args.weight_max,
            'residual_grids': residual_grids,
            'step_score_max_drop': step_grid,
            'score_guard_extra_drop': args.score_guard_extra_drop,
        },
    }

    for task in tasks:
        task_out = output_root / task
        candidates_root = task_out / 'candidates'
        task_out.mkdir(parents=True, exist_ok=True)
        candidates_root.mkdir(parents=True, exist_ok=True)

        pairs = gather_pairs(task, args.split, args.limit, args.seed)
        print(f'[TUNE][{task}] split={args.split} pairs={len(pairs)} seed={args.seed}')

        records_by_sample: Dict[str, List[CandidateRecord]] = {}
        for idx, (inp, gt) in enumerate(pairs, 1):
            sample = inp.stem
            records = ensure_candidate(
                task=task,
                sample=sample,
                inp=inp,
                gt=gt,
                candidates_root=candidates_root,
                metric_suite=metric_suite,
                evaluator=evaluator,
            )
            records_by_sample[sample] = records
            if idx % 10 == 0 or idx == len(pairs):
                print(f'  [{task}] candidate build {idx}/{len(pairs)}')

        rows = []
        for sample, recs in records_by_sample.items():
            for r in recs:
                rows.append({
                    'sample': r.sample,
                    'candidate': r.candidate,
                    'image_path': r.image_path,
                    'psnr': r.psnr,
                    'ssim': r.ssim,
                    'clipiqa': r.clipiqa,
                    'musiq': r.musiq,
                    'niqe': r.niqe,
                    'residual': r.residual,
                })
        with (task_out / 'candidate_metrics.csv').open('w', newline='', encoding='utf-8') as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

        p_grid = penalty_grid(task, residual_grids)
        best, top20 = evaluate_grid(
            records_by_sample=records_by_sample,
            task=task,
            w_grid=w_grid,
            p_grid=p_grid,
            step_score_grid=step_grid,
            extra_drop=args.score_guard_extra_drop,
        )

        task_result = {
            'best': best,
            'top20': top20,
            'num_samples': len(pairs),
        }
        (task_out / 'best_config.json').write_text(
            json.dumps(task_result, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        global_result['tasks'][task] = task_result
        print(f"[TUNE][{task}] best={best}")

        release_perception_model()

    global_result['elapsed_sec'] = time.time() - started
    (output_root / 'best_configs_all_tasks.json').write_text(
        json.dumps(global_result, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(f'[DONE] saved => {output_root}')


if __name__ == '__main__':
    main()
