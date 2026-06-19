#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import torch
from PIL import Image
import torchvision.transforms as transforms
import pyiqa

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from restoration_agent import RestorationAgent
from task_planner import TaskPlanner
from perception_module import predict_degradation


def parse_args():
    parser = argparse.ArgumentParser(description='Derain benchmark on two datasets with 5 metrics.')
    parser.add_argument('--dataset_root', default='dataset/rain', help='Root folder containing rain datasets.')
    parser.add_argument('--datasets', nargs='+', default=['Rain100H', 'rain100H_train'],
                        help='Dataset folder names under dataset_root.')
    parser.add_argument('--output_root', default='outputs_derain_benchmark', help='Benchmark output root directory.')
    parser.add_argument('--profile', choices=['fast', 'balanced', 'quality'], default='quality',
                        help='Runtime profile preset for dynamic replanning.')
    parser.add_argument('--force_derain', action='store_true', default=True,
                        help='Force running derain task only for each sample.')
    parser.add_argument('--no_force_derain', action='store_true',
                        help='Disable derain-only mode and use full planner.')
    parser.add_argument('--limit', type=int, default=0, help='Optional max samples per dataset (0 means all).')
    parser.add_argument('--skip_existing', action='store_true', default=True,
                        help='Skip samples that already have final_output.png.')
    parser.add_argument('--parallel_workers', type=int, default=1,
                        help='Parallel expert workers per step (quality unchanged, walltime faster).')
    parser.add_argument('--parallel_gpu_ids', default='',
                        help='Comma-separated GPU ids for parallel expert scheduling, e.g. "0,1".')
    parser.add_argument('--keep_intermediates', action='store_true',
                        help='Keep all intermediate tool outputs (default: cleanup unselected outputs).')
    return parser.parse_args()


def apply_profile(profile: str):
    if profile == 'fast':
        os.environ['ENABLE_DYNAMIC_REPLAN'] = '1'
        os.environ['DYNAMIC_REPLAN_MAX_STEPS'] = '2'
        os.environ['DYNAMIC_REPLAN_PERCEPTION_INTERVAL'] = '2'
        os.environ['DYNAMIC_REPLAN_MIN_IMPROVE'] = '0.0'
        os.environ['TASK_THRESHOLD'] = '0.72'
        os.environ['TASK_MIN_TRIGGER_SCORE'] = '0.55'
        os.environ['TASK_MAX_REPEAT_PER_STEP'] = '1'
        os.environ['ENABLE_STEP_SCORE_GUARD'] = '1'
        os.environ['STEP_SCORE_MAX_DROP'] = '0.01'
    elif profile == 'balanced':
        os.environ['ENABLE_DYNAMIC_REPLAN'] = '1'
        os.environ['DYNAMIC_REPLAN_MAX_STEPS'] = '4'
        os.environ['DYNAMIC_REPLAN_PERCEPTION_INTERVAL'] = '1'
        os.environ['DYNAMIC_REPLAN_MIN_IMPROVE'] = '0.0'
        os.environ['TASK_THRESHOLD'] = '0.65'
        os.environ['TASK_MIN_TRIGGER_SCORE'] = '0.45'
        os.environ['TASK_MAX_REPEAT_PER_STEP'] = '2'
        os.environ['ENABLE_STEP_SCORE_GUARD'] = '1'
        os.environ['STEP_SCORE_MAX_DROP'] = '0.005'
    else:
        os.environ['ENABLE_DYNAMIC_REPLAN'] = '1'
        os.environ['DYNAMIC_REPLAN_MAX_STEPS'] = '6'
        os.environ['DYNAMIC_REPLAN_PERCEPTION_INTERVAL'] = '1'
        os.environ['DYNAMIC_REPLAN_MIN_IMPROVE'] = '0.001'
        os.environ['TASK_THRESHOLD'] = '0.58'
        os.environ['TASK_MIN_TRIGGER_SCORE'] = '0.35'
        os.environ['TASK_MAX_REPEAT_PER_STEP'] = '2'
        os.environ['ENABLE_STEP_SCORE_GUARD'] = '1'
        os.environ['STEP_SCORE_MAX_DROP'] = '0.0'


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def collect_pairs_rain100h(flat_dir: Path) -> List[Tuple[Path, Path]]:
    pairs = []
    for p in sorted(flat_dir.iterdir()):
        if not p.is_file() or not _is_image(p):
            continue
        stem = p.stem.lower()
        if stem.startswith('norain-'):
            continue
        m = re.search(r'-(\d+)$', p.stem)
        if not m:
            continue
        idx = m.group(1)
        gt = flat_dir / f'norain-{idx}{p.suffix}'
        if not gt.exists():
            alt = list(flat_dir.glob(f'norain-{idx}.*'))
            if alt:
                gt = alt[0]
            else:
                continue
        pairs.append((p, gt))
    return pairs


def collect_pairs_split(split_dir: Path) -> List[Tuple[Path, Path]]:
    rain_dir = split_dir / 'rain'
    gt_dir = split_dir / 'norain'
    if not rain_dir.exists() or not gt_dir.exists():
        return []
    pairs = []
    for p in sorted(rain_dir.iterdir()):
        if not p.is_file() or not _is_image(p):
            continue
        gt = gt_dir / p.name
        if not gt.exists():
            alt = list(gt_dir.glob(f'{p.stem}.*'))
            if not alt:
                continue
            gt = alt[0]
        pairs.append((p, gt))
    return pairs


def collect_pairs(dataset_dir: Path) -> List[Tuple[Path, Path]]:
    split_pairs = collect_pairs_split(dataset_dir)
    if split_pairs:
        return split_pairs
    return collect_pairs_rain100h(dataset_dir)


class MetricSuite:
    def __init__(self, device: str):
        self.device = device
        self.to_tensor = transforms.ToTensor()
        self.psnr = pyiqa.create_metric('psnr', device=device)
        self.ssim = pyiqa.create_metric('ssim', device=device)
        self.vif = pyiqa.create_metric('vif', device=device)
        self.fsim = pyiqa.create_metric('fsim', device=device)
        self.niqe = pyiqa.create_metric('niqe', device=device)

    def _load_tensor(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert('RGB')
        return self.to_tensor(img).unsqueeze(0).to(self.device)

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
            'VIF': float(self.vif(pred, gt).item()),
            'FSIM': float(self.fsim(pred, gt).item()),
            'NIQE': float(self.niqe(pred).item()),
        }


def mean_std(values: List[float]) -> Dict[str, float]:
    if not values:
        return {'mean': 0.0, 'std': 0.0}
    arr = torch.tensor(values, dtype=torch.float32)
    return {
        'mean': float(arr.mean().item()),
        'std': float(arr.std(unbiased=False).item()),
    }


def main():
    args = parse_args()
    if args.no_force_derain:
        args.force_derain = False

    apply_profile(args.profile)
    os.environ['EXPERT_PARALLEL_WORKERS'] = str(max(1, int(args.parallel_workers)))
    if args.parallel_gpu_ids.strip():
        os.environ['EXPERT_PARALLEL_GPU_IDS'] = args.parallel_gpu_ids.strip()
    os.environ['KEEP_ALL_INTERMEDIATES'] = '1' if args.keep_intermediates else '0'

    dataset_root = (PROJECT_ROOT / args.dataset_root).resolve()
    output_root = (PROJECT_ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    metric_suite = MetricSuite(device=device)
    planner = TaskPlanner()
    agent = RestorationAgent()

    all_summary = {}

    for dataset_name in args.datasets:
        ds_dir = dataset_root / dataset_name
        if not ds_dir.exists():
            print(f'[WARN] Dataset not found, skip: {ds_dir}')
            continue

        pairs = collect_pairs(ds_dir)
        if args.limit > 0:
            pairs = pairs[:args.limit]

        if not pairs:
            print(f'[WARN] No valid pairs found in {ds_dir}')
            continue

        ds_out = output_root / dataset_name
        restored_root = ds_out / 'restored'
        restored_root.mkdir(parents=True, exist_ok=True)
        csv_path = ds_out / 'per_image_metrics.csv'
        summary_path = ds_out / 'summary.json'

        rows = []
        plan_source_counter = {}
        plan_task_counter = {}
        t0 = time.time()

        print(f'\n[DATASET] {dataset_name}: {len(pairs)} samples')
        for idx, (rainy, gt) in enumerate(pairs, 1):
            sample_name = rainy.stem
            sample_out = restored_root / sample_name
            sample_out.mkdir(parents=True, exist_ok=True)
            final_out = sample_out / 'final_output.png'

            run_start = time.time()
            if not (args.skip_existing and final_out.exists()):
                if args.force_derain:
                    plan = ['derain']
                    planner_source = 'forced_derain'
                    planner_meta = {}
                else:
                    perception = predict_degradation(str(rainy))
                    plan = planner.plan(perception, image_path=str(rainy))
                    planner_meta = getattr(planner, 'last_plan_metadata', {})
                    planner_source = planner_meta.get('planner_source', 'clip_direct')
                    if not plan:
                        plan = ['derain']
                        planner_source = 'clip_empty_fallback_derain'

                agent.execute_plan(plan, str(rainy), str(sample_out))

            for task_name in plan:
                plan_task_counter[task_name] = plan_task_counter.get(task_name, 0) + 1
            plan_source_counter[planner_source] = plan_source_counter.get(planner_source, 0) + 1

            if not final_out.exists():
                print(f'[FAIL] {dataset_name} {sample_name}: no final_output.png')
                continue

            metrics = metric_suite.evaluate(final_out, gt)
            elapsed = time.time() - run_start

            row = {
                'dataset': dataset_name,
                'sample': sample_name,
                'rainy_path': str(rainy),
                'gt_path': str(gt),
                'output_path': str(final_out),
                'PSNR': metrics['PSNR'],
                'SSIM': metrics['SSIM'],
                'VIF': metrics['VIF'],
                'FSIM': metrics['FSIM'],
                'NIQE': metrics['NIQE'],
                'time_sec': elapsed,
                'plan': json.dumps(plan, ensure_ascii=False),
                'planner_source': planner_source,
            }
            rows.append(row)

            if idx % 10 == 0 or idx == len(pairs):
                print(f'  [{idx}/{len(pairs)}] {sample_name} | '
                      f"PSNR={metrics['PSNR']:.3f} SSIM={metrics['SSIM']:.4f} NIQE={metrics['NIQE']:.3f}")

        fieldnames = [
            'dataset', 'sample', 'rainy_path', 'gt_path', 'output_path',
            'PSNR', 'SSIM', 'VIF', 'FSIM', 'NIQE', 'time_sec', 'plan', 'planner_source'
        ]
        with csv_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        summary = {
            'dataset': dataset_name,
            'num_samples': len(rows),
            'total_time_sec': time.time() - t0,
            'profile': args.profile,
            'force_derain': args.force_derain,
            'planner_source_distribution': plan_source_counter,
            'planned_task_distribution': plan_task_counter,
            'metrics': {
                'PSNR': mean_std([r['PSNR'] for r in rows]),
                'SSIM': mean_std([r['SSIM'] for r in rows]),
                'VIF': mean_std([r['VIF'] for r in rows]),
                'FSIM': mean_std([r['FSIM'] for r in rows]),
                'NIQE': mean_std([r['NIQE'] for r in rows]),
                'TIME_SEC': mean_std([r['time_sec'] for r in rows]),
            }
        }
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
        all_summary[dataset_name] = summary

        print(f"[SUMMARY] {dataset_name}: "
              f"PSNR={summary['metrics']['PSNR']['mean']:.3f}, "
              f"SSIM={summary['metrics']['SSIM']['mean']:.4f}, "
              f"VIF={summary['metrics']['VIF']['mean']:.4f}, "
              f"FSIM={summary['metrics']['FSIM']['mean']:.4f}, "
              f"NIQE={summary['metrics']['NIQE']['mean']:.3f}")

    overall_path = output_root / 'overall_summary.json'
    overall_path.write_text(json.dumps(all_summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\nDone. Overall summary saved to: {overall_path}')


if __name__ == '__main__':
    main()
