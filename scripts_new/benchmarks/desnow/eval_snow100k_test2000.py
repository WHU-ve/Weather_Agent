#!/usr/bin/env python3
import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Dict, List
import sys

import torch
from PIL import Image
import torchvision.transforms as transforms
import pyiqa

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from restoration_agent import RestorationAgent
from perception_module import (
    predict_degradation,
    predict_degradation_vector,
    release_model as release_perception_model,
)
from task_planner import TaskPlanner


def parse_args():
    parser = argparse.ArgumentParser(description='Snow100K test2000 desnow benchmark with full-reference metrics.')
    parser.add_argument('--snow_dir', default='dataset/snow/Snow100K/test2000/Snow', help='Snowy input directory.')
    parser.add_argument('--gt_dir', default='dataset/snow/Snow100K/test2000/Gt', help='Ground-truth directory.')
    parser.add_argument('--output_root', default='outputs_desnow_snow100k_test2000', help='Output root directory.')
    parser.add_argument('--profile', choices=['fast', 'balanced', 'quality'], default='quality')
    parser.add_argument('--limit', type=int, default=0, help='Optional max samples (0 means all).')
    parser.add_argument('--skip_existing', action='store_true', default=True)
    parser.add_argument('--parallel_workers', type=int, default=1)
    parser.add_argument('--parallel_gpu_ids', default='')
    parser.add_argument('--keep_intermediates', action='store_true')
    parser.add_argument('--execution_mode', choices=['full_pipeline', 'desnow_only'], default='full_pipeline',
                        help='full_pipeline: perception+planning+execution; desnow_only: force [desnow] only.')
    parser.add_argument('--planner_mode', choices=['legacy', 'perception_direct', 'qwen_only'],
                        default=os.getenv('TASK_PLANNER_MODE', 'qwen_only'),
                        help='Planner mode for full pipeline execution.')
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


class MetricSuite:
    def __init__(self, device: str):
        self.device = device
        self.to_tensor = transforms.ToTensor()
        self.psnr = pyiqa.create_metric('psnr', device=device)
        self.ssim = pyiqa.create_metric('ssim', device=device)
        self.vif = pyiqa.create_metric('vif', device=device)
        self.fsim = pyiqa.create_metric('fsim', device=device)
        self.niqe = pyiqa.create_metric('niqe', device=device)

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
    apply_profile(args.profile)
    os.environ['TASK_PLANNER_MODE'] = args.planner_mode
    os.environ['EXPERT_PARALLEL_WORKERS'] = str(max(1, int(args.parallel_workers)))
    if args.parallel_gpu_ids.strip():
        os.environ['EXPERT_PARALLEL_GPU_IDS'] = args.parallel_gpu_ids.strip()
    os.environ['KEEP_ALL_INTERMEDIATES'] = '1' if args.keep_intermediates else '0'

    snow_dir = (PROJECT_ROOT / args.snow_dir).resolve()
    gt_dir = (PROJECT_ROOT / args.gt_dir).resolve()
    output_root = (PROJECT_ROOT / args.output_root).resolve()
    restored_root = output_root / 'restored'
    output_root.mkdir(parents=True, exist_ok=True)
    restored_root.mkdir(parents=True, exist_ok=True)

    if not snow_dir.exists() or not gt_dir.exists():
        raise FileNotFoundError(f'Dataset path missing: snow={snow_dir}, gt={gt_dir}')

    all_inputs = sorted([p for p in snow_dir.iterdir() if p.is_file() and _is_image(p)])
    pairs = []
    for snow in all_inputs:
        gt = gt_dir / snow.name
        if gt.exists() and gt.is_file() and _is_image(gt):
            pairs.append((snow, gt))

    if args.limit > 0:
        pairs = pairs[:args.limit]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    metric_suite = MetricSuite(device)
    agent = RestorationAgent()

    rows = []
    failed_rows = []
    t0 = time.time()
    print(f'[Snow100K-test2000] total pairs: {len(pairs)}')

    for idx, (snow, gt) in enumerate(pairs, 1):
        sample_name = snow.stem
        sample_out = restored_root / sample_name
        sample_out.mkdir(parents=True, exist_ok=True)
        final_out = sample_out / 'final_output.png'

        run_start = time.time()
        if not (args.skip_existing and final_out.exists()):
            try:
                if args.execution_mode == 'desnow_only':
                    plan = ['desnow']
                    (sample_out / 'planner_info.json').write_text(
                        json.dumps({'execution_mode': 'desnow_only', 'plan': plan}, indent=2, ensure_ascii=False),
                        encoding='utf-8'
                    )
                    agent.execute_plan(plan, str(snow), str(sample_out))
                else:
                    degradation = predict_degradation(str(snow))
                    planner = TaskPlanner()
                    plan = planner.plan(degradation, image_path=str(snow))

                    if not plan:
                        raise RuntimeError(f'Planner returned empty plan. metadata={getattr(planner, "last_plan_metadata", {})}')

                    planner_info = {
                        'degradation': degradation,
                        'degradation_vector': [float(x) for x in predict_degradation_vector(str(snow), result=degradation).tolist()],
                        'plan': plan,
                        'planner_metadata': getattr(planner, 'last_plan_metadata', {}),
                    }
                    (sample_out / 'planner_info.json').write_text(
                        json.dumps(planner_info, indent=2, ensure_ascii=False),
                        encoding='utf-8'
                    )

                    if args.planner_mode in {'clip_direct', 'clip_vlm_fallback'}:
                        release_perception_model()

                    agent.execute_plan(plan, str(snow), str(sample_out))
            except Exception as e:
                print(f'[FAIL] {sample_name}: execute_plan failed: {e}')
                failed_rows.append({
                    'sample': sample_name,
                    'snow_path': str(snow),
                    'gt_path': str(gt),
                    'stage': 'execute_plan',
                    'reason': str(e),
                })
                continue

        if not final_out.exists():
            print(f'[FAIL] {sample_name}: no final_output.png')
            failed_rows.append({
                'sample': sample_name,
                'snow_path': str(snow),
                'gt_path': str(gt),
                'stage': 'missing_output',
                'reason': 'no final_output.png',
            })
            continue

        try:
            metrics = metric_suite.evaluate(final_out, gt)
        except Exception as e:
            print(f'[FAIL] {sample_name}: metric eval failed: {e}')
            failed_rows.append({
                'sample': sample_name,
                'snow_path': str(snow),
                'gt_path': str(gt),
                'stage': 'metric_eval',
                'reason': str(e),
            })
            continue

        elapsed = time.time() - run_start
        row = {
            'sample': sample_name,
            'snow_path': str(snow),
            'gt_path': str(gt),
            'output_path': str(final_out),
            'PSNR': metrics['PSNR'],
            'SSIM': metrics['SSIM'],
            'VIF': metrics['VIF'],
            'FSIM': metrics['FSIM'],
            'NIQE': metrics['NIQE'],
            'time_sec': elapsed,
        }
        rows.append(row)

        if idx % 50 == 0 or idx == len(pairs):
            print(f"[{idx}/{len(pairs)}] {sample_name} | PSNR={metrics['PSNR']:.3f} SSIM={metrics['SSIM']:.4f} NIQE={metrics['NIQE']:.3f}")

    csv_path = output_root / 'per_image_metrics.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'sample', 'snow_path', 'gt_path', 'output_path',
            'PSNR', 'SSIM', 'VIF', 'FSIM', 'NIQE', 'time_sec'
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    failed_csv_path = output_root / 'failed_samples.csv'
    with failed_csv_path.open('w', newline='', encoding='utf-8') as f:
        fieldnames = ['sample', 'snow_path', 'gt_path', 'stage', 'reason']
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(failed_rows)

    summary = {
        'dataset': 'Snow100K-test2000',
        'num_pairs': len(pairs),
        'num_success': len(rows),
        'num_failed': len(failed_rows),
        'profile': args.profile,
        'total_time_sec': time.time() - t0,
        'metrics': {
            'PSNR': mean_std([r['PSNR'] for r in rows]),
            'SSIM': mean_std([r['SSIM'] for r in rows]),
            'VIF': mean_std([r['VIF'] for r in rows]),
            'FSIM': mean_std([r['FSIM'] for r in rows]),
            'NIQE': mean_std([r['NIQE'] for r in rows]),
            'TIME_SEC': mean_std([r['time_sec'] for r in rows]),
        }
    }

    summary_path = output_root / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    print('[SUMMARY] Snow100K-test2000')
    print(f"  pairs={summary['num_pairs']}, success={summary['num_success']}, failed={summary['num_failed']}, profile={summary['profile']}")
    print(f"  PSNR={summary['metrics']['PSNR']['mean']:.3f}, SSIM={summary['metrics']['SSIM']['mean']:.4f}, VIF={summary['metrics']['VIF']['mean']:.4f}, FSIM={summary['metrics']['FSIM']['mean']:.4f}, NIQE={summary['metrics']['NIQE']['mean']:.3f}")
    print(f"Saved: {summary_path}")
    print(f"Failed list: {failed_csv_path}")


if __name__ == '__main__':
    main()
