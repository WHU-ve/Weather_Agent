#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
import torchvision.transforms as transforms
import pyiqa

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from restoration_agent import RestorationAgent
from perception_module import predict_degradation_vector


def parse_args():
    parser = argparse.ArgumentParser(description='Full RTTS dehaze benchmark (no-reference metrics).')
    parser.add_argument('--dataset_dir', default='dataset/haze/RTTS/JPEGImages', help='RTTS hazy images directory.')
    parser.add_argument('--output_root', default='outputs_dehaze_rtts_full', help='Output root for restored images and reports.')
    parser.add_argument('--profile', choices=['fast', 'balanced', 'quality'], default='quality')
    parser.add_argument('--limit', type=int, default=0, help='Optional max images (0 means all).')
    parser.add_argument('--skip_existing', action='store_true', default=True)
    parser.add_argument('--parallel_workers', type=int, default=1)
    parser.add_argument('--parallel_gpu_ids', default='')
    parser.add_argument('--keep_intermediates', action='store_true')
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


def _validate_image(path: Path) -> str:
    if not path.exists():
        return 'missing_file'
    if path.stat().st_size == 0:
        return 'empty_file'
    try:
        with Image.open(path) as img:
            img.verify()
    except Exception as e:
        return f'corrupted_image: {e}'
    return ''


class NRMetricSuite:
    def __init__(self, device: str):
        self.device = device
        self.to_tensor = transforms.ToTensor()
        self.niqe = pyiqa.create_metric('niqe', device=device)
        self.musiq = pyiqa.create_metric('musiq', device=device)
        self.clipiqa = pyiqa.create_metric('clipiqa+', device=device)

    def evaluate(self, image_path: Path) -> Dict[str, float]:
        img = Image.open(image_path).convert('RGB')
        ten = self.to_tensor(img).unsqueeze(0).to(self.device)
        return {
            'NIQE': float(self.niqe(ten).item()),
            'MUSIQ': float(self.musiq(ten).item()),
            'CLIPIQA': float(self.clipiqa(ten).item()),
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
    os.environ['EXPERT_PARALLEL_WORKERS'] = str(max(1, int(args.parallel_workers)))
    if args.parallel_gpu_ids.strip():
        os.environ['EXPERT_PARALLEL_GPU_IDS'] = args.parallel_gpu_ids.strip()
    os.environ['KEEP_ALL_INTERMEDIATES'] = '1' if args.keep_intermediates else '0'

    project_root = Path(__file__).resolve().parents[3]
    dataset_dir = (project_root / args.dataset_dir).resolve()
    output_root = (project_root / args.output_root).resolve()
    restored_root = output_root / 'restored'
    output_root.mkdir(parents=True, exist_ok=True)
    restored_root.mkdir(parents=True, exist_ok=True)

    all_images = sorted([p for p in dataset_dir.iterdir() if p.is_file() and _is_image(p)])
    if args.limit > 0:
        all_images = all_images[:args.limit]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    metric_suite = NRMetricSuite(device)
    agent = RestorationAgent()

    rows = []
    failed_rows = []
    t0 = time.time()
    n_total = len(all_images)
    print(f'[RTTS] total images: {n_total}')

    for idx, hazy in enumerate(all_images, 1):
        sample_name = hazy.stem
        sample_out = restored_root / sample_name
        sample_out.mkdir(parents=True, exist_ok=True)
        final_out = sample_out / 'final_output.png'

        input_status = _validate_image(hazy)
        if input_status:
            print(f'[FAIL] {sample_name}: invalid input image ({input_status})')
            failed_rows.append({
                'sample': sample_name,
                'input_path': str(hazy),
                'stage': 'input_validation',
                'reason': input_status,
            })
            continue

        run_start = time.time()
        try:
            haze_before = float(predict_degradation_vector(str(hazy))[1])
        except Exception as e:
            print(f'[FAIL] {sample_name}: predict_degradation(input) failed: {e}')
            failed_rows.append({
                'sample': sample_name,
                'input_path': str(hazy),
                'stage': 'predict_before',
                'reason': str(e),
            })
            continue

        if not (args.skip_existing and final_out.exists()):
            try:
                agent.execute_plan(['dehaze'], str(hazy), str(sample_out))
            except Exception as e:
                print(f'[FAIL] {sample_name}: execute_plan failed: {e}')
                failed_rows.append({
                    'sample': sample_name,
                    'input_path': str(hazy),
                    'stage': 'execute_plan',
                    'reason': str(e),
                })
                continue

        if not final_out.exists():
            print(f'[FAIL] {sample_name}: no final_output.png')
            failed_rows.append({
                'sample': sample_name,
                'input_path': str(hazy),
                'stage': 'missing_output',
                'reason': 'no final_output.png',
            })
            continue

        try:
            haze_after = float(predict_degradation_vector(str(final_out))[1])
            nr = metric_suite.evaluate(final_out)
        except Exception as e:
            print(f'[FAIL] {sample_name}: post-eval failed: {e}')
            failed_rows.append({
                'sample': sample_name,
                'input_path': str(hazy),
                'stage': 'post_eval',
                'reason': str(e),
            })
            continue
        elapsed = time.time() - run_start

        row = {
            'sample': sample_name,
            'input_path': str(hazy),
            'output_path': str(final_out),
            'HazeProbBefore': haze_before,
            'HazeProbAfter': haze_after,
            'DeltaHazeProb': haze_before - haze_after,
            'NIQE': nr['NIQE'],
            'MUSIQ': nr['MUSIQ'],
            'CLIPIQA': nr['CLIPIQA'],
            'time_sec': elapsed,
        }
        rows.append(row)

        if idx % 50 == 0 or idx == n_total:
            print(f"[{idx}/{n_total}] {sample_name} | ΔHaze={row['DeltaHazeProb']:.4f} NIQE={row['NIQE']:.3f} MUSIQ={row['MUSIQ']:.3f}")

    csv_path = output_root / 'per_image_metrics.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'sample', 'input_path', 'output_path',
            'HazeProbBefore', 'HazeProbAfter', 'DeltaHazeProb',
            'NIQE', 'MUSIQ', 'CLIPIQA', 'time_sec'
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    failed_csv_path = output_root / 'failed_samples.csv'
    with failed_csv_path.open('w', newline='', encoding='utf-8') as f:
        fieldnames = ['sample', 'input_path', 'stage', 'reason']
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(failed_rows)

    summary = {
        'dataset': 'RTTS',
        'num_samples': len(rows),
        'num_failed': len(failed_rows),
        'profile': args.profile,
        'total_time_sec': time.time() - t0,
        'metrics': {
            'HazeProbBefore': mean_std([r['HazeProbBefore'] for r in rows]),
            'HazeProbAfter': mean_std([r['HazeProbAfter'] for r in rows]),
            'DeltaHazeProb': mean_std([r['DeltaHazeProb'] for r in rows]),
            'NIQE': mean_std([r['NIQE'] for r in rows]),
            'MUSIQ': mean_std([r['MUSIQ'] for r in rows]),
            'CLIPIQA': mean_std([r['CLIPIQA'] for r in rows]),
            'TIME_SEC': mean_std([r['time_sec'] for r in rows]),
        }
    }

    summary_path = output_root / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    print('[SUMMARY] RTTS')
    print(f"  n={summary['num_samples']}, failed={summary['num_failed']}, profile={summary['profile']}")
    print(f"  ΔHazeProb={summary['metrics']['DeltaHazeProb']['mean']:.4f}")
    print(f"  NIQE={summary['metrics']['NIQE']['mean']:.4f}, MUSIQ={summary['metrics']['MUSIQ']['mean']:.4f}, CLIPIQA={summary['metrics']['CLIPIQA']['mean']:.4f}")
    print(f"Saved: {summary_path}")
    print(f"Failed list: {failed_csv_path}")


if __name__ == '__main__':
    main()
