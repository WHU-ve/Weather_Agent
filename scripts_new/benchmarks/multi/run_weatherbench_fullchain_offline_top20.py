#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
from pathlib import Path
from statistics import mean

import torch
from PIL import Image
import torchvision.transforms as transforms
import pyiqa

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from restoration_agent import RestorationAgent

DATA_ROOT = PROJECT_ROOT / 'dataset' / 'multi' / 'WeatherBench'
DEFAULT_OUT_ROOT = PROJECT_ROOT / 'output' / 'weatherbench_fullchain_offline_top20'
TASK_TO_STEP = {'rain': 'derain', 'haze': 'dehaze', 'snow': 'desnow'}


def parse_args():
    p = argparse.ArgumentParser(description='WeatherBench full test benchmark (fixed task experts, skip perception/planning).')
    p.add_argument('--output_root', default=str(DEFAULT_OUT_ROOT), help='Output root directory')
    p.add_argument('--topk', type=int, default=20, help='Top-k for PSNR/SSIM averaging')
    p.add_argument('--overwrite_existing', action='store_true', help='Re-run samples even if final_output exists')
    p.add_argument('--tasks', default='rain,haze,snow', help='Comma-separated tasks subset')
    return p.parse_args()


def _is_image(p: Path) -> bool:
    return p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def _collect_pairs(task: str):
    inp_dir = DATA_ROOT / task / 'test' / 'input'
    gt_dir = DATA_ROOT / task / 'test' / 'target'
    gt_map = {p.name: p for p in gt_dir.iterdir() if p.is_file() and _is_image(p)}
    pairs = []
    for inp in sorted(inp_dir.iterdir()):
        if not inp.is_file() or not _is_image(inp):
            continue
        gt = gt_map.get(inp.name)
        if gt is not None:
            pairs.append((inp, gt))
    return pairs


def _topk_avg(rows, key: str, k: int = 20) -> float:
    vals = sorted([float(r[key]) for r in rows], reverse=True)[:k]
    return float(mean(vals)) if vals else 0.0


def _init_metrics(device: str):
    to_tensor = transforms.ToTensor()
    psnr = pyiqa.create_metric('psnr', device=device)
    ssim = pyiqa.create_metric('ssim', device=device)

    def eval_pair(pred_path: Path, gt_path: Path):
        pred = Image.open(pred_path).convert('RGB')
        gt = Image.open(gt_path).convert('RGB')
        if pred.size != gt.size:
            pred = pred.resize(gt.size, Image.BICUBIC)
        t_pred = to_tensor(pred).unsqueeze(0).to(device)
        t_gt = to_tensor(gt).unsqueeze(0).to(device)
        with torch.no_grad():
            return {
                'PSNR': float(psnr(t_pred, t_gt).item()),
                'SSIM': float(ssim(t_pred, t_gt).item()),
            }

    return eval_pair


def run_all(args):
    os.environ.setdefault('QE_STRICT_OFFLINE', '1')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
    os.environ.setdefault('WEATHER_UTILS_DIR', 'utils_new')
    os.environ.setdefault('WEATHER_CKPT_DIR', 'pretrained_ckpts_new')
    os.environ.setdefault('USE_FLAX', '0')
    os.environ.setdefault('TRANSFORMERS_NO_FLAX', '1')
    # Critical: lock fixed one-step plan and skip dynamic re-perception.
    os.environ['ENABLE_DYNAMIC_REPLAN'] = '0'
    os.environ['TASK_PLANNER_LOCK_PLAN'] = '1'

    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_pair = _init_metrics(device)
    agent = RestorationAgent()

    tasks = [t.strip().lower() for t in args.tasks.split(',') if t.strip()]
    tasks = [t for t in tasks if t in TASK_TO_STEP]
    if not tasks:
        raise ValueError('No valid tasks selected.')

    task_summaries = {}
    all_rows = []

    for task in tasks:
        step = TASK_TO_STEP[task]
        pairs = _collect_pairs(task)
        task_out_root = out_root / task
        task_out_root.mkdir(parents=True, exist_ok=True)

        rows = []
        failures = []
        for idx, (inp, gt) in enumerate(pairs, 1):
            sample_out_dir = task_out_root / inp.stem
            sample_out_dir.mkdir(parents=True, exist_ok=True)
            final_out = sample_out_dir / 'final_output.png'

            try:
                if args.overwrite_existing and final_out.exists():
                    final_out.unlink()

                if not final_out.exists():
                    agent.execute_plan([step], str(inp), str(sample_out_dir))

                if not final_out.exists():
                    raise RuntimeError('final_output.png missing')

                metrics = eval_pair(final_out, gt)
                row = {
                    'task': task,
                    'sample': inp.name,
                    'fixed_step': step,
                    'input_path': str(inp),
                    'gt_path': str(gt),
                    'output_path': str(final_out),
                    'PSNR': metrics['PSNR'],
                    'SSIM': metrics['SSIM'],
                }
                rows.append(row)
                all_rows.append(row)
                print(f"[{task}] {idx}/{len(pairs)} done: {inp.name} PSNR={metrics['PSNR']:.3f} SSIM={metrics['SSIM']:.4f}")
            except Exception as e:
                failures.append({'task': task, 'sample': inp.name, 'reason': str(e)})
                print(f"[{task}] {idx}/{len(pairs)} fail: {inp.name} :: {e}")

        task_csv = out_root / f'{task}_per_image_metrics.csv'
        with task_csv.open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=['task', 'sample', 'fixed_step', 'input_path', 'gt_path', 'output_path', 'PSNR', 'SSIM'])
            w.writeheader()
            w.writerows(rows)

        fail_csv = out_root / f'{task}_failed.csv'
        with fail_csv.open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=['task', 'sample', 'reason'])
            w.writeheader()
            w.writerows(failures)

        task_summaries[task] = {
            'num_total': len(pairs),
            'num_success': len(rows),
            'num_failed': len(failures),
            'topk': int(args.topk),
            'topk_psnr_mean': _topk_avg(rows, 'PSNR', args.topk),
            'topk_ssim_mean': _topk_avg(rows, 'SSIM', args.topk),
        }

    macro_psnr = float(mean([task_summaries[t]['topk_psnr_mean'] for t in tasks]))
    macro_ssim = float(mean([task_summaries[t]['topk_ssim_mean'] for t in tasks]))

    summary = {
        'mode': 'fixed_task_experts_no_perception_no_planning',
        'tasks': task_summaries,
        'overall': {
            'topk': int(args.topk),
            'topk_psnr_mean_macro': macro_psnr,
            'topk_ssim_mean_macro': macro_ssim,
            'num_all_success': len(all_rows),
        },
    }

    summary_path = out_root / 'summary_topk_fixed_task.json'
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Saved summary to: {summary_path}')


if __name__ == '__main__':
    run_all(parse_args())
