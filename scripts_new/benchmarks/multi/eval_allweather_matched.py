#!/usr/bin/env python3
import argparse
import contextlib
import csv
import io
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

from perception_module import (
    predict_degradation,
    predict_degradation_vector,
    release_model as release_perception_model,
)
from restoration_agent import RestorationAgent
from task_planner import TaskPlanner


def parse_args():
    parser = argparse.ArgumentParser(description='Allweather matched subset full-pipeline benchmark.')
    parser.add_argument('--input_dir', default='dataset/multi/allweather/input', help='Allweather degraded input dir.')
    parser.add_argument('--gt_dir', default='dataset/multi/allweather/gt', help='Allweather GT dir.')
    parser.add_argument('--file_list', default='dataset/multi/allweather/allweather.txt', help='Optional file list.')
    parser.add_argument('--output_root', default='outputs_multi_allweather_full', help='Output root dir.')
    parser.add_argument('--profile', choices=['fast', 'balanced', 'quality'], default='quality')
    parser.add_argument('--planner_mode', choices=['legacy', 'perception_direct', 'qwen_only'], default='qwen_only')
    parser.add_argument('--disable_dynamic_replan', action='store_true',
                        help='Force disable dynamic replanning regardless of profile presets.')
    parser.add_argument('--limit', type=int, default=0, help='Optional max samples (0 for all).')
    parser.add_argument('--skip_existing', action='store_true', default=True)
    parser.add_argument('--parallel_workers', type=int, default=2)
    parser.add_argument('--parallel_gpu_ids', default='0,1')
    parser.add_argument('--keep_intermediates', action='store_true')
    parser.add_argument('--concise_log', action='store_true', help='Only print per-sample completion/failure lines.')
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
        os.environ['DYNAMIC_REPLAN_MAX_STEPS'] = '3'
        os.environ['DYNAMIC_REPLAN_PERCEPTION_INTERVAL'] = '1'
        os.environ['DYNAMIC_REPLAN_MIN_IMPROVE'] = '0.015'
        os.environ['TASK_THRESHOLD'] = '0.58'
        os.environ['TASK_MIN_TRIGGER_SCORE'] = '0.35'
        os.environ['TASK_MAX_REPEAT_PER_STEP'] = '2'
        os.environ['TASK_PRESENCE_PROB_THRESHOLD'] = '0.68'
        os.environ['TASK_DIRECT_MIN_TOP1'] = '0.68'
        os.environ['TASK_DIRECT_MIN_MARGIN'] = '0.18'
        os.environ['QE_ENABLE_TASK_AWARE'] = '1'
        os.environ['QE_ENABLE_RESIDUAL_PENALTY'] = '1'
        os.environ['QE_RESIDUAL_PENALTY_DERAIN'] = '4.0'
        os.environ['QE_RESIDUAL_PENALTY_DESNOW'] = '2.0'
        os.environ['QE_RESIDUAL_PENALTY_DEHAZE'] = '1.8'
        # Task-aware weights tuned for current project distribution (rain-heavy, then snow, then haze).
        os.environ['QE_WEIGHTS_DERAIN'] = '0.34,0.21,0.45'
        os.environ['QE_WEIGHTS_DESNOW'] = '0.33,0.27,0.40'
        os.environ['QE_WEIGHTS_DEHAZE'] = '0.28,0.37,0.35'
        os.environ['ENABLE_STEP_SCORE_GUARD'] = '1'
        os.environ['STEP_SCORE_MAX_DROP'] = '0.02'
        os.environ['PREFER_EXPERT_WHEN_CLOSE'] = '1'
        os.environ['PREFER_EXPERT_MARGIN'] = '0.02'


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
    return {'mean': float(arr.mean().item()), 'std': float(arr.std(unbiased=False).item())}


def collect_pairs(input_dir: Path, gt_dir: Path, file_list: Path) -> List[tuple[Path, Path]]:
    gt_map = {p.name: p for p in gt_dir.iterdir() if p.is_file() and _is_image(p)}

    if file_list.exists():
        names = []
        for line in file_list.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line:
                continue
            names.append(Path(line).name)
        pairs = []
        for name in names:
            inp = input_dir / name
            gt = gt_map.get(name)
            if inp.exists() and inp.is_file() and _is_image(inp) and gt is not None:
                pairs.append((inp, gt))
        return pairs

    pairs = []
    for inp in sorted(input_dir.iterdir()):
        if inp.is_file() and _is_image(inp) and inp.name in gt_map:
            pairs.append((inp, gt_map[inp.name]))
    return pairs


def main():
    args = parse_args()
    apply_profile(args.profile)
    if args.disable_dynamic_replan:
        os.environ['ENABLE_DYNAMIC_REPLAN'] = '0'

    os.environ['TASK_PLANNER_MODE'] = args.planner_mode
    os.environ['EXPERT_PARALLEL_WORKERS'] = str(max(1, int(args.parallel_workers)))
    if args.parallel_gpu_ids.strip():
        os.environ['EXPERT_PARALLEL_GPU_IDS'] = args.parallel_gpu_ids.strip()
    os.environ['KEEP_ALL_INTERMEDIATES'] = '1' if args.keep_intermediates else '0'

    input_dir = (PROJECT_ROOT / args.input_dir).resolve()
    gt_dir = (PROJECT_ROOT / args.gt_dir).resolve()
    file_list = (PROJECT_ROOT / args.file_list).resolve()
    output_root = (PROJECT_ROOT / args.output_root).resolve()
    restored_root = output_root / 'restored'
    output_root.mkdir(parents=True, exist_ok=True)
    restored_root.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists() or not gt_dir.exists():
        raise FileNotFoundError(f'Dataset path missing: input={input_dir}, gt={gt_dir}')

    pairs = collect_pairs(input_dir, gt_dir, file_list)
    if args.limit > 0:
        pairs = pairs[:args.limit]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    metric_suite = MetricSuite(device)
    agent = RestorationAgent()

    rows = []
    failed_rows = []
    task_presence_counter: Dict[str, int] = {}

    t0 = time.time()
    print(f'[Allweather matched] total pairs: {len(pairs)}')

    for idx, (inp, gt) in enumerate(pairs, 1):
        sample_name = inp.stem
        sample_out = restored_root / sample_name
        sample_out.mkdir(parents=True, exist_ok=True)
        final_out = sample_out / 'final_output.png'

        run_start = time.time()
        if not (args.skip_existing and final_out.exists()):
            try:
                if args.concise_log:
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        degradation = predict_degradation(str(inp))
                        planner = TaskPlanner()
                        plan = planner.plan(degradation, image_path=str(inp))
                else:
                    degradation = predict_degradation(str(inp))
                    planner = TaskPlanner()
                    plan = planner.plan(degradation, image_path=str(inp))
                if not plan:
                    raise RuntimeError(f'Planner returned empty plan. metadata={getattr(planner, "last_plan_metadata", {})}')

                planner_info = {
                    'degradation': degradation,
                    'degradation_vector': [float(x) for x in predict_degradation_vector(str(inp), result=degradation).tolist()],
                    'plan': plan,
                    'planner_metadata': getattr(planner, 'last_plan_metadata', {}),
                }
                (sample_out / 'planner_info.json').write_text(json.dumps(planner_info, ensure_ascii=False, indent=2), encoding='utf-8')

                release_perception_model()
                if args.concise_log:
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        agent.execute_plan(plan, str(inp), str(sample_out))
                else:
                    agent.execute_plan(plan, str(inp), str(sample_out))
            except Exception as e:
                if args.concise_log:
                    print(f'第{idx}张处理失败: {sample_name}')
                else:
                    print(f'[FAIL] {sample_name}: execute_plan failed: {e}')
                failed_rows.append({
                    'sample': sample_name,
                    'input_path': str(inp),
                    'gt_path': str(gt),
                    'stage': 'execute_plan',
                    'reason': str(e),
                })
                continue

        if not final_out.exists():
            if args.concise_log:
                print(f'第{idx}张处理失败: {sample_name}')
            else:
                print(f'[FAIL] {sample_name}: no final_output.png')
            failed_rows.append({
                'sample': sample_name,
                'input_path': str(inp),
                'gt_path': str(gt),
                'stage': 'missing_output',
                'reason': 'no final_output.png',
            })
            continue

        try:
            metrics = metric_suite.evaluate(final_out, gt)
        except Exception as e:
            if args.concise_log:
                print(f'第{idx}张处理失败: {sample_name}')
            else:
                print(f'[FAIL] {sample_name}: metric eval failed: {e}')
            failed_rows.append({
                'sample': sample_name,
                'input_path': str(inp),
                'gt_path': str(gt),
                'stage': 'metric_eval',
                'reason': str(e),
            })
            continue

        planner_info_path = sample_out / 'planner_info.json'
        plan = []
        if planner_info_path.exists():
            try:
                info = json.loads(planner_info_path.read_text(encoding='utf-8'))
                plan = info.get('plan', [])
            except Exception:
                plan = []
        for t in sorted(set(plan)):
            task_presence_counter[t] = task_presence_counter.get(t, 0) + 1

        elapsed = time.time() - run_start
        rows.append({
            'sample': sample_name,
            'input_path': str(inp),
            'gt_path': str(gt),
            'output_path': str(final_out),
            'PSNR': metrics['PSNR'],
            'SSIM': metrics['SSIM'],
            'VIF': metrics['VIF'],
            'FSIM': metrics['FSIM'],
            'NIQE': metrics['NIQE'],
            'time_sec': elapsed,
            'plan': '|'.join(plan),
        })

        if args.concise_log:
            print(f'第{idx}张处理完毕: {sample_name}')
        elif idx % 50 == 0 or idx == len(pairs):
            print(f"[{idx}/{len(pairs)}] {sample_name} | PSNR={metrics['PSNR']:.3f} SSIM={metrics['SSIM']:.4f} NIQE={metrics['NIQE']:.3f}")

    csv_path = output_root / 'per_image_metrics.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        fieldnames = ['sample', 'input_path', 'gt_path', 'output_path', 'PSNR', 'SSIM', 'VIF', 'FSIM', 'NIQE', 'time_sec', 'plan']
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    failed_csv_path = output_root / 'failed_samples.csv'
    with failed_csv_path.open('w', newline='', encoding='utf-8') as f:
        fieldnames = ['sample', 'input_path', 'gt_path', 'stage', 'reason']
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(failed_rows)

    summary = {
        'dataset': 'allweather-matched',
        'num_pairs': len(pairs),
        'num_success': len(rows),
        'num_failed': len(failed_rows),
        'profile': args.profile,
        'planner_mode': args.planner_mode,
        'total_time_sec': time.time() - t0,
        'task_allocation': {'sample_level_presence': task_presence_counter},
        'metrics': {
            'PSNR': mean_std([r['PSNR'] for r in rows]),
            'SSIM': mean_std([r['SSIM'] for r in rows]),
            'VIF': mean_std([r['VIF'] for r in rows]),
            'FSIM': mean_std([r['FSIM'] for r in rows]),
            'NIQE': mean_std([r['NIQE'] for r in rows]),
            'TIME_SEC': mean_std([r['time_sec'] for r in rows]),
        },
    }

    summary_path = output_root / 'summary.json'
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('[SUMMARY] allweather-matched')
    print(f"pairs={summary['num_pairs']}, success={summary['num_success']}, failed={summary['num_failed']}")
    print(f"Saved: {summary_path}")


if __name__ == '__main__':
    main()
