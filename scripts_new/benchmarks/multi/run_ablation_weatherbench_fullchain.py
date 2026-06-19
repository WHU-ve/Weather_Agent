#!/usr/bin/env python3
import csv
import json
import os
import random
import time
from pathlib import Path
from statistics import mean
from typing import Optional

import torch
from PIL import Image
import torchvision.transforms as transforms
import pyiqa

PROJECT_ROOT = Path(__file__).resolve().parents[3]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import main as run_pipeline


DATA_ROOT = PROJECT_ROOT / 'dataset' / 'multi' / 'WeatherBench'
OUT_ROOT = PROJECT_ROOT / 'output' / 'ablation_weatherbench_fullchain'

# E0 已由用户完成，这里从 E1 开始。
EXPERIMENTS = [
    {
        # E2: 无感知，直接规划；且不启用重规划
        'id': 'E2_no_perception_direct_planning_no_replan',
        'planner_mode': 'qwen_only',
        'profile': 'quality',
        'disable_dynamic_replan': True,
        'enable_local_replan': False,
        'disable_perception': True,
        'random_single_expert': False,
        'limit_per_task': 50,
    },
    {
        # E4: 每一步随机单专家执行；且不启用重规划
        'id': 'E4_random_single_expert_no_replan',
        'planner_mode': 'qwen_only',
        'profile': 'quality',
        'disable_dynamic_replan': True,
        'enable_local_replan': False,
        'disable_perception': False,
        'random_single_expert': True,
        'random_single_expert_seed': 2026,
        'limit_per_task': 50,
    },
]


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def _collect_pairs(task: str, limit_per_task: Optional[int] = None, sample_seed: int = 2026):
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
    if limit_per_task is not None and limit_per_task > 0 and len(pairs) > limit_per_task:
        rng = random.Random(sample_seed + sum(ord(c) for c in task))
        pairs = rng.sample(pairs, limit_per_task)
        pairs = sorted(pairs, key=lambda x: x[0].name)
    return pairs


def _apply_profile(profile: str):
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
        os.environ['QE_ENABLE_TASK_AWARE'] = '1'
        os.environ['QE_ENABLE_RESIDUAL_PENALTY'] = '1'
        os.environ['ENABLE_STEP_SCORE_GUARD'] = '1'
        os.environ['STEP_SCORE_MAX_DROP'] = '0.02'
        os.environ['PREFER_EXPERT_WHEN_CLOSE'] = '1'
        os.environ['PREFER_EXPERT_MARGIN'] = '0.02'


def _set_base_env():
    os.environ.setdefault('QE_STRICT_OFFLINE', '1')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
    os.environ.setdefault('WEATHER_PERCEPTION_SUBPROCESS', '1')
    os.environ.setdefault('WEATHER_PERCEPTION_TOPK_GPUS', '2')
    os.environ.setdefault('WEATHER_PERCEPTION_DEVICE_MAP', 'auto')
    os.environ.setdefault('WEATHER_PERCEPTION_RELEASE_AFTER_INFER', '1')
    os.environ.setdefault('WEATHER_UTILS_DIR', 'utils_new')
    os.environ.setdefault('WEATHER_CKPT_DIR', 'pretrained_ckpts_new')
    os.environ.setdefault('USE_FLAX', '0')
    os.environ.setdefault('TRANSFORMERS_NO_FLAX', '1')
    os.environ.setdefault('TASK_PLANNER_ISOLATED_ENV', 'weather_agent_planner')
    os.environ.setdefault('EXPERT_MIN_FREE_MB', '6000')
    os.environ.setdefault('EXPERT_GPU_WAIT_SECONDS', '60')
    os.environ.setdefault('EXPERT_GPU_POLL_SECONDS', '2')


def _topk(vals, k):
    arr = sorted(vals, reverse=True)[:k]
    return float(mean(arr)) if arr else 0.0


def _init_metrics(device: str):
    to_tensor = transforms.ToTensor()
    psnr = pyiqa.create_metric('psnr', device=device)
    ssim = pyiqa.create_metric('ssim', device=device)

    def evaluate(pred_path: Path, gt_path: Path):
        pred = Image.open(pred_path).convert('RGB')
        gt = Image.open(gt_path).convert('RGB')
        if pred.size != gt.size:
            pred = pred.resize(gt.size, Image.BICUBIC)
        t_pred = to_tensor(pred).unsqueeze(0).to(device)
        t_gt = to_tensor(gt).unsqueeze(0).to(device)
        with torch.no_grad():
            return float(psnr(t_pred, t_gt).item()), float(ssim(t_pred, t_gt).item())

    return evaluate


def run_experiment(exp: dict, eval_pair):
    exp_id = exp['id']
    out_dir = OUT_ROOT / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    _set_base_env()
    _apply_profile(exp['profile'])
    os.environ['TASK_PLANNER_MODE'] = exp['planner_mode']
    os.environ['ENABLE_LOCAL_REPLAN'] = '1' if exp.get('enable_local_replan', True) else '0'
    os.environ['LOCAL_REPLAN_MAX'] = str(int(exp.get('local_replan_max', 3)))
    os.environ['LOCAL_REPLAN_MAX_FIRST_PRIORITY'] = str(int(exp.get('local_replan_max_first_priority', 3)))
    os.environ['WEATHER_DISABLE_PERCEPTION'] = '1' if exp.get('disable_perception', False) else '0'
    os.environ['RANDOM_SINGLE_EXPERT'] = '1' if exp.get('random_single_expert', False) else '0'
    os.environ['RANDOM_SINGLE_EXPERT_SEED'] = str(int(exp.get('random_single_expert_seed', 2026)))
    if exp['disable_dynamic_replan']:
        os.environ['ENABLE_DYNAMIC_REPLAN'] = '0'

    manifest = {
        'experiment': exp,
        'resolved_env': {
            'TASK_PLANNER_MODE': os.environ.get('TASK_PLANNER_MODE', ''),
            'ENABLE_DYNAMIC_REPLAN': os.environ.get('ENABLE_DYNAMIC_REPLAN', ''),
            'DYNAMIC_REPLAN_MAX_STEPS': os.environ.get('DYNAMIC_REPLAN_MAX_STEPS', ''),
            'DYNAMIC_REPLAN_PERCEPTION_INTERVAL': os.environ.get('DYNAMIC_REPLAN_PERCEPTION_INTERVAL', ''),
            'DYNAMIC_REPLAN_MIN_IMPROVE': os.environ.get('DYNAMIC_REPLAN_MIN_IMPROVE', ''),
            'WEATHER_DISABLE_PERCEPTION': os.environ.get('WEATHER_DISABLE_PERCEPTION', ''),
            'RANDOM_SINGLE_EXPERT': os.environ.get('RANDOM_SINGLE_EXPERT', ''),
            'RANDOM_SINGLE_EXPERT_SEED': os.environ.get('RANDOM_SINGLE_EXPERT_SEED', ''),
            'ENABLE_LOCAL_REPLAN': os.environ.get('ENABLE_LOCAL_REPLAN', ''),
            'LOCAL_REPLAN_MAX': os.environ.get('LOCAL_REPLAN_MAX', ''),
            'LOCAL_REPLAN_MAX_FIRST_PRIORITY': os.environ.get('LOCAL_REPLAN_MAX_FIRST_PRIORITY', ''),
        },
    }
    (out_dir / 'run_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

    rows = []
    failures = []
    tasks = ['rain', 'haze', 'snow']
    t0 = time.time()

    limit_per_task = int(exp.get('limit_per_task', 0))
    sample_seed = int(exp.get('sample_seed', 2026))
    for task in tasks:
        task_out_dir = out_dir / task
        already_done = len(list(task_out_dir.glob('*/final_output.png'))) if task_out_dir.exists() else 0
        if limit_per_task > 0 and already_done >= limit_per_task:
            print(f"[{exp_id}] [{task}] skip: already have {already_done} outputs >= limit {limit_per_task}")
            continue

        pairs = _collect_pairs(task, limit_per_task=limit_per_task, sample_seed=sample_seed)
        for idx, (inp, gt) in enumerate(pairs, 1):
            sample_dir = out_dir / task / inp.stem
            sample_dir.mkdir(parents=True, exist_ok=True)
            final_out = sample_dir / 'final_output.png'
            try:
                if not final_out.exists():
                    run_pipeline(str(inp), str(sample_dir), planner_mode=exp['planner_mode'], run_restoration=True)
                if not final_out.exists():
                    raise RuntimeError('final_output.png missing')
                psnr, ssim = eval_pair(final_out, gt)
                rows.append({
                    'experiment': exp_id,
                    'task': task,
                    'sample': inp.name,
                    'input_path': str(inp),
                    'gt_path': str(gt),
                    'output_path': str(final_out),
                    'PSNR': psnr,
                    'SSIM': ssim,
                })
                print(f"[{exp_id}] [{task}] {idx}/{len(pairs)} done: {inp.name} PSNR={psnr:.3f} SSIM={ssim:.4f}")
            except Exception as e:
                failures.append({'experiment': exp_id, 'task': task, 'sample': inp.name, 'reason': str(e)})
                print(f"[{exp_id}] [{task}] {idx}/{len(pairs)} fail: {inp.name} :: {e}")

    per_image_csv = out_dir / 'per_image_metrics.csv'
    with per_image_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['experiment', 'task', 'sample', 'input_path', 'gt_path', 'output_path', 'PSNR', 'SSIM'])
        w.writeheader()
        w.writerows(rows)

    failed_csv = out_dir / 'failed_samples.csv'
    with failed_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['experiment', 'task', 'sample', 'reason'])
        w.writeheader()
        w.writerows(failures)

    task_summary = {}
    for task in tasks:
        tr = [r for r in rows if r['task'] == task]
        ps = [float(r['PSNR']) for r in tr]
        ss = [float(r['SSIM']) for r in tr]
        task_summary[task] = {
            'num_success': len(tr),
            'num_failed': len([x for x in failures if x['task'] == task]),
            'psnr_mean': float(mean(ps)) if ps else 0.0,
            'ssim_mean': float(mean(ss)) if ss else 0.0,
            'psnr_top20_mean': _topk(ps, 20),
            'ssim_top20_mean': _topk(ss, 20),
            'psnr_top50_mean': _topk(ps, 50),
            'ssim_top50_mean': _topk(ss, 50),
            'psnr_top100_mean': _topk(ps, 100),
            'ssim_top100_mean': _topk(ss, 100),
        }

    overall_ps = [float(r['PSNR']) for r in rows]
    overall_ss = [float(r['SSIM']) for r in rows]
    summary = {
        'experiment': exp,
        'num_success': len(rows),
        'num_failed': len(failures),
        'elapsed_sec': time.time() - t0,
        'tasks': task_summary,
        'overall': {
            'psnr_mean': float(mean(overall_ps)) if overall_ps else 0.0,
            'ssim_mean': float(mean(overall_ss)) if overall_ss else 0.0,
            'psnr_top20_mean_macro': float(mean([task_summary[t]['psnr_top20_mean'] for t in tasks])),
            'ssim_top20_mean_macro': float(mean([task_summary[t]['ssim_top20_mean'] for t in tasks])),
            'psnr_top50_mean_macro': float(mean([task_summary[t]['psnr_top50_mean'] for t in tasks])),
            'ssim_top50_mean_macro': float(mean([task_summary[t]['ssim_top50_mean'] for t in tasks])),
            'psnr_top100_mean_macro': float(mean([task_summary[t]['psnr_top100_mean'] for t in tasks])),
            'ssim_top100_mean_macro': float(mean([task_summary[t]['ssim_top100_mean'] for t in tasks])),
        },
        'sanity_check': {
            'planner_mode': exp['planner_mode'],
            'dynamic_replan_enabled': os.environ.get('ENABLE_DYNAMIC_REPLAN', '1') != '0',
        },
    }

    summary_path = out_dir / 'summary.json'
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"[{exp_id}] summary saved: {summary_path}")
    return summary


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_pair = _init_metrics(device)

    all_summary_rows = []
    for exp in EXPERIMENTS:
        s = run_experiment(exp, eval_pair)
        all_summary_rows.append({
            'experiment': exp['id'],
            'planner_mode': exp['planner_mode'],
            'profile': exp['profile'],
            'disable_dynamic_replan': int(bool(exp['disable_dynamic_replan'])),
            'enable_local_replan': int(bool(exp.get('enable_local_replan', True))),
            'num_success': s['num_success'],
            'num_failed': s['num_failed'],
            'overall_psnr_mean': s['overall']['psnr_mean'],
            'overall_ssim_mean': s['overall']['ssim_mean'],
            'overall_psnr_top20_macro': s['overall']['psnr_top20_mean_macro'],
            'overall_ssim_top20_macro': s['overall']['ssim_top20_mean_macro'],
            'overall_psnr_top50_macro': s['overall']['psnr_top50_mean_macro'],
            'overall_ssim_top50_macro': s['overall']['ssim_top50_mean_macro'],
            'overall_psnr_top100_macro': s['overall']['psnr_top100_mean_macro'],
            'overall_ssim_top100_macro': s['overall']['ssim_top100_mean_macro'],
            'elapsed_sec': s['elapsed_sec'],
        })

    summary_csv = OUT_ROOT / 'ablation_summary.csv'
    with summary_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(all_summary_rows[0].keys()))
        w.writeheader()
        w.writerows(all_summary_rows)
    print(f'Ablation summary csv saved: {summary_csv}')


if __name__ == '__main__':
    main()
