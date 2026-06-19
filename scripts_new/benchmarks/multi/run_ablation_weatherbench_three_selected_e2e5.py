#!/usr/bin/env python3
import csv
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_ablation_weatherbench_e2e4_isolated as iso
import run_ablation_weatherbench_fullchain as full
from main import main as run_pipeline

DATA_ROOT = PROJECT_ROOT / 'dataset' / 'multi' / 'WeatherBench'
OUT_ROOT = PROJECT_ROOT / 'output' / 'ablation_weatherbench_three_selected_e2e5'
SELECTED = {'rain': '323.jpg', 'haze': '097.jpg', 'snow': '563.jpg'}
TASKS = ['rain', 'haze', 'snow']
EXPERIMENTS = [
    {'id': 'E2_no_perception_direct_planning_no_replan', 'runner': 'isolated', 'kind': 'e2', 'seed': 2026},
    {'id': 'E3_no_planning_qwen_fast_dynamic', 'runner': 'fullchain', 'planner_mode': 'qwen_only', 'profile': 'fast', 'disable_dynamic_replan': False},
    {'id': 'E4_random_single_expert_no_replan', 'runner': 'isolated', 'kind': 'e4', 'seed': 2026},
    {'id': 'E5_no_dynamic_replan_qwen_quality', 'runner': 'fullchain', 'planner_mode': 'qwen_only', 'profile': 'quality', 'disable_dynamic_replan': True},
]


def selected_pairs():
    pairs = []
    for task in TASKS:
        name = SELECTED[task]
        inp = DATA_ROOT / task / 'test' / 'input' / name
        gt = DATA_ROOT / task / 'test' / 'target' / name
        if not inp.exists() or not gt.exists():
            raise FileNotFoundError(f'Missing pair: {inp} / {gt}')
        pairs.append((task, inp, gt))
    return pairs


def configure_fullchain(exp):
    full._set_base_env()
    full._apply_profile(exp['profile'])
    os.environ['TASK_PLANNER_MODE'] = exp['planner_mode']
    os.environ['ENABLE_LOCAL_REPLAN'] = '0'
    os.environ['WEATHER_DISABLE_PERCEPTION'] = '0'
    os.environ['RANDOM_SINGLE_EXPERT'] = '0'
    os.environ['ALLOW_INPUT_AS_CANDIDATE'] = '0'
    if exp.get('disable_dynamic_replan'):
        os.environ['ENABLE_DYNAMIC_REPLAN'] = '0'


def configure_isolated():
    iso._set_env_no_replan()
    os.environ['ALLOW_INPUT_AS_CANDIDATE'] = '0'


def run_one_experiment(exp, eval_pair):
    exp_id = exp['id']
    out_dir = OUT_ROOT / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, failures = [], []
    t0 = time.time()

    if exp['runner'] == 'isolated':
        configure_isolated()
    else:
        configure_fullchain(exp)

    manifest = {
        'experiment': exp,
        'selected': SELECTED,
        'resolved_env': {
            'TASK_PLANNER_MODE': os.environ.get('TASK_PLANNER_MODE', ''),
            'ENABLE_DYNAMIC_REPLAN': os.environ.get('ENABLE_DYNAMIC_REPLAN', ''),
            'ENABLE_LOCAL_REPLAN': os.environ.get('ENABLE_LOCAL_REPLAN', ''),
            'ALLOW_INPUT_AS_CANDIDATE': os.environ.get('ALLOW_INPUT_AS_CANDIDATE', ''),
            'RANDOM_SINGLE_EXPERT': os.environ.get('RANDOM_SINGLE_EXPERT', ''),
        },
    }
    (out_dir / 'run_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

    for task, inp, gt in selected_pairs():
        sample_dir = out_dir / task / inp.stem
        sample_dir.mkdir(parents=True, exist_ok=True)
        final_out = sample_dir / 'final_output.png'
        try:
            if not final_out.exists():
                if exp['runner'] == 'isolated':
                    if exp['kind'] == 'e2':
                        plan = iso._plan_e2_without_perception(str(inp))
                    else:
                        plan = iso._plan_e4_with_perception(str(inp))
                    final_out = iso._run_random_single_expert(plan, str(inp), sample_dir, int(exp.get('seed', 2026)))
                else:
                    run_pipeline(str(inp), str(sample_dir), planner_mode=exp['planner_mode'], run_restoration=True)
            if not final_out.exists():
                raise RuntimeError('final_output.png missing')
            psnr, ssim = eval_pair(final_out, gt)
            row = {'experiment': exp_id, 'task': task, 'sample': inp.name, 'input_path': str(inp), 'gt_path': str(gt), 'output_path': str(final_out), 'PSNR': psnr, 'SSIM': ssim}
            rows.append(row)
            print(f'[{exp_id}] {task}/{inp.name}: PSNR={psnr:.3f}, SSIM={ssim:.4f}')
        except Exception as exc:
            failures.append({'experiment': exp_id, 'task': task, 'sample': inp.name, 'reason': str(exc)})
            print(f'[{exp_id}] {task}/{inp.name} failed: {exc}')

    with (out_dir / 'per_image_metrics.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['experiment', 'task', 'sample', 'input_path', 'gt_path', 'output_path', 'PSNR', 'SSIM'])
        w.writeheader(); w.writerows(rows)
    with (out_dir / 'failed_samples.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['experiment', 'task', 'sample', 'reason'])
        w.writeheader(); w.writerows(failures)

    task_summary = {}
    for task in TASKS:
        tr = [r for r in rows if r['task'] == task]
        task_summary[task] = {
            'num_success': len(tr),
            'num_failed': len([f for f in failures if f['task'] == task]),
            'psnr_mean': float(mean([float(r['PSNR']) for r in tr])) if tr else 0.0,
            'ssim_mean': float(mean([float(r['SSIM']) for r in tr])) if tr else 0.0,
        }
    summary = {
        'experiment': exp,
        'selected': SELECTED,
        'num_success': len(rows),
        'num_failed': len(failures),
        'elapsed_sec': time.time() - t0,
        'tasks': task_summary,
        'overall': {
            'psnr_mean': float(mean([float(r['PSNR']) for r in rows])) if rows else 0.0,
            'ssim_mean': float(mean([float(r['SSIM']) for r in rows])) if rows else 0.0,
        },
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_pair = iso._init_eval(device)
    summary_rows = []
    for exp in EXPERIMENTS:
        s = run_one_experiment(exp, eval_pair)
        summary_rows.append({
            'experiment': exp['id'],
            'num_success': s['num_success'],
            'num_failed': s['num_failed'],
            'overall_psnr_mean': s['overall']['psnr_mean'],
            'overall_ssim_mean': s['overall']['ssim_mean'],
            'rain_psnr': s['tasks']['rain']['psnr_mean'],
            'rain_ssim': s['tasks']['rain']['ssim_mean'],
            'haze_psnr': s['tasks']['haze']['psnr_mean'],
            'haze_ssim': s['tasks']['haze']['ssim_mean'],
            'snow_psnr': s['tasks']['snow']['psnr_mean'],
            'snow_ssim': s['tasks']['snow']['ssim_mean'],
            'elapsed_sec': s['elapsed_sec'],
        })
    with (OUT_ROOT / 'ablation_summary.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)
    print(f'Ablation summary csv saved: {OUT_ROOT / "ablation_summary.csv"}')


if __name__ == '__main__':
    main()
