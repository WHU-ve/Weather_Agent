#!/usr/bin/env python3
import csv
import json
import os
import random
import shutil
import time
from pathlib import Path
from statistics import mean

import torch
from PIL import Image
import torchvision.transforms as transforms
import pyiqa

PROJECT_ROOT = Path(__file__).resolve().parents[3]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from task_planner import TaskPlanner
from quality_evaluator import QualityEvaluator
from perception_module import predict_degradation

DATA_ROOT = PROJECT_ROOT / 'dataset' / 'multi' / 'WeatherBench'
OUT_ROOT = PROJECT_ROOT / 'output' / 'ablation_weatherbench_isolated_e2e4'
TASKS = ['rain', 'haze', 'snow']
STEP_ORDER = ['desnow', 'derain', 'dehaze']

EXPERIMENTS = [
    {
        'id': 'E2_no_perception_direct_planning_no_replan',
        'kind': 'e2',
        'limit_per_task': 50,
        'sample_seed': 2026,
        'random_single_expert_seed': 2026,
    },
    {
        'id': 'E4_random_single_expert_no_replan',
        'kind': 'e4',
        'limit_per_task': 50,
        'sample_seed': 2026,
        'random_single_expert_seed': 2026,
    },
]


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def _set_env_no_replan():
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
    os.environ['TASK_PLANNER_MODE'] = 'qwen_only'
    os.environ['ENABLE_DYNAMIC_REPLAN'] = '0'
    os.environ['ENABLE_LOCAL_REPLAN'] = '0'


def _collect_pairs(task: str, limit_per_task: int, seed: int):
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
    if len(pairs) > limit_per_task:
        rng = random.Random(seed + sum(ord(c) for c in task))
        pairs = sorted(rng.sample(pairs, limit_per_task), key=lambda x: x[0].name)
    return pairs


def _init_eval(device: str):
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
            return float(psnr(t_pred, t_gt).item()), float(ssim(t_pred, t_gt).item())

    return eval_pair


def _toolbox(step_name: str):
    if step_name == 'derain':
        from utils_new.deraining import deraining_toolbox
        return deraining_toolbox
    if step_name == 'dehaze':
        from utils_new.dehazing import dehazing_toolbox
        return dehazing_toolbox
    if step_name == 'desnow':
        from utils_new.desnowing import desnowing_toolbox
        return desnowing_toolbox
    return []


def _plan_e2_without_perception(image_path: str):
    planner = TaskPlanner()
    explicit_inputs = {
        'C_I': 'Perception disabled for E2',
        'D_I': [],
        'A_I': list(STEP_ORDER),
        'I': image_path,
    }
    try:
        q = planner._plan_via_isolated_env(
            image_path=image_path,
            explicit_inputs=explicit_inputs,
            allowed_steps=list(STEP_ORDER),
        )
        plan = [x for x in q.get('plan', []) if x in STEP_ORDER]
        if plan:
            return plan
    except Exception:
        pass
    return ['derain']


def _plan_e4_with_perception(image_path: str):
    deg = predict_degradation(image_path)
    planner = TaskPlanner()
    plan = planner.plan(deg, image_path=image_path)
    return [x for x in plan if x in STEP_ORDER]


def _run_random_single_expert(plan, input_image: str, output_dir: Path, seed: int):
    evaluator = QualityEvaluator(normalize=False)
    cur = str(Path(input_image).resolve())
    output_dir.mkdir(parents=True, exist_ok=True)
    allow_input = os.getenv('ALLOW_INPUT_AS_CANDIDATE', '1').strip().lower() not in {'0', 'false', 'no'}

    for i, step in enumerate(plan):
        tools = [
            t for t in _toolbox(step)
            if t.work_dir is not None and t.work_dir.exists() and t.script_path is not None and t.script_path.exists()
        ]
        if not tools:
            raise RuntimeError(f'No available tools for {step}')

        rng = random.Random(seed + i)
        tool = rng.choice(tools)

        td = output_dir / f'temp_{step}_{tool.tool_name}'
        if td.exists():
            shutil.rmtree(td)
        inp_d = td / 'input'
        out_d = td / 'output'
        inp_d.mkdir(parents=True, exist_ok=True)
        out_d.mkdir(parents=True, exist_ok=True)
        shutil.copy(cur, inp_d / 'input.png')

        tool(input_dir=inp_d, output_dir=out_d, silent=True, run_gpu_id=None)
        cand = out_d / 'output.png'
        if not cand.exists():
            raise RuntimeError(f'Tool {tool.tool_name} produced no output')

        pool = [cur, str(cand)] if allow_input else [str(cand)]
        best, _ = evaluator.select_best(pool, task_name=step)
        stable = output_dir / f'selected_step_{i+1}_{step}.png'
        shutil.copy(best, stable)
        cur = str(stable)

    final_out = output_dir / 'final_output.png'
    shutil.copy(cur, final_out)
    return final_out


def _run_experiment(exp: dict, eval_pair):
    exp_id = exp['id']
    out_dir = OUT_ROOT / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        'experiment': exp,
        'resolved_env': {
            'TASK_PLANNER_MODE': os.environ.get('TASK_PLANNER_MODE', ''),
            'ENABLE_DYNAMIC_REPLAN': os.environ.get('ENABLE_DYNAMIC_REPLAN', ''),
            'ENABLE_LOCAL_REPLAN': os.environ.get('ENABLE_LOCAL_REPLAN', ''),
        },
    }
    (out_dir / 'run_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

    rows, failures = [], []
    t0 = time.time()

    for task in TASKS:
        pairs = _collect_pairs(task, exp['limit_per_task'], exp['sample_seed'])
        for i, (inp, gt) in enumerate(pairs, 1):
            sample_dir = out_dir / task / inp.stem
            sample_dir.mkdir(parents=True, exist_ok=True)
            final_out = sample_dir / 'final_output.png'
            try:
                if not final_out.exists():
                    if exp['kind'] == 'e2':
                        plan = _plan_e2_without_perception(str(inp))
                    else:
                        plan = _plan_e4_with_perception(str(inp))
                    final_out = _run_random_single_expert(plan, str(inp), sample_dir, exp['random_single_expert_seed'])
                psnr, ssim = eval_pair(final_out, gt)
                rows.append({
                    'experiment': exp_id, 'task': task, 'sample': inp.name,
                    'input_path': str(inp), 'gt_path': str(gt), 'output_path': str(final_out),
                    'PSNR': psnr, 'SSIM': ssim,
                })
                print(f'[{exp_id}] [{task}] {i}/{len(pairs)} done: {inp.name} PSNR={psnr:.3f} SSIM={ssim:.4f}')
            except Exception as e:
                failures.append({'experiment': exp_id, 'task': task, 'sample': inp.name, 'reason': str(e)})
                print(f'[{exp_id}] [{task}] {i}/{len(pairs)} fail: {inp.name} :: {e}')

    with (out_dir / 'per_image_metrics.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['experiment', 'task', 'sample', 'input_path', 'gt_path', 'output_path', 'PSNR', 'SSIM'])
        w.writeheader(); w.writerows(rows)

    with (out_dir / 'failed_samples.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['experiment', 'task', 'sample', 'reason'])
        w.writeheader(); w.writerows(failures)

    task_summary = {}
    for task in TASKS:
        tr = [r for r in rows if r['task'] == task]
        ps = [float(r['PSNR']) for r in tr]
        ss = [float(r['SSIM']) for r in tr]
        task_summary[task] = {
            'num_success': len(tr),
            'num_failed': len([x for x in failures if x['task'] == task]),
            'psnr_mean': float(mean(ps)) if ps else 0.0,
            'ssim_mean': float(mean(ss)) if ss else 0.0,
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
        },
        'sanity_check': {
            'dynamic_replan_enabled': os.environ.get('ENABLE_DYNAMIC_REPLAN', '1') != '0',
            'local_replan_enabled': os.environ.get('ENABLE_LOCAL_REPLAN', '1') != '0',
        },
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[{exp_id}] summary saved: {out_dir / "summary.json"}')
    return summary


def main():
    _set_env_no_replan()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_pair = _init_eval(device)

    summary_rows = []
    for exp in EXPERIMENTS:
        s = _run_experiment(exp, eval_pair)
        summary_rows.append({
            'experiment': exp['id'],
            'kind': exp['kind'],
            'num_success': s['num_success'],
            'num_failed': s['num_failed'],
            'overall_psnr_mean': s['overall']['psnr_mean'],
            'overall_ssim_mean': s['overall']['ssim_mean'],
            'elapsed_sec': s['elapsed_sec'],
        })

    with (OUT_ROOT / 'ablation_summary.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)
    print(f'Ablation summary csv saved: {OUT_ROOT / "ablation_summary.csv"}')


if __name__ == '__main__':
    main()
