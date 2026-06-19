#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quality_evaluator import QualityEvaluator
from utils_new.deraining import deraining_toolbox
from utils_new.dehazing import dehazing_toolbox
from utils_new.desnowing import desnowing_toolbox


TOOLBOX_BY_TASK = {
    'derain': deraining_toolbox,
    'dehaze': dehazing_toolbox,
    'desnow': desnowing_toolbox,
}

WEATHERBENCH_TASKS = {
    'rain': 'derain',
    'haze': 'dehaze',
    'snow': 'desnow',
}


def parse_gpu_ids():
    raw = os.getenv('EXPERT_PARALLEL_GPU_IDS', '').strip()
    if raw:
        out = []
        for tok in raw.split(','):
            tok = tok.strip()
            if tok:
                try:
                    out.append(int(tok))
                except ValueError:
                    pass
        return out
    if not torch.cuda.is_available():
        return []
    ids = list(range(torch.cuda.device_count()))
    if len(ids) > 1:
        ids = [i for i in ids if i != 0] or list(range(torch.cuda.device_count()))
    return ids


def psnr_ssim(pred_path: Path, gt_path: Path):
    pred_img = Image.open(pred_path).convert('RGB')
    gt_img = Image.open(gt_path).convert('RGB')
    if gt_img.size != pred_img.size:
        gt_img = gt_img.resize(pred_img.size, Image.Resampling.BICUBIC)
    pred = np.asarray(pred_img, dtype=np.float64)
    gt = np.asarray(gt_img, dtype=np.float64)
    mse = np.mean((pred - gt) ** 2)
    psnr = float('inf') if mse == 0 else float(20.0 * np.log10(255.0 / np.sqrt(mse)))
    ssim = structural_similarity(gt, pred, channel_axis=2, data_range=255)
    return psnr, float(ssim)


def run_tool(tool, input_dir: Path, output_dir: Path, run_gpu_id):
    try:
        tool(input_dir=input_dir, output_dir=output_dir, silent=True, run_gpu_id=run_gpu_id)
        out = output_dir / 'output.png'
        if out.exists():
            return str(out), None
        return None, f'{tool.tool_name} produced no output.png'
    except Exception as exc:
        return None, f'{tool.tool_name} failed: {exc}'


def run_multiexpert(task_name: str, lq_path: Path, out_dir: Path, evaluator: QualityEvaluator, gpu_ids: list[int]):
    toolbox = TOOLBOX_BY_TASK[task_name]
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for tool in toolbox:
        if tool.work_dir is None or not tool.work_dir.exists() or tool.script_path is None or not tool.script_path.exists():
            continue
        temp_dir = out_dir / f'tool_{tool.tool_name}'
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        in_dir = temp_dir / 'input'
        out_sub = temp_dir / 'output'
        in_dir.mkdir(parents=True, exist_ok=True)
        out_sub.mkdir(parents=True, exist_ok=True)
        shutil.copy(lq_path, in_dir / 'input.png')
        tasks.append((tool, temp_dir, in_dir, out_sub))

    temp_outputs = []
    errors = []
    max_workers_env = int(os.getenv('EXPERT_PARALLEL_WORKERS', str(max(1, len(gpu_ids))))) if gpu_ids else 1
    max_workers = max(1, min(max_workers_env, len(tasks))) if tasks else 1
    sched_gpu_ids = gpu_ids[:max_workers] if gpu_ids else []

    heavy = {'maxim', 'diffplugin'}
    light_tasks = []
    heavy_tasks = []
    for idx, info in enumerate(tasks):
        if info[0].tool_name.lower() in heavy:
            heavy_tasks.append((idx, info))
        else:
            light_tasks.append((idx, info))

    def run_one(idx, info):
        tool, _temp, in_dir, out_sub = info
        run_gpu_id = None
        if sched_gpu_ids and tool.tool_name.lower() not in heavy:
            run_gpu_id = sched_gpu_ids[idx % len(sched_gpu_ids)]
        return run_tool(tool, in_dir, out_sub, run_gpu_id)

    def record(result):
        out, err = result
        if err:
            errors.append(err)
        elif out:
            temp_outputs.append(out)

    if light_tasks:
        workers = min(max_workers, len(light_tasks))
        if workers <= 1:
            for idx, info in light_tasks:
                record(run_one(idx, info))
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(run_one, idx, info) for idx, info in light_tasks]
                for fut in as_completed(futs):
                    record(fut.result())

    for idx, info in heavy_tasks:
        record(run_one(idx, info))

    if not temp_outputs:
        raise RuntimeError('; '.join(errors) if errors else 'no expert outputs')

    best, score = evaluator.select_best(temp_outputs, task_name=task_name)
    selected = out_dir / 'selected.png'
    shutil.copy(best, selected)
    meta = {
        'selected': str(selected),
        'selected_source': str(best),
        'qe_score': float(score),
        'num_outputs': len(temp_outputs),
        'errors': errors,
    }
    (out_dir / 'selection.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    return selected, meta


def write_metrics_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def load_metrics_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open('r', newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in ['qe_score', 'psnr', 'ssim']:
            if key in row:
                row[key] = float(row[key])
        for key in ['num_outputs', 'num_errors']:
            if key in row:
                row[key] = int(row[key])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='dataset/multi/WeatherBench')
    ap.add_argument('--output', default='output/weatherbench_test50_single_task_multiexpert')
    ap.add_argument('--samples_per_task', type=int, default=50)
    ap.add_argument('--all', action='store_true', help='evaluate all paired test samples in each task')
    ap.add_argument('--resume', action='store_true', help='reuse existing metrics.csv rows and skip completed samples')
    ap.add_argument('--seed', type=int, default=20260510)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_root = Path(args.output).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    os.environ.setdefault('ALLOW_INPUT_AS_CANDIDATE', '0')
    os.environ.setdefault('KEEP_ALL_INTERMEDIATES', '1')
    evaluator = QualityEvaluator(normalize=False)
    gpu_ids = parse_gpu_ids()

    rows = load_metrics_csv(out_root / 'metrics.csv') if args.resume else []
    completed_keys = {(str(r.get('dataset')), str(r.get('name'))) for r in rows}
    sampled = {}
    start = time.time()
    for dataset_name, task_name in WEATHERBENCH_TASKS.items():
        input_dir = root / dataset_name / 'test' / 'input'
        target_dir = root / dataset_name / 'test' / 'target'
        pairs = sorted([
            p.name for p in input_dir.iterdir()
            if p.suffix.lower() in {'.jpg', '.jpeg', '.png'} and (target_dir / p.name).exists()
        ])
        if args.all:
            chosen = pairs
        else:
            existing_for_dataset = sorted({str(r.get('name')) for r in rows if str(r.get('dataset')) == dataset_name})
            existing_for_dataset = [name for name in existing_for_dataset if name in pairs]
            target_n = min(args.samples_per_task, len(pairs))
            if args.resume and existing_for_dataset:
                chosen = list(existing_for_dataset[:target_n])
                if len(chosen) < target_n:
                    remaining_pool = [name for name in pairs if name not in set(chosen)]
                    chosen.extend(random.sample(remaining_pool, target_n - len(chosen)))
            else:
                chosen = random.sample(pairs, target_n)
        sampled[dataset_name] = chosen
        pending = [name for name in chosen if (dataset_name, name) not in completed_keys]
        print(
            f'[{dataset_name}] total_target={len(chosen)} completed={len(chosen) - len(pending)} pending={len(pending)} task={task_name}',
            flush=True,
        )
        for i, name in enumerate(pending, 1):
            print(f'[{dataset_name}] pending {i}/{len(pending)} {name} task={task_name}', flush=True)
            sample_dir = out_root / dataset_name / Path(name).stem
            selected, meta = run_multiexpert(task_name, input_dir / name, sample_dir, evaluator, gpu_ids)
            psnr, ssim = psnr_ssim(selected, target_dir / name)
            row = {
                'dataset': dataset_name,
                'task': task_name,
                'name': name,
                'selected': str(selected),
                'selected_source': meta['selected_source'],
                'qe_score': meta['qe_score'],
                'psnr': psnr,
                'ssim': ssim,
                'num_outputs': meta['num_outputs'],
                'num_errors': len(meta['errors']),
            }
            rows.append(row)
            completed_keys.add((dataset_name, name))
            write_metrics_csv(out_root / 'metrics.csv', rows)

    summary = {}
    for dataset_name in WEATHERBENCH_TASKS:
        vals = [r for r in rows if r['dataset'] == dataset_name]
        summary[dataset_name] = {
            'num_samples': len(vals),
            'mean_psnr': float(np.mean([r['psnr'] for r in vals])),
            'mean_ssim': float(np.mean([r['ssim'] for r in vals])),
        }
    summary['all'] = {
        'num_samples': len(rows),
        'mean_psnr': float(np.mean([r['psnr'] for r in rows])),
        'mean_ssim': float(np.mean([r['ssim'] for r in rows])),
    }
    summary['sampled'] = sampled
    summary['elapsed_sec'] = time.time() - start
    (out_root / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
