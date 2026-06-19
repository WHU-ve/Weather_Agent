#!/usr/bin/env python3
import csv
import json
import os
import random
import sys
from pathlib import Path
from statistics import mean

import torch
from PIL import Image
import torchvision.transforms as transforms
import pyiqa

PROJECT_ROOT = Path('/root/project/huangchao/zhengyanggong/weather_agent').resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import main as run_pipeline

WB_ROOT = PROJECT_ROOT / 'dataset' / 'multi' / 'WeatherBench'
FI_ROOT = PROJECT_ROOT / 'dataset' / 'FoundIR-Weather'
OUT_ROOT = PROJECT_ROOT / 'output' / 'fullchain_allow_input0_eval'


def _is_image(p: Path) -> bool:
    return p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def _set_env():
    os.environ['TASK_PLANNER_MODE'] = 'qwen_only'
    os.environ['ENABLE_DYNAMIC_REPLAN'] = '1'
    os.environ['ENABLE_LOCAL_REPLAN'] = '1'
    os.environ['KEEP_ALL_INTERMEDIATES'] = '1'
    os.environ['ALLOW_INPUT_AS_CANDIDATE'] = '0'

    os.environ.setdefault('QE_STRICT_OFFLINE', '1')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
    os.environ.setdefault('WEATHER_UTILS_DIR', 'utils_new')
    os.environ.setdefault('WEATHER_CKPT_DIR', 'pretrained_ckpts_new')
    os.environ.setdefault('WEATHER_PERCEPTION_SUBPROCESS', '1')
    os.environ.setdefault('WEATHER_PERCEPTION_TOPK_GPUS', '2')
    os.environ.setdefault('WEATHER_PERCEPTION_DEVICE_MAP', 'auto')
    os.environ.setdefault('WEATHER_PERCEPTION_RELEASE_AFTER_INFER', '1')


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
            return float(psnr(t_pred, t_gt).item()), float(ssim(t_pred, t_gt).item())

    return eval_pair


def _sample_pairs(inp_dir: Path, gt_dir: Path, k: int, seed: int):
    gt_map = {p.name: p for p in gt_dir.iterdir() if p.is_file() and _is_image(p)}
    pairs = []
    for inp in sorted(inp_dir.iterdir()):
        if not inp.is_file() or not _is_image(inp):
            continue
        gt = gt_map.get(inp.name)
        if gt is not None:
            pairs.append((inp, gt))
    if len(pairs) <= k:
        return pairs
    rng = random.Random(seed)
    sampled = rng.sample(pairs, k)
    return sorted(sampled, key=lambda x: x[0].name)


def _run_dataset(dataset_name: str, task_pairs: dict, out_root: Path, eval_pair):
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for task, pairs in task_pairs.items():
        for idx, (inp, gt) in enumerate(pairs, 1):
            sample_out = out_root / task / inp.stem
            sample_out.mkdir(parents=True, exist_ok=True)
            final_out = sample_out / 'final_output.png'
            err = ''
            ok = 1
            try:
                if not final_out.exists():
                    run_pipeline(str(inp), str(sample_out), planner_mode='qwen_only', run_restoration=True)
                if not final_out.exists():
                    raise RuntimeError('final_output.png missing')
                psnr, ssim = eval_pair(final_out, gt)
            except Exception as e:
                ok = 0
                err = str(e)
                psnr, ssim = 0.0, 0.0

            rows.append({
                'dataset': dataset_name,
                'task': task,
                'sample': inp.name,
                'input_path': str(inp),
                'gt_path': str(gt),
                'output_path': str(final_out),
                'ok': ok,
                'error': err,
                'PSNR': psnr,
                'SSIM': ssim,
            })
            print(f'[{dataset_name}] [{task}] {idx}/{len(pairs)} {inp.name} ok={ok} PSNR={psnr:.3f} SSIM={ssim:.4f}')

    csv_path = out_root / 'per_image_metrics.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['dataset', 'task', 'sample', 'input_path', 'gt_path', 'output_path', 'ok', 'error', 'PSNR', 'SSIM'])
        w.writeheader()
        w.writerows(rows)

    task_summary = {}
    for task in sorted(task_pairs.keys()):
        tr = [r for r in rows if r['task'] == task and r['ok'] == 1]
        ps = [float(r['PSNR']) for r in tr]
        ss = [float(r['SSIM']) for r in tr]
        task_summary[task] = {
            'num_total': len([r for r in rows if r['task'] == task]),
            'num_success': len(tr),
            'psnr_mean': float(mean(ps)) if ps else 0.0,
            'ssim_mean': float(mean(ss)) if ss else 0.0,
        }

    succ = [r for r in rows if r['ok'] == 1]
    ps_all = [float(r['PSNR']) for r in succ]
    ss_all = [float(r['SSIM']) for r in succ]
    summary = {
        'dataset': dataset_name,
        'config': {
            'mode': 'fullchain',
            'allow_input_as_candidate': 0,
            'planner_mode': 'qwen_only',
            'dynamic_replan': 1,
            'local_replan': 1,
        },
        'tasks': task_summary,
        'overall': {
            'num_success': len(succ),
            'psnr_mean_micro': float(mean(ps_all)) if ps_all else 0.0,
            'ssim_mean_micro': float(mean(ss_all)) if ss_all else 0.0,
            'psnr_mean_macro': float(mean([v['psnr_mean'] for v in task_summary.values()])),
            'ssim_mean_macro': float(mean([v['ssim_mean'] for v in task_summary.values()])),
        },
    }

    summary_path = out_root / 'summary.json'
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


def main():
    _set_env()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_pair = _init_metrics(device)

    wb_tasks = {
        'rain': _sample_pairs(WB_ROOT / 'rain' / 'test' / 'input', WB_ROOT / 'rain' / 'test' / 'target', k=50, seed=20260503),
        'haze': _sample_pairs(WB_ROOT / 'haze' / 'test' / 'input', WB_ROOT / 'haze' / 'test' / 'target', k=50, seed=20260504),
        'snow': _sample_pairs(WB_ROOT / 'snow' / 'test' / 'input', WB_ROOT / 'snow' / 'test' / 'target', k=50, seed=20260505),
    }
    fi_tasks = {
        '08Haze': _sample_pairs(FI_ROOT / 'LQ' / '08Haze', FI_ROOT / 'GT' / '08Haze', k=25, seed=20260513),
        '10Rain': _sample_pairs(FI_ROOT / 'LQ' / '10Rain', FI_ROOT / 'GT' / '10Rain', k=25, seed=20260514),
        '11Raindrop': _sample_pairs(FI_ROOT / 'LQ' / '11Raindrop', FI_ROOT / 'GT' / '11Raindrop', k=25, seed=20260515),
        '12NightRain': _sample_pairs(FI_ROOT / 'LQ' / '12NightRain', FI_ROOT / 'GT' / '12NightRain', k=25, seed=20260516),
    }

    wb_summary = _run_dataset('WeatherBench', wb_tasks, OUT_ROOT / 'weatherbench_top50_each', eval_pair)
    fi_summary = _run_dataset('FoundIR-Weather', fi_tasks, OUT_ROOT / 'foundir_top25_each', eval_pair)

    merged = {'weatherbench': wb_summary, 'foundir_weather': fi_summary}
    (OUT_ROOT / 'summary_merged.json').write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Done. merged summary:', OUT_ROOT / 'summary_merged.json')


if __name__ == '__main__':
    main()
