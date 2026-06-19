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
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import restoration_agent as ra

DATA_ROOT = PROJECT_ROOT / 'dataset' / 'FoundIR-Weather'
OUT_ROOT = PROJECT_ROOT / 'output' / 'FoundIR-Weather'
SUBSET_TO_STEP = {
    '08Haze': 'dehaze',
    '10Rain': 'derain',
    '11Raindrop': 'derain',
    '12NightRain': 'derain',
}


def parse_args():
    p = argparse.ArgumentParser(description='Run FoundIR-Weather first-N benchmark with fixed expert steps.')
    p.add_argument('--data_root', default=str(DATA_ROOT), help='FoundIR-Weather root with LQ/ and GT/')
    p.add_argument('--output_root', default=str(OUT_ROOT), help='Output directory')
    p.add_argument('--subsets', default='08Haze,10Rain,11Raindrop,12NightRain', help='Comma-separated subsets')
    p.add_argument('--limit_per_subset', type=int, default=100, help='Number of images per subset')
    p.add_argument('--overwrite_existing', action='store_true', help='Force rerun samples with existing final_output')
    return p.parse_args()


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


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


def _set_env():
    os.environ.setdefault('QE_STRICT_OFFLINE', '1')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
    os.environ.setdefault('WEATHER_UTILS_DIR', 'utils_new')
    os.environ.setdefault('WEATHER_CKPT_DIR', 'pretrained_ckpts_new')
    os.environ.setdefault('USE_FLAX', '0')
    os.environ.setdefault('TRANSFORMERS_NO_FLAX', '1')
    os.environ['ENABLE_DYNAMIC_REPLAN'] = '0'
    os.environ['TASK_PLANNER_LOCK_PLAN'] = '1'

    os.environ.setdefault('WEATHER_AGENT_ENV', 'weather_agent')
    os.environ.setdefault('WEATHER_AGENT_RIDCP_ENV', 'weather_agent_ridcp')
    os.environ.setdefault('WEATHER_AGENT_NAFNET_ENV', 'weather_agent_nafnet')
    os.environ.setdefault('WEATHER_AGENT_MAXIM_ENV', 'weather_agent_maxim')
    os.environ.setdefault('WEATHER_AGENT_DIFFPLUGIN_ENV', 'weather_agent_diffplugin')
    os.environ.setdefault('WEATHER_AGENT_JSTASR_ENV', 'weather_agent_jstasr')
    os.environ.setdefault('WEATHER_AGENT_STARNET_ENV', 'weather_agent_starnet')
    os.environ.setdefault('WEATHER_AGENT_DDMSNET_ENV', 'weather_agent_DDMSNet')


def _collect_pairs(lq_dir: Path, gt_dir: Path, limit: int):
    gt_map = {p.name: p for p in gt_dir.iterdir() if p.is_file() and _is_image(p)}
    pairs = []
    for inp in sorted(lq_dir.iterdir()):
        if not inp.is_file() or not _is_image(inp):
            continue
        gt = gt_map.get(inp.name)
        if gt is not None:
            pairs.append((inp, gt))
    if limit > 0:
        pairs = pairs[:limit]
    return pairs


def main():
    args = parse_args()
    _set_env()

    data_root = Path(args.data_root).resolve()
    out_root = Path(args.output_root).resolve()
    lq_root = data_root / 'LQ'
    gt_root = data_root / 'GT'

    if not lq_root.exists() or not gt_root.exists():
        raise FileNotFoundError(f'Missing LQ/GT under {data_root}')

    out_root.mkdir(parents=True, exist_ok=True)

    subsets = [s.strip() for s in args.subsets.split(',') if s.strip()]
    subsets = [s for s in subsets if s in SUBSET_TO_STEP]
    if not subsets:
        raise ValueError('No valid subsets selected.')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_pair = _init_metrics(device)
    class FixedRestorationAgent(ra.RestorationAgent):
        def execute_plan(self, plan, input_image_path, output_dir):
            old_predict = ra.predict_degradation
            try:
                ra.predict_degradation = lambda *_args, **_kwargs: {'degradations': [], 'image_description': ''}
                return super().execute_plan(plan, input_image_path, output_dir)
            finally:
                ra.predict_degradation = old_predict

    agent = FixedRestorationAgent()

    all_rows = []
    subset_summary_rows = []

    for subset in subsets:
        step = SUBSET_TO_STEP[subset]
        lq_dir = lq_root / subset
        gt_dir = gt_root / subset
        if not lq_dir.exists() or not gt_dir.exists():
            raise FileNotFoundError(f'Subset missing: {subset}')

        pairs = _collect_pairs(lq_dir, gt_dir, args.limit_per_subset)
        subset_out = out_root / subset
        subset_out.mkdir(parents=True, exist_ok=True)

        rows = []
        failures = []

        for idx, (inp, gt) in enumerate(pairs, 1):
            sample_out = subset_out / inp.stem
            sample_out.mkdir(parents=True, exist_ok=True)
            final_out = sample_out / 'final_output.png'

            try:
                if args.overwrite_existing and final_out.exists():
                    final_out.unlink()

                if not final_out.exists():
                    agent.execute_plan([step], str(inp), str(sample_out))

                if not final_out.exists():
                    raise RuntimeError('final_output.png missing')

                metrics = eval_pair(final_out, gt)
                row = {
                    'subset': subset,
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
                print(f"[{subset}] {idx}/{len(pairs)} done: {inp.name} PSNR={metrics['PSNR']:.3f} SSIM={metrics['SSIM']:.4f}")
            except Exception as e:
                failures.append({'subset': subset, 'sample': inp.name, 'reason': str(e)})
                print(f"[{subset}] {idx}/{len(pairs)} fail: {inp.name} :: {e}")

        per_subset_csv = out_root / f'{subset}_per_image_metrics.csv'
        with per_subset_csv.open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=['subset', 'sample', 'fixed_step', 'input_path', 'gt_path', 'output_path', 'PSNR', 'SSIM'])
            w.writeheader()
            w.writerows(rows)

        fail_csv = out_root / f'{subset}_failed.csv'
        with fail_csv.open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=['subset', 'sample', 'reason'])
            w.writeheader()
            w.writerows(failures)

        subset_summary_rows.append({
            'subset': subset,
            'fixed_step': step,
            'num_total': len(pairs),
            'num_success': len(rows),
            'num_failed': len(failures),
            'psnr_mean': float(mean([float(r['PSNR']) for r in rows])) if rows else 0.0,
            'ssim_mean': float(mean([float(r['SSIM']) for r in rows])) if rows else 0.0,
        })

    merged_csv = out_root / 'per_image_metrics_all.csv'
    with merged_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['subset', 'sample', 'fixed_step', 'input_path', 'gt_path', 'output_path', 'PSNR', 'SSIM'])
        w.writeheader()
        w.writerows(all_rows)

    subset_summary_csv = out_root / 'subset_summary.csv'
    with subset_summary_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['subset', 'fixed_step', 'num_total', 'num_success', 'num_failed', 'psnr_mean', 'ssim_mean'])
        w.writeheader()
        w.writerows(subset_summary_rows)

    psnr_micro = float(mean([float(r['PSNR']) for r in all_rows])) if all_rows else 0.0
    ssim_micro = float(mean([float(r['SSIM']) for r in all_rows])) if all_rows else 0.0
    psnr_macro = float(mean([float(r['psnr_mean']) for r in subset_summary_rows])) if subset_summary_rows else 0.0
    ssim_macro = float(mean([float(r['ssim_mean']) for r in subset_summary_rows])) if subset_summary_rows else 0.0

    summary = {
        'dataset': 'FoundIR-Weather',
        'data_root': str(data_root),
        'limit_per_subset': int(args.limit_per_subset),
        'subsets': {r['subset']: {
            'fixed_step': r['fixed_step'],
            'num_total': r['num_total'],
            'num_success': r['num_success'],
            'num_failed': r['num_failed'],
            'psnr_mean': r['psnr_mean'],
            'ssim_mean': r['ssim_mean'],
        } for r in subset_summary_rows},
        'overall': {
            'num_all_success': len(all_rows),
            'psnr_mean_micro': psnr_micro,
            'ssim_mean_micro': ssim_micro,
            'psnr_mean_macro': psnr_macro,
            'ssim_mean_macro': ssim_macro,
        },
    }

    summary_json = out_root / 'summary_foundir_weather.json'
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'Saved: {summary_json}')
    print(f'Saved: {subset_summary_csv}')
    print(f'Saved: {merged_csv}')


if __name__ == '__main__':
    main()
