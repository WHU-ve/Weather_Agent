#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path('/root/project/huangchao/zhengyanggong/weather_agent').resolve()
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_pipeline():
    from main import main as run_pipeline
    return run_pipeline

DATA_ROOT = PROJECT_ROOT / 'dataset' / 'FoundIR-Weather'
DEFAULT_OUT = PROJECT_ROOT / 'output' / 'foundir_10rain_fullchain_rand25'


def parse_args():
    p = argparse.ArgumentParser(description='Run fullchain on FoundIR-Weather 10Rain random 25 samples.')
    p.add_argument('--data_root', default=str(DATA_ROOT), help='FoundIR-Weather root with LQ/ and GT/')
    p.add_argument('--output_root', default=str(DEFAULT_OUT), help='Output directory')
    p.add_argument('--subset', default='10Rain', help='Subset name under LQ/ and GT/')
    p.add_argument('--num_samples', type=int, default=25, help='Random sample count')
    p.add_argument('--seed', type=int, default=20260514, help='Random seed')
    p.add_argument('--allow_input_as_candidate', type=int, default=0, choices=[0, 1], help='Whether input image can be a candidate')
    p.add_argument('--overwrite_existing', action='store_true', help='Force rerun samples with existing final_output.png')
    return p.parse_args()


def _is_image(p: Path) -> bool:
    return p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def _set_env(allow_input_as_candidate: int):
    os.environ['TASK_PLANNER_MODE'] = 'qwen_only'
    os.environ['ENABLE_DYNAMIC_REPLAN'] = '1'
    os.environ['ENABLE_LOCAL_REPLAN'] = '1'
    os.environ['KEEP_ALL_INTERMEDIATES'] = '1'
    os.environ['ALLOW_INPUT_AS_CANDIDATE'] = str(int(allow_input_as_candidate))

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

    os.environ.setdefault('WEATHER_AGENT_ENV', 'weather_agent')
    os.environ.setdefault('WEATHER_AGENT_RIDCP_ENV', 'weather_agent_ridcp')
    os.environ.setdefault('WEATHER_AGENT_NAFNET_ENV', 'weather_agent_nafnet')
    os.environ.setdefault('WEATHER_AGENT_MAXIM_ENV', 'weather_agent_maxim')
    os.environ.setdefault('WEATHER_AGENT_DIFFPLUGIN_ENV', 'weather_agent_diffplugin')
    os.environ.setdefault('WEATHER_AGENT_JSTASR_ENV', 'weather_agent_jstasr')
    os.environ.setdefault('WEATHER_AGENT_STARNET_ENV', 'weather_agent_starnet')
    os.environ.setdefault('WEATHER_AGENT_DDMSNET_ENV', 'weather_agent_DDMSNet')


def _collect_pairs(lq_dir: Path, gt_dir: Path):
    gt_map = {p.name: p for p in gt_dir.iterdir() if p.is_file() and _is_image(p)}
    pairs = []
    for inp in sorted(lq_dir.iterdir()):
        if not inp.is_file() or not _is_image(inp):
            continue
        gt = gt_map.get(inp.name)
        if gt is not None:
            pairs.append((inp, gt))
    return pairs


def _pick_random(pairs, k: int, seed: int):
    if len(pairs) <= k:
        return pairs
    rng = random.Random(seed)
    sampled = rng.sample(pairs, k)
    return sorted(sampled, key=lambda x: x[0].name)


def main():
    args = parse_args()
    _set_env(args.allow_input_as_candidate)

    print('[BOOT] Loading pipeline entry from main.py ...', flush=True)
    run_pipeline = _load_pipeline()
    print('[BOOT] Pipeline loaded.', flush=True)

    data_root = Path(args.data_root).resolve()
    out_root = Path(args.output_root).resolve()
    subset = args.subset

    lq_dir = data_root / 'LQ' / subset
    gt_dir = data_root / 'GT' / subset
    if not lq_dir.exists() or not gt_dir.exists():
        raise FileNotFoundError(f'Subset missing: {subset} under {data_root}/LQ and {data_root}/GT')

    all_pairs = _collect_pairs(lq_dir, gt_dir)
    picked_pairs = _pick_random(all_pairs, args.num_samples, args.seed)

    run_root = out_root / subset
    run_root.mkdir(parents=True, exist_ok=True)

    rows = []
    print(f'[INFO] subset={subset} total_candidates={len(all_pairs)} picked={len(picked_pairs)} seed={args.seed}', flush=True)
    print(f'[INFO] allow_input_as_candidate={args.allow_input_as_candidate}', flush=True)
    print(f'[INFO] output_root={run_root}', flush=True)

    for idx, (inp, gt) in enumerate(picked_pairs, 1):
        sample_out = run_root / inp.stem
        sample_out.mkdir(parents=True, exist_ok=True)
        final_out = sample_out / 'final_output.png'

        ok = 1
        err = ''
        start_t = time.time()
        try:
            if args.overwrite_existing and final_out.exists():
                final_out.unlink()

            if not final_out.exists():
                run_pipeline(str(inp), str(sample_out), planner_mode='qwen_only', run_restoration=True)

            if not final_out.exists():
                raise RuntimeError('final_output.png missing')
        except Exception as e:
            ok = 0
            err = str(e)

        elapsed = time.time() - start_t
        rows.append({
            'subset': subset,
            'sample': inp.name,
            'input_path': str(inp),
            'gt_path': str(gt),
            'output_path': str(final_out),
            'ok': ok,
            'elapsed_sec': round(elapsed, 3),
            'error': err,
        })

        print(f'[{subset}] {idx}/{len(picked_pairs)} {inp.name} ok={ok} time={elapsed:.1f}s', flush=True)
        if err:
            print(f'  error: {err}', flush=True)

    csv_path = out_root / 'per_image_status.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['subset', 'sample', 'input_path', 'gt_path', 'output_path', 'ok', 'elapsed_sec', 'error'])
        w.writeheader()
        w.writerows(rows)

    ok_rows = [r for r in rows if int(r['ok']) == 1]
    summary = {
        'dataset': 'FoundIR-Weather',
        'subset': subset,
        'num_total': len(rows),
        'num_success': len(ok_rows),
        'num_failed': len(rows) - len(ok_rows),
        'allow_input_as_candidate': int(args.allow_input_as_candidate),
        'seed': int(args.seed),
        'num_samples': int(args.num_samples),
        'output_root': str(out_root),
        'status_csv': str(csv_path),
    }

    summary_path = out_root / 'summary_status.json'
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    manifest_path = out_root / 'picked_samples.txt'
    manifest_path.write_text('\n'.join([p[0].name for p in picked_pairs]) + '\n', encoding='utf-8')

    print(f'[DONE] summary: {summary_path}', flush=True)
    print(f'[DONE] status_csv: {csv_path}', flush=True)
    print(f'[DONE] picked_samples: {manifest_path}', flush=True)


if __name__ == '__main__':
    main()
