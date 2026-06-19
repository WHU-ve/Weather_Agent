#!/usr/bin/env python3
"""Re-sweep alpha using current scoring formulas, reusing cached expert outputs."""
import csv, json, os, sys, time
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quality_evaluator import QualityEvaluator

CANDIDATE_DIR = ROOT / 'output' / 'alpha_sweep_weatherbench_train150'
OUT_DIR = ROOT / 'output' / 'alpha_sweep_final'

TASKS = ['derain', 'dehaze', 'desnow']
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SSIM_WEIGHT = 30.0


EXCLUDED_TOOLS = {'MWFormer', 'jstasr'}

def load_candidates(task: str) -> Dict[str, List[dict]]:
    """Read candidate_metrics.csv, group by sample, excluding removed experts."""
    path = CANDIDATE_DIR / task / 'candidate_metrics.csv'
    by_sample: Dict[str, List[dict]] = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r['tool'] in EXCLUDED_TOOLS:
                continue
            r['psnr'] = float(r['psnr'])
            r['ssim'] = float(r['ssim'])
            r['objective'] = r['psnr'] + SSIM_WEIGHT * r['ssim']
            by_sample.setdefault(r['sample'], []).append(r)
    return by_sample


def sweep(task: str, by_sample: Dict[str, List[dict]], out_dir: Path):
    qe = QualityEvaluator(normalize=False)
    valid = {k: v for k, v in by_sample.items() if v}
    oracle = {k: max(v, key=lambda x: x['objective']) for k, v in valid.items()}

    # Pre-load input features for every sample (needed for delta & relative scores).
    input_features_cache: Dict[str, dict] = {}
    for sid in valid:
        inp_png = CANDIDATE_DIR / task / 'candidates' / sid / 'input.png'
        if inp_png.exists():
            try:
                input_features_cache[sid] = qe._extract_features(str(inp_png))
            except Exception:
                input_features_cache[sid] = None
        else:
            input_features_cache[sid] = None

    curve, sels = [], []
    for a in ALPHAS:
        qe.alpha_by_task[task] = a
        chosen, hits, picks = [], 0, {}
        print(f'[{task}] alpha={a:.1f} samples={len(valid)}', flush=True)
        for sid, rs in valid.items():
            path_map = {str(Path(r['image_path']).resolve()): r for r in rs}
            _input_feat = input_features_cache.get(sid)
            bp, qs = qe.select_best(list(path_map), task_name=task,
                                     input_features=_input_feat)
            r = path_map[str(Path(bp).resolve())]
            o = oracle[sid]
            chosen.append(r)
            hits += int(r['tool'] == o['tool'])
            picks[r['tool']] = picks.get(r['tool'], 0) + 1
            sels.append({
                'task': task, 'alpha': a, 'sample': sid,
                'chosen_tool': r['tool'], 'qe_score': qs,
                'psnr': r['psnr'], 'ssim': r['ssim'],
                'objective': r['objective'],
                'oracle_tool': o['tool'],
                'oracle_objective': o['objective'],
                'oracle_hit': int(r['tool'] == o['tool']),
                'oracle_gap': o['objective'] - r['objective'],
            })
        if chosen:
            mo = float(np.mean([r['objective'] for r in chosen]))
            oo = float(np.mean([r['objective'] for r in oracle.values()]))
            row = {
                'task': task, 'alpha': a, 'num_samples': len(chosen),
                'mean_psnr': float(np.mean([r['psnr'] for r in chosen])),
                'mean_ssim': float(np.mean([r['ssim'] for r in chosen])),
                'mean_objective': mo,
                'mean_oracle_objective': oo,
                'oracle_gap': oo - mo,
                'oracle_hit_rate': hits / len(chosen),
                'pick_distribution': json.dumps(picks, sort_keys=True),
            }
            curve.append(row)
            print(f"  obj={mo:.4f} gap={row['oracle_gap']:.4f} hit={row['oracle_hit_rate']:.4f} picks={picks}", flush=True)

    curve_sorted = sorted(curve, key=lambda x: x['alpha'])
    best = max(curve, key=lambda x: (x['mean_objective'], -abs(x['alpha'] - 0.5)))

    # Write CSV
    _write_csv(out_dir / 'alpha_curve.csv', curve_sorted,
               ['task', 'alpha', 'num_samples', 'mean_psnr', 'mean_ssim',
                'mean_objective', 'mean_oracle_objective', 'oracle_gap',
                'oracle_hit_rate', 'pick_distribution'])
    _write_csv(out_dir / 'selection_by_alpha.csv', sels,
               ['task', 'alpha', 'sample', 'chosen_tool', 'qe_score', 'psnr',
                'ssim', 'objective', 'oracle_tool', 'oracle_objective',
                'oracle_hit', 'oracle_gap'])

    (out_dir / 'best_alpha.json').write_text(
        json.dumps({'task': task, 'best': best, 'alphas': curve_sorted},
                   indent=2, ensure_ascii=False), encoding='utf-8')
    return best


def _write_csv(path: Path, rows: list, fields: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def main():
    t0 = time.time()
    os.environ['QE_STRICT_OFFLINE'] = '0'
    # Load deep IQA models from cache, no network needed.
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
    os.environ.setdefault('ALLOW_INPUT_AS_CANDIDATE', '0')

    summary = {
        'description': 'Re-sweep with current quality_evaluator.py formulas',
        'alphas': ALPHAS,
        'objective': f'PSNR+{SSIM_WEIGHT}*SSIM',
        'candidate_source': str(CANDIDATE_DIR),
        'tasks': {},
    }

    for task in TASKS:
        od = OUT_DIR / task
        od.mkdir(parents=True, exist_ok=True)
        by_sample = load_candidates(task)
        print(f'\n{"="*60}\n{task}: {len(by_sample)} samples\n{"="*60}', flush=True)
        summary['tasks'][task] = sweep(task, by_sample, od)

    summary['elapsed_sec'] = time.time() - t0
    (OUT_DIR / 'best_alpha_all_tasks.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\n[DONE] Results: {OUT_DIR}', flush=True)


if __name__ == '__main__':
    main()
