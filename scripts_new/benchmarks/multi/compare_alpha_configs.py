#!/usr/bin/env python3
"""Compare alpha configs on existing expert outputs — with input_features."""
import json, os, sys
from pathlib import Path
from statistics import mean
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quality_evaluator import QualityEvaluator

TEST_DIR = ROOT / 'output' / 'weatherbench_test50_single_task_multiexpert'
DATA_ROOT = ROOT / 'dataset' / 'multi' / 'WeatherBench'

CONFIGS = {
    'current (0.8/0.8/0.6)': {'derain': 0.8, 'dehaze': 0.8, 'desnow': 0.6},
    'sweep  (0.9/0.8/0.8)': {'derain': 0.9, 'dehaze': 0.8, 'desnow': 0.8},
    'alt    (0.9/0.8/0.5)': {'derain': 0.9, 'dehaze': 0.8, 'desnow': 0.5},
}

TASK_QE_MAP = {'rain': 'derain', 'haze': 'dehaze', 'snow': 'desnow'}


def find_samples(task: str):
    d = TEST_DIR / task
    return sorted([p for p in d.iterdir() if p.is_dir() and (p / 'selection.json').exists()])


def get_expert_outputs(sample_dir: Path):
    outputs = {}
    for tool_dir in sorted(sample_dir.glob('tool_*')):
        out_png = tool_dir / 'output' / 'output.png'
        if out_png.exists():
            outputs[tool_dir.name] = out_png
    return outputs


def metric(pred: Path, gt: Path):
    gi = Image.open(gt).convert('RGB')
    pi = Image.open(pred).convert('RGB')
    if pi.size != gi.size:
        pi = pi.resize(gi.size, Image.BICUBIC)
    ga, pa = np.asarray(gi), np.asarray(pi)
    return (float(peak_signal_noise_ratio(ga, pa, data_range=255)),
            float(structural_similarity(ga, pa, channel_axis=2, data_range=255)))


def main():
    os.environ['QE_STRICT_OFFLINE'] = '0'
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
    os.environ.setdefault('ALLOW_INPUT_AS_CANDIDATE', '0')

    qe = QualityEvaluator(normalize=False)
    all_results = {}  # (task, cfg) -> {psnr, ssim, objective, n, alpha}

    for task in ['rain', 'haze', 'snow']:
        samples = find_samples(task)
        if not samples:
            print(f'{task}: no samples')
            continue
        qe_task = TASK_QE_MAP[task]

        # Pre-load input features for every sample.
        input_feats = {}
        for sd in samples:
            inp_png = sd / 'input.png'
            if inp_png.exists():
                try:
                    input_feats[sd.name] = qe._extract_features(str(inp_png))
                except Exception:
                    input_feats[sd.name] = None
            else:
                input_feats[sd.name] = None

        print(f'\n{"="*60}')
        print(f'{task} ({qe_task}): {len(samples)} samples')
        print(f'{"="*60}')

        for cfg_name, alphas in CONFIGS.items():
            alpha = alphas[qe_task]
            qe.alpha_by_task[qe_task] = alpha

            results_p, results_s, results_obj = [], [], []
            for sd in samples:
                sid = sd.name
                experts = get_expert_outputs(sd)
                if not experts:
                    continue
                path_map = {str(v.resolve()): k for k, v in experts.items()}
                bp, qs = qe.select_best(list(path_map), task_name=qe_task,
                                         input_features=input_feats.get(sid))

                gt = DATA_ROOT / task / 'test' / 'target' / f'{sid}.jpg'
                if not gt.exists():
                    gt = DATA_ROOT / task / 'test' / 'target' / f'{sid}.png'
                p, s = metric(Path(bp), gt)
                results_p.append(p); results_s.append(s)
                results_obj.append(p + 30.0 * s)

            if results_p:
                all_results[(task, cfg_name)] = {
                    'psnr': mean(results_p), 'ssim': mean(results_s),
                    'objective': mean(results_obj), 'n': len(results_p), 'alpha': alpha,
                }
                print(f"  {cfg_name} (α={alpha:.1f}): PSNR={mean(results_p):.2f}  "
                      f"SSIM={mean(results_s):.4f}  obj={mean(results_obj):.2f}")

    # Overall summary
    print(f'\n{"="*60}')
    print('Overall comparison')
    print(f'{"="*60}')
    for task in ['rain', 'haze', 'snow']:
        qe_task = TASK_QE_MAP[task]
        print(f'\n{task}:')
        best_cfg, best_obj = None, -1
        for cfg_name in CONFIGS:
            r = all_results.get((task, cfg_name))
            if r:
                marker = ' ← BEST' if r['objective'] > best_obj else ''
                if r['objective'] > best_obj:
                    best_obj = r['objective']; best_cfg = cfg_name
                print(f"  {cfg_name}: obj={r['objective']:.2f}  PSNR={r['psnr']:.2f}  SSIM={r['ssim']:.4f}{marker}")


if __name__ == '__main__':
    main()
