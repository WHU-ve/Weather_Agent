from pathlib import Path
import csv
import json
import sys
from collections import Counter

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

ROOT = Path('/root/project/huangchao/zhengyanggong/weather_agent')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quality_evaluator import QualityEvaluator

out = ROOT / 'output/weatherbench_test50_single_task_multiexpert'
rows = list(csv.DictReader((out / 'metrics.csv').open('r', encoding='utf-8')))
snow_names = list(dict.fromkeys([r['name'] for r in rows if r['dataset'] == 'snow']))
gt_dir = ROOT / 'dataset/multi/WeatherBench/snow/test/target'


def psnr_ssim(pred_path, gt_path):
    pred_img = Image.open(pred_path).convert('RGB')
    gt_img = Image.open(gt_path).convert('RGB')
    if gt_img.size != pred_img.size:
        gt_img = gt_img.resize(pred_img.size, Image.Resampling.BICUBIC)
    pred = np.asarray(pred_img, dtype=np.float64)
    gt = np.asarray(gt_img, dtype=np.float64)
    mse = np.mean((pred - gt) ** 2)
    psnr = float('inf') if mse == 0 else float(20 * np.log10(255.0 / np.sqrt(mse)))
    ssim = float(structural_similarity(gt, pred, channel_axis=2, data_range=255))
    return psnr, ssim


evaluator = QualityEvaluator(normalize=False)
selected = []
missing = []
for name in snow_names:
    sample_dir = out / 'snow' / Path(name).stem
    candidates = []
    for tool_dir in sorted(sample_dir.glob('tool_*')):
        pred = tool_dir / 'output/output.png'
        if pred.exists():
            candidates.append(str(pred))
        else:
            missing.append({'name': name, 'expert': tool_dir.name.replace('tool_', ''), 'path': str(pred)})
    if not candidates:
        continue
    best, score = evaluator.select_best(candidates, task_name='desnow')
    psnr, ssim = psnr_ssim(Path(best), gt_dir / name)
    selected.append({
        'dataset': 'snow',
        'task': 'desnow',
        'name': name,
        'selected_source': best,
        'expert': Path(best).parts[-3].replace('tool_', ''),
        'qe_score': float(score),
        'psnr': psnr,
        'ssim': ssim,
    })

ps = [r['psnr'] for r in selected]
ss = [r['ssim'] for r in selected]
summary = {
    'config': {
        'QE_ALPHA_DESNOW': 0.10,
        'QE_STRICT_OFFLINE': '0',
        'source': 'existing expert outputs, no expert rerun',
    },
    'num_samples': len(selected),
    'mean_psnr': float(np.mean(ps)),
    'mean_ssim': float(np.mean(ss)),
    'median_psnr': float(np.median(ps)),
    'median_ssim': float(np.median(ss)),
    'min_psnr': float(np.min(ps)),
    'max_psnr': float(np.max(ps)),
    'min_ssim': float(np.min(ss)),
    'max_ssim': float(np.max(ss)),
    'selected_experts': dict(Counter(r['expert'] for r in selected)),
    'missing_outputs': missing,
}
json_path = out / 'snow_reselect_alpha_0p1_summary.json'
csv_path = out / 'snow_reselect_alpha_0p1_metrics.csv'
json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
with csv_path.open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(selected[0].keys()))
    writer.writeheader()
    writer.writerows(selected)
print(json.dumps(summary, ensure_ascii=False, indent=2))
print('json_path', json_path)
print('csv_path', csv_path)
