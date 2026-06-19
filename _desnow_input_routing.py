"""Route desnow to best expert based on INPUT image features."""
import csv, sys
from collections import defaultdict, Counter
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from quality_evaluator import QualityEvaluator

qe = QualityEvaluator(normalize=False)

# Load data
samples = defaultdict(list)
with open('output/alpha_sweep_v2_train150/desnow/candidate_metrics.csv') as f:
    for r in csv.DictReader(f):
        r['psnr']=float(r['psnr']); r['ssim']=float(r['ssim'])
        r['objective']=r['psnr']+30*r['ssim']
        samples[r['sample']].append(r)

# Compute INPUT features and oracle per sample
inputs = {}
for sid, rs in samples.items():
    inp = str(Path(f'output/alpha_sweep_v2_train150/desnow/candidates/{sid}/input.png'))
    oracle = max(rs, key=lambda x: x['objective'])
    if Path(inp).exists():
        f = qe._extract_features(inp)
        inputs[sid] = {
            'oracle': oracle['tool'],
            'rr': f['rain_residual_score'], 'fog': f['fog_density_score'],
            'lc': f['local_contrast'], 'detail': f['detail_score'],
            'texture': f['texture_retention'], 'snow': f['snow_artifact'],
        }

print(f"{len(inputs)} samples")

# What input features distinguish which expert is best?
for tool in ['diffplugin', 'starnet', 'ddmsnet', 'jstasr', 'desnownet']:
    subset = {sid: d for sid, d in inputs.items() if d['oracle'] == tool}
    if not subset: continue
    print(f"\n=== {tool} ({len(subset)} samples) ===")
    for k in ['rr','fog','lc','detail','texture','snow']:
        vals = [d[k] for d in subset.values()]
        print(f"  {k}: mean={np.mean(vals):.3f}  std={np.std(vals):.3f}")

# Try routing rules
print(f"\n=== Simple routing rules ===")
def test_route(rule_fn):
    hits, objs = 0, []
    for sid, rs in samples.items():
        if sid not in inputs: continue
        predicted = rule_fn(inputs[sid])
        # Pick the predicted expert's output
        candidates = [r for r in rs if r['tool'] == predicted]
        if not candidates: candidates = rs
        best = max(candidates, key=lambda x: x['objective'])
        oracle = max(rs, key=lambda x: x['objective'])
        if best['tool'] == oracle['tool']: hits += 1
        objs.append(best['objective'])
    return hits/len(inputs), np.mean(objs)

# Try manual rules
for rule_name, rule in [
    ('always diffplugin', lambda d: 'diffplugin'),
    ('always starnet', lambda d: 'starnet'),
    ('detail > 0.5 → diffplugin, else starnet', lambda d: 'diffplugin' if d['detail'] > 0.5 else 'starnet'),
    ('detail > 0.4 → diffplugin, else starnet', lambda d: 'diffplugin' if d['detail'] > 0.4 else 'starnet'),
    ('lc > 0.7 → diffplugin, else starnet', lambda d: 'diffplugin' if d['lc'] > 0.7 else 'starnet'),
    ('snow > 0.01 → diffplugin, else starnet', lambda d: 'diffplugin' if d['snow'] > 0.01 else 'starnet'),
    ('rr > 0.5 → diffplugin, else starnet', lambda d: 'diffplugin' if d['rr'] > 0.5 else 'starnet'),
    ('fog > 0.3 → diffplugin, else starnet', lambda d: 'diffplugin' if d['fog'] > 0.3 else 'starnet'),
]:
    hit, obj = test_route(rule)
    print(f"  {rule_name:50s}: hit={hit:.3f}  obj={obj:.2f}")

# Best 2-expert rule
for t1, t2 in [('diffplugin','starnet'), ('diffplugin','ddmsnet'), ('starnet','ddmsnet')]:
    for thresh in np.linspace(0.35, 0.65, 7):
        rule = lambda d, t1=t1, t2=t2, t=thresh: t1 if d['detail'] > t else t2
        hit, obj = test_route(rule)
        if hit > 0.35:
            print(f"  detail>{thresh:.2f}→{t1} else {t2}: hit={hit:.3f}  obj={obj:.2f}")

# Best overall
print(f"\nBest single expert: diffplugin at {inputs_dp/len(inputs):.3f}" if (inputs_dp := len([d for d in inputs.values() if d['oracle']=='diffplugin'])) else "")
