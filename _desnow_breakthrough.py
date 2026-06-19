"""Analyze what makes a good desnow output — feature correlation with PSNR."""
import csv, sys
from collections import defaultdict, Counter
import numpy as np
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from quality_evaluator import QualityEvaluator

qe = QualityEvaluator(normalize=False)
samples = defaultdict(list)

with open('output/alpha_sweep_v2_train150/desnow/candidate_metrics.csv') as f:
    for r in csv.DictReader(f):
        r['psnr']=float(r['psnr']); r['ssim']=float(r['ssim'])
        r['objective']=r['psnr']+30*r['ssim']
        samples[r['sample']].append(r)

# Pre-compute standard features
feature_cache = {}
all_paths = set()
for sid, rs in samples.items():
    for r in rs: all_paths.add(r['image_path'])
    inp = Path(f'output/alpha_sweep_v2_train150/desnow/candidates/{sid}/input.png')
    if inp.exists(): all_paths.add(str(inp))

for i, p in enumerate(sorted(all_paths)):
    if i%300==0: print(f'  feat {i}/{len(all_paths)}')
    if Path(p).exists():
        try: feature_cache[p] = qe._extract_features(p)
        except: pass
print(f'{len(feature_cache)} features')

# Compute EXTRA metrics for each output: snow-removal-specific
main_key = qe.TASK_MAIN_KEY['desnow']
ml, mu = qe.task_main_bounds['desnow'][main_key]

print("Computing extra snow metrics...")
all_data = []
for sid, rs in samples.items():
    inp = str(Path(f'output/alpha_sweep_v2_train150/desnow/candidates/{sid}/input.png'))
    inp_f = feature_cache.get(inp, {})
    oracle = max(rs, key=lambda x: x['objective'])

    for r in rs:
        f = feature_cache.get(r['image_path'], {})
        if not f: continue
        inp_vals = {}
        out_vals = {}

        # Open images for extra pixel-level metrics
        try:
            arr_out = np.asarray(Image.open(r['image_path']).convert('RGB'))/255.0
            g_out = arr_out[:,:,0]*0.299 + arr_out[:,:,1]*0.587 + arr_out[:,:,2]*0.114
            arr_in = np.asarray(Image.open(inp).convert('RGB'))/255.0
            g_in = arr_in[:,:,0]*0.299 + arr_in[:,:,1]*0.587 + arr_in[:,:,2]*0.114

            # 1. Bright spot reduction (snow-like: bright + isolated)
            bright_out = g_out > 0.85
            bright_in = g_in > 0.85
            inp_vals['bright_ratio'] = np.mean(bright_in)
            out_vals['bright_ratio'] = np.mean(bright_out)
            # Reduction
            inp_vals['bright_reduction'] = max(0, inp_vals['bright_ratio'] - out_vals['bright_ratio'])

            # 2. Edge preservation after snow removal
            gy_out,gx_out = np.gradient(g_out); gm_out = np.sqrt(gx_out**2+gy_out**2)
            gy_in,gx_in = np.gradient(g_in); gm_in = np.sqrt(gx_in**2+gy_in**2)
            inp_vals['edge_mean'] = np.mean(gm_in)
            out_vals['edge_mean'] = np.mean(gm_out)
            inp_vals['edge_preserve'] = np.clip(np.mean(gm_out)/max(np.mean(gm_in),1e-6), 0, 2)

            # 3. Local variance preservation (texture)
            from numpy.lib.stride_tricks import sliding_window_view
            for label, g in [('out',g_out), ('in',g_in)]:
                pad = np.pad(g, 2, mode='reflect')
                p = sliding_window_view(pad, (5,5))
                lstd = np.std(p, axis=(-2,-1))
                if label == 'out': out_vals['local_std_mean'] = np.mean(lstd)
                else: inp_vals['local_std_mean'] = np.mean(lstd)
            inp_vals['texture_preserve'] = np.clip(out_vals['local_std_mean']/max(inp_vals['local_std_mean'],1e-6), 0, 2)

            # 4. Combined quality: remove snow but keep edges+texture
            inp_vals['quality_score'] = inp_vals['bright_reduction'] * 3.0 + inp_vals['edge_preserve'] * 0.5
            out_vals['quality_score'] = inp_vals['quality_score']  # same formula

            # 5. Snow removal: did we reduce the "snow-like" regions?
            # Snow regions = bright AND moderately textured
            for label, g in [('out',g_out), ('in',g_in)]:
                pad = np.pad(g, 2, mode='reflect')
                p = sliding_window_view(pad, (5,5))
                lstd = np.std(p, axis=(-2,-1))
                snow_region = (g > 0.75) & (lstd > 0.02)
                if label == 'out': out_vals['snow_region_ratio'] = np.mean(snow_region)
                else: inp_vals['snow_region_ratio'] = np.mean(snow_region)
            inp_vals['snow_removal'] = max(0, inp_vals['snow_region_ratio'] - out_vals.get('snow_region_ratio', 0))

        except Exception as e:
            pass

        m = {}
        mv = f.get(main_key, 0)
        m['main_good'] = 1.0 - np.clip((mv - ml)/max(mu-ml, 1e-6), 0, 1)
        m['delta'] = np.clip((inp_f.get(main_key,0)-mv)/max(inp_f.get(main_key,0),1e-6),0,1) if inp_f.get(main_key,0)>1e-6 else 0
        for key, name in [('local_contrast','lc'),('detail_score','detail'),('texture_retention','texture')]:
            v = f.get(key, 0)
            l,u = qe.shared_bounds[key]
            m[f'{name}_raw'] = v
            m[f'{name}_abs'] = np.clip((v - l)/max(u-l,1e-6), 0, 1)
            if inp_f.get(key, 0) > 1e-6:
                m[f'{name}_rel'] = np.clip(v/max(inp_f.get(key,0),1e-6), 0, 3)
        # Add extra metrics
        for k, v in inp_vals.items():
            m[f'extra_{k}'] = v
        for k, v in out_vals.items():
            m[f'extra_{k}'] = v

        all_data.append({'sid': sid, 'tool': r['tool'], 'obj': r['objective'], 'is_oracle': r['tool']==oracle['tool'], 'm': m})

# Now: what features best separate oracle from non-oracle?
print(f"\n=== Feature importance for predicting oracle ===")
oracle_yes = [d for d in all_data if d['is_oracle']]
oracle_no = [d for d in all_data if not d['is_oracle']]

feature_scores = []
for k in oracle_yes[0]['m'].keys():
    y = np.mean([d['m'].get(k, 0) for d in oracle_yes])
    n = np.mean([d['m'].get(k, 0) for d in oracle_no])
    diff = y - n
    # Separation power
    y_std = np.std([d['m'].get(k, 0) for d in oracle_yes])
    n_std = np.std([d['m'].get(k, 0) for d in oracle_no])
    sep = abs(diff) / max((y_std + n_std) / 2, 1e-6)
    feature_scores.append((sep, diff, k))

feature_scores.sort(key=lambda x: -x[0])
print("Top 20 most discriminative features:")
for sep, diff, k in feature_scores[:20]:
    direction = 'oracle HIGHER' if diff > 0 else 'oracle LOWER'
    print(f"  {k:35s}: sep={sep:.3f}  diff={diff:+.4f}  ({direction})")

# Try combining top features
print(f"\n=== Trying top feature combinations ===")
def test_scoring(metric_fns, weights):
    hits, objs = 0, []
    for sid in set(d['sid'] for d in all_data):
        cands = [d for d in all_data if d['sid']==sid]
        best = max(cands, key=lambda c: sum(w*fn(c['m']) for fn,w in zip(metric_fns, weights)))
        if best['is_oracle']: hits += 1
        objs.append(best['obj'])
    return hits/len(set(d['sid'] for d in all_data)), np.mean(objs)

# Best 3 features as single metrics
for name, fn in [
    ('bright_reduction', lambda m: m.get('extra_bright_reduction',0)),
    ('edge_preserve', lambda m: m.get('extra_edge_preserve',1)),
    ('texture_preserve', lambda m: m.get('extra_texture_preserve',1)),
    ('snow_removal', lambda m: m.get('extra_snow_removal',0)),
    ('bright_ratio (lower)', lambda m: -m.get('extra_bright_ratio',0)),
]:
    hit, obj = test_scoring([fn], [1.0])
    print(f"  {name:30s}: hit={hit:.3f}  obj={obj:.2f}")

# Combine: snow_removal + edge_preserve + texture_preserve
for wr in np.linspace(0.3,0.6,7):
    for we in np.linspace(0.1,0.3,5):
        for wt in np.linspace(0.1,0.3,5):
            if abs(wr+we+wt-1.0) > 0.01: continue
            fns = [
                lambda m: m.get('extra_snow_removal',0),
                lambda m: m.get('extra_edge_preserve',1),
                lambda m: m.get('extra_texture_preserve',1),
            ]
            hit, obj = test_scoring(fns, [wr, we, wt])
            if hit > 0.29:
                print(f"  snow_removal*{wr:.2f}+edge*{we:.2f}+texture*{wt:.2f}: hit={hit:.3f}  obj={obj:.2f}")
