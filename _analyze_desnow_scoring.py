"""Find optimal desnow scoring by testing all metric combinations on training data."""
import csv, sys, json
from collections import defaultdict, Counter
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Load data
samples = defaultdict(list)
with open('output/alpha_sweep_v2_train150/desnow/candidate_metrics.csv') as f:
    for r in csv.DictReader(f):
        r['psnr']=float(r['psnr']); r['ssim']=float(r['ssim'])
        r['objective']=r['psnr']+30*r['ssim']
        samples[r['sample']].append(r)

print(f"Loaded {len(samples)} desnow samples")

# Compute features for all candidate images
from quality_evaluator import QualityEvaluator
qe = QualityEvaluator(normalize=False)
feature_cache = {}
all_paths = set()
for sid, rs in samples.items():
    for r in rs: all_paths.add(r['image_path'])
    inp = Path(f'output/alpha_sweep_v2_train150/desnow/candidates/{sid}/input.png')
    if inp.exists(): all_paths.add(str(inp))

print(f"Computing features for {len(all_paths)} images...")
for i, p in enumerate(sorted(all_paths)):
    if i % 300 == 0: print(f"  {i}/{len(all_paths)}")
    if Path(p).exists():
        try: feature_cache[p] = qe._extract_features(p)
        except: pass
print(f"  {len(feature_cache)} features computed")

# Build per-sample data with rich metrics
main_key = qe.TASK_MAIN_KEY['desnow']
ml, mu = qe.task_main_bounds['desnow'][main_key]

data = []
for sid, rs in samples.items():
    inp = str(Path(f'output/alpha_sweep_v2_train150/desnow/candidates/{sid}/input.png'))
    inp_f = feature_cache.get(inp, {})
    oracle = max(rs, key=lambda x: x['objective'])
    cands = []
    for r in rs:
        f = feature_cache.get(r['image_path'], {})
        if not f: continue
        m = {}
        # Snow metrics: both old (bright_artifact) and new (snow_artifact)
        m['snow_artifact'] = f.get('snow_artifact', 0) or 0
        # Compute bright_artifact on the fly from image
        from PIL import Image as PILImage
        arr = np.asarray(PILImage.open(r['image_path']).convert('RGB'))/255.0
        gray = arr[:,:,0]*0.299 + arr[:,:,1]*0.587 + arr[:,:,2]*0.114
        m['bright_artifact'] = float(np.mean(gray > 0.92))
        m['bright_85'] = float(np.mean(gray > 0.85))
        gy,gx = np.gradient(gray); gm = np.sqrt(gx**2+gy**2)
        m['edge_continuity'] = float(np.clip(1.0 - np.std(gm)*2.0, 0, 1.0))

        # Standard QE metrics
        mv = f.get(main_key, 0)
        m['main_good'] = 1.0 - np.clip((mv - ml)/max(mu-ml, 1e-6), 0, 1)
        for key, name in [('local_contrast','lc'),('detail_score','detail'),('texture_retention','texture')]:
            l,u = qe.shared_bounds[key]; v = f.get(key, 0)
            m[f'{name}_raw'] = v
            m[f'{name}_abs'] = np.clip((v - l)/max(u-l, 1e-6), 0, 1)
        # General IQA
        for key in ['maniqa','clipiqa','topiq_nr']:
            l,u = qe.shared_bounds[key]; v = f.get(key, 0) or 0
            m[f'{key}_abs'] = np.clip((v - l)/max(u-l, 1e-6), 0, 1)
        l,u = qe.shared_bounds['niqe']; v = f.get('niqe',0) or 0
        m['niqe_good'] = 1.0 - np.clip((v - l)/max(u-l, 1e-6), 0, 1)
        m['general'] = 0.25*(m.get('maniqa_abs',0)+m.get('clipiqa_abs',0)+m.get('topiq_nr_abs',0)+m.get('niqe_good',0))
        # Rel + delta
        if inp_f:
            for key,name in [('local_contrast','lc'),('detail_score','detail'),('texture_retention','texture')]:
                iv=inp_f.get(key,0); ov=f.get(key,0)
                m[f'{name}_rel'] = np.clip(ov/max(iv,1e-6),0,3)
            im=inp_f.get(main_key,0); om=f.get(main_key,0)
            m['delta'] = np.clip((im-om)/max(im,1e-6),0,1) if im>1e-6 else 0
            # Delta for bright_artifact too
            ib = inp_f.get('snow_artifact',0) or 0
            ob = m['snow_artifact']
            m['delta_snow'] = np.clip((ib-ob)/max(ib,1e-6),0,1) if ib>1e-6 else 0
        cands.append({'tool':r['tool'],'obj':r['objective'],'psnr':r['psnr'],'ssim':r['ssim'],'m':m})
    if cands:
        data.append({'sid':sid,'oracle':oracle['tool'],'cands':cands})

N = len(data)
print(f"\n{N} samples with features")

# Oracle distribution
od = Counter(d['oracle'] for d in data)
print(f"Oracle: {dict(od.most_common())}")
print(f"Best single expert: {max(od.values())/N:.3f}")

# Test single metrics
print(f"\n=== Single metric performance ===")
def test_metric(name, fn):
    hits, objs, worst = 0, [], 0
    for d in data:
        best = max(d['cands'], key=lambda c: fn(c['m']))
        if best['tool'] == d['oracle']: hits += 1
        objs.append(best['obj'])
        if best['obj'] == min(c['obj'] for c in d['cands']): worst += 1
    return hits/N, np.mean(objs), worst/N

best_results = []
for label, fn in [
    ('snow_artifact (lower)', lambda m: -m['snow_artifact']),
    ('bright_artifact (lower)', lambda m: -m['bright_artifact']),
    ('bright_85 (lower)', lambda m: -m['bright_85']),
    ('1 - bright_artifact', lambda m: 1.0 - m['bright_artifact']),
    ('1 - bright_85', lambda m: 1.0 - m['bright_85']),
    ('main_good (current)', lambda m: m['main_good']),
    ('delta (current)', lambda m: m.get('delta',0)),
    ('delta_snow', lambda m: m.get('delta_snow',0)),
    ('edge_continuity', lambda m: m['edge_continuity']),
    ('detail_raw', lambda m: m['detail_raw']),
    ('texture_raw', lambda m: m['texture_raw']),
    ('general', lambda m: m['general']),
    ('lc_raw', lambda m: m['lc_raw']),
    ('detail_rel', lambda m: m.get('detail_rel',1)),
    ('texture_rel', lambda m: m.get('texture_rel',1)),
]:
    hit, obj, wr = test_metric(label, fn)
    best_results.append((hit, obj, wr, label))
    print(f"  {label:30s}: hit={hit:.3f}  obj={obj:.2f}  worst={wr:.3f}")

# Best combos
print(f"\n=== Two-metric combos ===")
best_combos = []
metrics_pool = [
    ('bright_artifact_low', lambda m: 1.0 - m['bright_artifact']),
    ('bright_85_low', lambda m: 1.0 - m['bright_85']),
    ('main_good', lambda m: m['main_good']),
    ('delta', lambda m: m.get('delta',0)),
    ('edge_continuity', lambda m: m['edge_continuity']),
    ('detail_raw', lambda m: m['detail_raw']),
    ('general', lambda m: m['general']),
    ('detail_rel', lambda m: m.get('detail_rel',1)),
]

for (n1, f1) in metrics_pool:
    for (n2, f2) in metrics_pool:
        if n1 >= n2: continue
        for w1 in np.linspace(0.2, 0.8, 13):
            w2 = 1.0 - w1
            def mk_fn(f1,w1,f2,w2):
                return lambda m: w1*f1(m) + w2*f2(m)
            hit, obj, wr = test_metric(f"{w1:.1f}*{n1}+{w2:.1f}*{n2}", mk_fn(f1,w1,f2,w2))
            if hit > 0.25:  # Only keep promising ones
                best_combos.append((hit, obj, wr, f"{w1:.1f}*{n1}+{w2:.1f}*{n2}"))

best_combos.sort(key=lambda x: (-x[0], -x[1]))
print("Top 10:")
for hit, obj, wr, formula in best_combos[:10]:
    print(f"  hit={hit:.3f}  obj={obj:.2f}  worst={wr:.3f}  {formula}")

print(f"\nCurrent desnow (main+delta): hit=0.287  obj=42.59  worst=0.147")
