"""Deep analysis: per-sample, is task_score fixable or fundamentally limited?"""
import csv, sys
from collections import defaultdict
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

rows = []
with open('output/alpha_sweep_v2_train150/dehaze/candidate_metrics.csv') as f:
    for r in csv.DictReader(f):
        r['psnr'] = float(r['psnr']); r['ssim'] = float(r['ssim']); r['objective'] = float(r['objective'])
        rows.append(r)

by_sample = defaultdict(list)
for r in rows:
    by_sample[r['sample']].append(r)

# For each sample: find oracle, find RIDCP, compare
print("=== Per-sample: when is Oracle DIFFERENT from RIDCP? ===\n")
results = []
for sid, rs in sorted(by_sample.items()):
    oracle = max(rs, key=lambda x: x['objective'])
    ridcp_list = [r for r in rs if r['tool'] == 'ridcp']
    if not ridcp_list: continue
    ridcp = ridcp_list[0]

    # Only analyze samples where oracle != RIDCP
    if oracle['tool'] == 'ridcp':
        continue  # RIDCP IS the best, skip

    # Check: how much better is oracle than RIDCP?
    obj_gap = oracle['objective'] - ridcp['objective']
    psnr_gap = oracle['psnr'] - ridcp['psnr']
    results.append((sid, oracle['tool'], obj_gap, psnr_gap, ridcp['psnr'], ridcp['ssim']))

# Sort by how much worse RIDCP is
results.sort(key=lambda x: -x[2])

print(f"Samples where RIDCP is NOT the best: {len(results)}/150")
print()

# Group by gap size
small_gap = [r for r in results if r[2] < 2.0]
med_gap = [r for r in results if 2.0 <= r[2] < 5.0]
big_gap = [r for r in results if r[2] >= 5.0]
print(f"Small gap (<2 obj):  {len(small_gap)} samples — RIDCP is close to best, not a big problem")
print(f"Medium gap (2-5):    {len(med_gap)} samples — moderate room for improvement")
print(f"Large gap (>=5):     {len(big_gap)} samples — RIDCP is clearly wrong")
print()

# Show some large-gap examples
print("=== Worst RIDCP failures (largest gaps) ===")
for sid, tool, gap, pgap, rp, rs in results[:10]:
    print(f"  {sid}: oracle={tool}, RIDCP PSNR={rp:.1f}, gap={gap:.1f} obj")

# NOW: for each sample with large gap, what SHOULD the scoring have noticed?
print("\n=== For large-gap samples, compare QE metrics ===")
from quality_evaluator import QualityEvaluator
qe = QualityEvaluator(normalize=False)

for sid, oracle_tool, gap, pgap, rp, rs in results[:5]:
    rs_list = by_sample[sid]
    oracle = [r for r in rs_list if r['tool'] == oracle_tool][0]
    ridcp = [r for r in rs_list if r['tool'] == 'ridcp'][0]

    print(f"\n  Sample {sid}: oracle={oracle_tool}(PSNR={oracle['psnr']:.1f}) vs RIDCP(PSNR={ridcp['psnr']:.1f})")

    for label, r in [('Oracle', oracle), ('RIDCP', ridcp)]:
        if Path(r['image_path']).exists():
            feat = qe._extract_features(r['image_path'])
            print(f"    {label:8s}: fog={feat['fog_density_score']:.3f}  lc={feat['local_contrast']:.3f}  "
                  f"detail={feat['detail_score']:.3f}  texture={feat['texture_retention']:.3f}  "
                  f"halo_estimate={feat.get('rain_residual_score',0):.3f}")

# Key analysis: can we construct a SIMPLE binary feature that separates oracle from RIDCP?
print("\n=== Can a SIMPLE rule separate oracle from RIDCP? ===")
# Try: "lc > 0.7 AND detail > 0.35" → likely RIDCP over-processing
ridcp_flag1 = 0  # RIDCP images that match rule
oracle_flag1 = 0  # Oracle images that match rule
for sid, oracle_tool, gap, pgap, rp, rs in results:
    rs_list = by_sample[sid]
    oracle = [r for r in rs_list if r['tool'] == oracle_tool][0]
    ridcp = [r for r in rs_list if r['tool'] == 'ridcp'][0]
    if Path(ridcp['image_path']).exists() and Path(oracle['image_path']).exists():
        of = qe._extract_features(oracle['image_path'])
        rf = qe._extract_features(ridcp['image_path'])
        if rf['local_contrast'] > 0.70 and rf['detail_score'] > 0.35:
            ridcp_flag1 += 1
        if of['local_contrast'] > 0.70 and of['detail_score'] > 0.35:
            oracle_flag1 += 1

print(f"  Rule 'lc>0.7 AND detail>0.35' catches RIDCP: {ridcp_flag1}/{len(results)}")
print(f"  Same rule would ALSO catch oracle: {oracle_flag1}/{len(results)}")
print(f"  Net benefit: {ridcp_flag1 - oracle_flag1}")

# What about lc ratio to input?
print("\n=== What about lc INPUT vs OUTPUT? ===")
# For a few samples, check the input lc
input_dir = Path('output/alpha_sweep_v2_train150/dehaze/candidates')
count = 0
lc_ratios_ridcp = []
lc_ratios_oracle = []
for sid, oracle_tool, gap, pgap, rp, rs in results[:20]:
    inp = input_dir / sid / 'input.png'
    if not inp.exists(): continue
    inp_feat = qe._extract_features(str(inp))
    inp_lc = inp_feat['local_contrast']

    rs_list = by_sample[sid]
    oracle = [r for r in rs_list if r['tool'] == oracle_tool][0]
    ridcp = [r for r in rs_list if r['tool'] == 'ridcp'][0]

    if Path(oracle['image_path']).exists() and Path(ridcp['image_path']).exists():
        of = qe._extract_features(oracle['image_path'])
        rf = qe._extract_features(ridcp['image_path'])
        lc_ratios_oracle.append(of['local_contrast'] / max(inp_lc, 0.01))
        lc_ratios_ridcp.append(rf['local_contrast'] / max(inp_lc, 0.01))

if lc_ratios_ridcp:
    print(f"  Avg lc ratio (output/input): Oracle={np.mean(lc_ratios_oracle):.2f}  RIDCP={np.mean(lc_ratios_ridcp):.2f}")
    # What threshold separates them?
    for thresh in [1.5, 2.0, 2.5, 3.0, 4.0]:
        r_above = sum(1 for x in lc_ratios_ridcp if x > thresh)
        o_above = sum(1 for x in lc_ratios_oracle if x > thresh)
        net = r_above - o_above
        print(f"  lc_ratio > {thresh:.1f}: catches RIDCP={r_above}, oracle={o_above}, net={net}")
