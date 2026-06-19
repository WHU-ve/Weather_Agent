#!/usr/bin/env python3
import argparse, csv, json, os, random, shutil, sys, time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quality_evaluator import QualityEvaluator
from utils_new.deraining import deraining_toolbox
from utils_new.dehazing import dehazing_toolbox
from utils_new.desnowing import desnowing_toolbox

TASKS = {'derain': ('rain', deraining_toolbox), 'dehaze': ('haze', dehazing_toolbox), 'desnow': ('snow', desnowing_toolbox)}


def args():
    p = argparse.ArgumentParser(description='Online-QE clean alpha sweep on WeatherBench train split.')
    p.add_argument('--split', default='train', choices=['train', 'test'])
    p.add_argument('--limit', type=int, default=100)
    p.add_argument('--seed', type=int, default=20260509)
    p.add_argument('--tasks', default='derain,dehaze,desnow')
    p.add_argument('--alphas', default='0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1')
    p.add_argument('--output_root', default='output/alpha_sweep_weatherbench_train150')
    p.add_argument('--ssim_weight', type=float, default=30.0)
    p.add_argument('--strict_offline', action='store_true')
    p.add_argument('--overwrite', action='store_true')
    return p.parse_args()


def is_img(p: Path):
    return p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def pairs(task: str, split: str, limit: int, seed: int) -> List[Tuple[Path, Path]]:
    d = TASKS[task][0]
    base = ROOT / 'dataset/multi/WeatherBench' / d / split
    ins, gts = base / 'input', base / 'target'
    ps = [(p, gts / p.name) for p in sorted(ins.iterdir()) if p.is_file() and is_img(p) and (gts / p.name).exists()]
    r = random.Random(seed + sum(map(ord, task)))
    if limit > 0 and len(ps) > limit:
        ps = sorted(r.sample(ps, limit), key=lambda x: x[0].name)
    return ps


def metric(pred: Path, gt: Path):
    gi = Image.open(gt).convert('RGB')
    pi = Image.open(pred).convert('RGB')
    if pi.size != gi.size:
        pi = pi.resize(gi.size, Image.BICUBIC)
    ga, pa = np.asarray(gi), np.asarray(pi)
    return float(peak_signal_noise_ratio(ga, pa, data_range=255)), float(structural_similarity(ga, pa, channel_axis=2, data_range=255))


def write_csv(path: Path, rows: List[dict], fields: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def run_tool(tool, inp: Path, sample_dir: Path):
    td, od = sample_dir / f'tool_{tool.tool_name}', sample_dir / f'tool_{tool.tool_name}' / 'output'
    out = od / 'output.png'
    if out.exists():
        return out, ''
    if td.exists():
        shutil.rmtree(td)
    idir = td / 'input'; idir.mkdir(parents=True); od.mkdir(parents=True)
    shutil.copy(inp, idir / 'input.png')
    if tool.work_dir is None or not tool.work_dir.exists() or tool.script_path is None or not tool.script_path.exists():
        return None, 'missing tool script'
    try:
        tool(input_dir=idir, output_dir=od, silent=True)
        return (out, '') if out.exists() else (None, 'no output.png')
    except Exception as e:
        return None, str(e)


def build(task: str, ps: List[Tuple[Path, Path]], out_dir: Path, sw: float) -> Dict[str, List[dict]]:
    all_rows, fails, by_sample = [], [], {}
    tools = TASKS[task][1]
    for i, (inp, gt) in enumerate(ps, 1):
        sid = inp.stem; sd = out_dir / 'candidates' / sid; sd.mkdir(parents=True, exist_ok=True)
        sin, sgt = sd / 'input.png', sd / 'gt.png'
        if not sin.exists(): shutil.copy(inp, sin)
        if not sgt.exists(): shutil.copy(gt, sgt)
        rows = []
        print(f'[{task}] {i}/{len(ps)} {inp.name} experts={len(tools)}', flush=True)
        for tool in tools:
            op, err = run_tool(tool, sin, sd)
            if err or op is None:
                fails.append({'task': task, 'sample': sid, 'tool': tool.tool_name, 'error': err}); print(f'  fail {tool.tool_name}: {err}', flush=True); continue
            try:
                p, s = metric(op, sgt)
                row = {'task': task, 'sample': sid, 'tool': tool.tool_name, 'image_path': str(op.resolve()), 'gt_path': str(sgt.resolve()), 'psnr': p, 'ssim': s, 'objective': p + sw * s}
                rows.append(row); all_rows.append(row)
            except Exception as e:
                fails.append({'task': task, 'sample': sid, 'tool': tool.tool_name, 'error': f'metric: {e}'})
        by_sample[sid] = rows
        if rows:
            b = max(rows, key=lambda x: x['objective']); print(f"  oracle {b['tool']} obj={b['objective']:.6f}", flush=True)
    write_csv(out_dir / 'candidate_metrics.csv', all_rows, ['task','sample','tool','image_path','gt_path','psnr','ssim','objective'])
    write_csv(out_dir / 'failures.csv', fails, ['task','sample','tool','error'])
    return by_sample


def sweep(task: str, by_sample: Dict[str, List[dict]], alphas: List[float], out_dir: Path):
    qe, curve, sels = QualityEvaluator(normalize=False), [], []
    valid = {k: v for k, v in by_sample.items() if v}
    oracle = {k: max(v, key=lambda x: x['objective']) for k, v in valid.items()}
    for a in alphas:
        qe.alpha_by_task[task] = a
        chosen, hits, picks = [], 0, {}
        print(f'[{task}] alpha={a:.1f} samples={len(valid)}', flush=True)
        for sid, rs in valid.items():
            path_map = {str(Path(r['image_path']).resolve()): r for r in rs}
            bp, qs = qe.select_best(list(path_map), task_name=task)
            r, o = path_map[str(Path(bp).resolve())], oracle[sid]
            chosen.append(r); hits += int(r['tool'] == o['tool']); picks[r['tool']] = picks.get(r['tool'], 0) + 1
            sels.append({'task': task, 'alpha': a, 'sample': sid, 'chosen_tool': r['tool'], 'qe_score': qs, 'psnr': r['psnr'], 'ssim': r['ssim'], 'objective': r['objective'], 'oracle_tool': o['tool'], 'oracle_objective': o['objective'], 'oracle_hit': int(r['tool'] == o['tool']), 'oracle_gap': o['objective'] - r['objective']})
        if chosen:
            mo = float(np.mean([r['objective'] for r in chosen])); oo = float(np.mean([r['objective'] for r in oracle.values()]))
            row = {'task': task, 'alpha': a, 'num_samples': len(chosen), 'mean_psnr': float(np.mean([r['psnr'] for r in chosen])), 'mean_ssim': float(np.mean([r['ssim'] for r in chosen])), 'mean_objective': mo, 'mean_oracle_objective': oo, 'oracle_gap': oo - mo, 'oracle_hit_rate': hits / len(chosen), 'pick_distribution': json.dumps(picks, sort_keys=True)}
            curve.append(row); print(f"  obj={mo:.6f} gap={row['oracle_gap']:.6f} hit={row['oracle_hit_rate']:.4f}", flush=True)
    curve_sorted = sorted(curve, key=lambda x: x['alpha'])
    best = sorted(curve, key=lambda x: (x['mean_objective'], -abs(x['alpha'] - 0.5)), reverse=True)[0] if curve else None
    write_csv(out_dir / 'alpha_curve.csv', curve_sorted, ['task','alpha','num_samples','mean_psnr','mean_ssim','mean_objective','mean_oracle_objective','oracle_gap','oracle_hit_rate','pick_distribution'])
    write_csv(out_dir / 'selection_by_alpha.csv', sels, ['task','alpha','sample','chosen_tool','qe_score','psnr','ssim','objective','oracle_tool','oracle_objective','oracle_hit','oracle_gap'])
    (out_dir / 'best_alpha.json').write_text(json.dumps({'task': task, 'best': best, 'alphas': curve_sorted}, indent=2, ensure_ascii=False), encoding='utf-8')
    return best


def main():
    a = args(); t0 = time.time()
    os.environ['QE_STRICT_OFFLINE'] = '1' if a.strict_offline else '0'
    os.environ.setdefault('ALLOW_INPUT_AS_CANDIDATE', '0')
    out = (ROOT / a.output_root).resolve()
    if a.overwrite and out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    alphas = sorted({float(x) for x in a.alphas.split(',') if x.strip()})
    summary = {'split': a.split, 'limit': a.limit, 'seed': a.seed, 'alphas': alphas, 'objective': f'PSNR+{a.ssim_weight}*SSIM', 'tasks': {}}
    for task in [x.strip() for x in a.tasks.split(',') if x.strip()]:
        od = out / task; od.mkdir(parents=True, exist_ok=True)
        sampled_path = od / 'sampled_pairs.json'
        if sampled_path.exists():
            saved = json.loads(sampled_path.read_text(encoding='utf-8'))
            ps = [(Path(x['input']), Path(x['gt'])) for x in saved[:a.limit]]
        else:
            ps = pairs(task, a.split, a.limit, a.seed)
            sampled_path.write_text(json.dumps([{'input': str(i), 'gt': str(g)} for i, g in ps], indent=2), encoding='utf-8')
        if len(ps) > a.limit > 0:
            ps = ps[:a.limit]
            sampled_path.write_text(json.dumps([{'input': str(i), 'gt': str(g)} for i, g in ps], indent=2), encoding='utf-8')
        by = build(task, ps, od, a.ssim_weight)
        summary['tasks'][task] = sweep(task, by, alphas, od)
    summary['elapsed_sec'] = time.time() - t0
    (out / 'best_alpha_all_tasks.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[DONE] {out}', flush=True)

if __name__ == '__main__':
    main()
