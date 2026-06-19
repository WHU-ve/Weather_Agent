#!/usr/bin/env python3
"""Ablation: E1-E5 with PSNR/SSIM, latency, GPU memory, replanning triggers."""
import csv, json, os, random, sys, time, threading, subprocess
from pathlib import Path
from statistics import mean

# Must be set BEFORE pyiqa import to prevent network retries on model load.
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
os.environ.setdefault('TIMM_OFFLINE', '1')

import torch
from PIL import Image
import torchvision.transforms as transforms
import pyiqa

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_ROOT = PROJECT_ROOT / 'dataset' / 'multi' / 'WeatherBench'
OUT_ROOT = PROJECT_ROOT / 'output' / 'ablation_v6'

class GPUMonitor:
    def __init__(self, interval=0.2):
        self.interval = interval; self.stop_flag = False; self.peak = 0.0
    def _query(self):
        try:
            out = subprocess.check_output(
                ["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],
                text=True, stderr=subprocess.DEVNULL)
            return max(float(l.strip()) for l in out.strip().splitlines() if l.strip())
        except: return 0.0
    def _loop(self):
        while not self.stop_flag:
            self.peak = max(self.peak, self._query()); time.sleep(self.interval)
    def start(self):
        self.stop_flag = False; self.peak = 0.0
        self.t = threading.Thread(target=self._loop, daemon=True); self.t.start()
    def stop(self):
        self.stop_flag = True; self.t.join(timeout=2)

EXPERIMENTS = [
    # E1, E4, E5 done — skip
    # {'id': 'E1_full', ...},
    # {'id': 'E4_no_multiexpert', ...},
    # {'id': 'E5_no_replan', ...},
    {'id': 'E3_no_planning', 'planner_mode': 'perception_direct',
     'disable_perception': False, 'random_single_expert': False, 'disable_replan': False,
     'limit_per_task': 50, 'track_system': True},
]

def _set_env(exp):
    defaults = {
        'QE_STRICT_OFFLINE': '0', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'TIMM_OFFLINE': '1',
        'WEATHER_TOOL_VERBOSE_ERRORS': '1',
        # diffplugin multi-GPU: auto-select best GPUs at runtime (not pre-assigned).
        'WEATHER_DIFFPLUGIN_GPU_IDS': '0,1,2,3,4,5',
        'HF_DATASETS_OFFLINE': '1', 'WEATHER_PERCEPTION_SUBPROCESS': '1',
        'TASK_PLANNER_ISOLATED_ENV': 'weather_agent_planner',
        'TASK_PLANNER_MODE': exp['planner_mode'],
        'ENABLE_LOCAL_REPLAN': '0' if exp['disable_replan'] else '1',
        'LOCAL_REPLAN_MAX': '3',
        'RANDOM_SINGLE_EXPERT': '1' if exp.get('random_single_expert', False) else '0',
        'RANDOM_SINGLE_EXPERT_SEED': str(int(exp.get('random_single_expert_seed', 2026))),
        # Lock L/U bounds: never recalibrate during ablation.
        'QE_CLIP_CONSEC_WINDOWS': '999999',
    }
    for k, v in defaults.items():
        os.environ[k] = v

def _collect_pairs(task, limit, seed=2026):
    inp_dir = DATA_ROOT / task / 'test' / 'input'
    gt_dir = DATA_ROOT / task / 'test' / 'target'
    gt_map = {p.name: p for p in gt_dir.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'}}
    pairs = [(inp, gt_map[inp.name]) for inp in sorted(inp_dir.iterdir())
             if inp.suffix.lower() in {'.jpg','.jpeg','.png'} and inp.name in gt_map]
    if limit > 0 and len(pairs) > limit:
        rng = random.Random(seed + sum(ord(c) for c in task))
        pairs = sorted(rng.sample(pairs, limit), key=lambda x: x[0].name)
    return pairs

def _init_metrics(device):
    to_tensor = transforms.ToTensor()
    psnr_rgb = pyiqa.create_metric('psnr', device=device, test_y_channel=False)
    ssim_rgb = pyiqa.create_metric('ssim', device=device, test_y_channel=False)
    psnr_y = pyiqa.create_metric('psnr', device=device, test_y_channel=True)
    ssim_y = pyiqa.create_metric('ssim', device=device, test_y_channel=True)
    def fn(pred, gt):
        pimg = Image.open(pred).convert('RGB'); gimg = Image.open(gt).convert('RGB')
        if pimg.size != gimg.size: pimg = pimg.resize(gimg.size, Image.BICUBIC)
        tp = to_tensor(pimg).unsqueeze(0).to(device); tg = to_tensor(gimg).unsqueeze(0).to(device)
        with torch.no_grad():
            return (float(psnr_rgb(tp, tg).item()), float(ssim_rgb(tp, tg).item()),
                    float(psnr_y(tp, tg).item()), float(ssim_y(tp, tg).item()))
    return fn

def _count_replan(sample_dir):
    """Count replanning from selected_step files.
    One replan = one step has multiple different tasks."""
    step_files = sorted(sample_dir.glob('selected_step_*.png'))
    if not step_files: return 0
    # Group by step number, count unique tasks per step
    step_tasks = {}
    for sf in step_files:
        name = sf.stem.replace('selected_step_','')
        parts = name.split('_', 1)
        if len(parts) >= 2:
            step_num = parts[0]
            task_name = parts[1]
            if step_num not in step_tasks: step_tasks[step_num] = set()
            step_tasks[step_num].add(task_name)
    # Replan count = sum of (extra tasks per step)
    total = 0
    for step_num, tasks in step_tasks.items():
        if len(tasks) > 1:
            total += len(tasks) - 1  # e.g., step1 has {derain, dehaze} → 1 replan
    return total

def run_e2(inp, out_dir):
    """E2: Qwen plans from image alone (no perception), via isolated env."""
    from task_planner import TaskPlanner
    from restoration_agent import RestorationAgent
    prompt = (
        'You are an expert in weather image restoration planning. '
        'Look at this image and identify what adverse weather conditions are present. '
        'Your task: select the appropriate subset from [derain, dehaze, desnow] '
        'and arrange them into an ordered restoration sequence. '
        'You may select one, two, or all three tasks. '
        'The "plan" must be a complete ordered list of all selected tasks. '
        'Output ONLY a JSON object with key "plan". '
        'Example: {"plan": ["derain", "dehaze"]}'
    )
    planner = TaskPlanner()
    explicit_inputs = {
        'C_I': '',
        'D_I': [],
        'A_I': ['derain', 'dehaze', 'desnow'],
        'I': str(inp),
    }
    result = planner._plan_via_isolated_env(
        image_path=str(inp),
        explicit_inputs=explicit_inputs,
        allowed_steps=['derain', 'dehaze', 'desnow'],
        prompt=prompt,
    )
    plan = [s for s in result.get('plan', []) if s in {'derain','dehaze','desnow'}]
    if not plan: plan = ['derain']
    print(f"    [E2] plan: {plan}")
    return RestorationAgent().execute_plan(plan, str(inp), str(out_dir))

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_fn = _init_metrics(device)
    gpu_mon = GPUMonitor()
    all_summaries = []

    for exp in EXPERIMENTS:
        exp_id = exp['id']
        out_dir = OUT_ROOT / exp_id; out_dir.mkdir(parents=True, exist_ok=True)
        _set_env(exp)
        track = exp.get('track_system', False)
        limit = exp['limit_per_task']
        print(f"\n{'='*60}\n{exp_id} ({limit}/task, track_system={track})\n{'='*60}", flush=True)

        csv_path = out_dir / 'per_image_metrics.csv'
        csv_fields = ['experiment', 'task', 'sample', 'PSNR', 'SSIM', 'PSNR_Y', 'SSIM_Y']
        if track:
            csv_fields += ['latency_sec', 'peak_gpu_mem_gb', 'replan_count']
        csv_header_written = False

        for task in ['rain', 'haze', 'snow']:
            pairs = _collect_pairs(task, limit)
            task_dir = out_dir / task; task_dir.mkdir(parents=True, exist_ok=True)
            print(f"[{task}] {len(pairs)} samples", flush=True)
            for idx, (inp, gt) in enumerate(pairs, 1):
                sample_dir = task_dir / inp.stem
                sample_dir.mkdir(parents=True, exist_ok=True)
                final_out = sample_dir / 'final_output.png'

                t0 = time.time()
                gpu_mon.start()
                ok = True
                try:
                    if not final_out.exists():
                        if exp['disable_perception']:
                            final = run_e2(inp, sample_dir)
                        else:
                            from main import main as run_pipeline
                            run_pipeline(str(inp), str(sample_dir),
                                        planner_mode=exp['planner_mode'], run_restoration=True)
                except Exception as e:
                    ok = False
                    print(f"    [{idx}/{len(pairs)}] {inp.name} FAIL: {e}", flush=True)
                finally:
                    gpu_mon.stop()
                dt = time.time() - t0

                if not final_out.exists():
                    print(f"    [{idx}/{len(pairs)}] {inp.name} MISSING OUTPUT", flush=True)
                    continue

                psnr, ssim, psnr_y, ssim_y = eval_fn(final_out, gt)
                replan_count = _count_replan(sample_dir)

                row = {'experiment': exp_id, 'task': task, 'sample': inp.name,
                       'PSNR': psnr, 'SSIM': ssim,
                       'PSNR_Y': psnr_y, 'SSIM_Y': ssim_y}
                if track:
                    row['latency_sec'] = dt
                    row['peak_gpu_mem_gb'] = gpu_mon.peak / 1024.0
                    row['replan_count'] = replan_count

                # Write immediately — survive crashes.
                if not csv_header_written:
                    with csv_path.open('w', newline='') as f:
                        w = csv.DictWriter(f, fieldnames=csv_fields); w.writeheader()
                    csv_header_written = True
                with csv_path.open('a', newline='') as f:
                    csv.DictWriter(f, fieldnames=csv_fields).writerow(row)

                extra = f" t={dt:.0f}s M={gpu_mon.peak/1024:.1f}G R={replan_count}" if track else ""
                print(f"  [{idx}/{len(pairs)}] {inp.name} PSNR={psnr:.2f}/{psnr_y:.2f} SSIM={ssim:.4f}/{ssim_y:.4f}{extra}", flush=True)

        if not csv_path.exists():
            print(f"  WARNING: no valid results for {exp_id}")
            continue

        # Read back from CSV for summary
        written_rows = []
        with csv_path.open('r') as f:
            for r in csv.DictReader(f):
                r['PSNR'] = float(r['PSNR']); r['SSIM'] = float(r['SSIM'])
                r['PSNR_Y'] = float(r.get('PSNR_Y', 0) or 0)
                r['SSIM_Y'] = float(r.get('SSIM_Y', 0) or 0)
                if track:
                    r['latency_sec'] = float(r.get('latency_sec', 0) or 0)
                    r['peak_gpu_mem_gb'] = float(r.get('peak_gpu_mem_gb', 0) or 0)
                    r['replan_count'] = float(r.get('replan_count', 0) or 0)
                written_rows.append(r)

        # Summary
        ok_rows = [r for r in written_rows if r['PSNR'] > 0]
        task_stats = {}
        for task in ['rain', 'haze', 'snow']:
            tr = [r for r in ok_rows if r['task'] == task]
            ps = [r['PSNR'] for r in tr]; ss = [r['SSIM'] for r in tr]
            psy = [r['PSNR_Y'] for r in tr]; ssy = [r['SSIM_Y'] for r in tr]
            s = {'n': len(tr),
                 'psnr': mean(ps) if ps else 0, 'ssim': mean(ss) if ss else 0,
                 'psnr_y': mean(psy) if psy else 0, 'ssim_y': mean(ssy) if ssy else 0}
            if track:
                s['latency'] = mean([r['latency_sec'] for r in tr]) if tr else 0
                s['mem'] = mean([r['peak_gpu_mem_gb'] for r in tr]) if tr else 0
                s['replan'] = mean([r['replan_count'] for r in tr]) if tr else 0
            task_stats[task] = s

        overall_ps = [r['PSNR'] for r in ok_rows]; overall_ss = [r['SSIM'] for r in ok_rows]
        overall_ps_y = [r['PSNR_Y'] for r in ok_rows]; overall_ss_y = [r['SSIM_Y'] for r in ok_rows]
        summary = {'experiment': exp_id, 'tasks': task_stats,
                   'overall_psnr': mean(overall_ps) if overall_ps else 0,
                   'overall_ssim': mean(overall_ss) if overall_ss else 0,
                   'overall_psnr_y': mean(overall_ps_y) if overall_ps_y else 0,
                   'overall_ssim_y': mean(overall_ss_y) if overall_ss_y else 0}
        if track:
            summary['overall_latency'] = mean([r['latency_sec'] for r in ok_rows]) if ok_rows else 0
            summary['overall_mem'] = mean([r['peak_gpu_mem_gb'] for r in ok_rows]) if ok_rows else 0
            summary['overall_replan'] = mean([r['replan_count'] for r in ok_rows]) if ok_rows else 0

        print(f"\n  Summary:")
        for task in ['rain','haze','snow']:
            ts = task_stats[task]
            print(f"  {task}: n={ts['n']} PSNR={ts['psnr']:.2f}/{ts['psnr_y']:.2f} SSIM={ts['ssim']:.4f}/{ts['ssim_y']:.4f}", end='')
            if track: print(f" t={ts['latency']:.0f}s M={ts['mem']:.2f}G R={ts['replan']:.2f}", end='')
            print()
        print(f"  Overall: PSNR={summary['overall_psnr']:.2f}/{summary['overall_psnr_y']:.2f} SSIM={summary['overall_ssim']:.4f}/{summary['overall_ssim_y']:.4f}", flush=True)
        (out_dir/'summary.json').write_text(json.dumps(summary, indent=2))

        s_row = {'experiment': exp_id,
                 'rain_psnr': task_stats['rain']['psnr'], 'rain_ssim': task_stats['rain']['ssim'],
                 'rain_psnr_y': task_stats['rain']['psnr_y'], 'rain_ssim_y': task_stats['rain']['ssim_y'],
                 'haze_psnr': task_stats['haze']['psnr'], 'haze_ssim': task_stats['haze']['ssim'],
                 'haze_psnr_y': task_stats['haze']['psnr_y'], 'haze_ssim_y': task_stats['haze']['ssim_y'],
                 'snow_psnr': task_stats['snow']['psnr'], 'snow_ssim': task_stats['snow']['ssim'],
                 'snow_psnr_y': task_stats['snow']['psnr_y'], 'snow_ssim_y': task_stats['snow']['ssim_y'],
                 'overall_psnr': summary['overall_psnr'], 'overall_ssim': summary['overall_ssim'],
                 'overall_psnr_y': summary['overall_psnr_y'], 'overall_ssim_y': summary['overall_ssim_y']}
        if track:
            s_row.update({'overall_latency_sec': summary['overall_latency'],
                          'overall_mem_gb': summary['overall_mem'],
                          'overall_replan_per_img': summary['overall_replan']})
        all_summaries.append(s_row)

    # Overall table
    keys = ['experiment',
            'rain_psnr','rain_ssim','rain_psnr_y','rain_ssim_y',
            'haze_psnr','haze_ssim','haze_psnr_y','haze_ssim_y',
            'snow_psnr','snow_ssim','snow_psnr_y','snow_ssim_y',
            'overall_psnr','overall_ssim','overall_psnr_y','overall_ssim_y']
    if any('overall_latency_sec' in s for s in all_summaries):
        keys += ['overall_latency_sec','overall_mem_gb','overall_replan_per_img']
    with (OUT_ROOT/'ablation_summary.csv').open('w',newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore'); w.writeheader(); w.writerows(all_summaries)
    print(f"\nDone. Results: {OUT_ROOT}")

if __name__ == '__main__':
    main()
