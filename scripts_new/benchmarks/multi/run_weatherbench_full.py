#!/usr/bin/env python3
"""Full pipeline on all WeatherBench test images (200 per task) — E1, E2, E3, E4, E5."""
import csv, json, os, random, sys, time, threading, subprocess
from pathlib import Path
from statistics import mean

# ── MUST be before pyiqa import ──────────────────────────
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
os.environ.setdefault('TIMM_OFFLINE', '1')
# ──────────────────────────────────────────────────────────

import torch
from PIL import Image
import torchvision.transforms as transforms
import pyiqa

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_ROOT = PROJECT_ROOT / 'dataset' / 'multi' / 'WeatherBench'
OUT_ROOT = PROJECT_ROOT / 'output' / 'weatherbench_full_200'

EXPERIMENTS = [
    {'id': 'E1_full',             'planner_mode': 'qwen_only',
     'disable_perception': False, 'random_single_expert': False, 'disable_replan': False},
    {'id': 'E2_no_perception',    'planner_mode': 'qwen_only',
     'disable_perception': True,  'random_single_expert': False, 'disable_replan': False},
    {'id': 'E3_no_planning',      'planner_mode': 'perception_direct',
     'disable_perception': False, 'random_single_expert': False, 'disable_replan': False},
    {'id': 'E4_no_multiexpert',   'planner_mode': 'qwen_only',
     'disable_perception': False, 'random_single_expert': True,  'disable_replan': False,
     'random_single_expert_seed': 2026},
    {'id': 'E5_no_replan',        'planner_mode': 'qwen_only',
     'disable_perception': False, 'random_single_expert': False, 'disable_replan': True},
]


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


def _set_env(exp):
    defaults = {
        'QE_STRICT_OFFLINE': '0',
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'TIMM_OFFLINE': '1',
        'HF_DATASETS_OFFLINE': '1',
        'WEATHER_PERCEPTION_SUBPROCESS': '1',
        'TASK_PLANNER_ISOLATED_ENV': 'weather_agent_planner',
        'TASK_PLANNER_MODE': exp['planner_mode'],
        'ENABLE_LOCAL_REPLAN': '0' if exp['disable_replan'] else '1',
        'LOCAL_REPLAN_MAX': '3',
        'WEATHER_TOOL_VERBOSE_ERRORS': '1',
        # diffplugin multi-GPU with tiling
        'WEATHER_DIFFPLUGIN_GPU_IDS': '0,1,2,3,4,5',
        'WEATHER_DIFFPLUGIN_TOPK_GPUS': '4',
        # ridcp auto-select GPU at runtime
        'WEATHER_RIDCP_GPU_IDS': '0,1,2,3,4',
        # maxim: disable tiling on single GPU
        'WEATHER_MAXIM_ENABLE_MULTI_GPU_TILING': '0',
        # Lock L/U bounds
        'QE_CLIP_CONSEC_WINDOWS': '999999',
    }
    if exp.get('random_single_expert'):
        defaults['RANDOM_SINGLE_EXPERT'] = '1'
        defaults['RANDOM_SINGLE_EXPERT_SEED'] = str(int(exp.get('random_single_expert_seed', 2026)))
    for k, v in defaults.items():
        os.environ[k] = v


def _collect_pairs(task):
    inp_dir = DATA_ROOT / task / 'test' / 'input'
    gt_dir = DATA_ROOT / task / 'test' / 'target'
    gt_map = {p.name: p for p in gt_dir.iterdir()
              if p.suffix.lower() in {'.jpg','.jpeg','.png'}}
    return sorted([(inp, gt_map[inp.name]) for inp in sorted(inp_dir.iterdir())
                   if inp.suffix.lower() in {'.jpg','.jpeg','.png'} and inp.name in gt_map],
                  key=lambda x: x[0].name)


def _init_metrics(device):
    to_tensor = transforms.ToTensor()
    psnr_rgb = pyiqa.create_metric('psnr', device=device, test_y_channel=False)
    ssim_rgb = pyiqa.create_metric('ssim', device=device, test_y_channel=False)
    psnr_y  = pyiqa.create_metric('psnr', device=device, test_y_channel=True)
    ssim_y  = pyiqa.create_metric('ssim', device=device, test_y_channel=True)
    def fn(pred, gt):
        pimg = Image.open(pred).convert('RGB'); gimg = Image.open(gt).convert('RGB')
        if pimg.size != gimg.size: pimg = pimg.resize(gimg.size, Image.BICUBIC)
        tp = to_tensor(pimg).unsqueeze(0).to(device)
        tg = to_tensor(gimg).unsqueeze(0).to(device)
        with torch.no_grad():
            return (float(psnr_rgb(tp, tg).item()), float(ssim_rgb(tp, tg).item()),
                    float(psnr_y(tp, tg).item()),   float(ssim_y(tp, tg).item()))
    return fn


def _count_replan(sample_dir):
    step_files = sorted(sample_dir.glob('selected_step_*.png'))
    if not step_files: return 0
    step_tasks = {}
    for sf in step_files:
        name = sf.stem.replace('selected_step_','')
        parts = name.split('_', 1)
        if len(parts) >= 2:
            step_tasks.setdefault(parts[0], set()).add(parts[1])
    return sum(max(0, len(v)-1) for v in step_tasks.values())


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
    explicit_inputs = {'C_I': '', 'D_I': [], 'A_I': ['derain','dehaze','desnow'], 'I': str(inp)}
    result = planner._plan_via_isolated_env(
        image_path=str(inp), explicit_inputs=explicit_inputs,
        allowed_steps=['derain','dehaze','desnow'], prompt=prompt)
    plan = [s for s in result.get('plan', []) if s in {'derain','dehaze','desnow'}]
    if not plan: plan = ['derain']
    return RestorationAgent().execute_plan(plan, str(inp), str(out_dir))


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_fn = _init_metrics(device)
    gpu_mon = GPUMonitor()

    TASKS = ['rain', 'haze', 'snow']
    all_summaries = []

    for exp in EXPERIMENTS:
        exp_id = exp['id']
        _set_env(exp)
        print(f"\n{'='*60}\n{exp_id}\n{'='*60}", flush=True)

        exp_out = OUT_ROOT / exp_id
        exp_out.mkdir(parents=True, exist_ok=True)
        csv_path = exp_out / 'per_image_metrics.csv'
        csv_fields = ['experiment','task','sample','PSNR','SSIM','PSNR_Y','SSIM_Y',
                      'latency_sec','peak_gpu_mem_gb','replan_count']
        csv_header_written = False

        for task in TASKS:
            pairs = _collect_pairs(task)
            task_dir = exp_out / task; task_dir.mkdir(parents=True, exist_ok=True)
            print(f"[{task}] {len(pairs)} samples", flush=True)

            for idx, (inp, gt) in enumerate(pairs, 1):
                sample_dir = task_dir / inp.stem
                sample_dir.mkdir(parents=True, exist_ok=True)
                final_out = sample_dir / 'final_output.png'

                t0 = time.time()
                gpu_mon.start()
                try:
                    if not final_out.exists():
                        if exp['disable_perception']:
                            run_e2(inp, sample_dir)
                        else:
                            from main import main as run_pipeline
                            run_pipeline(str(inp), str(sample_dir),
                                         planner_mode=exp['planner_mode'],
                                         run_restoration=True)
                except Exception as e:
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
                       'PSNR_Y': psnr_y, 'SSIM_Y': ssim_y,
                       'latency_sec': dt,
                       'peak_gpu_mem_gb': gpu_mon.peak / 1024.0,
                       'replan_count': replan_count}

                if not csv_header_written:
                    with csv_path.open('w', newline='') as f:
                        csv.DictWriter(f, fieldnames=csv_fields).writeheader()
                    csv_header_written = True
                with csv_path.open('a', newline='') as f:
                    csv.DictWriter(f, fieldnames=csv_fields).writerow(row)

                print(f"  [{idx}/{len(pairs)}] {inp.name} PSNR={psnr:.2f}/{psnr_y:.2f} "
                      f"SSIM={ssim:.4f}/{ssim_y:.4f} t={dt:.0f}s "
                      f"M={gpu_mon.peak/1024:.1f}G R={replan_count}", flush=True)

        # Summary
        if not csv_path.exists():
            print(f"  WARNING: no valid results for {exp_id}")
            continue
        written_rows = []
        with csv_path.open('r') as f:
            for r in csv.DictReader(f):
                for k in ['PSNR','SSIM','PSNR_Y','SSIM_Y','latency_sec',
                          'peak_gpu_mem_gb','replan_count']:
                    r[k] = float(r.get(k, 0) or 0)
                written_rows.append(r)

        for task in TASKS:
            tr = [r for r in written_rows if r['task'] == task and r['PSNR'] > 0]
            if tr:
                s = {'experiment': exp_id, 'task': task, 'n': len(tr),
                     'psnr': mean([r['PSNR'] for r in tr]),
                     'ssim': mean([r['SSIM'] for r in tr]),
                     'psnr_y': mean([r['PSNR_Y'] for r in tr]),
                     'ssim_y': mean([r['SSIM_Y'] for r in tr]),
                     'latency': mean([r['latency_sec'] for r in tr]),
                     'mem': mean([r['peak_gpu_mem_gb'] for r in tr]),
                     'replan': mean([r['replan_count'] for r in tr])}
                print(f"  {task}: n={s['n']} PSNR={s['psnr']:.2f} SSIM={s['ssim']:.4f} "
                      f"t={s['latency']:.0f}s M={s['mem']:.2f}G R={s['replan']:.2f}", flush=True)
                all_summaries.append(s)

    keys = ['experiment','task','n','psnr','ssim','psnr_y','ssim_y','latency','mem','replan']
    with (OUT_ROOT / 'summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader(); w.writerows(all_summaries)
    print(f"\nDone. Results: {OUT_ROOT}")


if __name__ == '__main__':
    main()
