#!/usr/bin/env python3
"""Run remaining FoundIR images (skip already done)."""
import csv, json, os, sys, time, threading, subprocess
from pathlib import Path
from statistics import mean

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

DATA_ROOT = PROJECT_ROOT / 'dataset' / 'FoundIR-Weather'
OUT_ROOT = PROJECT_ROOT / 'output' / 'foundir_full_50'  # same dir, append to existing CSV

SUBSETS = ['08Haze', '10Rain', '11Raindrop', '12NightRain']


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


def _set_env():
    defaults = {
        'QE_STRICT_OFFLINE': '0',
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'TIMM_OFFLINE': '1',
        'HF_DATASETS_OFFLINE': '1',
        'WEATHER_PERCEPTION_SUBPROCESS': '1',
        'TASK_PLANNER_ISOLATED_ENV': 'weather_agent_planner',
        'TASK_PLANNER_MODE': 'qwen_only',
        'ENABLE_LOCAL_REPLAN': '1', 'LOCAL_REPLAN_MAX': '3',
        'WEATHER_TOOL_VERBOSE_ERRORS': '1',
        'WEATHER_DIFFPLUGIN_GPU_IDS': '1,2,3,4,5',
        'WEATHER_DIFFPLUGIN_TOPK_GPUS': '4',
        'QE_CLIP_CONSEC_WINDOWS': '999999',
    }
    for k, v in defaults.items():
        os.environ[k] = v


def _collect_all(subset):
    lq_dir = DATA_ROOT / 'LQ' / subset
    gt_dir = DATA_ROOT / 'GT' / subset
    gt_map = {p.name: p for p in gt_dir.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'}}
    return sorted([(inp, gt_map[inp.name]) for inp in sorted(lq_dir.iterdir())
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
                    float(psnr_y(tp, tg).item()), float(ssim_y(tp, tg).item()))
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


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    _set_env()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_fn = _init_metrics(device)
    gpu_mon = GPUMonitor()

    csv_path = OUT_ROOT / 'per_image_metrics.csv'
    csv_fields = ['subset', 'task', 'sample', 'PSNR', 'SSIM', 'PSNR_Y', 'SSIM_Y',
                  'latency_sec', 'peak_gpu_mem_gb', 'replan_count']

    for subset in SUBSETS:
        all_pairs = _collect_all(subset)
        # Read already-done samples from CSV (survives directory cleanup)
        done_samples = set()
        if csv_path.exists():
            with csv_path.open('r') as f:
                for r in csv.DictReader(f):
                    if r['subset'] == subset and float(r.get('PSNR', 0) or 0) > 0:
                        done_samples.add(Path(r['sample']).stem)
        # Filter: skip existing CSV entries or existing final_output.png
        pairs = []
        for inp, gt in all_pairs:
            if inp.stem not in done_samples:
                sample_dir = OUT_ROOT / subset / inp.stem
                if not (sample_dir / 'final_output.png').exists():
                    pairs.append((inp, gt))
        if not pairs:
            print(f'{subset}: all {len(all_pairs)} already done')
            continue
        print(f"\n{'='*60}\n{subset}: {len(all_pairs)} total, {len(pairs)} remaining\n{'='*60}", flush=True)

        for idx, (inp, gt) in enumerate(pairs, 1):
            sample_dir = OUT_ROOT / subset / inp.stem
            sample_dir.mkdir(parents=True, exist_ok=True)
            final_out = sample_dir / 'final_output.png'

            t0 = time.time()
            gpu_mon.start()
            try:
                from main import main as run_pipeline
                run_pipeline(str(inp), str(sample_dir),
                             planner_mode='qwen_only', run_restoration=True)
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

            row = {'subset': subset, 'task': '', 'sample': inp.name,
                   'PSNR': psnr, 'SSIM': ssim, 'PSNR_Y': psnr_y, 'SSIM_Y': ssim_y,
                   'latency_sec': dt, 'peak_gpu_mem_gb': gpu_mon.peak / 1024.0,
                   'replan_count': replan_count}

            with csv_path.open('a', newline='') as f:
                csv.DictWriter(f, fieldnames=csv_fields).writerow(row)

            print(f"  [{idx}/{len(pairs)}] {inp.name} PSNR={psnr:.2f}/{psnr_y:.2f} "
                  f"SSIM={ssim:.4f}/{ssim_y:.4f} t={dt:.0f}s "
                  f"M={gpu_mon.peak/1024:.1f}G R={replan_count}", flush=True)

    # Summary
    written_rows = []
    with csv_path.open('r') as f:
        for r in csv.DictReader(f):
            for k in ['PSNR','SSIM','PSNR_Y','SSIM_Y','latency_sec','peak_gpu_mem_gb','replan_count']:
                r[k] = float(r.get(k, 0) or 0)
            written_rows.append(r)
    print(f"\nFinal summary ({len(written_rows)} images):")
    for subset in ['08Haze','10Rain','11Raindrop','12NightRain']:
        tr = [r for r in written_rows if r['subset'] == subset and r['PSNR'] > 0]
        if tr:
            ps = [r['PSNR'] for r in tr]; ss = [r['SSIM'] for r in tr]
            lt = [r['latency_sec'] for r in tr]
            rp = len([r for r in tr if r['replan_count'] > 0])
            print(f"  {subset}: {len(tr)}  PSNR={mean(ps):.2f}  SSIM={mean(ss):.4f}  "
                  f"Lat={mean(lt):.0f}s  Replan={rp}/{len(tr)}")
    print(f"\nDone. Results: {OUT_ROOT}")


if __name__ == '__main__':
    main()
