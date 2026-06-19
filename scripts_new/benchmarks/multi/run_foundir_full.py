#!/usr/bin/env python3
"""Full pipeline on FoundIR-Weather: 08Haze, 10Rain, 11Raindrop, 12NightRain × 50."""
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

DATA_ROOT = PROJECT_ROOT / 'dataset' / 'FoundIR-Weather'
OUT_ROOT = PROJECT_ROOT / 'output' / 'foundir_full_50'

# FoundIR subset → WeatherBench task mapping
SUBSET_TASK_MAP = {
    '08Haze':      'haze',     # → dehaze
    '10Rain':      'rain',     # → derain
    '11Raindrop':  'rain',     # → derain (raindrop is a rain subtype)
    '12NightRain': 'rain',     # → derain (night rain)
}


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
        'ENABLE_LOCAL_REPLAN': '1',
        'LOCAL_REPLAN_MAX': '3',
        'WEATHER_TOOL_VERBOSE_ERRORS': '1',
        # diffplugin multi-GPU with tiling (--gpu_ids conflict fixed, top 4 GPUs)
        'WEATHER_DIFFPLUGIN_GPU_IDS': '0,1,2,3,4,5',
        'WEATHER_DIFFPLUGIN_TOPK_GPUS': '4',
        # ridcp auto-select GPU at runtime (avoids bad GPU 5)
        'WEATHER_RIDCP_GPU_IDS': '0,1,2,3,4',
        # maxim: disable multi-GPU tiling (OOM on large FoundIR images with single GPU)
        'WEATHER_MAXIM_ENABLE_MULTI_GPU_TILING': '0',
        # Lock L/U bounds: never recalibrate
        'QE_CLIP_CONSEC_WINDOWS': '999999',
    }
    for k, v in defaults.items():
        os.environ[k] = v


def _collect_pairs(subset, limit, seed=2026):
    """Read LQ/GT pairs from FoundIR."""
    lq_dir = DATA_ROOT / 'LQ' / subset
    gt_dir = DATA_ROOT / 'GT' / subset
    gt_map = {p.name: p for p in gt_dir.iterdir()
              if p.suffix.lower() in {'.jpg','.jpeg','.png'}}
    pairs = [(inp, gt_map[inp.name]) for inp in sorted(lq_dir.iterdir())
             if inp.suffix.lower() in {'.jpg','.jpeg','.png'} and inp.name in gt_map]
    if 0 < limit < len(pairs):
        rng = random.Random(seed + sum(ord(c) for c in subset))
        pairs = sorted(rng.sample(pairs, limit), key=lambda x: x[0].name)
    return pairs


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
            sn = parts[0]; tn = parts[1]
            step_tasks.setdefault(sn, set()).add(tn)
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
    csv_header_written = False

    all_summaries = []

    for subset, task in SUBSET_TASK_MAP.items():
        pairs = _collect_pairs(subset, limit=50)
        task_dir = OUT_ROOT / subset
        task_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*60}\n{subset} → task={task}, {len(pairs)} samples\n{'='*60}", flush=True)

        for idx, (inp, gt) in enumerate(pairs, 1):
            sample_dir = task_dir / inp.stem
            sample_dir.mkdir(parents=True, exist_ok=True)
            final_out = sample_dir / 'final_output.png'

            t0 = time.time()
            gpu_mon.start()
            try:
                if not final_out.exists():
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

            row = {'subset': subset, 'task': task, 'sample': inp.name,
                   'PSNR': psnr, 'SSIM': ssim, 'PSNR_Y': psnr_y, 'SSIM_Y': ssim_y,
                   'latency_sec': dt, 'peak_gpu_mem_gb': gpu_mon.peak / 1024.0,
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

        if not csv_path.exists():
            print(f"  WARNING: no valid results for {subset}")
            continue

        # Summary for this subset
        written_rows = []
        with csv_path.open('r') as f:
            for r in csv.DictReader(f):
                r['PSNR'] = float(r['PSNR']); r['SSIM'] = float(r['SSIM'])
                r['PSNR_Y'] = float(r.get('PSNR_Y', 0) or 0)
                r['SSIM_Y'] = float(r.get('SSIM_Y', 0) or 0)
                r['latency_sec'] = float(r.get('latency_sec', 0) or 0)
                r['peak_gpu_mem_gb'] = float(r.get('peak_gpu_mem_gb', 0) or 0)
                r['replan_count'] = float(r.get('replan_count', 0) or 0)
                written_rows.append(r)

        subset_rows = [r for r in written_rows if r['subset'] == subset and r['PSNR'] > 0]
        if not subset_rows: continue
        ps = [r['PSNR'] for r in subset_rows]; ss = [r['SSIM'] for r in subset_rows]
        lt = [r['latency_sec'] for r in subset_rows]
        gm = [r['peak_gpu_mem_gb'] for r in subset_rows]
        rp = [r['replan_count'] for r in subset_rows]
        s = {'subset': subset, 'n': len(subset_rows),
             'psnr': mean(ps), 'ssim': mean(ss),
             'psnr_y': mean([r['PSNR_Y'] for r in subset_rows]),
             'ssim_y': mean([r['SSIM_Y'] for r in subset_rows]),
             'latency': mean(lt), 'mem': mean(gm), 'replan': mean(rp)}
        print(f"\n  {subset}: n={s['n']} PSNR={s['psnr']:.2f} SSIM={s['ssim']:.4f} "
              f"t={s['latency']:.0f}s M={s['mem']:.2f}G R={s['replan']:.2f}", flush=True)
        (task_dir / 'summary.json').write_text(json.dumps(s, indent=2))
        all_summaries.append(s)

    # Overall table
    keys = ['subset','n','psnr','ssim','psnr_y','ssim_y','latency','mem','replan']
    with (OUT_ROOT / 'summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader(); w.writerows(all_summaries)
    print(f"\nDone. Results: {OUT_ROOT}")


if __name__ == '__main__':
    main()
