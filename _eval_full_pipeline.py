#!/usr/bin/env python3
"""Full pipeline eval: WeatherBench 3 tasks + FoundIR-Weather 4 tasks, 50 images each."""
import argparse, csv, os, subprocess, threading, time, random, json
from pathlib import Path
from statistics import mean, stdev
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

class GPUMonitor:
    def __init__(self, interval=0.1, gpu_ids=None):
        self.interval=interval; self.gpu_ids=gpu_ids; self.stop_flag=False; self.peak=0.0
    def _query(self):
        try:
            out=subprocess.check_output(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True,stderr=subprocess.DEVNULL)
            vals=[]
            for i,l in enumerate(out.strip().splitlines()):
                try:v=float(l.strip())
                except:continue
                if self.gpu_ids is None or i in self.gpu_ids: vals.append(v)
            return max(vals) if vals else 0.0
        except: return 0.0
    def _loop(self):
        while not self.stop_flag:
            m=self._query(); self.peak=max(self.peak,m); time.sleep(self.interval)
    def start(self):
        self.stop_flag=False; self.peak=0.0; self.t=threading.Thread(target=self._loop,daemon=True); self.t.start()
    def stop(self):
        self.stop_flag=True; self.t.join(timeout=2)

def compute_psnr_ssim(output_path, gt_path):
    oi = Image.open(output_path).convert('RGB')
    gi = Image.open(gt_path).convert('RGB')
    if oi.size != gi.size: oi = oi.resize(gi.size, Image.BICUBIC)
    oa, ga = np.asarray(oi), np.asarray(gi)
    psnr = float(peak_signal_noise_ratio(ga, oa, data_range=255))
    ssim = float(structural_similarity(ga, oa, channel_axis=2, data_range=255))
    return psnr, ssim

def sample_images(img_dir, gt_dir, n, seed):
    exts = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}
    imgs = sorted([x for x in Path(img_dir).iterdir() if x.suffix.lower() in exts])
    imgs = [x for x in imgs if (Path(gt_dir)/x.name).exists()]
    if n <= 0 or len(imgs) <= n: return imgs
    rng = random.Random(seed)
    return sorted(rng.sample(imgs, n), key=lambda x: x.name)

def run_one(py, main_py, inp, out_dir, extra_env, poll=0.2, gpu_ids=None, timeout_sec=1800):
    env = dict(os.environ)
    env.update(dict(QE_STRICT_OFFLINE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        HF_DATASETS_OFFLINE='1', WEATHER_UTILS_DIR='utils_new', WEATHER_CKPT_DIR='pretrained_ckpts_new',
        WEATHER_PERCEPTION_SUBPROCESS='1', WEATHER_PERCEPTION_TOPK_GPUS='2',
        WEATHER_PERCEPTION_DEVICE_MAP='auto', WEATHER_PERCEPTION_RELEASE_AFTER_INFER='1'))
    if gpu_ids:
        env['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in gpu_ids)
    env.update(extra_env)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [py, str(main_py), '--input', str(inp), '--output', str(out_dir),
           '--planner_mode', 'qwen_only', '--run_restoration']
    mon = GPUMonitor(interval=poll, gpu_ids=gpu_ids)
    t0 = time.time()
    mon.start()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, env=env,
                          cwd=str(main_py.parent), timeout=timeout_sec)
        return_code = p.returncode; stdout = p.stdout or ''
        timed_out = 0
    except subprocess.TimeoutExpired:
        timed_out = 1; return_code = -9; stdout = ''
    finally:
        mon.stop()
    dt = time.time() - t0
    has_final = int((out_dir/'final_output.png').exists())
    ok = int(timed_out == 0 and return_code == 0 and has_final == 1)
    replan = stdout.count('Local replan triggered')
    return {'ok': ok, 'timed_out': timed_out, 'latency_sec': dt,
            'peak_gpu_mem_mb': mon.peak, 'replan_calls': replan}

def print_summary(name, rows):
    ok = [r for r in rows if r['ok']==1 and r['psnr']>0]
    if not ok: return {}
    s = {
        'count': len(ok),
        'psnr_mean': mean(r['psnr'] for r in ok),
        'ssim_mean': mean(r['ssim'] for r in ok),
        'latency_mean_sec': mean(r['latency_sec'] for r in ok),
        'peak_mem_mean_gb': mean(r['peak_gpu_mem_mb'] for r in ok)/1024.0,
        'replan_mean': mean(r['replan_calls'] for r in ok),
    }
    print(f"  {name:25s}: n={s['count']:3d}  PSNR={s['psnr_mean']:.2f}  SSIM={s['ssim_mean']:.4f}  "
          f"T={s['latency_mean_sec']:.0f}s/img  M={s['peak_mem_mean_gb']:.2f}GB  R={s['replan_mean']:.2f}/img")
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project_root', default=str(Path(__file__).resolve().parent))
    ap.add_argument('--samples', type=int, default=50)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--gpu_indices', default='0,1,2,3,4,5')
    ap.add_argument('--timeout_sec', type=int, default=1200)
    ap.add_argument('--output_dir', default=str(Path(__file__).resolve().parent/'output'/'eval_full_pipeline_50each'))
    a = ap.parse_args()

    project = Path(a.project_root)
    main_py = project/'main.py'
    py = '/root/project/huangchao/anaconda3/envs/weather_agent/bin/python'
    gids = [int(x.strip()) for x in a.gpu_indices.split(',') if x.strip()] if a.gpu_indices.strip() else None
    out_dir = Path(a.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    wb_root = project/'dataset'/'multi'/'WeatherBench'

    env = {'TASK_PLANNER_MODE': 'qwen_only', 'ENABLE_LOCAL_REPLAN': '1',
           'LOCAL_REPLAN_MAX': '3', 'KEEP_ALL_INTERMEDIATES': '1'}

    # ====================
    # WeatherBench: derain, dehaze, desnow
    # ====================
    wb_tasks = [
        ('WeatherBench-derain', wb_root/'rain'/'test'/'input', wb_root/'rain'/'test'/'target', 'derain'),
        ('WeatherBench-dehaze', wb_root/'haze'/'test'/'input', wb_root/'haze'/'test'/'target', 'dehaze'),
        ('WeatherBench-desnow', wb_root/'snow'/'test'/'input', wb_root/'snow'/'test'/'target', 'desnow'),
    ]

    # ====================
    # FoundIR-Weather: 08Haze, 10Rain, 11Raindrop, 12NightRain
    # ====================
    fir_root = project/'dataset'/'FoundIR-Weather'
    fir_tasks = [
        ('FoundIR-08Haze',       fir_root/'LQ'/'08Haze',       fir_root/'GT'/'08Haze',       'dehaze'),
        ('FoundIR-10Rain',       fir_root/'LQ'/'10Rain',       fir_root/'GT'/'10Rain',       'derain'),
        ('FoundIR-11Raindrop',   fir_root/'LQ'/'11Raindrop',   fir_root/'GT'/'11Raindrop',   'derain'),
        ('FoundIR-12NightRain',  fir_root/'LQ'/'12NightRain',  fir_root/'GT'/'12NightRain',  'derain'),
    ]

    all_rows = []

    for dataset_label, task_list in [('WeatherBench', wb_tasks), ('FoundIR-Weather', fir_tasks)]:
        print(f"\n{'#'*60}")
        print(f"# {dataset_label}")
        print(f"{'#'*60}")

        for task_label, img_dir, gt_dir, pipeline_task in task_list:
            print(f"\n{'='*60}")
            print(f"=== {task_label} ===")
            imgs = sample_images(img_dir, gt_dir, a.samples, a.seed + hash(task_label))
            print(f"{len(imgs)} images")

            for i, inp in enumerate(imgs):
                task_seed = a.seed + hash(f"{task_label}-{inp.stem}")
                sample_out = out_dir / task_label.replace(' ','_') / inp.stem
                r = run_one(py, main_py, inp, sample_out, env, 0.2, gids, a.timeout_sec)

                psnr, ssim = -1.0, -1.0
                final_png = sample_out / 'final_output.png'
                gt_png = Path(gt_dir) / inp.name
                if r['ok'] and final_png.exists() and gt_png.exists():
                    try: psnr, ssim = compute_psnr_ssim(final_png, gt_png)
                    except: pass

                row = {'dataset': dataset_label, 'task': task_label, 'pipeline_task': pipeline_task,
                       'sample': inp.stem, 'psnr': psnr, 'ssim': ssim, **r}
                all_rows.append(row)
                status = f"PSNR={psnr:.1f}" if psnr>0 else "FAIL"
                print(f"  [{i+1}/{len(imgs)}] {inp.stem}  {status}  "
                      f"t={r['latency_sec']:.0f}s  M={r['peak_gpu_mem_mb']/1024:.1f}G  R={r['replan_calls']}",
                      flush=True)

    # ====================
    # Save per-image CSV
    # ====================
    fields = ['dataset','task','sample','psnr','ssim','ok','timed_out','latency_sec','peak_gpu_mem_mb','replan_calls']
    per_csv = out_dir / 'per_image.csv'
    with per_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in all_rows: w.writerow({k: r.get(k,'') for k in fields})

    # ====================
    # Summary
    # ====================
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    summary = {'per_task': {}, 'overall': {}}

    for dataset_label, task_list in [('WeatherBench', wb_tasks), ('FoundIR-Weather', fir_tasks)]:
        print(f"\n[{dataset_label}]")
        ds_rows_all = [r for r in all_rows if r['dataset']==dataset_label]
        for task_label, _, _, _ in task_list:
            task_rows = [r for r in ds_rows_all if r['task']==task_label]
            s = print_summary(f"  {task_label}", task_rows)
            if s: summary['per_task'][task_label] = s

        ds_ok = [r for r in ds_rows_all if r['ok']==1 and r['psnr']>0]
        if ds_ok:
            s = print_summary(f"  ** {dataset_label} OVERALL **", ds_ok)
            if s: summary['overall'][dataset_label] = s

    # Grand total
    print(f"\n[GRAND TOTAL]")
    all_ok = [r for r in all_rows if r['ok']==1 and r['psnr']>0]
    s = print_summary(f"  ALL", all_ok)
    if s: summary['overall']['ALL'] = s

    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved to {out_dir}")

if __name__ == '__main__':
    main()
