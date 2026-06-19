"""Continue eval: desnow + FoundIR with new code, reusing existing sample list."""
import csv, os, subprocess, threading, time, random, json
from pathlib import Path
from statistics import mean
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
    return (float(peak_signal_noise_ratio(ga,oa,data_range=255)),
            float(structural_similarity(ga,oa,channel_axis=2,data_range=255)))

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
    if gpu_ids: env['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in gpu_ids)
    env.update(extra_env)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [py, str(main_py), '--input', str(inp), '--output', str(out_dir),
           '--planner_mode', 'qwen_only', '--run_restoration']
    mon = GPUMonitor(interval=poll, gpu_ids=gpu_ids)
    t0 = time.time(); mon.start()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(main_py.parent), timeout=timeout_sec)
        return_code = p.returncode; stdout = p.stdout or ''; timed_out = 0
    except subprocess.TimeoutExpired:
        timed_out = 1; return_code = -9; stdout = ''
    finally: mon.stop()
    dt = time.time() - t0
    has_final = int((out_dir/'final_output.png').exists())
    ok = int(timed_out == 0 and return_code == 0 and has_final == 1)
    return {'ok': ok, 'timed_out': timed_out, 'latency_sec': dt,
            'peak_gpu_mem_mb': mon.peak, 'replan_calls': stdout.count('Local replan triggered')}

project = Path(__file__).resolve().parent
main_py = project/'main.py'
py = '/root/project/huangchao/anaconda3/envs/weather_agent/bin/python'
gids = [0,1,2,3,4,5]
out_dir = project/'output'/'eval_full_pipeline_50each'
wb_root = project/'dataset'/'multi'/'WeatherBench'
fir_root = project/'dataset'/'FoundIR-Weather'
env = {'TASK_PLANNER_MODE': 'qwen_only', 'ENABLE_LOCAL_REPLAN': '1', 'LOCAL_REPLAN_MAX': '3', 'KEEP_ALL_INTERMEDIATES': '1'}

tasks = [
    ('WeatherBench-desnow', wb_root/'snow'/'test'/'input', wb_root/'snow'/'test'/'target', 'desnow'),
    ('FoundIR-08Haze', fir_root/'LQ'/'08Haze', fir_root/'GT'/'08Haze', 'dehaze'),
    ('FoundIR-10Rain', fir_root/'LQ'/'10Rain', fir_root/'GT'/'10Rain', 'derain'),
    ('FoundIR-11Raindrop', fir_root/'LQ'/'11Raindrop', fir_root/'GT'/'11Raindrop', 'derain'),
    ('FoundIR-12NightRain', fir_root/'LQ'/'12NightRain', fir_root/'GT'/'12NightRain', 'derain'),
]

all_rows = []
for task_label, img_dir, gt_dir, pipeline_task in tasks:
    print(f"\n{'='*60}")
    print(f"=== {task_label} ===")
    imgs = sample_images(img_dir, gt_dir, 50, 2026 + hash(task_label))
    print(f"{len(imgs)} images")

    for i, inp in enumerate(imgs):
        sample_out = out_dir / task_label.replace(' ','_') / inp.stem
        r = run_one(py, main_py, inp, sample_out, env, 0.2, gids, 1200)

        psnr, ssim = -1.0, -1.0
        final_png = sample_out / 'final_output.png'
        gt_png = Path(gt_dir) / inp.name
        if r['ok'] and final_png.exists() and gt_png.exists():
            try: psnr, ssim = compute_psnr_ssim(final_png, gt_png)
            except: pass

        row = {'dataset': task_label.split('-')[0], 'task': task_label, 'sample': inp.stem,
               'psnr': psnr, 'ssim': ssim, **r}
        all_rows.append(row)
        status = f"PSNR={psnr:.1f}" if psnr>0 else "FAIL"
        print(f"  [{i+1}/{len(imgs)}] {inp.stem}  {status}  "
              f"t={r['latency_sec']:.0f}s  M={r['peak_gpu_mem_mb']/1024:.1f}G  R={r['replan_calls']}", flush=True)

# Save
fields = ['dataset','task','sample','psnr','ssim','ok','timed_out','latency_sec','peak_gpu_mem_mb','replan_calls']
csv_path = out_dir / 'per_image_desnow_foundir.csv'
with csv_path.open('w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
    for r in all_rows: w.writerow({k: r.get(k,'') for k in fields})

# Summary
print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
for task_label, _, _, _ in tasks:
    tr = [r for r in all_rows if r['task']==task_label and r['ok']==1 and r['psnr']>0]
    if tr:
        p = mean(r['psnr'] for r in tr); s = mean(r['ssim'] for r in tr)
        t = mean(r['latency_sec'] for r in tr)
        m = mean(r['peak_gpu_mem_mb'] for r in tr)/1024.0
        rp = mean(r['replan_calls'] for r in tr)
        print(f"  {task_label:30s}: n={len(tr):2d}  PSNR={p:.2f}  SSIM={s:.4f}  T={t:.0f}s  M={m:.2f}G  R={rp:.2f}")

print(f"\nSaved: {csv_path}")
