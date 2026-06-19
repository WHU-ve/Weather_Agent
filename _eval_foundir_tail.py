"""Run FoundIR 11Raindrop + 12NightRain only."""
import os, subprocess, threading, time, random
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

def psnr_ssim(out, gt):
    oi=Image.open(out).convert('RGB'); gi=Image.open(gt).convert('RGB')
    if oi.size!=gi.size: oi=oi.resize(gi.size,Image.BICUBIC)
    oa,ga=np.asarray(oi),np.asarray(gi)
    return (float(peak_signal_noise_ratio(ga,oa,data_range=255)),
            float(structural_similarity(ga,oa,channel_axis=2,data_range=255)))

def sample_images(img_dir, gt_dir, n, seed):
    exts={'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}
    imgs=sorted([x for x in Path(img_dir).iterdir() if x.suffix.lower() in exts])
    imgs=[x for x in imgs if (Path(gt_dir)/x.name).exists()]
    if n<=0 or len(imgs)<=n: return imgs
    rng=random.Random(seed)
    return sorted(rng.sample(imgs,n), key=lambda x:x.name)

def run_one(py, main_py, inp, out_dir, env, gpu_ids):
    e=dict(os.environ)
    e.update(dict(QE_STRICT_OFFLINE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        HF_DATASETS_OFFLINE='1',WEATHER_UTILS_DIR='utils_new',WEATHER_CKPT_DIR='pretrained_ckpts_new',
        WEATHER_PERCEPTION_SUBPROCESS='1',WEATHER_PERCEPTION_TOPK_GPUS='2',
        WEATHER_PERCEPTION_DEVICE_MAP='auto',WEATHER_PERCEPTION_RELEASE_AFTER_INFER='1'))
    if gpu_ids: e['CUDA_VISIBLE_DEVICES']=','.join(str(i) for i in gpu_ids)
    e.update(env)
    out_dir.mkdir(parents=True,exist_ok=True)
    cmd=[py,str(main_py),'--input',str(inp),'--output',str(out_dir),'--planner_mode','qwen_only','--run_restoration']
    mon=GPUMonitor(interval=0.2,gpu_ids=gpu_ids)
    t0=time.time(); mon.start()
    try:
        p=subprocess.run(cmd,capture_output=True,text=True,env=e,cwd=str(main_py.parent),timeout=1200)
        rc=p.returncode; stdout=p.stdout or ''; to=0
    except subprocess.TimeoutExpired:
        to=1; rc=-9; stdout=''
    finally: mon.stop()
    dt=time.time()-t0
    ok=int(to==0 and rc==0 and (out_dir/'final_output.png').exists())
    return {'ok':ok,'latency_sec':dt,'peak_gpu_mem_mb':mon.peak,'replan_calls':stdout.count('Local replan triggered')}

project=Path(__file__).resolve().parent
main_py=project/'main.py'
py='/root/project/huangchao/anaconda3/envs/weather_agent/bin/python'
gids=[0,1,2,3,4,5]
out_dir=project/'output'/'eval_full_pipeline_50each'
fir_root=project/'dataset'/'FoundIR-Weather'
env={'TASK_PLANNER_MODE':'qwen_only','ENABLE_LOCAL_REPLAN':'1','LOCAL_REPLAN_MAX':'3','KEEP_ALL_INTERMEDIATES':'1'}

tasks=[
    ('FoundIR-11Raindrop', fir_root/'LQ'/'11Raindrop', fir_root/'GT'/'11Raindrop', 'derain'),
    ('FoundIR-12NightRain', fir_root/'LQ'/'12NightRain', fir_root/'GT'/'12NightRain', 'derain'),
]

for task_label, img_dir, gt_dir, pipeline_task in tasks:
    print(f"\n{'='*60}\n=== {task_label} ===\n{'='*60}")
    imgs=sample_images(img_dir, gt_dir, 50, 2026+hash(task_label))
    print(f'{len(imgs)} images')
    results=[]
    for i, inp in enumerate(imgs):
        sample_out=out_dir/task_label.replace(' ','_')/inp.stem
        r=run_one(py, main_py, inp, sample_out, env, gids)
        ps,ss=-1.0,-1.0
        final_png=sample_out/'final_output.png'
        gt_png=Path(gt_dir)/inp.name
        if r['ok'] and final_png.exists() and gt_png.exists():
            try: ps,ss=psnr_ssim(final_png,gt_png)
            except: pass
        results.append({'psnr':ps,'ssim':ss,**r})
        print(f"  [{i+1}/{len(imgs)}] {inp.stem}  PSNR={ps:.1f}  t={r['latency_sec']:.0f}s  M={r['peak_gpu_mem_mb']/1024:.1f}G  R={r['replan_calls']}",flush=True)
    ok=[r for r in results if r['ok'] and r['psnr']>0]
    if ok:
        p=mean(r['psnr'] for r in ok); s=mean(r['ssim'] for r in ok)
        print(f"\n  {task_label}: n={len(ok)}  PSNR={p:.2f}  SSIM={s:.4f}")
