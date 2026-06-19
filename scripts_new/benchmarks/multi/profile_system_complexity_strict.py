#!/usr/bin/env python3
import argparse, csv, os, subprocess, threading, time, random
from pathlib import Path
from statistics import mean, stdev

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

def pct(xs,q):
    if not xs:return 0.0
    ys=sorted(xs); i=round((len(ys)-1)*q); return ys[int(max(0,min(i,len(ys)-1)))]

def list_inputs(root,n,sample_mode='random',sample_seed=2026):
    exts={'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}
    p=Path(root)/'rain'/'test'/'input'
    imgs=sorted([x for x in p.iterdir() if x.suffix.lower() in exts])
    if n <= 0:
        chosen = imgs
    elif sample_mode == 'head':
        chosen = imgs[:n]
    else:
        rng=random.Random(sample_seed)
        chosen = sorted(rng.sample(imgs, min(n, len(imgs))), key=lambda x:x.name)
    return {'rain': chosen}

def methods(selected=None):
    all_methods = [
        ('E1_full',{'TASK_PLANNER_MODE':'qwen_only','ENABLE_DYNAMIC_REPLAN':'1','ENABLE_LOCAL_REPLAN':'1','KEEP_ALL_INTERMEDIATES':'1'}),
        ('E4_random_single_no_replan',{'TASK_PLANNER_MODE':'qwen_only','ENABLE_DYNAMIC_REPLAN':'0','ENABLE_LOCAL_REPLAN':'0','RANDOM_SINGLE_EXPERT':'1','RANDOM_SINGLE_EXPERT_SEED':'2026','KEEP_ALL_INTERMEDIATES':'1'}),
        ('E5_no_replan',{'TASK_PLANNER_MODE':'qwen_only','ENABLE_DYNAMIC_REPLAN':'0','ENABLE_LOCAL_REPLAN':'0','KEEP_ALL_INTERMEDIATES':'1'}),
    ]
    if not selected:
        return all_methods
    selected_set = {x.strip() for x in selected.split(',') if x.strip()}
    return [m for m in all_methods if m[0] in selected_set]

def run_one(py, main_py, inp, out_dir, extra_env, poll=0.1, gpu_ids=None, timeout_sec=1800):
    env=dict(os.environ)
    env.update({'QE_STRICT_OFFLINE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','WEATHER_UTILS_DIR':'utils_new','WEATHER_CKPT_DIR':'pretrained_ckpts_new','USE_FLAX':'0','TRANSFORMERS_NO_FLAX':'1','WEATHER_PERCEPTION_SUBPROCESS':'1','WEATHER_PERCEPTION_TOPK_GPUS':'2','WEATHER_PERCEPTION_DEVICE_MAP':'auto','WEATHER_PERCEPTION_RELEASE_AFTER_INFER':'1'})
    env.update(extra_env)
    out_dir.mkdir(parents=True,exist_ok=True)
    cmd=[py,str(main_py),'--input',str(inp),'--output',str(out_dir),'--planner_mode','qwen_only','--run_restoration']
    mon=GPUMonitor(interval=poll,gpu_ids=gpu_ids)

    t0=time.time()
    mon.start()
    timed_out=0
    stdout=''
    stderr=''
    try:
        p=subprocess.run(cmd,capture_output=True,text=True,env=env,cwd=str(main_py.parent),timeout=timeout_sec)
        return_code=p.returncode
        stdout=p.stdout or ''
        stderr=p.stderr or ''
    except subprocess.TimeoutExpired:
        timed_out=1
        return_code=-9
    finally:
        mon.stop()
    dt=time.time()-t0

    has_final=int((out_dir/'final_output.png').exists())
    ok=int(timed_out==0 and return_code==0 and has_final==1)
    completed=int(timed_out==0)
    experts=len(list(out_dir.glob('temp_*')))
    replan=stdout.count('Local replan triggered')
    return {
        'ok':ok,
        'completed':completed,
        'has_final_output':has_final,
        'timed_out':timed_out,
        'return_code':return_code,
        'latency_sec':dt,
        'peak_gpu_mem_mb':mon.peak,
        'expert_calls':experts,
        'replan_calls':replan,
        'stdout_tail':'\\n'.join(stdout.splitlines()[-8:]),
        'stderr_tail':'\\n'.join(stderr.splitlines()[-8:]),
    }

def read_quality_rows(project_root):
    cand=[Path(project_root)/'output'/'ablation_weatherbench_fullchain'/'ablation_summary.csv',Path(project_root)/'output'/'ablation_weatherbench_isolated_e2e4'/'ablation_summary.csv']
    rows=[]
    for p in cand:
        if p.exists():
            with p.open('r',encoding='utf-8') as f: rows.extend(list(csv.DictReader(f)))
    return rows

def merge_quality(project_root, summary_rows, out_dir):
    qrows=read_quality_rows(project_root)
    m={'E1_full':['E1_perception_direct_quality'],'E4_random_single_no_replan':['E4_random_single_expert_no_replan'],'E5_no_replan':['E5_qwen_only_quality_no_replan']}
    qmap={}
    for method,ids in m.items():
        for r in qrows:
            if r.get('experiment','') in ids:
                psnr=r.get('overall_psnr_mean') or r.get('psnr_mean') or ''
                ssim=r.get('overall_ssim_mean') or r.get('ssim_mean') or ''
                if psnr!='' and ssim!='': qmap[method]={'PSNR':float(psnr),'SSIM':float(ssim)}; break
    final=[]
    for r in summary_rows:
        q=qmap.get(r['method'],{'PSNR':'','SSIM':''})
        final.append({'method':r['method'],'PSNR':q['PSNR'],'SSIM':q['SSIM'],'E2E_Latency_mean_sec':r['latency_mean_sec'],'E2E_Latency_std_sec':r['latency_std_sec'],'E2E_Latency_p95_sec':r['latency_p95_sec'],'PeakMem_mean_gb':r['peak_mem_mean_gb'],'PeakMem_max_gb':r['peak_mem_max_gb'],'Experts_per_img':r['experts_per_img_mean'],'Replan_per_img':r['replan_per_img_mean'],'num_success':r['num_success']})
    fp=Path(out_dir)/'final_table.csv'
    with fp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(final[0].keys())); w.writeheader(); w.writerows(final)
    return fp

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--project_root',default='/root/project/huangchao/zhengyanggong/weather_agent')
    ap.add_argument('--dataset_root',default='/root/project/huangchao/zhengyanggong/weather_agent/dataset/multi/WeatherBench')
    ap.add_argument('--samples_per_task',type=int,default=20)
    ap.add_argument('--sample_mode',choices=['random','head'],default='random')
    ap.add_argument('--sample_seed',type=int,default=2026)
    ap.add_argument('--repeats',type=int,default=3)
    ap.add_argument('--warmup_per_method',type=int,default=2)
    ap.add_argument('--poll_interval_sec',type=float,default=0.1)
    ap.add_argument('--python_bin',default='/root/project/huangchao/anaconda3/envs/weather_agent/bin/python')
    ap.add_argument('--gpu_indices',default='')
    ap.add_argument('--timeout_sec',type=int,default=1200)
    ap.add_argument('--output_dir',default='/root/project/huangchao/zhengyanggong/weather_agent/output/complexity_strict_5_timeout')
    ap.add_argument('--methods',default='E1_full,E4_random_single_no_replan,E5_no_replan')
    a=ap.parse_args()

    project=Path(a.project_root); main_py=project/'main.py'; out_dir=Path(a.output_dir); out_dir.mkdir(parents=True,exist_ok=True)
    gids=[int(x.strip()) for x in a.gpu_indices.split(',') if x.strip()] if a.gpu_indices.strip() else None
    inputs=list_inputs(a.dataset_root,a.samples_per_task,a.sample_mode,a.sample_seed)
    per=[]
    selected_methods = methods(a.methods)

    for mname,menv in selected_methods:
        print(f'===== {mname} =====')
        for rep in range(a.repeats):
            warm=[]
            for t in ['rain']: warm.extend(inputs[t][:1])
            for inp in warm[:a.warmup_per_method]:
                run_one(a.python_bin,main_py,inp,out_dir/'_warmup'/mname/f'rep{rep}'/inp.stem,menv,a.poll_interval_sec,gids,a.timeout_sec)
            for t in ['rain']:
                for inp in inputs[t]:
                    r=run_one(a.python_bin,main_py,inp,out_dir/mname/f'rep{rep}'/t/inp.stem,menv,a.poll_interval_sec,gids,a.timeout_sec)
                    if r['ok'] == 0:
                        fail_fp = out_dir / mname / f'rep{rep}' / t / inp.stem / 'failure.txt'
                        fail_fp.write_text(
                            f"return_code={r['return_code']} timed_out={r['timed_out']} has_final_output={r['has_final_output']} latency_sec={r['latency_sec']:.3f} replan_calls={r['replan_calls']}\n"
                            f"stdout_tail:\n{r['stdout_tail']}\n\n"
                            f"stderr_tail:\n{r['stderr_tail']}\n",
                            encoding='utf-8'
                        )
                    per.append({'method':mname,'repeat':rep,'task':t,'sample':inp.name,**r})
                    print(f"[{mname}][rep{rep}][{t}] {inp.name} ok={r['ok']} timeout={r['timed_out']} t={r['latency_sec']:.3f}s peak={r['peak_gpu_mem_mb']:.0f}MB", flush=True)

    per_csv=out_dir/'per_image.csv'
    with per_csv.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(per[0].keys())); w.writeheader(); w.writerows(per)

    summ=[]
    for mname,_ in selected_methods:
        all_rows=[r for r in per if r['method']==mname]
        rs=[r for r in all_rows if r.get('completed',0)==1]
        succ=[r for r in all_rows if r.get('ok',0)==1]
        lat=[float(r['latency_sec']) for r in rs]
        mem=[float(r['peak_gpu_mem_mb']) for r in rs]
        ex=[float(r['expert_calls']) for r in rs]
        rp=[float(r['replan_calls']) for r in rs]
        summ.append({'method':mname,'num_success':len(succ),'num_completed':len(rs),'num_timeout':sum(int(r.get('timed_out',0)) for r in all_rows),'latency_mean_sec':mean(lat) if lat else 0.0,'latency_std_sec':stdev(lat) if len(lat)>1 else 0.0,'latency_p95_sec':pct(lat,0.95) if lat else 0.0,'peak_mem_mean_gb':(mean(mem)/1024.0) if mem else 0.0,'peak_mem_max_gb':(max(mem)/1024.0) if mem else 0.0,'experts_per_img_mean':mean(ex) if ex else 0.0,'replan_per_img_mean':mean(rp) if rp else 0.0})

    sum_csv=out_dir/'summary.csv'
    with sum_csv.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(summ[0].keys())); w.writeheader(); w.writerows(summ)

    final_csv=merge_quality(project,summ,out_dir)
    print(f'Saved:\n- {per_csv}\n- {sum_csv}\n- {final_csv}')

if __name__=='__main__':
    main()
