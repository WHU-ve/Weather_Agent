#!/usr/bin/env python3
import io
import json
import os
import random
import statistics
import subprocess
import sys
import threading
import time
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path('/root/project/huangchao/zhengyanggong/weather_agent').resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import main as run_pipeline
import restoration_agent as ra
from utils_new.deraining import deraining_toolbox
from utils_new.dehazing import dehazing_toolbox
from utils_new.desnowing import desnowing_toolbox

OUT_ROOT = PROJECT_ROOT / 'output' / 'tmp_profile_12samples_singleexpert_dynamic'
RUN_LOG = OUT_ROOT / 'run.log'


def _log(msg: str):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with RUN_LOG.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def _pick_one(toolbox, name: str):
    for t in toolbox:
        if getattr(t, 'tool_name', '').lower() == name.lower():
            return t
    raise RuntimeError(f'Expert {name} not found in toolbox {[getattr(x, "tool_name", "") for x in toolbox]}')


def _patch_single_experts():
    ra.deraining_toolbox[:] = [_pick_one(deraining_toolbox, 'diffplugin')]
    ra.dehazing_toolbox[:] = [_pick_one(dehazing_toolbox, 'dehazeformer')]
    ra.desnowing_toolbox[:] = [_pick_one(desnowing_toolbox, 'starnet')]


def _gpu_peak_mb_during(func):
    stop = {'v': False}
    peak = {'v': 0.0}

    def query_once():
        try:
            out = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            vals = [float(x.strip()) for x in out.strip().splitlines() if x.strip()]
            if vals:
                peak['v'] = max(peak['v'], max(vals))
        except Exception:
            pass

    def loop():
        while not stop['v']:
            query_once()
            time.sleep(0.2)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    try:
        return func(), peak['v']
    finally:
        stop['v'] = True
        t.join(timeout=1.0)
        query_once()


def _p95(xs):
    if not xs:
        return 0.0
    ys = sorted(xs)
    i = round((len(ys) - 1) * 0.95)
    return ys[max(0, min(i, len(ys) - 1))]


def _sample_inputs(input_dir: Path, k: int, seed: int):
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    imgs = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])
    rng = random.Random(seed)
    if len(imgs) <= k:
        return imgs
    return sorted(rng.sample(imgs, k), key=lambda x: x.name)


def run_profile():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    RUN_LOG.write_text('', encoding='utf-8')

    task_dirs = {
        'rain': PROJECT_ROOT / 'dataset/multi/WeatherBench/rain/test/input',
        'haze': PROJECT_ROOT / 'dataset/multi/WeatherBench/haze/test/input',
        'snow': PROJECT_ROOT / 'dataset/multi/WeatherBench/snow/test/input',
    }
    base_seed = 20260503
    per_task = 4

    samples = []
    for idx, (task, d) in enumerate(task_dirs.items()):
        picked = _sample_inputs(d, k=per_task, seed=base_seed + idx)
        for p in picked:
            samples.append((task, p))

    _log('=== CONFIG ===')
    _log('Pipeline: perception + planning + dynamic/local replan + single fixed expert')
    _log('Fixed experts: derain=diffplugin, dehaze=dehazeformer, desnow=starnet')
    _log(f'Samples({len(samples)}): {[f"{t}:{p.name}" for t, p in samples]}')

    os.environ['TASK_PLANNER_MODE'] = 'qwen_only'
    os.environ['ENABLE_DYNAMIC_REPLAN'] = '1'
    os.environ['ENABLE_LOCAL_REPLAN'] = '1'
    os.environ['KEEP_ALL_INTERMEDIATES'] = '1'
    os.environ['ALLOW_INPUT_AS_CANDIDATE'] = '0'

    os.environ.setdefault('EXPERT_PARALLEL_GPU_IDS', '')
    os.environ.setdefault('EXPERT_PARALLEL_WORKERS', '6')
    os.environ.setdefault('EXPERT_MIN_FREE_MB', '3000')
    os.environ.setdefault('EXPERT_GPU_WAIT_SECONDS', '30')
    os.environ.setdefault('EXPERT_GPU_POLL_SECONDS', '2')

    os.environ.setdefault('WEATHER_PERCEPTION_DEVICE_MAP', 'auto')
    os.environ.setdefault('WEATHER_PERCEPTION_TOPK_GPUS', '2')
    os.environ.setdefault('WEATHER_PERCEPTION_SUBPROCESS', '1')
    os.environ.setdefault('WEATHER_PERCEPTION_RELEASE_AFTER_INFER', '1')

    os.environ.setdefault('QE_STRICT_OFFLINE', '1')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
    os.environ.setdefault('WEATHER_UTILS_DIR', 'utils_new')
    os.environ.setdefault('WEATHER_CKPT_DIR', 'pretrained_ckpts_new')

    _patch_single_experts()

    rows = []
    for idx, (task, inp) in enumerate(samples, 1):
        row = {'task': task, 'input': str(inp)}
        sample_out = OUT_ROOT / task / inp.stem
        sample_out.mkdir(parents=True, exist_ok=True)

        _log(f'[{idx}/{len(samples)}] START {task}:{inp.name}')
        t0 = time.time()
        out_buf = io.StringIO()
        err_buf = io.StringIO()

        if not inp.exists():
            row.update({'ok': 0, 'error': 'input_not_found'})
            rows.append(row)
            _log(f'[{idx}/{len(samples)}] FAIL {task}:{inp.name} input_not_found')
            continue

        try:
            def _run():
                with redirect_stdout(out_buf), redirect_stderr(err_buf):
                    run_pipeline(str(inp), str(sample_out), planner_mode='qwen_only', run_restoration=True)

            _, peak_mb = _gpu_peak_mb_during(_run)
            final_out = sample_out / 'final_output.png'
            ok = int(final_out.exists())
            err = ''
        except Exception as e:
            peak_mb = 0.0
            ok = 0
            err = str(e)
            final_out = ''

        dt = time.time() - t0
        experts = len(list(sample_out.glob('temp_*')))
        stdout_txt = out_buf.getvalue()
        stderr_txt = err_buf.getvalue()
        replan = stdout_txt.count('Local replan triggered')

        row.update({
            'ok': ok,
            'error': err,
            'output': str(final_out) if final_out else '',
            'latency_sec': dt,
            'peak_mem_mb': peak_mb,
            'expert_calls': experts,
            'replan_calls': replan,
            'stdout_tail': '\n'.join(stdout_txt.splitlines()[-40:]),
            'stderr_tail': '\n'.join(stderr_txt.splitlines()[-40:]),
        })
        rows.append(row)

        if ok:
            _log(f'[{idx}/{len(samples)}] DONE {task}:{inp.name} t={dt:.1f}s peak={peak_mb:.0f}MB experts={experts} replan={replan}')
        else:
            _log(f'[{idx}/{len(samples)}] FAIL {task}:{inp.name} t={dt:.1f}s err={err}')

    succ = [r for r in rows if r.get('ok') == 1]
    lat = [float(r['latency_sec']) for r in succ]
    mem = [float(r['peak_mem_mb']) for r in succ]
    ex = [float(r['expert_calls']) for r in succ]
    rp = [float(r['replan_calls']) for r in succ]

    summary = {
        'avg_peak_mem_gb': (statistics.mean(mem) / 1024.0) if mem else 0.0,
        'max_peak_mem_gb': (max(mem) / 1024.0) if mem else 0.0,
        'experts_per_img': statistics.mean(ex) if ex else 0.0,
        'replan_per_img': statistics.mean(rp) if rp else 0.0,
        'lat_mean_sec': statistics.mean(lat) if lat else 0.0,
        'lat_std_sec': statistics.stdev(lat) if len(lat) > 1 else 0.0,
        'lat_p95_sec': _p95(lat),
        'num_success': len(succ),
        'num_total': len(rows),
        'fixed_experts': {'derain': 'diffplugin', 'dehaze': 'dehazeformer', 'desnow': 'starnet'},
        'sampling': {'per_task': per_task, 'seed_base': base_seed},
    }

    out = {'summary': summary, 'rows': rows}
    out_fp = OUT_ROOT / 'profile_12samples_summary.json'
    out_fp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    _log('=== FINISHED ===')
    _log(json.dumps(summary, ensure_ascii=False))
    _log(f'saved_json={out_fp}')


if __name__ == '__main__':
    run_profile()
