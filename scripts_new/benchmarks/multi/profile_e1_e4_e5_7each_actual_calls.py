#!/usr/bin/env python3
import io
import json
import os
import random
import re
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

OUT_ROOT = PROJECT_ROOT / 'output' / 'complexity_e1_e4_e5_7each_actual_calls'
RUN_LOG = OUT_ROOT / 'run.log'

TASK_DIRS = {
    'rain': PROJECT_ROOT / 'dataset/multi/WeatherBench/rain/test/input',
    'haze': PROJECT_ROOT / 'dataset/multi/WeatherBench/haze/test/input',
    'snow': PROJECT_ROOT / 'dataset/multi/WeatherBench/snow/test/input',
}

STEP_TO_TASK = {
    'derain': 'derain',
    'dehaze': 'dehaze',
    'desnow': 'desnow',
}

POOL_SIZE = {
    'derain': len(deraining_toolbox),
    'dehaze': len(dehazing_toolbox),
    'desnow': len(desnowing_toolbox),
}

FIXED_EXPERTS = {
    'derain': 'diffplugin',
    'dehaze': 'dehazeformer',
    'desnow': 'starnet',
}


def _log(msg: str):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with RUN_LOG.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def _pick_one(toolbox, name: str):
    for t in toolbox:
        if getattr(t, 'tool_name', '').lower() == name.lower():
            return t
    raise RuntimeError(f'Expert {name} not found in toolbox {[getattr(x, "tool_name", "") for x in toolbox]}')


def _set_full_experts():
    ra.deraining_toolbox[:] = list(deraining_toolbox)
    ra.dehazing_toolbox[:] = list(dehazing_toolbox)
    ra.desnowing_toolbox[:] = list(desnowing_toolbox)


def _set_fixed_single_experts():
    ra.deraining_toolbox[:] = [_pick_one(deraining_toolbox, FIXED_EXPERTS['derain'])]
    ra.dehazing_toolbox[:] = [_pick_one(dehazing_toolbox, FIXED_EXPERTS['dehaze'])]
    ra.desnowing_toolbox[:] = [_pick_one(desnowing_toolbox, FIXED_EXPERTS['desnow'])]


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


def _actual_expert_calls(stdout_txt: str, *, single_expert: bool):
    steps = re.findall(r'^Step\s+\d+:\s*(derain|dehaze|desnow)\s*$', stdout_txt, flags=re.MULTILINE)
    if single_expert:
        return len(steps), steps
    return sum(POOL_SIZE.get(step, 0) for step in steps), steps


def _configure_common_env():
    os.environ['TASK_PLANNER_MODE'] = 'qwen_only'
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


def _configure_method(method: str):
    _configure_common_env()
    if method == 'E1_full':
        _set_full_experts()
        os.environ['ENABLE_DYNAMIC_REPLAN'] = '1'
        os.environ['ENABLE_LOCAL_REPLAN'] = '1'
        return False
    if method == 'E4_fixed_single':
        _set_fixed_single_experts()
        os.environ['ENABLE_DYNAMIC_REPLAN'] = '1'
        os.environ['ENABLE_LOCAL_REPLAN'] = '1'
        return True
    if method == 'E5_no_replan':
        _set_full_experts()
        os.environ['ENABLE_DYNAMIC_REPLAN'] = '0'
        os.environ['ENABLE_LOCAL_REPLAN'] = '0'
        return False
    raise ValueError(method)


def _run_one(method: str, task: str, inp: Path, sample_out: Path):
    single_expert = _configure_method(method)
    sample_out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    out_buf = io.StringIO()
    err_buf = io.StringIO()
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
    stdout_txt = out_buf.getvalue()
    stderr_txt = err_buf.getvalue()
    actual_calls, executed_steps = _actual_expert_calls(stdout_txt, single_expert=single_expert)
    unique_temp_dirs = len(list(sample_out.glob('temp_*')))
    replan = stdout_txt.count('Local replan triggered')

    (sample_out / 'stdout_full.log').write_text(stdout_txt, encoding='utf-8')
    (sample_out / 'stderr_full.log').write_text(stderr_txt, encoding='utf-8')

    return {
        'method': method,
        'task': task,
        'input': str(inp),
        'sample': inp.name,
        'ok': ok,
        'error': err,
        'output': str(final_out) if final_out else '',
        'latency_sec': dt,
        'peak_mem_mb': peak_mb,
        'expert_calls': actual_calls,
        'unique_temp_dirs': unique_temp_dirs,
        'executed_steps': json.dumps(executed_steps, ensure_ascii=False),
        'num_executed_steps': len(executed_steps),
        'replan_calls': replan,
        'stdout_tail': '\n'.join(stdout_txt.splitlines()[-60:]),
        'stderr_tail': '\n'.join(stderr_txt.splitlines()[-60:]),
    }


def run_profile():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    RUN_LOG.write_text('', encoding='utf-8')

    base_seed = 20260510
    per_task = 7
    samples = []
    for idx, (task, d) in enumerate(TASK_DIRS.items()):
        picked = _sample_inputs(d, k=per_task, seed=base_seed + idx)
        for p in picked:
            samples.append((task, p))

    methods = ['E1_full', 'E4_fixed_single', 'E5_no_replan']
    _log('=== CONFIG: E1/E4/E5, 7 samples each task, actual expert calls ===')
    _log(f'POOL_SIZE={POOL_SIZE}')
    _log(f'FIXED_EXPERTS={FIXED_EXPERTS}')
    _log(f'Samples({len(samples)}): {[f"{t}:{p.name}" for t, p in samples]}')

    rows = []
    total = len(methods) * len(samples)
    done = 0
    for method in methods:
        _log(f'===== START {method} =====')
        for task, inp in samples:
            done += 1
            sample_out = OUT_ROOT / method / task / inp.stem
            _log(f'[{done}/{total}] START {method} {task}:{inp.name}')
            row = _run_one(method, task, inp, sample_out)
            rows.append(row)
            if row['ok']:
                _log(
                    f'[{done}/{total}] DONE {method} {task}:{inp.name} '
                    f't={row["latency_sec"]:.1f}s peak={row["peak_mem_mb"]:.0f}MB '
                    f'expert_calls={row["expert_calls"]} steps={row["num_executed_steps"]} replan={row["replan_calls"]}'
                )
            else:
                _log(f'[{done}/{total}] FAIL {method} {task}:{inp.name} t={row["latency_sec"]:.1f}s err={row["error"]}')

    per_fp = OUT_ROOT / 'per_image_actual_calls.json'
    per_fp.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')

    summary_rows = []
    for method in methods:
        rs = [r for r in rows if r['method'] == method and r.get('ok') == 1]
        lat = [float(r['latency_sec']) for r in rs]
        mem = [float(r['peak_mem_mb']) for r in rs]
        ex = [float(r['expert_calls']) for r in rs]
        rp = [float(r['replan_calls']) for r in rs]
        steps = [float(r['num_executed_steps']) for r in rs]
        summary_rows.append({
            'method': method,
            'num_images': len([r for r in rows if r['method'] == method]),
            'num_success': len(rs),
            'latency_mean_sec': statistics.mean(lat) if lat else 0.0,
            'latency_std_sec': statistics.stdev(lat) if len(lat) > 1 else 0.0,
            'latency_p95_sec': _p95(lat),
            'avg_peak_mem_gb': (statistics.mean(mem) / 1024.0) if mem else 0.0,
            'max_peak_mem_gb': (max(mem) / 1024.0) if mem else 0.0,
            'expert_calls_per_img_mean': statistics.mean(ex) if ex else 0.0,
            'replan_per_img_mean': statistics.mean(rp) if rp else 0.0,
            'executed_steps_per_img_mean': statistics.mean(steps) if steps else 0.0,
        })

    out = {
        'config': {
            'per_task': per_task,
            'seed_base': base_seed,
            'pool_size': POOL_SIZE,
            'fixed_experts': FIXED_EXPERTS,
            'expert_calls_definition': 'Sum over all executed restoration steps from pipeline start to final output. Multi-expert step counts all experts in the task toolbox; fixed-single step counts one expert. Repeated steps after replan are included.',
        },
        'summary': summary_rows,
        'rows': rows,
    }
    summary_fp = OUT_ROOT / 'profile_7each_actual_calls_summary.json'
    summary_fp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    _log('=== FINISHED ===')
    _log(json.dumps(summary_rows, ensure_ascii=False))
    _log(f'saved_json={summary_fp}')


if __name__ == '__main__':
    run_profile()
