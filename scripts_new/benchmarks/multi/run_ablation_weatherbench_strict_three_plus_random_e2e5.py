#!/usr/bin/env python3
import csv, json, os, random, shutil, sys, time
from pathlib import Path
from statistics import mean

import torch
from PIL import Image
import torchvision.transforms as transforms
import pyiqa

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perception_module import predict_degradation
from task_planner import TaskPlanner
from restoration_agent import RestorationAgent
from quality_evaluator import QualityEvaluator
from utils_new.deraining import deraining_toolbox
from utils_new.dehazing import dehazing_toolbox
from utils_new.desnowing import desnowing_toolbox

DATA = ROOT / 'dataset/multi/WeatherBench'
OUT = ROOT / 'output/ablation_weatherbench_strict_three_plus_random_e2e5'
TASKS = ['rain', 'haze', 'snow']
LABEL_TO_STEP = {'rain': 'derain', 'haze': 'dehaze', 'snow': 'desnow'}
TOOLBOX = {'derain': deraining_toolbox, 'dehaze': dehazing_toolbox, 'desnow': desnowing_toolbox}
FIXED = {'rain': '323.jpg', 'haze': '097.jpg', 'snow': '563.jpg'}
SEED = 20260510

EXPS = [
    ('E2_planner_only_no_perception', 'e2'),
    ('E3_perception_only_no_planning', 'e3'),
    ('E4_random_single_expert', 'e4'),
    ('E5_no_replan_otherwise_normal', 'e5'),
]


def set_base_env():
    os.environ.setdefault('QE_STRICT_OFFLINE', '1')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
    os.environ.setdefault('WEATHER_PERCEPTION_SUBPROCESS', '1')
    os.environ.setdefault('WEATHER_PERCEPTION_TOPK_GPUS', '2')
    os.environ.setdefault('WEATHER_PERCEPTION_DEVICE_MAP', 'auto')
    os.environ.setdefault('WEATHER_PERCEPTION_RELEASE_AFTER_INFER', '1')
    os.environ.setdefault('TASK_PLANNER_ISOLATED_ENV', 'weather_agent_planner')
    os.environ.setdefault('EXPERT_MIN_FREE_MB', '6000')
    os.environ.setdefault('EXPERT_GPU_WAIT_SECONDS', '60')
    os.environ.setdefault('EXPERT_GPU_POLL_SECONDS', '2')
    os.environ['ALLOW_INPUT_AS_CANDIDATE'] = '0'
    os.environ['KEEP_ALL_INTERMEDIATES'] = '1'
    os.environ['TASK_PLANNER_MODE'] = 'qwen_only'
    os.environ['ENABLE_LOCAL_REPLAN'] = '1'
    os.environ['LOCAL_REPLAN_MAX'] = '3'
    os.environ['ENABLE_STEP_SCORE_GUARD'] = '1'
    os.environ['PREFER_EXPERT_WHEN_CLOSE'] = '1'


def is_img(p):
    return p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def choose_samples():
    selected = {}
    for task in TASKS:
        inp_dir = DATA / task / 'test/input'
        gt_dir = DATA / task / 'test/target'
        names = sorted(p.name for p in inp_dir.iterdir() if p.is_file() and is_img(p) and (gt_dir / p.name).exists())
        fixed = FIXED[task]
        pool = [n for n in names if n != fixed]
        rng = random.Random(SEED + sum(ord(c) for c in task))
        selected[task] = [fixed] + sorted(rng.sample(pool, 2))
    return selected


def init_eval(device):
    to_tensor = transforms.ToTensor()
    psnr = pyiqa.create_metric('psnr', device=device)
    ssim = pyiqa.create_metric('ssim', device=device)
    def eval_pair(pred_path, gt_path):
        pred = Image.open(pred_path).convert('RGB')
        gt = Image.open(gt_path).convert('RGB')
        if pred.size != gt.size:
            pred = pred.resize(gt.size, Image.BICUBIC)
        t_pred = to_tensor(pred).unsqueeze(0).to(device)
        t_gt = to_tensor(gt).unsqueeze(0).to(device)
        with torch.no_grad():
            return float(psnr(t_pred, t_gt).item()), float(ssim(t_pred, t_gt).item())
    return eval_pair


def qwen_plan_without_perception(image_path):
    planner = TaskPlanner()
    explicit = {
        'C_I': 'Perception module is removed in E2. Generate restoration sequence from image and candidate task set.',
        'D_I': [],
        'A_I': ['desnow', 'derain', 'dehaze'],
        'I': str(image_path),
    }
    try:
        res = planner._plan_via_isolated_env(str(image_path), explicit, ['desnow', 'derain', 'dehaze'])
        plan = [x for x in res.get('plan', []) if x in {'desnow', 'derain', 'dehaze'}]
        return plan or ['derain']
    except Exception:
        return ['derain']


def perception_plan_without_planning(image_path):
    deg = predict_degradation(str(image_path))
    planner = TaskPlanner()
    plan = planner.direct_plan(deg)
    return [x for x in plan if x in {'desnow', 'derain', 'dehaze'}] or [LABEL_TO_STEP.get(image_path.parent.parent.name, 'derain')]


def normal_plan(image_path):
    deg = predict_degradation(str(image_path))
    planner = TaskPlanner()
    plan = planner.plan(deg, image_path=str(image_path))
    return [x for x in plan if x in {'desnow', 'derain', 'dehaze'}] or [LABEL_TO_STEP.get(image_path.parent.parent.name, 'derain')]


def execute_multi(plan, inp, sample_dir, no_replan=False):
    set_base_env()
    os.environ['ENABLE_LOCAL_REPLAN'] = '0' if no_replan else '1'
    if no_replan:
        os.environ['LOCAL_REPLAN_MAX'] = '0'
    agent = RestorationAgent()
    return Path(agent.execute_plan(plan, str(inp), str(sample_dir)))


def execute_random_single(task, inp, sample_dir, seed):
    set_base_env()
    step = LABEL_TO_STEP[task]
    tools = [t for t in TOOLBOX[step] if t.work_dir is not None and t.work_dir.exists() and t.script_path is not None and t.script_path.exists()]
    if not tools:
        raise RuntimeError(f'No available tools for {step}')
    tool = random.Random(seed).choice(tools)
    td = sample_dir / f'temp_{step}_{tool.tool_name}'
    if td.exists():
        shutil.rmtree(td)
    inp_d, out_d = td / 'input', td / 'output'
    inp_d.mkdir(parents=True, exist_ok=True)
    out_d.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(inp), inp_d / 'input.png')
    tool(input_dir=inp_d, output_dir=out_d, silent=True, run_gpu_id=None)
    cand = out_d / 'output.png'
    if not cand.exists():
        raise RuntimeError(f'{tool.tool_name} produced no output.png')
    final = sample_dir / 'final_output.png'
    shutil.copy(cand, final)
    (sample_dir / 'chosen_expert.json').write_text(json.dumps({'step': step, 'expert': tool.tool_name}, indent=2), encoding='utf-8')
    return final


def run_exp(exp_id, kind, samples, eval_pair):
    exp_dir = OUT / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    rows, fails = [], []
    t0 = time.time()
    for task in TASKS:
        for name in samples[task]:
            inp = DATA / task / 'test/input' / name
            gt = DATA / task / 'test/target' / name
            sd = exp_dir / task / Path(name).stem
            sd.mkdir(parents=True, exist_ok=True)
            final = sd / 'final_output.png'
            try:
                if not final.exists():
                    if kind == 'e2':
                        plan = qwen_plan_without_perception(inp)
                        final = execute_multi(plan, inp, sd, no_replan=True)
                    elif kind == 'e3':
                        plan = perception_plan_without_planning(inp)
                        final = execute_multi(plan, inp, sd, no_replan=False)
                    elif kind == 'e4':
                        plan = [LABEL_TO_STEP[task]]
                        final = execute_random_single(task, inp, sd, SEED + sum(ord(c) for c in task))
                    else:
                        plan = normal_plan(inp)
                        final = execute_multi(plan, inp, sd, no_replan=True)
                    (sd / 'plan.json').write_text(json.dumps({'plan': plan}, ensure_ascii=False, indent=2), encoding='utf-8')
                ps, ss = eval_pair(final, gt)
                rows.append({'experiment': exp_id, 'task': task, 'sample': name, 'output_path': str(final), 'PSNR': ps, 'SSIM': ss})
                print(f'[{exp_id}] {task}/{name} PSNR={ps:.3f} SSIM={ss:.4f}', flush=True)
            except Exception as e:
                fails.append({'experiment': exp_id, 'task': task, 'sample': name, 'reason': str(e)})
                print(f'[{exp_id}] {task}/{name} FAIL {e}', flush=True)
    with (exp_dir / 'per_image_metrics.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['experiment', 'task', 'sample', 'output_path', 'PSNR', 'SSIM'])
        w.writeheader(); w.writerows(rows)
    with (exp_dir / 'failed_samples.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['experiment', 'task', 'sample', 'reason'])
        w.writeheader(); w.writerows(fails)
    task_sum = {}
    for task in TASKS:
        tr = [r for r in rows if r['task'] == task]
        task_sum[task] = {'num_success': len(tr), 'num_failed': len([x for x in fails if x['task'] == task]), 'psnr_mean': mean([float(r['PSNR']) for r in tr]) if tr else 0.0, 'ssim_mean': mean([float(r['SSIM']) for r in tr]) if tr else 0.0}
    summary = {'experiment': exp_id, 'kind': kind, 'samples': samples, 'num_success': len(rows), 'num_failed': len(fails), 'elapsed_sec': time.time() - t0, 'tasks': task_sum, 'overall': {'psnr_mean': mean([float(r['PSNR']) for r in rows]) if rows else 0.0, 'ssim_mean': mean([float(r['SSIM']) for r in rows]) if rows else 0.0}}
    (exp_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


def main():
    set_base_env()
    OUT.mkdir(parents=True, exist_ok=True)
    samples = choose_samples()
    (OUT / 'selected_samples.json').write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding='utf-8')
    eval_pair = init_eval('cuda' if torch.cuda.is_available() else 'cpu')
    rows = []
    for exp_id, kind in EXPS:
        s = run_exp(exp_id, kind, samples, eval_pair)
        rows.append({'experiment': exp_id, 'num_success': s['num_success'], 'num_failed': s['num_failed'], 'overall_psnr_mean': s['overall']['psnr_mean'], 'overall_ssim_mean': s['overall']['ssim_mean'], 'rain_psnr': s['tasks']['rain']['psnr_mean'], 'rain_ssim': s['tasks']['rain']['ssim_mean'], 'haze_psnr': s['tasks']['haze']['psnr_mean'], 'haze_ssim': s['tasks']['haze']['ssim_mean'], 'snow_psnr': s['tasks']['snow']['psnr_mean'], 'snow_ssim': s['tasks']['snow']['ssim_mean'], 'elapsed_sec': s['elapsed_sec']})
    with (OUT / 'ablation_summary.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    (OUT / 'status.txt').write_text('DONE\n', encoding='utf-8')
    print(f'DONE {OUT / "ablation_summary.csv"}', flush=True)

if __name__ == '__main__':
    main()
