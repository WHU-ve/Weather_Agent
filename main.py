from perception_module import (
    predict_degradation,
    release_model as release_perception_model,
)
from task_planner import TaskPlanner
from restoration_agent import RestorationAgent
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def _apply_default_runtime_settings(planner_mode=None):
    # Keep perception and planning defaults consistent with the project baseline.
    os.environ.setdefault('WEATHER_PERCEPTION_DEVICE_MAP', 'auto')
    os.environ.setdefault('WEATHER_PERCEPTION_TORCH_DTYPE', 'fp16')
    os.environ.setdefault('TASK_PLANNER_ISOLATED_ENV', 'weather_agent_planner')
    # Project default: always-on Qwen planner (no confidence-gated fallback path).
    os.environ.setdefault('TASK_PLANNER_MODE', 'qwen_only')
    # Mitigate known RTX4000 P2P driver issue and suppress non-actionable advisory warnings.
    os.environ.setdefault('NCCL_P2P_DISABLE', '1')
    os.environ.setdefault('TRANSFORMERS_NO_ADVISORY_WARNINGS', '1')

    if planner_mode:
        os.environ['TASK_PLANNER_MODE'] = planner_mode


def _parse_json_from_text(text: str):
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            return json.loads(line)
        except Exception:
            continue

    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            return None
    return None


def _rank_gpus_by_free_memory() -> list[int]:
    try:
        out = subprocess.check_output(
            [
                'nvidia-smi',
                '--query-gpu=index,memory.free',
                '--format=csv,noheader,nounits',
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []

    rows = []
    for line in out.strip().splitlines():
        parts = [x.strip() for x in line.split(',')]
        if len(parts) != 2:
            continue
        try:
            idx = int(parts[0])
            free_mb = int(parts[1])
        except ValueError:
            continue
        rows.append((idx, free_mb))

    rows.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _free in rows]


def _build_perception_subprocess_env() -> dict:
    env = dict(os.environ)
    env.setdefault('WEATHER_PERCEPTION_FORCE_SINGLE_GPU', '0')
    env.setdefault('WEATHER_PERCEPTION_DEVICE_MAP', 'auto')
    env.setdefault('WEATHER_PERCEPTION_RELEASE_AFTER_INFER', '1')

    if not env.get('CUDA_VISIBLE_DEVICES', '').strip():
        ranked = _rank_gpus_by_free_memory()
        if ranked:
            top_k = int(env.get('WEATHER_PERCEPTION_TOPK_GPUS', '2'))
            top_k = max(1, min(top_k, len(ranked)))
            selected = ranked[:top_k]
            env['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in selected)
            # Restrict accelerate auto-sharding strictly to selected cards.
            env.setdefault('WEATHER_PERCEPTION_CANDIDATE_GPU_IDS', ','.join(str(i) for i in range(top_k)))
    return env


def _predict_degradation_subprocess(input_image: str):
    script = (
        'import json,sys; '
        'from perception_module import predict_degradation; '
        'res = predict_degradation(sys.argv[1]); '
        'print(json.dumps(res, ensure_ascii=False))'
    )
    cmd = [sys.executable, '-c', script, input_image]

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(Path(__file__).resolve().parent),
            env=_build_perception_subprocess_env(),
            capture_output=True,
            text=True,
            timeout=int(os.getenv('WEATHER_PERCEPTION_SUBPROCESS_TIMEOUT', '900')),
            check=False,
        )
    except Exception:
        return None

    if os.getenv('WEATHER_PERCEPTION_DEBUG_SUBPROCESS', '0').strip().lower() in {'1', 'true', 'yes'}:
        try:
            stderr_path = Path(__file__).resolve().parent / 'output' / 'perception_subprocess_stderr.log'
            stderr_path.parent.mkdir(parents=True, exist_ok=True)
            stderr_path.write_text(proc.stderr or '', encoding='utf-8')
        except Exception:
            pass

    if proc.returncode != 0:
        return None

    parsed = _parse_json_from_text(proc.stdout)
    if isinstance(parsed, dict):
        return parsed
    return None


def main(input_image, output_dir, planner_mode=None, run_restoration=False):
    _apply_default_runtime_settings(planner_mode=planner_mode)

    # 1. Perception (zero-shot via VLM)
    use_subprocess = os.getenv('WEATHER_PERCEPTION_SUBPROCESS', '1').strip().lower() not in {'0', 'false', 'no'}
    degradation = _predict_degradation_subprocess(input_image) if use_subprocess else None
    used_subprocess = isinstance(degradation, dict)
    if not used_subprocess:
        degradation = predict_degradation(input_image)

    if isinstance(degradation, dict):
        degradations = degradation.get('degradations', [])
        image_description = degradation.get('image_description', '')
    else:
        degradations = []
        image_description = ''
    print(f"degradations: {degradations}")
    print(f"image_description: {image_description}")

    # 2. Planning
    planner = TaskPlanner()
    plan = planner.plan(degradation, image_path=input_image)
    print(f"Restoration plan: {plan}")
    if getattr(planner, 'last_plan_metadata', None):
        print(f"Planner metadata: {planner.last_plan_metadata}")

    effective_planner_mode = planner_mode or os.getenv('TASK_PLANNER_MODE', 'qwen_only')
    if (not used_subprocess) and effective_planner_mode in {'perception_direct', 'qwen_only'}:
        release_perception_model()

    if not run_restoration:
        print('Restoration stage skipped (default behavior). Use --run_restoration to enable full pipeline.')
        return

    # 3. Restoration
    agent = RestorationAgent()
    final_output = agent.execute_plan(plan, input_image, output_dir)
    print(f"Final output: {final_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--planner_mode", choices=["legacy", "perception_direct", "qwen_only"], default=None,
                        help="Planner mode override. Recommended: qwen_only.")
    parser.add_argument("--run_restoration", action="store_true", help="Enable restoration stage (disabled by default).")
    args = parser.parse_args()
    main(args.input, args.output, planner_mode=args.planner_mode, run_restoration=args.run_restoration)
