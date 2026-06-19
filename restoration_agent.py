import os
import subprocess
import shutil
import time
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import torch
from utils_new.multitask_tools import *
from utils_new.deraining import deraining_toolbox
from utils_new.dehazing import dehazing_toolbox
from utils_new.denoising import denoising_toolbox
from utils_new.desnowing import desnowing_toolbox
from quality_evaluator import QualityEvaluator
from perception_module import predict_degradation
from task_planner import TaskPlanner

class RestorationAgent:
    """
    按 planner 给出的顺序逐步恢复。
    每个步骤用多个专家，选最优。
    """
    def __init__(self):
        self.toolbox_router = {
            'desnow': desnowing_toolbox,
            'derain': deraining_toolbox,
            'dehaze': dehazing_toolbox,
            'denoise': denoising_toolbox,
        }
        self.evaluator = QualityEvaluator(normalize=False)
        self.dynamic_planner = TaskPlanner()

    @staticmethod
    def _parse_gpu_ids() -> list[int]:
        raw = os.getenv('EXPERT_PARALLEL_GPU_IDS', '').strip()
        if raw:
            ids = []
            for token in raw.split(','):
                token = token.strip()
                if not token:
                    continue
                try:
                    ids.append(int(token))
                except ValueError:
                    continue
            return ids

        if not torch.cuda.is_available():
            return []

        # Default: exclude GPU0 because it is often occupied by other services.
        all_ids = list(range(torch.cuda.device_count()))
        if len(all_ids) > 1:
            all_ids = [i for i in all_ids if i != 0] or list(range(torch.cuda.device_count()))

        min_free_mb = int(os.getenv('EXPERT_MIN_FREE_MB', '3000'))
        try:
            out = subprocess.check_output(
                [
                    'nvidia-smi',
                    '--query-gpu=index,memory.free,utilization.gpu',
                    '--format=csv,noheader,nounits',
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            rows = []
            for line in out.strip().splitlines():
                parts = [x.strip() for x in line.split(',')]
                if len(parts) != 3:
                    continue
                idx, free_mb, util = int(parts[0]), int(parts[1]), int(parts[2])
                if idx not in all_ids:
                    continue
                rows.append((idx, free_mb, util))

            if not rows:
                return all_ids

            eligible = [r for r in rows if r[1] >= min_free_mb] or rows
            eligible.sort(key=lambda r: (r[1] - 20 * r[2]), reverse=True)
            return [idx for idx, _free, _util in eligible]
        except Exception:
            return all_ids

    @staticmethod
    def _query_gpu_rows() -> list[tuple[int, int, int]]:
        try:
            out = subprocess.check_output(
                [
                    'nvidia-smi',
                    '--query-gpu=index,memory.free,utilization.gpu',
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
            if len(parts) != 3:
                continue
            try:
                rows.append((int(parts[0]), int(parts[1]), int(parts[2])))
            except ValueError:
                continue
        return rows

    def _wait_for_available_gpus(self, gpu_ids: list[int], required: int, min_free_mb: int) -> list[int]:
        if not gpu_ids:
            return []

        wait_sec = float(os.getenv('EXPERT_GPU_WAIT_SECONDS', '30'))
        poll_sec = float(os.getenv('EXPERT_GPU_POLL_SECONDS', '2'))
        deadline = time.time() + max(0.0, wait_sec)

        while True:
            rows = self._query_gpu_rows()
            if rows:
                free_map = {idx: free for idx, free, _util in rows}
                ranked = sorted(gpu_ids, key=lambda x: free_map.get(x, -1), reverse=True)
                eligible = [idx for idx in ranked if free_map.get(idx, 0) >= min_free_mb]
                if len(eligible) >= required:
                    return eligible[:required]

                # If not enough eligible GPUs, try best-effort with ranked list when wait budget is exhausted.
                if time.time() >= deadline:
                    return ranked[:max(1, min(len(ranked), required))]

            if time.time() >= deadline:
                return gpu_ids[:max(1, min(len(gpu_ids), required))]
            time.sleep(max(0.2, poll_sec))


    @staticmethod
    def _validate_image_readable(image_path: str) -> None:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Input image not found: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"Input image is empty (0 bytes): {path}")
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception as err:
            raise ValueError(f"Input image is unreadable/corrupted: {path} ({err})") from err

    def execute_plan(self, plan, input_image_path, output_dir):
        """
        执行恢复计划。
        plan: 列表，如 ['derain', 'dehaze']
        input_image_path: 输入图像路径
        output_dir: 输出目录
        返回最终输出图像路径
        """
        output_root = Path(output_dir).resolve()
        output_root.mkdir(parents=True, exist_ok=True)

        # Clear stale artifacts from previous runs in the same sample dir.
        for stale in output_root.glob('selected_step_*_*.png'):
            try:
                stale.unlink()
            except Exception:
                pass
        stale_final = output_root / 'final_output.png'
        if stale_final.exists():
            try:
                stale_final.unlink()
            except Exception:
                pass

        current_image = str(Path(input_image_path).resolve())
        self._validate_image_readable(current_image)
        active_plan = list(plan)

        denoise_enabled = os.getenv('ENABLE_DENOISE_STEP', '0').strip().lower() in {'1', 'true', 'yes'}
        if not denoise_enabled:
            original_len = len(active_plan)
            active_plan = [step_name for step_name in active_plan if step_name != 'denoise']
            if len(active_plan) < original_len:
                print('Denoise step is disabled by ENABLE_DENOISE_STEP=0, skipping denoise in current plan.')

        if len(active_plan) == 0:
            final_output = output_root / "final_output.png"
            shutil.copy(current_image, final_output)
            return str(final_output)

        planner_mode = os.getenv('TASK_PLANNER_MODE', 'legacy').strip().lower()
        lock_initial_plan_raw = os.getenv('TASK_PLANNER_LOCK_PLAN', '').strip().lower()
        if lock_initial_plan_raw:
            lock_initial_plan = lock_initial_plan_raw not in {'0', 'false', 'no'}
        else:
            lock_initial_plan = False  # perception_direct no longer locks replanning

        local_replan_enabled = os.getenv('ENABLE_LOCAL_REPLAN', '1').strip().lower() not in {'0', 'false', 'no'}
        if lock_initial_plan:
            local_replan_enabled = False

        max_local_replans = int(os.getenv('LOCAL_REPLAN_MAX', '3'))
        local_replan_count = 0
        completed_tasks: list[str] = []
        failed_experience: list[str] = []
        failed_first_tasks: list[str] = []
        must_check_degradations: list[str] = []
        round_original_task: str | None = None
        round_original_output: str | None = None
        round_original_input: str | None = None
        round_original_deg: list[str] = []
        round_original_remaining_plan: list[str] = []
        round_local_task_scope: list[str] = []
        round_must_check_strict = False
        round_candidates: list[tuple[str, str, list[str]]] = []  # (task_name, output_path, remaining_plan)

        initial_plan = list(active_plan)
        step = 0
        # Perception of step input, updated after each successful step from
        # the post-step perception result.  Replanning reuses it directly —
        # no redundant predict_degradation call needed.
        input_perception: dict | None = None

        step_to_degradation = {
            'derain': 'rain',
            'dehaze': 'haze',
            'desnow': 'snow',
            'denoise': 'noise',
        }

        def _task_is_still_needed(task_name: str, detected_degradations: list[str]) -> bool:
            mapped = step_to_degradation.get(task_name)
            if mapped is None:
                return True
            return mapped in detected_degradations

        def _improvement_score(output_path: str, baseline_input: str, degs: list[str]) -> float:
            """Score output by how much it improved relative to baseline input.
            Degradation-agnostic: rewards removing detected degradations and preserving quality."""
            out_f = self.evaluator._extract_features(output_path)
            inp_f = self.evaluator._extract_features(baseline_input)
            if not out_f or not inp_f:
                return 0.0

            deg_map = {'rain': 'rain_residual_score', 'haze': 'fog_density_score', 'snow': 'snow_artifact'}
            active_degs = [d for d in degs if d in deg_map]
            n = max(len(active_degs), 1)

            score = 0.0
            # Degradation reduction
            for d in active_degs:
                key = deg_map[d]
                iv = inp_f.get(key, 0); ov = out_f.get(key, 0)
                if iv > 1e-6:
                    reduction = (iv - ov) / iv
                    score += max(0.0, reduction) * (0.20 if n <= 2 else 0.15)
            # Detail preservation
            idet = inp_f.get('detail_score', 0); odet = out_f.get('detail_score', 0)
            if idet > 1e-6:
                score += min(odet / idet, 1.0) * 0.20
            # Texture preservation
            itex = inp_f.get('texture_retention', 0); otex = out_f.get('texture_retention', 0)
            if itex > 1e-6:
                score += min(otex / itex, 1.0) * 0.20
            # Contrast improvement
            ilc = inp_f.get('local_contrast', 0); olc = out_f.get('local_contrast', 0)
            if ilc > 1e-6:
                lc_delta = (olc - ilc) / ilc
                score += max(0.0, min(lc_delta * 0.5, 0.20))

            return float(score)

        while step < len(active_plan):
            subtask = active_plan[step]

            print(f"Step {step+1}: {subtask}")
            toolbox = list(self.toolbox_router[subtask])

            # E4 ablation: random single expert replaces the full toolbox.
            if os.getenv('RANDOM_SINGLE_EXPERT', '0').strip().lower() in {'1', 'true', 'yes'}:
                import random as _random, hashlib
                _seed = int(os.getenv('RANDOM_SINGLE_EXPERT_SEED', '2026'))
                # Deterministic per-sample hash so different images get different experts.
                _sample_hash = int(hashlib.md5(current_image.encode()).hexdigest(), 16) % (10 ** 8)
                _rng = _random.Random(_seed + step + _sample_hash)
                valid = [t for t in toolbox if t.work_dir and t.work_dir.exists() and t.script_path and t.script_path.exists()]
                toolbox = [_rng.choice(valid)] if valid else toolbox[:1]

            input_image_for_step = current_image
            # Cache perception of step input for potential replanning use.
            # Step 1+ reuses previous step's post-perception (set below).
            if input_perception is None:
                try:
                    input_perception = predict_degradation(input_image_for_step)
                except Exception:
                    input_perception = None
            step_input_features = self.evaluator._extract_features(input_image_for_step)
            score_guard_enabled = os.getenv('ENABLE_STEP_SCORE_GUARD', '1').strip().lower() not in {'0', 'false', 'no'}
            allow_input_as_candidate = os.getenv('ALLOW_INPUT_AS_CANDIDATE', '0').strip().lower() not in {'0', 'false', 'no'}
            score_max_drop = float(os.getenv('STEP_SCORE_MAX_DROP', '0.02'))
            score_guard_extra_drop = float(os.getenv('STEP_SCORE_GUARD_EXTRA_DROP', '0.01'))
            prefer_expert_when_close = os.getenv('PREFER_EXPERT_WHEN_CLOSE', '1').strip().lower() not in {'0', 'false', 'no'}
            prefer_expert_margin = float(os.getenv('PREFER_EXPERT_MARGIN', '0.02'))

            tasks = []
            for tool in toolbox:
                if tool.work_dir is None or not tool.work_dir.exists() or tool.script_path is None or not tool.script_path.exists():
                    print(f"Tool {tool.tool_name} skipped: missing tool directory or script")
                    continue

                temp_dir = output_root / f"temp_{subtask}_{tool.tool_name}"
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                temp_dir.mkdir(parents=True, exist_ok=True)

                input_dir = temp_dir / "input"
                output_subdir = temp_dir / "output"
                input_dir.mkdir(parents=True, exist_ok=True)
                output_subdir.mkdir(parents=True, exist_ok=True)
                shutil.copy(current_image, input_dir / "input.png")

                tasks.append((tool, temp_dir, input_dir, output_subdir))

            temp_outputs = []
            tool_errors = []
            gpu_ids = self._parse_gpu_ids()
            min_free_mb = int(os.getenv('EXPERT_MIN_FREE_MB', '3000'))
            max_workers_env = int(os.getenv('EXPERT_PARALLEL_WORKERS', str(max(1, len(gpu_ids))))) if gpu_ids else 1
            if max_workers_env < 1:
                max_workers_env = 1
            max_workers = min(max_workers_env, len(tasks))

            if gpu_ids:
                sched_required = min(max_workers, len(gpu_ids))
                sched_gpu_ids = self._wait_for_available_gpus(gpu_ids, required=sched_required, min_free_mb=min_free_mb)
                if not sched_gpu_ids:
                    sched_gpu_ids = gpu_ids[:1]
                max_workers = min(max_workers, len(sched_gpu_ids))
            else:
                sched_gpu_ids = []

            def _run_tool(task_idx: int, tool, input_dir, output_subdir):
                run_gpu_id = None
                requested_multi_gpu_tools = {'maxim', 'diffplugin'}
                tool_gpu_ids_map = {
                    'maxim': os.getenv('WEATHER_MAXIM_GPU_IDS', '').strip(),
                    'diffplugin': os.getenv('WEATHER_DIFFPLUGIN_GPU_IDS', '').strip(),
                }
                use_tool_level_multi_gpu = (
                    tool.tool_name in requested_multi_gpu_tools
                    and bool(tool_gpu_ids_map.get(tool.tool_name, ''))
                )
                # Tools with own GPU routing skip the scheduler (runtime auto-select).
                has_own_routing = bool(tool_gpu_ids_map.get(tool.tool_name, '')) or bool(
                    os.getenv(f'WEATHER_{tool.tool_name.upper()}_GPU_IDS', '').strip())
                if (not use_tool_level_multi_gpu) and sched_gpu_ids and not has_own_routing:
                    run_gpu_id = sched_gpu_ids[task_idx % len(sched_gpu_ids)]
                try:
                    tool(input_dir=input_dir, output_dir=output_subdir, silent=True, run_gpu_id=run_gpu_id)
                    output_path = output_subdir / "output.png"
                    if output_path.exists():
                        return str(output_path), None
                    return None, f"Tool {tool.tool_name} produced no output.png"
                except Exception as err:
                    return None, f"Tool {tool.tool_name} failed: {err}"

            heavy_serial_tools = {'maxim', 'diffplugin'} if subtask in {'derain', 'dehaze', 'desnow'} else set()
            if subtask == 'desnow':
                heavy_serial_tools.add('ddmsnet')
            light_tasks = []
            heavy_tasks = []
            for idx, task_info in enumerate(tasks):
                tool = task_info[0]
                if tool.tool_name.lower() in heavy_serial_tools:
                    heavy_tasks.append((idx, task_info))
                else:
                    light_tasks.append((idx, task_info))

            def _record_tool_result(out_path, err):
                if err:
                    print(err)
                    tool_errors.append(err)
                elif out_path is not None:
                    temp_outputs.append(out_path)

            def _run_task_subset_parallel(task_subset):
                if not task_subset:
                    return
                subset_workers = min(max_workers, len(task_subset))
                if subset_workers <= 1:
                    for idx, (tool, _temp_dir, input_dir, output_subdir) in task_subset:
                        _record_tool_result(*_run_tool(idx, tool, input_dir, output_subdir))
                    return

                with ThreadPoolExecutor(max_workers=subset_workers) as executor:
                    futures = {}
                    for idx, (tool, _temp_dir, input_dir, output_subdir) in task_subset:
                        fut = executor.submit(_run_tool, idx, tool, input_dir, output_subdir)
                        futures[fut] = idx
                    for fut in as_completed(futures):
                        _record_tool_result(*fut.result())

            def _run_task_subset_serial(task_subset):
                for idx, (tool, _temp_dir, input_dir, output_subdir) in task_subset:
                    _record_tool_result(*_run_tool(idx, tool, input_dir, output_subdir))

            _run_task_subset_parallel(light_tasks)
            _run_task_subset_serial(heavy_tasks)

            if temp_outputs:
                candidate_pool = [input_image_for_step] + temp_outputs if (score_guard_enabled and allow_input_as_candidate) else temp_outputs
                best_output, score = self.evaluator.select_best(candidate_pool, task_name=subtask, input_features=step_input_features)
                selected_prev_by_guard = (score_guard_enabled and allow_input_as_candidate and Path(best_output).resolve() == Path(input_image_for_step).resolve())

                if selected_prev_by_guard and prefer_expert_when_close:
                    expert_best_output, _expert_score = self.evaluator.select_best(temp_outputs, task_name=subtask, input_features=step_input_features)
                    prev_score_raw = self.evaluator.evaluate(input_image_for_step, task_name=subtask)
                    expert_score_raw = self.evaluator.evaluate(expert_best_output, task_name=subtask)
                    if expert_score_raw >= prev_score_raw - prefer_expert_margin:
                        best_output, score = expert_best_output, _expert_score
                        selected_prev_by_guard = False

                if score_guard_enabled and allow_input_as_candidate and not selected_prev_by_guard:
                    prev_score_raw = self.evaluator.evaluate(input_image_for_step, task_name=subtask)
                    best_score_raw = self.evaluator.evaluate(best_output, task_name=subtask)
                    guard_drop_threshold = score_max_drop + score_guard_extra_drop
                    if best_score_raw < prev_score_raw - guard_drop_threshold:
                        selected_prev_by_guard = True

                stable_best = output_root / f"selected_step_{step+1}_{subtask}.png"
                if selected_prev_by_guard:
                    src_path = Path(input_image_for_step).resolve()
                    dst_path = stable_best.resolve()
                    if src_path != dst_path:
                        shutil.copy(src_path, dst_path)
                else:
                    src_path = Path(best_output).resolve()
                    dst_path = stable_best.resolve()
                    if src_path != dst_path:
                        shutil.copy(src_path, dst_path)
                current_image = str(stable_best)

                # Track this step's output for end-of-round comparison.
                remaining_after_this = list(active_plan[step + 1:])
                round_candidates.append((subtask, current_image, remaining_after_this))

                keep_intermediates = os.getenv('KEEP_ALL_INTERMEDIATES', '0').strip().lower() in {'1', 'true', 'yes'}
                if not keep_intermediates:
                    selected_path = Path(best_output).resolve()
                    for _tool, temp_dir, _input_dir, _output_subdir in tasks:
                        if selected_path.is_relative_to(temp_dir.resolve()):
                            continue
                        try:
                            shutil.rmtree(temp_dir)
                        except Exception:
                            pass
            else:
                if tool_errors:
                    detail = '; '.join(tool_errors[:4])
                    raise RuntimeError(f"No valid outputs for {subtask}. Tool errors: {detail}")
                raise RuntimeError(f"No valid outputs for {subtask}. No available tools or all tools skipped.")

            # Terminal only when no initial-plan tasks remain unfinished.
            # Must not break if replanning shortened active_plan
            # while other initial_plan tasks are still pending.
            pending_initial = [t for t in initial_plan
                               if t not in completed_tasks and t != subtask]
            is_terminal_task = len(initial_plan) <= 1 or (
                step >= len(active_plan) - 1 and not pending_initial)
            if is_terminal_task:
                break

            rep = None
            current_deg: list[str] = []
            current_description = ''
            try:
                rep = predict_degradation(current_image)
                if isinstance(rep, dict):
                    current_deg = [str(x).strip().lower() for x in rep.get('degradations', []) if str(x).strip()]
                    current_description = str(rep.get('image_description', '')).strip()
            except Exception as e:
                print(f'Post-step perception failed, keep remaining plan: {e}')

            task_deg = step_to_degradation.get(subtask)
            check_degradations = []
            strict_must_check_active = bool(must_check_degradations) and round_must_check_strict
            if strict_must_check_active:
                for deg in must_check_degradations:
                    if deg not in check_degradations:
                        check_degradations.append(deg)
            if task_deg is not None and task_deg not in check_degradations:
                check_degradations.append(task_deg)

            unresolved_degradations = [deg for deg in check_degradations if deg in current_deg]
            task_succeeded = task_deg is None or len(unresolved_degradations) == 0
            print(
                f"Post-step perception: current_deg={current_deg}; "
                f"check={check_degradations}; unresolved={unresolved_degradations}; "
                f"strict_must_check={strict_must_check_active}; success={task_succeeded}"
            )
            if task_succeeded:
                # If replanning happened this round, compare all candidates.
                if len(round_candidates) > 1:
                    candidate_paths = [p for _, p, _ in round_candidates]
                    scores = [_improvement_score(p, round_original_input or input_image_for_step, round_original_deg) for p in candidate_paths]
                    best_idx = max(range(len(scores)), key=lambda i: scores[i])
                    best_task, best_path, best_suffix = round_candidates[best_idx]
                    print(f"Round comparison (success): {len(round_candidates)} candidates, "
                          f"scores={[f'{s:.4f}' for s in scores]}, "
                          f"selected={best_task}(idx={best_idx})")
                    current_image = best_path
                    completed_tasks.append(best_task)
                else:
                    if subtask not in completed_tasks:
                        completed_tasks.append(subtask)
                    best_suffix = active_plan[step + 1:]
                    best_task = subtask

                had_replanning = len(round_candidates) > 1

                failed_experience = []
                failed_first_tasks = []
                must_check_degradations = []
                round_original_task = None
                round_original_output = None
                round_original_input = None
                round_original_deg = []
                round_original_remaining_plan = []
                round_local_task_scope = []
                round_must_check_strict = False
                round_candidates = []

                if had_replanning:
                    remain = best_suffix
                    # Re-perceive on the selected (winning) image to correctly filter remaining tasks.
                    try:
                        rep = predict_degradation(current_image)
                        if isinstance(rep, dict):
                            selected_deg = [str(x).strip().lower() for x in rep.get('degradations', []) if str(x).strip()]
                        else:
                            selected_deg = current_deg
                    except Exception:
                        selected_deg = current_deg
                    input_perception = rep if isinstance(rep, dict) else None
                else:
                    remain = active_plan[step + 1:]
                    selected_deg = current_deg
                    # rep from post-step perception is the output = next step's input.
                    input_perception = rep if isinstance(rep, dict) else None
                filtered_remain = [t for t in remain if t not in completed_tasks and _task_is_still_needed(t, selected_deg)]
                active_plan = active_plan[:step + 1] + filtered_remain
                if step + 1 >= len(active_plan):
                    break
                step += 1
                continue

            if unresolved_degradations:
                if round_original_task is None:
                    round_original_task = subtask
                    round_original_output = current_image
                    round_original_input = input_image_for_step
                    round_original_deg = list(current_deg)
                    round_original_remaining_plan = list(active_plan[step + 1:])
                    round_local_task_scope = [subtask] + list(round_original_remaining_plan)
                    round_must_check_strict = False

                exp_msg = (
                    f"{subtask} 作为当前第一优先任务后，"
                    f"检测仍存在本轮必须解决的残留退化: {', '.join(unresolved_degradations) or 'none'}; "
                    f"当前检测结果: {', '.join(current_deg) or 'none'}"
                )
                failed_experience.append(exp_msg)
                if subtask not in failed_first_tasks:
                    failed_first_tasks.append(subtask)
                for deg in check_degradations:
                    if deg not in must_check_degradations:
                        must_check_degradations.append(deg)
                for deg in current_deg:
                    if deg in {'rain', 'haze', 'snow'} and deg not in must_check_degradations:
                        must_check_degradations.append(deg)

                # Replanning only uses tasks from the INITIAL plan, not from dynamic scope.
                # All unfinished initial-plan tasks are candidates; Qwen can reorder them.
                # Only constraint: the FIRST task must not be a previously-failed one.
                all_unfinished = [t for t in initial_plan
                                  if t in {'derain', 'dehaze', 'desnow'}
                                  and t not in completed_tasks]
                has_alternative = any(t not in failed_first_tasks for t in all_unfinished)
                can_replan = local_replan_enabled and local_replan_count <= max_local_replans and has_alternative

                if can_replan:
                    # Use cached perception of step input (set at loop start,
                    # updated after each successful step). No redundant call.
                    step_degradation_vector = (
                        input_perception
                        if isinstance(input_perception, dict)
                        else {'degradations': current_deg, 'image_description': current_description}
                    )
                    replanned_suffix = self.dynamic_planner.local_replan(
                        image_path=input_image_for_step,
                        remaining_tasks=all_unfinished,
                        failed_task=subtask,
                        failed_experience=failed_experience,
                        degradation_vector=step_degradation_vector,
                        completed_tasks=completed_tasks,
                        failed_first_tasks=failed_first_tasks,
                        must_check_degradations=must_check_degradations,
                    )
                    if replanned_suffix:
                        active_plan = active_plan[:step] + replanned_suffix
                        local_replan_count += 1
                        # Roll back to the original input: replanning means trying a
                        # different first task from scratch, not from the failed result.
                        current_image = round_original_input or input_image_for_step
                        print(
                            f"Local replan triggered (count={local_replan_count}): {replanned_suffix}; "
                            f"failed_first={failed_first_tasks}; must_check={must_check_degradations}; "
                            f"local_scope={all_unfinished}; strict_must_check={round_must_check_strict}"
                        )
                        continue

                # Round exhausted: compare all outputs from this round, pick the best.
                if len(round_candidates) > 1:
                    candidate_paths = [p for _, p, _ in round_candidates]
                    scores = [_improvement_score(p, round_original_input or input_image_for_step, round_original_deg) for p in candidate_paths]
                    best_idx = max(range(len(scores)), key=lambda i: scores[i])
                    best_task, best_path, best_suffix = round_candidates[best_idx]
                    print(f"Round comparison: {len(round_candidates)} candidates, "
                          f"scores={[f'{s:.4f}' for s in scores]}, "
                          f"selected={best_task}(idx={best_idx})")
                    current_image = best_path
                elif len(round_candidates) == 1:
                    best_task, best_path, best_suffix = round_candidates[0]
                else:
                    best_task, best_suffix = (round_original_task or subtask), round_original_remaining_plan

                if best_task not in completed_tasks:
                    completed_tasks.append(best_task)
                # Re-perceive on the selected image to correctly filter remaining tasks.
                try:
                    rep = predict_degradation(current_image)
                    if isinstance(rep, dict):
                        selected_deg = [str(x).strip().lower() for x in rep.get('degradations', []) if str(x).strip()]
                    else:
                        selected_deg = current_deg
                except Exception:
                    selected_deg = current_deg
                input_perception = rep if isinstance(rep, dict) else None
                fallback_suffix = [t for t in best_suffix if t not in completed_tasks
                                   and _task_is_still_needed(t, selected_deg)]
                active_plan = active_plan[:step] + [best_task] + fallback_suffix
                failed_experience = []
                failed_first_tasks = []
                must_check_degradations = []
                round_original_task = None
                round_original_output = None
                round_original_input = None
                round_original_deg = []
                round_original_remaining_plan = []
                round_local_task_scope = []
                round_must_check_strict = False
                round_candidates = []
                print(f"Round ended: selected={best_task}, continuing suffix: {fallback_suffix}")
                if step + 1 >= len(active_plan):
                    break
                step += 1
                continue

            step += 1

        final_output = output_root / "final_output.png"
        shutil.copy(current_image, final_output)
        return str(final_output)