import numpy as np
import os
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional


class TaskPlanner:
    """
    根据检测出的退化列表，生成恢复顺序。
    经验规则库基于常见的退化组合和恢复顺序。
    """

    def __init__(self):
        raw_mode = os.getenv('TASK_PLANNER_MODE', 'qwen_only').strip().lower()
        self.planner_mode = self._normalize_planner_mode(raw_mode)
        self.presence_prob_threshold = float(os.getenv('TASK_PRESENCE_PROB_THRESHOLD', '0.68'))
        self.direct_min_top1 = float(os.getenv('TASK_DIRECT_MIN_TOP1', '0.68'))
        self.direct_min_margin = float(os.getenv('TASK_DIRECT_MIN_MARGIN', '0.18'))
        self.direct_min_active_tasks = int(os.getenv('TASK_DIRECT_MIN_ACTIVE_TASKS', '1'))
        self.direct_require_snow_when_present = os.getenv('TASK_DIRECT_REQUIRE_SNOW_WHEN_PRESENT', '1').strip().lower() not in {'0', 'false', 'no'}
        self.snow_presence_threshold = float(os.getenv('TASK_SNOW_PRESENCE_THRESHOLD', '0.75'))
        self.vlm_isolated_env = os.getenv('TASK_PLANNER_ISOLATED_ENV', 'weather_agent_planner').strip()
        self.vlm_subprocess_timeout = int(os.getenv('TASK_PLANNER_SUBPROCESS_TIMEOUT', '600'))
        self.vlm_bridge_script = str((Path(__file__).resolve().parent / 'vlm_planner_bridge.py'))
        self.last_plan_metadata = {}
        # 固定顺序：先去雪/去雨，最后去雾
        # 通过阈值控制每个步骤是否启用
        self.step_order = ['desnow', 'derain', 'dehaze']

        self.label_to_step = {
            'rain': 'derain',
            'haze': 'dehaze',
            'snow': 'desnow',
        }
        self.step_to_label = {v: k for k, v in self.label_to_step.items()}

    @staticmethod
    def _normalize_planner_mode(mode: str) -> str:
        # Keep legacy aliases for backward compatibility, but unify to current naming.
        alias = {
            'clip_direct': 'perception_direct',
            'clip_vlm_fallback': 'qwen_only',
            'perception_vlm_fallback': 'qwen_only',
        }
        normalized = alias.get(mode, mode)
        if normalized not in {'legacy', 'perception_direct', 'qwen_only'}:
            return 'qwen_only'
        return normalized

    def planner_io_definition(self):
        """
        规划器接口定义（P_I）。
        """
        return {
            'input': {
                'C_I': 'image content description (str)',
                'D_I': 'degradation list (list[str])',
                'A_I': 'candidate task set generated from D_I (list[str])',
                'I': 'original input image path (str)',
                'prompt': 'planner prompt text (str)',
            },
            'output': {
                'P_I': 'ordered task sequence (list[str])',
            },
        }

    def rollback_replan_io_definition(self):
        """
        回滚重规划接口定义（P_I^{adj}）。当前仅定义，不启用。
        """
        return {
            'enabled': False,
            'input': {
                'D_I': 'degradation list (list[str])',
                'A_I_R': 'remaining task set (list[str])',
                'S_I': 'failed step information (dict or str)',
            },
            'output': {
                'P_I_adj': 'adjusted ordered task sequence (list[str])',
            },
        }

    def _weather_probs(self, probs):
        return {
            'desnow': probs['snow'],
            'derain': probs['rain'],
            'dehaze': probs['haze'],
        }

    def _fixed_priority_steps(self):
        return ['desnow', 'derain', 'dehaze']

    def _active_tasks_to_degradations(self, active_tasks):
        degradations = []
        for step in active_tasks:
            for label, mapped_step in self.label_to_step.items():
                if step == mapped_step and label not in degradations:
                    degradations.append(label)
        return degradations

    def build_perception_context(self, degradation_vector):
        probs = self._parse_probs(degradation_vector)
        weather_probs = self._weather_probs(probs)
        ranked = sorted(weather_probs.items(), key=lambda x: x[1], reverse=True)
        visible = [f"{step}:{score:.3f}" for step, score in ranked]
        active = [step for step, score in ranked if score >= self.presence_prob_threshold]
        if not active and ranked:
            active = [ranked[0][0]]

        if active:
            summary = f"Perception suggests active degradations: {', '.join(active)}."
        else:
            summary = f"Perception finds no degradation above presence threshold {self.presence_prob_threshold:.2f}."

        return {
            'probabilities': {step: float(score) for step, score in ranked},
            'severity_scores': {step: float(score) for step, score in ranked},
            'tasks': active,
            'image_description': summary + ' Scores=' + ', '.join(visible),
        }

    def build_explicit_planner_inputs(self, degradation_vector, image_path: Optional[str] = None):
        perception_info = self.build_perception_context(degradation_vector)
        direct_plan = self.direct_plan(degradation_vector)
        degradations = self._active_tasks_to_degradations(perception_info.get('tasks', []))
        image_description = str(perception_info.get('image_description', '')).strip()

        # In qwen_only mode, the current perception module already returns a model-generated
        # JSON degradation list. Use that list directly instead of converting it to the
        # legacy pseudo-probability representation and thresholding it again.
        if self.planner_mode == 'qwen_only' and isinstance(degradation_vector, dict):
            raw_degradations = []
            for item in degradation_vector.get('degradations', []) or []:
                if isinstance(item, dict):
                    dtype = str(item.get('type', '')).strip().lower()
                else:
                    dtype = str(item).strip().lower()
                dtype = {'fog': 'haze', 'smog': 'haze', 'mist': 'haze', 'rainy': 'rain', 'snowy': 'snow'}.get(dtype, dtype)
                if dtype in self.label_to_step and dtype not in raw_degradations:
                    raw_degradations.append(dtype)

            if raw_degradations:
                degradations = raw_degradations
                direct_plan = [self.label_to_step[d] for d in raw_degradations]
                image_description = str(degradation_vector.get('image_description', '')).strip() or image_description
                perception_info = {
                    **perception_info,
                    'tasks': list(direct_plan),
                    'raw_degradations': list(raw_degradations),
                    'image_description': image_description,
                    'source': 'model_json_degradations',
                }

        return {
            'C_I': image_description,
            'D_I': degradations,
            'A_I': list(direct_plan),
            'I': image_path or '',
            'perception_info': perception_info,
            'direct_plan': list(direct_plan),
        }

    def direct_plan(self, degradation_vector):
        probs = self._parse_probs(degradation_vector)
        weather_probs = self._weather_probs(probs)
        ranked = sorted(weather_probs.items(), key=lambda x: x[1], reverse=True)

        perceived = [step for step, score in ranked if score >= self.presence_prob_threshold]
        if not perceived and ranked:
            perceived = [ranked[0][0]]

        # 严重程度仅用于排序，不用于“是否执行”的阈值裁剪
        return perceived

    def evaluate_direct_plan(self, degradation_vector, direct_plan):
        probs = self._parse_probs(degradation_vector)
        weather_probs = self._weather_probs(probs)
        ranked = sorted(weather_probs.items(), key=lambda x: x[1], reverse=True)
        top1_name, top1 = ranked[0] if ranked else ('', 0.0)
        top2_name, top2 = ranked[1] if len(ranked) > 1 else ('', 0.0)
        margin = top1 - top2

        reasons = []
        if len(direct_plan) < self.direct_min_active_tasks:
            reasons.append('insufficient_active_tasks')
        if top1 < self.direct_min_top1:
            reasons.append('top1_confidence_too_low')
        if margin < self.direct_min_margin:
            reasons.append('top1_top2_margin_too_small')

        snow_present = weather_probs.get('desnow', 0.0) >= self.snow_presence_threshold
        if self.direct_require_snow_when_present and snow_present and 'desnow' not in direct_plan:
            reasons.append('snow_detected_but_not_in_direct_plan')

        return {
            'direct_acceptable': len(reasons) == 0,
            'reasons': reasons,
            'top1_name': top1_name,
            'top1_score': float(top1),
            'top2_name': top2_name,
            'top2_score': float(top2),
            'margin': float(margin),
            'active_count': len(direct_plan),
        }

    def _plan_legacy(self, degradation_vector):
        probs = self._parse_probs(degradation_vector)
        plan = self.initial_candidates(degradation_vector)

        high_rain_th = float(os.getenv('TASK_HIGH_RAIN_THRESHOLD', '0.9'))
        if probs['rain'] > high_rain_th and 'derain' in plan:
            first_derain_idx = plan.index('derain')
            plan.insert(first_derain_idx + 1, 'derain')

        high_snow_th = float(os.getenv('TASK_HIGH_SNOW_THRESHOLD', '0.9'))
        if probs['snow'] > high_snow_th and 'desnow' in plan:
            first_desnow_idx = plan.index('desnow')
            plan.insert(first_desnow_idx + 1, 'desnow')

        self.last_plan_metadata = {
            'planner_mode': 'legacy',
            'planner_source': 'legacy_rules',
        }
        return plan

    def _plan_hybrid(self, degradation_vector, image_path: Optional[str] = None):
        explicit_inputs = self.build_explicit_planner_inputs(degradation_vector, image_path=image_path)
        perception_info = explicit_inputs['perception_info']
        direct_plan = explicit_inputs['direct_plan']
        direct_eval = self.evaluate_direct_plan(degradation_vector, direct_plan)
        self.last_plan_metadata = {
            'planner_mode': self.planner_mode,
            'planner_source': 'perception_direct',
            'perception_info': perception_info,
            'direct_plan': list(direct_plan),
            'direct_eval': direct_eval,
        }

        # perception_direct 模式：直接使用感知初始计划
        if self.planner_mode == 'perception_direct':
            return direct_plan

        # qwen_only 模式：始终使用 Qwen 生成计划（不进行置信度门控）。
        if image_path and self.planner_mode == 'qwen_only':
            try:
                if self.vlm_isolated_env:
                    qwen_result = self._plan_via_isolated_env(
                        image_path=image_path,
                        explicit_inputs=explicit_inputs,
                        allowed_steps=self._fixed_priority_steps(),
                    )
                    self.last_plan_metadata['qwen_execution_mode'] = 'isolated_subprocess'
                else:
                    from vlm_planner import QwenVLPlanner

                    vlm_planner = QwenVLPlanner()
                    qwen_result = vlm_planner.plan(
                        image_path=image_path,
                        C_I=explicit_inputs['C_I'],
                        D_I=explicit_inputs['D_I'],
                        A_I=explicit_inputs['A_I'],
                        prompt=None,
                        allowed_steps=self._fixed_priority_steps(),
                    )
                    self.last_plan_metadata['qwen_execution_mode'] = 'in_process'

                qwen_plan = [step for step in qwen_result.get('plan', []) if step in self._fixed_priority_steps()]
                if qwen_plan:
                    self.last_plan_metadata = {
                        **self.last_plan_metadata,
                        'planner_source': 'qwen_only',
                        'qwen_result': qwen_result,
                        'final_plan': list(qwen_plan),
                    }
                    return qwen_plan
                self.last_plan_metadata['qwen_result'] = qwen_result
            except Exception as e:
                self.last_plan_metadata['qwen_error'] = str(e)

        # Qwen 不可用或返回空计划时，回退 direct_plan 以保证系统可运行
        self.last_plan_metadata['planner_source'] = 'perception_direct_fallback'
        return direct_plan

    def build_local_replan_prompt(
        self,
        image_description: str,
        remaining_tasks: list[str],
        failed_task: str,
        failed_experience: list[str],
        degradations: list[str],
        completed_tasks: Optional[list[str]] = None,
        failed_first_tasks: Optional[list[str]] = None,
        must_check_degradations: Optional[list[str]] = None,
    ) -> str:
        completed_tasks = completed_tasks or []
        failed_first_tasks = failed_first_tasks or []
        must_check_degradations = must_check_degradations or []
        degrade_text = ', '.join(degradations) if degradations else 'unknown'
        remaining_text = json.dumps(remaining_tasks, ensure_ascii=False)
        completed_text = json.dumps(completed_tasks, ensure_ascii=False)
        failed_first_text = json.dumps(failed_first_tasks, ensure_ascii=False)
        must_check_text = ', '.join(must_check_degradations) if must_check_degradations else 'none'
        tried_text = json.dumps(failed_experience, ensure_ascii=False)
        failed_label = self.step_to_label.get(failed_task, failed_task)
        return (
            'You are an expert in weather image restoration planning. '
            'Generate a local replan from the current step onward. Output JSON only.\n\n'
            f'Current image description: {image_description}\n'
            f'Current detected degradations: {degrade_text}\n'
            f'Completed tasks that must NOT appear again: {completed_text}\n'
            f'Candidate unfinished tasks for the new plan: {remaining_text}\n'
            f'Tasks that already failed as the current first-priority task: {failed_first_text}\n'
            f'Must-check unresolved degradations for this replan round: {must_check_text}\n'
            f'Current failed task: {failed_task} (degradation: {failed_label})\n'
            f'Failure experiences: {tried_text}\n\n'
            'Rules:\n'
            '1) Return JSON with key "plan" only.\n'
            '2) Replan from the current step onward, not from the beginning.\n'
            '3) Use only candidate unfinished tasks; do not include completed tasks.\n'
            '4) If an unfinished alternative exists, do not put any failed-first task as the first task.\n'
            '5) Failed-first tasks may appear later only when there is at least one alternative first task in the current local scope.\n'
            '6) The first task should consider current detected degradations and the must-check unresolved degradations, but may choose a different first task when the original order may be wrong.\n'
            '7) If the first task is an alternative task, keep the failed task later unless it is completed or no longer useful.\n'
            '8) Keep plan concise and deterministic.\n'
        )

    def local_replan(
        self,
        image_path: str,
        remaining_tasks: list[str],
        failed_task: str,
        failed_experience: list[str],
        degradation_vector,
        completed_tasks: Optional[list[str]] = None,
        failed_first_tasks: Optional[list[str]] = None,
        must_check_degradations: Optional[list[str]] = None,
    ) -> list[str]:
        completed_tasks = completed_tasks or []
        failed_first_tasks = failed_first_tasks or []
        must_check_degradations = must_check_degradations or []
        remaining_tasks = [t for t in remaining_tasks if t not in completed_tasks]
        if not remaining_tasks:
            return []

        explicit_inputs = self.build_explicit_planner_inputs(degradation_vector, image_path=image_path)
        degradations = explicit_inputs.get('D_I', [])
        image_description = explicit_inputs.get('C_I', '')
        prompt = self.build_local_replan_prompt(
            image_description=image_description,
            remaining_tasks=remaining_tasks,
            failed_task=failed_task,
            failed_experience=failed_experience,
            degradations=degradations,
            completed_tasks=completed_tasks,
            failed_first_tasks=failed_first_tasks,
            must_check_degradations=must_check_degradations,
        )

        def _sanitize_plan(raw_plan: list[str]) -> list[str]:
            plan = []
            seen = set()
            for step in raw_plan:
                if step not in remaining_tasks or step in completed_tasks or step in seen:
                    continue
                plan.append(step)
                seen.add(step)
            return plan

        def _first_task_is_allowed(plan: list[str]) -> bool:
            if not plan:
                return False
            return plan[0] not in failed_first_tasks

        def _move_first_failed_task_back(plan: list[str]) -> list[str]:
            alternatives = [t for t in plan if t not in failed_first_tasks]
            if not alternatives:
                return []
            first = alternatives[0]
            return [first] + [t for t in plan if t != first]

        try:
            result = self._plan_via_isolated_env(
                image_path=image_path,
                explicit_inputs=explicit_inputs,
                allowed_steps=remaining_tasks,
                prompt=prompt,
            )
            plan = _sanitize_plan(result.get('plan', []))
            if plan and _first_task_is_allowed(plan):
                return plan
            if plan:
                reordered = _move_first_failed_task_back(plan)
                if reordered:
                    return reordered
        except Exception:
            pass

        alternatives = [t for t in remaining_tasks if t not in failed_first_tasks]
        if alternatives:
            first = alternatives[0]
            return [first] + [t for t in remaining_tasks if t != first]
        return []

    @staticmethod
    def _extract_json_from_text(text: str):
        if not text:
            return {}
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
                return {}
        return {}

    def _plan_via_isolated_env(self, image_path, explicit_inputs, allowed_steps, prompt: Optional[str] = None):
        payload = {
            'image_path': image_path,
            'C_I': explicit_inputs['C_I'],
            'D_I': explicit_inputs['D_I'],
            'A_I': explicit_inputs['A_I'],
            'I': explicit_inputs['I'],
            'allowed_steps': allowed_steps,
        }
        if prompt:
            payload['prompt'] = prompt

        isolated_python = ''
        candidates = []

        # 1) Explicit override takes highest priority.
        explicit_python = os.getenv('TASK_PLANNER_ISOLATED_PYTHON', '').strip()
        if explicit_python:
            candidates.append(Path(explicit_python))

        # 2) Derive from current conda prefix if available.
        conda_prefix = os.getenv('CONDA_PREFIX', '').strip()
        if conda_prefix:
            prefix_path = Path(conda_prefix).resolve()
            candidates.append(prefix_path.parent / self.vlm_isolated_env / 'bin' / 'python')
            candidates.append(prefix_path.parent / 'envs' / self.vlm_isolated_env / 'bin' / 'python')

        # 3) Derive from current interpreter location (works when launched via absolute env python).
        exe = Path(sys.executable).resolve()
        candidates.append(exe.parent.parent.parent / self.vlm_isolated_env / 'bin' / 'python')
        candidates.append(exe.parent.parent.parent / 'envs' / self.vlm_isolated_env / 'bin' / 'python')

        # 4) Derive from conda executable if present.
        conda_exe = os.getenv('CONDA_EXE', '').strip()
        if conda_exe:
            conda_root = Path(conda_exe).resolve().parent.parent
            candidates.append(conda_root / 'envs' / self.vlm_isolated_env / 'bin' / 'python')

        seen = set()
        for candidate in candidates:
            c = str(candidate)
            if c in seen:
                continue
            seen.add(c)
            if candidate.exists():
                isolated_python = c
                break

        if isolated_python:
            cmd = [isolated_python, self.vlm_bridge_script]
        else:
            cmd = [
                'conda', 'run', '-n', self.vlm_isolated_env,
                'python', self.vlm_bridge_script,
            ]
        env = os.environ.copy()
        env.setdefault('TRANSFORMERS_VERBOSITY', 'error')
        env.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
        env.setdefault('NCCL_P2P_DISABLE', '1')
        env.setdefault('TRANSFORMERS_NO_ADVISORY_WARNINGS', '1')
        env.setdefault('WEATHER_SUPPRESS_RUNTIME_WARNINGS', '1')

        completed = subprocess.run(
            cmd,
            input=json.dumps(payload, ensure_ascii=False),
            text=True,
            capture_output=True,
            timeout=self.vlm_subprocess_timeout,
            env=env,
            cwd=str(Path(__file__).resolve().parent),
        )

        stdout_json = self._extract_json_from_text(completed.stdout)
        stderr_text = (completed.stderr or '').strip()

        if completed.returncode != 0:
            raise RuntimeError(
                f'isolated planner failed (code={completed.returncode}): '
                f'{stderr_text or completed.stdout[:500]}'
            )

        if not isinstance(stdout_json, dict) or not stdout_json:
            raise RuntimeError('isolated planner returned empty/non-json output')

        if stderr_text:
            stdout_json['stderr'] = stderr_text[-2000:]
        return stdout_json

    def _parse_probs(self, degradation_vector):
        if isinstance(degradation_vector, dict):
            probs_json = degradation_vector.get('probabilities', {})
            if isinstance(probs_json, dict) and len(probs_json) > 0:
                rain_p = float(probs_json.get('rain', probs_json.get('derain', 0.0)))
                haze_p = float(probs_json.get('haze', probs_json.get('dehaze', 0.0)))
                snow_p = float(probs_json.get('snow', probs_json.get('desnow', 0.0)))
                return {
                    'rain': rain_p,
                    'haze': haze_p,
                    'snow': snow_p,
                }

            severity_map = {'mild': 0.35, 'moderate': 0.65, 'severe': 0.9}
            rain_p = haze_p = snow_p = 0.0
            degradations = degradation_vector.get('degradations', [])
            if isinstance(degradations, list):
                for item in degradations:
                    if isinstance(item, dict):
                        d_type = str(item.get('type', '')).strip().lower()
                        sev = severity_map.get(str(item.get('severity', '')).strip().lower(), 0.65)
                    else:
                        d_type = str(item).strip().lower()
                        sev = 0.65
                    if d_type == 'rain':
                        rain_p = max(rain_p, sev)
                    elif d_type == 'haze':
                        haze_p = max(haze_p, sev)
                    elif d_type == 'snow':
                        snow_p = max(snow_p, sev)
            return {
                'rain': float(rain_p),
                'haze': float(haze_p),
                'snow': float(snow_p),
            }

        degradation_vector = np.asarray(degradation_vector, dtype=np.float32).reshape(-1)
        if degradation_vector.size < 3:
            raise ValueError("degradation_vector 至少需要包含 [rain, haze, snow]")

        if degradation_vector.size == 3:
            rain_p, haze_p, snow_p = degradation_vector.tolist()
        else:
            # Backward compatibility for legacy 4D vector [rain, haze, noise, snow].
            rain_p, haze_p, _, snow_p = degradation_vector[:4].tolist()
        return {
            'rain': float(rain_p),
            'haze': float(haze_p),
            'snow': float(snow_p),
        }

    def _thresholds(self):
        threshold = float(os.getenv('TASK_THRESHOLD', '0.65'))
        return {
            'rain': float(os.getenv('TASK_RAIN_THRESHOLD', str(threshold))),
            'haze': float(os.getenv('TASK_HAZE_THRESHOLD', str(threshold))),
            'snow': float(os.getenv('TASK_SNOW_THRESHOLD', str(threshold))),
        }

    def _scores(self, probs):
        return {
            'desnow': probs['snow'] * 1.05,
            'derain': probs['rain'] * 1.00,
            'dehaze': probs['haze'] * 0.95,
        }

    def initial_candidates(self, degradation_vector):
        probs = self._parse_probs(degradation_vector)
        th = self._thresholds()
        candidates = []
        for step in self.step_order:
            if step == 'desnow' and probs['snow'] > th['snow']:
                candidates.append(step)
            elif step == 'derain' and probs['rain'] > th['rain']:
                candidates.append(step)
            elif step == 'dehaze' and probs['haze'] > th['haze']:
                candidates.append(step)

        if not candidates:
            scores = self._scores(probs)
            best_step = max(scores, key=scores.get)
            if scores[best_step] > float(os.getenv('TASK_MIN_TRIGGER_SCORE', '0.45')):
                candidates = [best_step]
        return candidates

    def choose_next_step(self, degradation_vector, allowed_steps, executed_counts=None):
        if executed_counts is None:
            executed_counts = {}

        probs = self._parse_probs(degradation_vector)
        th = self._thresholds()
        scores = self._scores(probs)
        max_repeat = int(os.getenv('TASK_MAX_REPEAT_PER_STEP', '2'))
        high_repeat_boost_th = float(os.getenv('TASK_HIGH_REPEAT_THRESHOLD', '0.9'))

        candidates = []
        for step in self.step_order:
            if step not in allowed_steps:
                continue

            count = int(executed_counts.get(step, 0))
            if count >= max_repeat:
                continue

            should_run = False
            if step == 'desnow':
                should_run = probs['snow'] > th['snow'] or (probs['snow'] > high_repeat_boost_th and count < max_repeat)
            elif step == 'derain':
                should_run = probs['rain'] > th['rain'] or (probs['rain'] > high_repeat_boost_th and count < max_repeat)
            elif step == 'dehaze':
                should_run = probs['haze'] > th['haze'] or (probs['haze'] > high_repeat_boost_th and count < max_repeat)

            if should_run:
                penalty = 0.12 * count
                candidates.append((scores[step] - penalty, step))

        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def plan(self, degradation_vector, image_path: Optional[str] = None):
        """
        输入退化概率向量 [Prain, Phaze, Psnow]，阈值化后生成计划。
        degradation_vector: numpy array 或 list, [Prain, Phaze, Psnow]
        返回恢复顺序列表，如 ['derain', 'dehaze']
        """
        if self.planner_mode == 'legacy':
            return self._plan_legacy(degradation_vector)
        return self._plan_hybrid(degradation_vector, image_path=image_path)

if __name__ == "__main__":
    planner = TaskPlanner()
    degradation = [0.8, 0.2, 0.7]
    plan = planner.plan(np.array(degradation))
    print(f"Plan: {plan}")