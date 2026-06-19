import json
import os
import importlib
import gc
from typing import Any
from pathlib import Path

from PIL import Image


class QwenVLPlanner:
    def __init__(self):
        project_root = Path(__file__).resolve().parent
        local_model_dir = project_root / 'models' / 'Qwen2.5-VL-7B-Instruct'
        default_model_id = str(local_model_dir) if local_model_dir.exists() else 'Qwen/Qwen2.5-VL-7B-Instruct'
        self.model_id = os.getenv('WEATHER_VLM_MODEL_ID', default_model_id)
        torch = importlib.import_module('torch')
        self.device = os.getenv('WEATHER_VLM_INPUT_DEVICE', 'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device_map = os.getenv('WEATHER_VLM_DEVICE_MAP', 'auto').strip().lower()
        self.max_new_tokens = int(os.getenv('WEATHER_VLM_MAX_NEW_TOKENS', '256'))
        self.temperature = float(os.getenv('WEATHER_VLM_TEMPERATURE', '0.0'))
        self.unload_after_plan = os.getenv('WEATHER_VLM_UNLOAD_AFTER_PLAN', '1').strip().lower() not in {'0', 'false', 'no'}
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None and self._processor is not None:
            return self._model, self._processor

        torch = importlib.import_module('torch')
        transformers = importlib.import_module('transformers')
        AutoProcessor = getattr(transformers, 'AutoProcessor')
        qwen_vl_cls = getattr(transformers, 'Qwen2_5_VLForConditionalGeneration', None)
        AutoModelForVision2Seq = getattr(transformers, 'AutoModelForVision2Seq', None)

        model_kwargs: dict[str, Any] = {
            'torch_dtype': torch.float16 if torch.cuda.is_available() else torch.float32,
        }
        # Use explicit single-device placement when WEATHER_VLM_DEVICE_MAP is disabled.
        if self.device_map not in {'', 'none', 'null', 'off', 'false', '0'}:
            model_kwargs['device_map'] = self.device_map
        if qwen_vl_cls is not None:
            self._model = qwen_vl_cls.from_pretrained(self.model_id, **model_kwargs)
        elif AutoModelForVision2Seq is not None:
            # Compatibility path for older transformers (e.g. 4.45.x).
            self._model = AutoModelForVision2Seq.from_pretrained(self.model_id, **model_kwargs)
        else:
            raise RuntimeError('No available VL model loader found in transformers; please upgrade transformers.')
        if 'device_map' not in model_kwargs and hasattr(self._model, 'to'):
            self._model = self._model.to(self.device)
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        return self._model, self._processor

    def _unload(self):
        torch = importlib.import_module('torch')
        if self._model is not None:
            try:
                self._model = self._model.cpu()
            except Exception:
                pass
        self._model = None
        self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _extract_json(text: str) -> dict:
        text = text.strip()
        if not text:
            return {}

        try:
            return json.loads(text)
        except Exception:
            pass

        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                return {}
        return {}

    @staticmethod
    def _active_tasks_to_degradations(active_tasks: list[str]) -> list[str]:
        step_to_deg = {
            'desnow': 'snow',
            'derain': 'rain',
            'dehaze': 'haze',
        }
        degradations = []
        for step in active_tasks:
            deg = step_to_deg.get(step)
            if deg and deg not in degradations:
                degradations.append(deg)
        return degradations

    def build_replan_io_definition(self) -> dict:
        """
        定义回滚重规划接口（当前仅定义，不在主流程中启用）。
        """
        return {
            'enabled': False,
            'input': {
                'D_I': 'degradation list (list[str])',
                'A_I_R': 'remaining restoration task set (list[str])',
                'S_I': 'failed step info (dict or str)',
            },
            'output': {
                'P_I_adj': 'adjusted ordered restoration plan (list[str])',
            },
        }

    def build_replan_prompt(self, image_description: str, degradations: list[str], remaining_tasks: list[str],
                            failed_tries: list[str], experience: str = '') -> str:
        """
        仅提供模板，不直接在现有流程中调用。
        """
        return (
            'You are an expert in weather image restoration planning. Given a low-quality image, '
            'your task is to generate an adjusted restoration plan after rollback. '
            'The final output must be a JSON object with key "plan" only.\n\n'
            'Information about the input image:\n'
            f'Its description is: {image_description} (C_I),\n'
            f'It suffers from degradations {json.dumps(degradations, ensure_ascii=False)} (D_I),\n'
            f'The remaining restoration tasks are: {json.dumps(remaining_tasks, ensure_ascii=False)} (A_I^R),\n'
            f'Failure step information is: {json.dumps(failed_tries, ensure_ascii=False)} (S_I),\n'
            f'Restoration experience: {experience} (E).\n\n'
            'Please provide the corrected order of remaining tasks. '
            'The plan must be a permutation/subset of remaining tasks only and should avoid placing failed_tries first. '
            'Do not output any explanation. Strictly return only JSON with key "plan".'
        )

    def build_prompt(self, C_I: str, D_I: list[str], A_I: list[str], I: str,
                     allowed_steps: list[str] | None = None,
                     restoration_experience: str | None = None,
                     perception_probabilities: dict | None = None) -> str:
        allowed_steps = allowed_steps or []
        restoration_experience = restoration_experience or os.getenv(
            'WEATHER_PLANNER_EXPERIENCE',
            'Prefer removing snow/rain before dehaze when both exist; keep plans short and deterministic.'
        )
        perception_probabilities = perception_probabilities or {}

        return (
            'You are an expert in weather image restoration planning. '\
            'Follow the explicit 4KAgent-style schema below and return a JSON plan only.\n\n'
            'Explicit inputs:\n'
            f'C_I: {C_I}\n'
            f'D_I: {json.dumps(D_I, ensure_ascii=False)}\n'
            f'A_I: {json.dumps(A_I, ensure_ascii=False)}\n'
            f'I: {I}\n'
            f'Allowed tasks: {json.dumps(allowed_steps, ensure_ascii=False)}\n'
            f'Restoration experience: {restoration_experience}\n\n'
            'Task: generate the ordered restoration plan using only tasks in A_I and only tasks in Allowed tasks. '
            'You may choose to only order the current candidate tasks, and you may omit tasks that you consider unnecessary, but avoid adding new tasks whenever possible. '
            'The output must be a JSON object with the single key "plan". '
            'Do not output any explanation, analysis, or extra text.'
        )

    def plan(self, image_path: str, C_I: str | None = None, D_I: list[str] | None = None,
             A_I: list[str] | None = None, I: str | None = None, prompt: str | None = None,
             allowed_steps: list[str] | None = None, perception_info: dict | None = None,
             direct_plan: list[str] | None = None, clip_info: dict | None = None) -> dict:
        torch = importlib.import_module('torch')
        try:
            model, processor = self._load()
            # clip_info is kept as a backward-compatible alias.
            if perception_info is None:
                perception_info = clip_info or {}
            if allowed_steps is None:
                allowed_steps = []

            if prompt is None:
                if C_I is None:
                    C_I = str(perception_info.get('image_description', '')).strip()
                if D_I is None:
                    active_tasks = perception_info.get('tasks', []) if isinstance(perception_info.get('tasks', []), list) else []
                    D_I = self._active_tasks_to_degradations(active_tasks)
                if A_I is None:
                    if direct_plan is not None:
                        A_I = list(direct_plan)
                    else:
                        A_I = list(perception_info.get('tasks', [])) if isinstance(perception_info.get('tasks', []), list) else []
                if I is None:
                    I = image_path
                prompt = self.build_prompt(
                    C_I=C_I or '',
                    D_I=D_I or [],
                    A_I=A_I or [],
                    I=I or image_path,
                    allowed_steps=allowed_steps,
                    perception_probabilities=perception_info.get('probabilities', {}),
                )

            image = Image.open(image_path).convert('RGB')
            messages = [
                {
                    'role': 'system',
                    'content': 'You are a precise visual restoration planner. Output valid JSON only.'
                },
                {
                    'role': 'user',
                    'content': [
                        {'type': 'image'},
                        {'type': 'text', 'text': prompt},
                    ],
                },
            ]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], padding=True, return_tensors='pt')
            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

            generate_kwargs = {
                'max_new_tokens': self.max_new_tokens,
                'do_sample': self.temperature > 0,
            }
            if self.temperature > 0:
                generate_kwargs['temperature'] = self.temperature

            with torch.no_grad():
                generated_ids = model.generate(**inputs, **generate_kwargs)
            prompt_len = inputs['input_ids'].shape[1]
            trimmed_ids = generated_ids[:, prompt_len:]
            output_text = processor.batch_decode(trimmed_ids, skip_special_tokens=True)[0]
            result = self._extract_json(output_text)

            plan = result.get('plan', []) if isinstance(result, dict) else []
            if not isinstance(plan, list):
                plan = []
            filtered_plan = [step for step in plan if step in allowed_steps]

            return {
                'plan': filtered_plan,
                'replan_io_definition': self.build_replan_io_definition(),
                'raw_text': output_text,
                'model_id': self.model_id,
            }
        finally:
            if self.unload_after_plan:
                self._unload()
