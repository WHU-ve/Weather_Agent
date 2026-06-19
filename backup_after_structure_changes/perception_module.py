"""
Perception module based on Llama-3.2-Vision for adverse weather degradation analysis.

Primary interface:
    predict_degradation(image_path) -> JSON dict
The historical vector helper is kept only for backward compatibility with older code paths.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import re
import sys
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image
import torch

_project_root = Path(__file__).resolve().parent
_local_llama_root = Path(
    os.getenv('WEATHER_PERCEPTION_MODEL_DIR', str(_project_root / 'models' / 'Llama-3.2-Vision-11B-Instruct'))
).resolve()
_default_modelscope_id = 'LLM-Research/Llama-3.2-11B-Vision-Instruct'

_device = None
_model = None
_processor = None

# Restrict the perception task space to rain/haze/snow.
_labels = ['rain', 'haze', 'snow']

_DEGRADATION_SCORE = {
    'rain': 0.75,
    'haze': 0.75,
    'snow': 0.75,
}

_PERCEPTION_PROMPT_TEMPLATE = """You are an expert in image quality assessment (IQA) and adverse weather degradation analysis. You are given an input image along with a set of quality-related metrics. The metrics include:
General perceptual quality metrics: MANIQA (higher is better)、CLIPIQA (higher is better)、TOPIQ-NR (higher is better)、NIQE (lower is better).
Weather-specific quality indicators: Rain residual score (lower streak energy → better rain removal)、Fog density score / FADE (lower means less fog)、Local contrast (higher means better visibility)、Snow artifact score (isolated bright spots — higher means more snow-like artifacts)、Detail score (P90 gradient magnitude — higher means stronger edges)、Texture retention (mean local std — higher means more texture preserved). Typically, high Rain residual score suggests rain degradation, high Snow artifact score suggests snow degradation. When cues conflict, output all plausible degradations rather than forcing a single label.
Your task consists of two steps:
Step 1: Describe the content and scene of the image (e.g., indoor/outdoor, objects, environment, weather context). Do NOT mention image quality in this step.
Step 2: Based on both the provided metrics and your visual reasoning, identify the degradation types affecting the image.
Possible degradations include: rain、haze、snow.
IMPORTANT: Use both metric signals AND visual understanding.
Output strictly in JSON format with the following keys:
{{
    "degradations": ["...", "..."],
    "image_description": "..."
}}
Information about the input image: IQA metrics:
{iqa_result}"""


def _parse_candidate_gpu_ids() -> list[int]:
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
        return []

    raw = os.getenv('WEATHER_PERCEPTION_CANDIDATE_GPU_IDS', '').strip()
    if not raw:
        return list(range(torch.cuda.device_count()))

    ids: list[int] = []
    max_idx = torch.cuda.device_count() - 1
    for token in raw.split(','):
        token = token.strip()
        if not token:
            continue
        try:
            idx = int(token)
        except ValueError:
            continue
        if 0 <= idx <= max_idx and idx not in ids:
            ids.append(idx)
    return ids if ids else list(range(torch.cuda.device_count()))


def _rank_candidate_cuda_devices() -> list[int]:
    ids = _parse_candidate_gpu_ids()
    scored = []
    for idx in ids:
        try:
            free_mem, total_mem = torch.cuda.mem_get_info(idx)
            scored.append((idx, free_mem, total_mem))
        except Exception:
            continue
    if not scored:
        return ids
    scored.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _free, _total in scored]


def _pick_best_cuda_device() -> str:
    ranked = _rank_candidate_cuda_devices()
    if not ranked:
        return 'cpu'
    return f'cuda:{ranked[0]}'


def _resolve_device():
    requested = os.getenv('WEATHER_PERCEPTION_DEVICE', os.getenv('WEATHER_CLIP_DEVICE', '')).strip().lower()
    if requested:
        if requested.startswith('cuda') and not torch.cuda.is_available():
            return 'cpu'
        return requested
    return _pick_best_cuda_device() if torch.cuda.is_available() else 'cpu'


def _resolve_model_id() -> str:
    default_id = str(_local_llama_root) if _local_llama_root.exists() else _default_modelscope_id
    return os.getenv('WEATHER_PERCEPTION_MODEL_ID', default_id)


def _resolve_model_source() -> str:
    # auto: local path if exists; otherwise use WEATHER_PERCEPTION_MODEL_ID as-is.
    # modelscope: resolve model from ModelScope hub and cache locally.
    default_source = 'auto' if _local_llama_root.exists() else 'modelscope'
    return os.getenv('WEATHER_PERCEPTION_MODEL_SOURCE', default_source).strip().lower()


def _resolve_offline_mode() -> bool:
    # Default to offline-first inference for stability and reproducibility.
    return os.getenv('WEATHER_PERCEPTION_OFFLINE', '1').strip().lower() not in {'0', 'false', 'no'}


@lru_cache(maxsize=1)
def _resolve_model_location() -> str:
    model_source = _resolve_model_source()
    model_id = _resolve_model_id()

    # Local path always wins when present.
    p = Path(model_id)
    if p.exists():
        return str(p.resolve())

    if _resolve_offline_mode():
        raise FileNotFoundError(
            f'Perception model path not found in offline mode: {model_id}. '
            'Please place model files locally or set WEATHER_PERCEPTION_OFFLINE=0 explicitly.'
        )

    if model_source in {'auto', 'modelscope'}:
        try:
            from modelscope.hub.snapshot_download import snapshot_download
        except Exception as exc:
            raise ImportError('需要安装 modelscope 才能从 ModelScope 下载模型：pip install modelscope') from exc

        ms_model_id = os.getenv('WEATHER_PERCEPTION_MODELSCOPE_ID', model_id or _default_modelscope_id).strip()
        # Download only files needed for transformers inference to avoid very large optional originals.
        local_dir = snapshot_download(
            ms_model_id,
            allow_patterns=[
                'config.json',
                'generation_config.json',
                'model.safetensors.index.json',
                'model-*.safetensors',
                'tokenizer.json',
                'tokenizer_config.json',
                'special_tokens_map.json',
                'chat_template.json',
                'preprocessor_config.json',
            ],
            ignore_patterns=['original/*'],
        )
        return str(Path(local_dir).resolve())

    return model_id


def _resolve_quantization_mode() -> str:
    # Disable bitsandbytes quantization by default (force full-precision / torch native path).
    default_mode = 'off'
    return os.getenv('WEATHER_PERCEPTION_QUANTIZATION', default_mode).strip().lower()


def _is_bitsandbytes_cuda_usable() -> bool:
    if not torch.cuda.is_available():
        return False
    cuda_version = (torch.version.cuda or '').replace('.', '')
    if not cuda_version:
        return False
    try:
        import bitsandbytes as bnb  # type: ignore
    except Exception:
        return False
    lib_path = Path(bnb.__file__).resolve().parent / f'libbitsandbytes_cuda{cuda_version}.so'
    return lib_path.exists()


def _resolve_device_map(device: str):
    mode = os.getenv('WEATHER_PERCEPTION_DEVICE_MAP', 'auto').strip().lower()
    if mode == 'auto' and torch.cuda.is_available():
        return 'auto'
    if mode in {'', 'none', 'null', 'off', 'false', '0'}:
        return {'': device}
    return mode


def _resolve_torch_dtype():
    if not torch.cuda.is_available():
        return torch.float32
    mode = os.getenv('WEATHER_PERCEPTION_TORCH_DTYPE', 'fp16').strip().lower()
    if mode in {'fp16', 'float16'}:
        return torch.float16
    if mode in {'fp32', 'float32'}:
        return torch.float32
    return torch.bfloat16


def _resolve_metric_device() -> str:
    return os.getenv('WEATHER_PERCEPTION_PYIQA_DEVICE', 'cpu').strip().lower() or 'cpu'


def _enable_pyiqa_metrics() -> bool:
    # Default off to avoid external timm/huggingface weight pulls during perception.
    return os.getenv('WEATHER_PERCEPTION_ENABLE_PYIQA', '0').strip().lower() in {'1', 'true', 'yes', 'on'}


def _apply_warning_controls() -> None:
    if os.getenv('WEATHER_SUPPRESS_RUNTIME_WARNINGS', '1').strip().lower() not in {'1', 'true', 'yes', 'on'}:
        return
    warnings.filterwarnings('ignore', message='The model weights are not tied.*')
    warnings.filterwarnings('ignore', message="We've detected an older driver with an RTX 4000 series GPU.*")
    logging.getLogger('transformers').setLevel(logging.ERROR)
    logging.getLogger('accelerate').setLevel(logging.ERROR)


@lru_cache(maxsize=1)
def _metric_factory(metric_name: str):
    if not _enable_pyiqa_metrics():
        return None
    try:
        import pyiqa

        return pyiqa.create_metric(metric_name, device=_resolve_metric_device())
    except Exception:
        return None


@lru_cache(maxsize=1)
def _get_model_and_processor():
    global _device, _model, _processor
    if _model is not None and _processor is not None:
        return _model, _processor, _device

    _apply_warning_controls()

    _device = _resolve_device()
    model_id = _resolve_model_location()
    offline_mode = _resolve_offline_mode()

    if offline_mode:
        os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
        os.environ.setdefault('HF_HUB_OFFLINE', '1')
        os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
    quantization_mode = _resolve_quantization_mode()
    if quantization_mode in {'4bit', '8bit'} and not _is_bitsandbytes_cuda_usable():
        quantization_mode = 'off'

    try:
        transformers = __import__('transformers', fromlist=['AutoProcessor'])
    except ImportError as exc:
        raise ImportError('需要安装 transformers 才能加载 Llama-3.2-Vision') from exc

    AutoProcessor = getattr(transformers, 'AutoProcessor')
    model_cls = None
    for candidate in (
        'MllamaForConditionalGeneration',
        'AutoModelForImageTextToText',
        'AutoModelForVision2Seq',
        'AutoModelForCausalLM',
    ):
        model_cls = getattr(transformers, candidate, None)
        if model_cls is not None:
            break
    if model_cls is None:
        raise ImportError('当前 transformers 版本不支持 Llama-Vision 模型类，请升级 transformers')

    model_kwargs: Dict[str, Any] = {
        'torch_dtype': _resolve_torch_dtype(),
        'low_cpu_mem_usage': True,
        'device_map': _resolve_device_map(_device),
        'local_files_only': offline_mode,
    }

    if quantization_mode in {'4bit', '8bit'}:
        try:
            BitsAndBytesConfig = getattr(transformers, 'BitsAndBytesConfig')
            model_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=(quantization_mode == '4bit'),
                load_in_8bit=(quantization_mode == '8bit'),
            )
        except Exception:
            # Fall back to regular loading when bitsandbytes is unavailable.
            model_kwargs.pop('quantization_config', None)

    try:
        _model = model_cls.from_pretrained(model_id, **model_kwargs)
    except Exception:
        # If quantized loading fails (e.g., bitsandbytes runtime mismatch), retry without quantization.
        if 'quantization_config' not in model_kwargs:
            raise
        retry_kwargs = dict(model_kwargs)
        retry_kwargs.pop('quantization_config', None)
        retry_kwargs['device_map'] = 'auto'
        retry_kwargs['offload_state_dict'] = True
        retry_kwargs['offload_folder'] = str((_project_root / '.offload').resolve())
        _model = model_cls.from_pretrained(model_id, **retry_kwargs)
    if hasattr(_model, 'eval'):
        _model.eval()
    _processor = AutoProcessor.from_pretrained(model_id, local_files_only=offline_mode)
    return _model, _processor, _device


def release_model():
    global _device, _model, _processor
    if _model is not None:
        try:
            _model = _model.cpu()
        except Exception:
            pass
    _model = None
    _processor = None
    _device = None
    _metric_factory.cache_clear()
    _get_model_and_processor.cache_clear()
    _resolve_model_location.cache_clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _to_tensor(image: Image.Image) -> torch.Tensor:
    try:
        from torchvision import transforms

        return transforms.ToTensor()(image).unsqueeze(0)
    except Exception:
        arr = np.asarray(image, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr).unsqueeze(0)


def _safe_metric(metric_name: str, image_tensor: torch.Tensor) -> Optional[float]:
    metric = _metric_factory(metric_name)
    if metric is None:
        return None
    try:
        with torch.no_grad():
            value = metric(image_tensor.to(_device)).item()
        return float(value)
    except Exception:
        return None


def _estimate_weather_indicators(image: Image.Image) -> Dict[str, float]:
    arr = np.asarray(image.convert('RGB'), dtype=np.float32) / 255.0
    gray = arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114

    gy, gx = np.gradient(gray)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    local_contrast = float(np.clip(gray.std() * 4.0, 0.0, 1.0))
    bright_ratio = float(np.mean(gray > 0.92))
    edge_density = float(np.mean(grad_mag > np.percentile(grad_mag, 75)))

    # Heuristic weather indicators used only to enrich the prompt when external metrics are unavailable.
    rain_streak_energy = float(np.clip((np.mean(np.abs(gx)) / (np.mean(np.abs(gy)) + 1e-6)) * 0.5 + edge_density * 0.3, 0.0, 1.0))
    fog_density = float(np.clip(1.0 - local_contrast + bright_ratio * 0.25, 0.0, 1.0))
    fade = fog_density

    # detail_score: P90 gradient magnitude — higher means stronger edges.
    detail_score = float(np.clip(np.percentile(grad_mag, 90) * 6.0, 0.0, 1.0))

    # snow_artifact: isolated bright spots (snow-like) vs large bright areas.
    bright_mask = (gray > 0.92).astype(np.float32)
    h, w = gray.shape
    neighbor_sum = np.zeros_like(bright_mask)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            shifted = np.zeros_like(bright_mask)
            si0, si1 = max(0, di), min(h, h + di)
            di0, di1 = max(0, -di), min(h, h - di)
            sj0, sj1 = max(0, dj), min(w, w + dj)
            dj0, dj1 = max(0, -dj), min(w, w - dj)
            shifted[si0:si1, sj0:sj1] = bright_mask[di0:di1, dj0:dj1]
            neighbor_sum += shifted
    neighbor_ratio = neighbor_sum / 8.0
    isolated_bright = np.mean((bright_mask > 0.5) & (neighbor_ratio < 0.30))
    snow_artifact = float(np.clip(isolated_bright * 5.0, 0.0, 1.0))

    # texture_retention: mean local std in 5x5 patches.
    patch_size = 5
    pad = patch_size // 2
    padded = np.pad(gray, pad, mode='reflect')
    patches = np.lib.stride_tricks.sliding_window_view(padded, (patch_size, patch_size))
    local_std_map = np.std(patches, axis=(-2, -1))
    texture_retention = float(np.clip(np.mean(local_std_map) * 10.0, 0.0, 1.0))

    return {
        'rain_residual_score': rain_streak_energy,
        'fog_density_score': fog_density,
        'fade': fade,
        'local_contrast': local_contrast,
        'snow_artifact': snow_artifact,
        'detail_score': detail_score,
        'texture_retention': texture_retention,
    }


def _build_iqa_result(image: Image.Image, iqa_result: Optional[str] = None) -> str:
    # Keep the argument only for backward compatibility. IQA text is always computed internally.
    _ = iqa_result

    image_tensor = _to_tensor(image)
    maniqa = _safe_metric('maniqa', image_tensor)
    clipiqa = _safe_metric('clipiqa+', image_tensor)
    topiq = _safe_metric('topiq_nr', image_tensor)
    niqe = _safe_metric('niqe', image_tensor)
    weather_indicators = _estimate_weather_indicators(image)

    def _fmt(value: Optional[float]) -> str:
        return 'N/A' if value is None else f'{value:.4f}'

    return (
        f'MANIQA={_fmt(maniqa)}, '
        f'CLIPIQA={_fmt(clipiqa)}, '
        f'TOPIQ-NR={_fmt(topiq)}, '
        f'NIQE={_fmt(niqe)}, '
        f"Rain residual score={weather_indicators['rain_residual_score']:.4f}, "
        f"Fog density score/FADE={weather_indicators['fog_density_score']:.4f}, "
        f"Local contrast={weather_indicators['local_contrast']:.4f}, "
        f"Snow artifact score={weather_indicators['snow_artifact']:.4f}, "
        f"Detail score={weather_indicators['detail_score']:.4f}, "
        f"Texture retention={weather_indicators['texture_retention']:.4f}"
    )


def _build_prompt(image: Image.Image, iqa_result: Optional[str] = None) -> str:
    metrics_text = _build_iqa_result(image, iqa_result=iqa_result)
    return _PERCEPTION_PROMPT_TEMPLATE.format(iqa_result=metrics_text)


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        return {}

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r'\{.*\}', text, flags=re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return {}
    return {}


def _response_to_vector(result: Dict[str, Any]) -> np.ndarray:
    scores = {label: 0.0 for label in _labels}
    degradations = result.get('degradations', []) if isinstance(result, dict) else []
    if isinstance(degradations, list):
        for item in degradations:
            if isinstance(item, dict):
                degradation_type = str(item.get('type', '')).strip().lower()
            else:
                degradation_type = str(item).strip().lower()
            if degradation_type in scores:
                scores[degradation_type] = max(scores[degradation_type], _DEGRADATION_SCORE.get(degradation_type, 0.0))
    return np.array([scores['rain'], scores['haze'], scores['snow']], dtype=np.float32)


def _fallback_vector(image: Image.Image) -> np.ndarray:
    indicators = _estimate_weather_indicators(image)
    rain = float(np.clip(indicators['rain_residual_score'], 0.0, 1.0))
    haze = float(np.clip(indicators['fog_density_score'], 0.0, 1.0))
    snow = float(np.clip(indicators['snow_artifact'] * 0.8 + (1.0 - indicators['detail_score']) * 0.2, 0.0, 1.0))
    return np.array([rain, haze, snow], dtype=np.float32)


def _fallback_json(image: Image.Image) -> Dict[str, Any]:
    vec = _fallback_vector(image)
    degradations = []
    for name, score in [('rain', float(vec[0])), ('haze', float(vec[1])), ('snow', float(vec[2]))]:
        if score >= 0.25:
            degradations.append(name)
    return {
        'degradations': degradations,
        'image_description': 'Fallback perception output from heuristic weather indicators.',
    }


def _debug_fallback_reason(reason: str):
    if os.getenv('WEATHER_PERCEPTION_DEBUG_FALLBACK', '0').strip().lower() in {'1', 'true', 'yes', 'on'}:
        print(f'[weather_agent] perception fallback: {reason}', file=sys.stderr)


def _normalize_result_json(result: Dict[str, Any]) -> Dict[str, Any]:
    degradations = result.get('degradations', []) if isinstance(result, dict) else []
    image_description = result.get('image_description', '') if isinstance(result, dict) else ''

    normalized = []
    alias = {
        'fog': 'haze',
        'smog': 'haze',
        'mist': 'haze',
        'rainy': 'rain',
        'snowy': 'snow',
    }
    valid = {'rain', 'haze', 'snow'}
    if isinstance(degradations, list):
        for item in degradations:
            if isinstance(item, dict):
                dtype = str(item.get('type', '')).strip().lower()
            else:
                dtype = str(item).strip().lower()
            dtype = alias.get(dtype, dtype)
            if dtype not in valid:
                continue
            if dtype not in normalized:
                normalized.append(dtype)

    return {
        'degradations': normalized,
        'image_description': str(image_description).strip(),
    }


def _generate_and_normalize(
    model,
    processor,
    inputs: Dict[str, Any],
    max_new_tokens: int,
    use_cache: bool = True,
) -> Dict[str, Any]:
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            use_cache=use_cache,
        )
    prompt_len = inputs['input_ids'].shape[1]
    trimmed_ids = generated_ids[:, prompt_len:]
    output_text = processor.batch_decode(trimmed_ids, skip_special_tokens=True)[0]
    result = _extract_json(output_text)
    return _normalize_result_json(result)


def _build_inputs(
    processor,
    image: Image.Image,
    prompt: str,
    device: str,
    max_image_edge: Optional[int] = None,
) -> Dict[str, Any]:
    work_image = image
    if max_image_edge is not None and max_image_edge > 0:
        w, h = image.size
        longest = max(w, h)
        if longest > max_image_edge:
            scale = max_image_edge / float(longest)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            work_image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)

    messages = [
        {
            'role': 'system',
            'content': 'You are a precise image perception model for adverse weather analysis. Output valid JSON only.'
        },
        {
            'role': 'user',
            'content': [
                {'type': 'image'},
                {'type': 'text', 'text': prompt},
            ],
        },
    ]

    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat_text], images=[work_image], padding=True, return_tensors='pt')
    return {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}


def predict_degradation(image_path, iqa_result: Optional[str] = None):
    """
    Use Llama-3.2-Vision to analyze the image and return structured JSON.

    Note:
        IQA metrics are always computed internally by this module.
        The `iqa_result` argument is kept only for backward compatibility and is ignored.

    Returns:
        dict with keys:
            - degradations: ["rain|haze|snow", ...]
            - image_description: str
    """
    original_device_env = os.getenv('WEATHER_PERCEPTION_DEVICE', '').strip()
    original_device_map_env = os.getenv('WEATHER_PERCEPTION_DEVICE_MAP', '').strip()

    force_single_gpu = os.getenv('WEATHER_PERCEPTION_FORCE_SINGLE_GPU', '0').strip().lower() in {'1', 'true', 'yes'}
    release_after_infer = os.getenv('WEATHER_PERCEPTION_RELEASE_AFTER_INFER', '1').strip().lower() in {'1', 'true', 'yes'}

    if torch.cuda.is_available():
        if force_single_gpu:
            os.environ['WEATHER_PERCEPTION_DEVICE_MAP'] = 'none'
            ranked = _rank_candidate_cuda_devices()
            if ranked:
                os.environ['WEATHER_PERCEPTION_DEVICE'] = f'cuda:{ranked[0]}'
        else:
            if not original_device_map_env:
                os.environ['WEATHER_PERCEPTION_DEVICE_MAP'] = 'auto'

    try:
        model, processor, device = _get_model_and_processor()
        image = Image.open(image_path).convert('RGB')
        prompt = _build_prompt(image, iqa_result=iqa_result)

        primary_tokens = int(os.getenv('WEATHER_PERCEPTION_MAX_NEW_TOKENS', '256'))
        retry_tokens = int(os.getenv('WEATHER_PERCEPTION_RETRY_MAX_NEW_TOKENS', '64'))
        retry_tokens = max(16, min(retry_tokens, primary_tokens))

        primary_max_edge = int(os.getenv('WEATHER_PERCEPTION_MAX_IMAGE_EDGE', '0'))
        retry_max_edge = int(os.getenv('WEATHER_PERCEPTION_RETRY_MAX_IMAGE_EDGE', '896'))

        inputs = _build_inputs(
            processor,
            image,
            prompt,
            device,
            max_image_edge=primary_max_edge if primary_max_edge > 0 else None,
        )

        try:
            normalized = _generate_and_normalize(
                model=model,
                processor=processor,
                inputs=inputs,
                max_new_tokens=primary_tokens,
                use_cache=True,
            )
            if normalized:
                return normalized
            _debug_fallback_reason('normalized_result_empty')
        except torch.cuda.OutOfMemoryError as exc:
            _debug_fallback_reason(f'generate_oom_primary:{exc}')
            retry_plans = [
                (retry_tokens, retry_max_edge if retry_max_edge > 0 else None),
                (32, 640),
                (24, 512),
                (16, 384),
            ]
            seen = set()

            candidate_devices = [device]
            if force_single_gpu and device.startswith('cuda'):
                ranked = _rank_candidate_cuda_devices()
                if ranked:
                    candidate_devices = [f'cuda:{idx}' for idx in ranked]

            for cur_tokens, cur_edge in retry_plans:
                key = (int(cur_tokens), int(cur_edge) if cur_edge is not None else -1)
                if key in seen:
                    continue
                seen.add(key)

                for dev_try in candidate_devices:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    try:
                        if dev_try != device and force_single_gpu:
                            _debug_fallback_reason(f'reload_model_on_retry_device:{dev_try}')
                            release_model()
                            os.environ['WEATHER_PERCEPTION_DEVICE'] = dev_try
                            os.environ['WEATHER_PERCEPTION_DEVICE_MAP'] = 'none'
                            model, processor, device = _get_model_and_processor()

                        retry_inputs = _build_inputs(
                            processor,
                            image,
                            prompt,
                            device,
                            max_image_edge=cur_edge,
                        )
                        normalized = _generate_and_normalize(
                            model=model,
                            processor=processor,
                            inputs=retry_inputs,
                            max_new_tokens=int(cur_tokens),
                            use_cache=False,
                        )
                        if normalized:
                            _debug_fallback_reason(
                                f'oom_recovered(tokens={cur_tokens},edge={cur_edge},device={device})'
                            )
                            return normalized
                        _debug_fallback_reason(
                            f'normalized_result_empty_after_oom_retry(tokens={cur_tokens},edge={cur_edge},device={device})'
                        )
                    except torch.cuda.OutOfMemoryError as retry_oom:
                        _debug_fallback_reason(
                            f'generate_retry_oom(tokens={cur_tokens},edge={cur_edge},device={dev_try}):{retry_oom}'
                        )
                    except Exception as retry_exc:
                        _debug_fallback_reason(
                            f'generate_retry_exception(tokens={cur_tokens},edge={cur_edge},device={dev_try}):'
                            f'{type(retry_exc).__name__}:{retry_exc}'
                        )
        except Exception as exc:
            _debug_fallback_reason(f'generate_or_parse_exception:{type(exc).__name__}:{exc}')

        _debug_fallback_reason('using_heuristic_fallback')
        return _fallback_json(image)
    finally:
        if original_device_env:
            os.environ['WEATHER_PERCEPTION_DEVICE'] = original_device_env
        else:
            os.environ.pop('WEATHER_PERCEPTION_DEVICE', None)

        if original_device_map_env:
            os.environ['WEATHER_PERCEPTION_DEVICE_MAP'] = original_device_map_env
        else:
            os.environ.pop('WEATHER_PERCEPTION_DEVICE_MAP', None)

        if release_after_infer:
            release_model()




def predict_degradation_vector(image_path, iqa_result: Optional[str] = None, result: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Compatibility helper that converts perception JSON into the historical 4D vector.
    """
    if result is None:
        result = predict_degradation(image_path, iqa_result=iqa_result)
    if not isinstance(result, dict):
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    return _response_to_vector(_normalize_result_json(result))


# Compatibility wrapper kept for older call sites.
def load_model(model_path=None):
    return _get_model_and_processor()[0]
