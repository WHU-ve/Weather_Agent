"""Quality evaluator for multi-expert selection.

Default policy implemented in this file:
1) All metrics are mapped to [0,1] with fixed L/U bounds.
2) Task priors remain:
     - derain: 0.55*RainResidual + 0.25*(1-LocalContrast) + 0.20*(1-EdgeContinuity)
     - dehaze: 0.55*FogDensity   + 0.30*(1-LocalContrast) + 0.15*(1-EdgeContinuity)
     - desnow: 0.55*BrightArtifactRatio + 0.30*(1-EdgeContinuity) + 0.15*(1-LocalContrast)
3) Reliability modulation is bounded to +/-20% by default.
4) Fusion uses alpha=0.8 by default: Final = alpha*Task + (1-alpha)*General.
5) Tie-break uses epsilon=0.02: compare Task score first, then task main metric.
6) Verbose mode prints contribution percentages for later calibration.

L/U bounds below are from spread-sampled calibration with 15% margin expansion.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms


class QualityEvaluator:
    # Shared L/U for general metrics and shared structure metrics.
    SHARED_BOUNDS_15 = {
        'maniqa': (0.14061265103518966, 0.4265043143182992),
        'clipiqa': (0.1837959674373269, 0.5603283370658756),
        'topiq_nr': (0.1679230612516403, 0.48506730973720547),
        'niqe': (3.881363810143916, 9.841173193833804),
        'local_contrast': (-0.024944932628422983, 1.1336884694732725),
        'edge_continuity': (0.8980515179177746, 0.9984523044945672),
    }

    # Task-main L/U (only these are task-specific).
    TASK_MAIN_BOUNDS_15 = {
        'derain': {'rain_residual_score': (0.43104312752365326, 1.0742117659751758)},
        'dehaze': {'fog_density_score': (0.17350409436970948, 1.0234778311476112)},
        'desnow': {'bright_artifact_ratio': (-0.011568946838378908, 0.0886952590942383)},
    }

    TASK_MAIN_KEY = {
        'derain': 'rain_residual_score',
        'dehaze': 'fog_density_score',
        'desnow': 'bright_artifact_ratio',
    }

    TASK_BY_MAIN_KEY = {v: k for k, v in TASK_MAIN_KEY.items()}

    def __init__(self, weights=None, normalize=True, norm_method='minmax'):
        _ = weights  # kept for backward compatibility of constructor signature
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.normalize = bool(normalize)
        self.norm_method = norm_method
        self.verbose_selection = os.getenv('QE_VERBOSE_SELECTION', '0').strip().lower() in {'1', 'true', 'yes'}

        # Strict offline mode: never pull weights from network at runtime.
        self.strict_offline = os.getenv('QE_STRICT_OFFLINE', '1').strip().lower() in {'1', 'true', 'yes'}
        if self.strict_offline:
            os.environ.setdefault('HF_HUB_OFFLINE', '1')
            os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
            os.environ.setdefault('HF_DATASETS_OFFLINE', '1')

        # Fusion defaults. Each task uses the alpha selected by WeatherBench train100 sweep; env vars can still override them.
        self.alpha_task = float(np.clip(float(os.getenv('QE_ALPHA_TASK', '0.80')), 0.0, 1.0))
        self.alpha_by_task = {
            'derain': float(np.clip(float(os.getenv('QE_ALPHA_DERAIN', '0.40')), 0.0, 1.0)),
            'dehaze': float(np.clip(float(os.getenv('QE_ALPHA_DEHAZE', '0.80')), 0.0, 1.0)),
            'desnow': float(np.clip(float(os.getenv('QE_ALPHA_DESNOW', '0.00')), 0.0, 1.0)),
        }
        self.general_weight = float(np.clip(1.0 - self.alpha_task, 0.0, 1.0))
        self.task_weight = self.alpha_task

        # Reliability modulation band (+/-20% by default).
        self.reliability_band = float(np.clip(float(os.getenv('QE_RELIABILITY_BAND', '0.20')), 0.0, 0.5))

        # Tie-break threshold.
        self.tie_epsilon = float(max(0.0, float(os.getenv('QE_TIE_EPS', '0.02'))))

        # Active bounds are mutable and can be updated by auto recalibration.
        self.shared_bounds = {k: (float(v[0]), float(v[1])) for k, v in self.SHARED_BOUNDS_15.items()}
        self.task_main_bounds = {
            task: {k: (float(v[0]), float(v[1])) for k, v in bounds.items()}
            for task, bounds in self.TASK_MAIN_BOUNDS_15.items()
        }

        # Auto-recalibration trigger (per metric): clip rate > threshold for consecutive windows.
        self.clip_rate_threshold = float(np.clip(float(os.getenv('QE_CLIP_RATE_THRESHOLD', '0.10')), 0.0, 1.0))
        self.clip_consecutive_windows = int(max(1, int(os.getenv('QE_CLIP_CONSEC_WINDOWS', '2'))))
        self.clip_window_size = int(max(1, int(os.getenv('QE_CLIP_WINDOW_SIZE', '200'))))
        self.recalib_margin = float(np.clip(float(os.getenv('QE_RECALIB_MARGIN', '0.15')), 0.0, 0.5))
        self.recalib_min_samples = int(max(20, int(os.getenv('QE_RECALIB_MIN_SAMPLES', '100'))))
        self.recalib_sample_cap = int(max(self.recalib_min_samples, int(os.getenv('QE_RECALIB_SAMPLE_CAP', '2000'))))
        self._bounds_version = int(max(1, int(os.getenv('QE_BOUNDS_VERSION_START', '1'))))

        default_version_log = Path(__file__).resolve().parent / 'update_log' / 'qe_bounds_versions.jsonl'
        self.bounds_version_log = Path(os.getenv('QE_BOUNDS_VERSION_LOG', str(default_version_log)))
        self.bounds_snapshot_path = self.bounds_version_log.with_name('qe_bounds_latest.json')

        self._metric_history: Dict[str, List[float]] = {}
        self._metric_monitor: Dict[str, Dict[str, object]] = {}
        self._metric_recalib_events: List[Dict[str, object]] = []
        self._init_metric_monitor()

        # Per-task formula weights (default from the agreed template).
        self.w_rain_main = float(os.getenv('QE_RAIN_W_MAIN', '0.55'))
        self.w_rain_lc = float(os.getenv('QE_RAIN_W_LOCAL_CONTRAST', '0.25'))
        self.w_rain_ec = float(os.getenv('QE_RAIN_W_EDGE_CONTINUITY', '0.20'))

        self.w_haze_main = float(os.getenv('QE_HAZE_W_MAIN', '0.55'))
        self.w_haze_lc = float(os.getenv('QE_HAZE_W_LOCAL_CONTRAST', '0.30'))
        self.w_haze_ec = float(os.getenv('QE_HAZE_W_EDGE_CONTINUITY', '0.15'))

        self.w_snow_main = float(os.getenv('QE_SNOW_W_MAIN', '0.55'))
        self.w_snow_ec = float(os.getenv('QE_SNOW_W_EDGE_CONTINUITY', '0.30'))
        self.w_snow_lc = float(os.getenv('QE_SNOW_W_LOCAL_CONTRAST', '0.15'))

        self._pyiqa = None
        self.clipiqa_model = None
        self.niqe_model = None
        self.maniqa_model = None
        self.topiq_model = None
        self._feature_cache: Dict[str, Dict[str, float]] = {}

        try:
            import pyiqa

            self._pyiqa = pyiqa
        except ImportError as exc:
            raise ImportError('需要安装 pyiqa: pip install pyiqa') from exc

    def _ensure_models(self) -> None:
        if self._pyiqa is None:
            raise RuntimeError('pyiqa is unavailable')

        # Strict offline fast-path: do not initialize any pyiqa-backed deep models.
        # This avoids network retries from dependent backbones and keeps evaluator fully local.
        if self.strict_offline:
            self.clipiqa_model = None
            self.niqe_model = None
            self.maniqa_model = None
            self.topiq_model = None
            return

        def _create_metric_safe(name: str):
            try:
                return self._pyiqa.create_metric(name, device=self.device)
            except Exception as exc:
                print(f"[QE] skip metric {name}: {exc}")
                return None

        if self.clipiqa_model is None:
            self.clipiqa_model = _create_metric_safe('clipiqa+')
        if self.niqe_model is None:
            self.niqe_model = _create_metric_safe('niqe')
        if self.maniqa_model is None:
            self.maniqa_model = _create_metric_safe('maniqa')
        if self.topiq_model is None:
            self.topiq_model = _create_metric_safe('topiq_nr')

    @staticmethod
    def _safe_metric_value(model, image_tensor: torch.Tensor) -> Optional[float]:
        if model is None:
            return None
        try:
            with torch.no_grad():
                return float(model(image_tensor).item())
        except Exception:
            return None

    @staticmethod
    def _estimate_weather_indicators(image: Image.Image) -> Dict[str, float]:
        arr = np.asarray(image.convert('RGB'), dtype=np.float32) / 255.0
        gray = arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114

        gy, gx = np.gradient(gray)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)

        local_contrast = float(np.clip(gray.std() * 4.0, 0.0, 1.0))
        bright_ratio = float(np.mean(gray > 0.92))
        edge_density = float(np.mean(grad_mag > np.percentile(grad_mag, 75)))

        rain_residual = float(
            np.clip((np.mean(np.abs(gx)) / (np.mean(np.abs(gy)) + 1e-6)) * 0.5 + edge_density * 0.3, 0.0, 1.0)
        )
        fog_density = float(np.clip(1.0 - local_contrast + bright_ratio * 0.25, 0.0, 1.0))
        bright_artifact = float(np.clip(bright_ratio, 0.0, 1.0))
        edge_continuity = float(np.clip(1.0 - (np.std(grad_mag) * 2.0), 0.0, 1.0))

        return {
            'rain_residual_score': rain_residual,
            'fog_density_score': fog_density,
            'fade': fog_density,
            'local_contrast': local_contrast,
            'bright_artifact_ratio': bright_artifact,
            'edge_continuity': edge_continuity,
        }

    def _extract_features(self, image_path: str) -> Dict[str, float]:
        self._ensure_models()
        resolved = str(Path(image_path).resolve())
        cached = self._feature_cache.get(resolved)
        if cached is not None:
            return cached

        image = Image.open(resolved).convert('RGB')
        image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(self.device)
        weather = self._estimate_weather_indicators(image)

        features = {
            'maniqa': self._safe_metric_value(self.maniqa_model, image_tensor),
            'clipiqa': self._safe_metric_value(self.clipiqa_model, image_tensor),
            'topiq_nr': self._safe_metric_value(self.topiq_model, image_tensor),
            'niqe': self._safe_metric_value(self.niqe_model, image_tensor),
            'rain_residual_score': weather['rain_residual_score'],
            'fog_density_score': weather['fog_density_score'],
            'bright_artifact_ratio': weather['bright_artifact_ratio'],
            'local_contrast': weather['local_contrast'],
            'edge_continuity': weather['edge_continuity'],
        }
        self._feature_cache[resolved] = features
        return features

    @staticmethod
    def _reliability_from_bounds(value: Optional[float], lower: float, upper: float) -> float:
        if value is None:
            return 0.0
        span = max(upper - lower, 1e-6)
        below = max(0.0, (lower - float(value)) / span)
        above = max(0.0, (float(value) - upper) / span)
        overflow = below + above
        return float(np.clip(1.0 - overflow, 0.0, 1.0))

    @staticmethod
    def _to_unit_interval(value: Optional[float], lower: float, upper: float, larger_better: bool = True) -> float:
        if value is None:
            return 0.0
        span = max(upper - lower, 1e-6)
        unit = float(np.clip((float(value) - lower) / span, 0.0, 1.0))
        if not larger_better:
            unit = 1.0 - unit
        return unit

    def _init_metric_monitor(self) -> None:
        all_keys = list(self.shared_bounds.keys()) + list(self.TASK_BY_MAIN_KEY.keys())
        for key in all_keys:
            self._metric_history[key] = []
            self._metric_monitor[key] = {
                'window_total': 0,
                'window_clipped': 0,
                'recent_rates': [],
                'recent_high_flags': [],
                'consecutive_high_windows': 0,
                'triggered': False,
            }

    def _append_metric_history(self, key: str, value: Optional[float]) -> None:
        if value is None:
            return
        hist = self._metric_history[key]
        hist.append(float(value))
        if len(hist) > self.recalib_sample_cap:
            del hist[:-self.recalib_sample_cap]

    def _bound_owner(self, key: str) -> Tuple[str, Optional[str]]:
        if key in self.shared_bounds:
            return 'shared', None
        task = self.TASK_BY_MAIN_KEY.get(key)
        if task is not None:
            return 'task_main', task
        raise KeyError(f'Unknown metric key for bounds: {key}')

    def _get_bounds(self, key: str, task_name: Optional[str] = None) -> Tuple[float, float]:
        owner, owner_task = self._bound_owner(key)
        if owner == 'shared':
            return self.shared_bounds[key]
        task = owner_task or (task_name or '').strip().lower()
        if task not in self.task_main_bounds or key not in self.task_main_bounds[task]:
            raise KeyError(f'No task-main bounds for task={task}, key={key}')
        return self.task_main_bounds[task][key]

    def _set_bounds(self, key: str, lower: float, upper: float) -> None:
        owner, owner_task = self._bound_owner(key)
        if owner == 'shared':
            self.shared_bounds[key] = (float(lower), float(upper))
            return
        if owner_task is None:
            raise KeyError(f'Missing owner task for key={key}')
        self.task_main_bounds[owner_task][key] = (float(lower), float(upper))

    def _save_bounds_snapshot(self) -> None:
        payload = {
            'version': self._bounds_version,
            'saved_at_utc': datetime.now(timezone.utc).isoformat(),
            'shared_bounds': self.shared_bounds,
            'task_main_bounds': self.task_main_bounds,
            'config': {
                'clip_rate_threshold': self.clip_rate_threshold,
                'clip_consecutive_windows': self.clip_consecutive_windows,
                'clip_window_size': self.clip_window_size,
                'recalib_margin': self.recalib_margin,
                'recalib_min_samples': self.recalib_min_samples,
                'recalib_sample_cap': self.recalib_sample_cap,
            },
        }
        self.bounds_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        self.bounds_snapshot_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding='utf-8')

    def _append_bounds_version_log(self, event: Dict[str, object]) -> None:
        self.bounds_version_log.parent.mkdir(parents=True, exist_ok=True)
        with self.bounds_version_log.open('a', encoding='utf-8') as f:
            f.write(json.dumps(event, ensure_ascii=True) + '\n')

    def _try_recalibrate_metric(self, key: str, trigger_clip_rate: float) -> bool:
        hist = self._metric_history.get(key, [])
        if len(hist) < self.recalib_min_samples:
            return False

        p5 = float(np.percentile(hist, 5))
        p95 = float(np.percentile(hist, 95))
        span = max(p95 - p5, 1e-6)
        lower = float(p5 - self.recalib_margin * span)
        upper = float(p95 + self.recalib_margin * span)
        if upper <= lower:
            upper = lower + 1e-6

        old_lower, old_upper = self._get_bounds(key)
        self._bounds_version += 1
        self._set_bounds(key, lower, upper)

        event = {
            'version': self._bounds_version,
            'updated_at_utc': datetime.now(timezone.utc).isoformat(),
            'metric': key,
            'trigger': {
                'clip_rate': float(trigger_clip_rate),
                'clip_rate_threshold': self.clip_rate_threshold,
                'clip_consecutive_windows': self.clip_consecutive_windows,
                'window_size': self.clip_window_size,
            },
            'samples_used': len(hist),
            'percentiles': {'p5': p5, 'p95': p95},
            'margin': self.recalib_margin,
            'old_bounds': [float(old_lower), float(old_upper)],
            'new_bounds': [float(lower), float(upper)],
        }
        self._append_bounds_version_log(event)
        self._save_bounds_snapshot()
        self._metric_recalib_events.append(event)
        if len(self._metric_recalib_events) > 20:
            self._metric_recalib_events = self._metric_recalib_events[-20:]
        return True

    def _record_clip_event(self, key: str, value: Optional[float], clipped: bool) -> None:
        self._append_metric_history(key, value)
        monitor = self._metric_monitor[key]
        monitor['window_total'] = int(monitor['window_total']) + 1
        if clipped:
            monitor['window_clipped'] = int(monitor['window_clipped']) + 1

        window_total = int(monitor['window_total'])
        window_clipped = int(monitor['window_clipped'])
        if window_total < self.clip_window_size:
            return

        rate = float(window_clipped) / float(max(window_total, 1))
        is_high = rate > self.clip_rate_threshold

        rates = monitor['recent_rates']
        flags = monitor['recent_high_flags']
        if isinstance(rates, list):
            rates.append(rate)
            if len(rates) > 20:
                del rates[:-20]
        if isinstance(flags, list):
            flags.append(is_high)
            if len(flags) > 20:
                del flags[:-20]

        consec = int(monitor['consecutive_high_windows'])
        consec = consec + 1 if is_high else 0
        monitor['consecutive_high_windows'] = consec
        monitor['triggered'] = False

        if consec >= self.clip_consecutive_windows:
            recalib_ok = self._try_recalibrate_metric(key, trigger_clip_rate=rate)
            monitor['triggered'] = bool(recalib_ok)
            if recalib_ok:
                monitor['consecutive_high_windows'] = 0

        monitor['window_total'] = 0
        monitor['window_clipped'] = 0

    def get_clip_monitor_status(self) -> Dict[str, object]:
        return {
            'window_size': self.clip_window_size,
            'clip_rate_threshold': self.clip_rate_threshold,
            'clip_consecutive_windows': self.clip_consecutive_windows,
            'recalib_margin': self.recalib_margin,
            'recalib_min_samples': self.recalib_min_samples,
            'bounds_version': self._bounds_version,
            'version_log_path': str(self.bounds_version_log),
            'latest_snapshot_path': str(self.bounds_snapshot_path),
            'metrics': {
                k: {
                    'current_window_total': int(v['window_total']),
                    'current_window_clipped': int(v['window_clipped']),
                    'recent_window_clip_rates': list(v['recent_rates']) if isinstance(v['recent_rates'], list) else [],
                    'recent_window_is_high': list(v['recent_high_flags']) if isinstance(v['recent_high_flags'], list) else [],
                    'consecutive_high_windows': int(v['consecutive_high_windows']),
                    'triggered': bool(v['triggered']),
                    'history_size': len(self._metric_history.get(k, [])),
                    'bounds': list(self._get_bounds(k)),
                }
                for k, v in self._metric_monitor.items()
            },
            'recent_recalibration_events': list(self._metric_recalib_events),
        }

    def consume_recalibration_trigger(self) -> bool:
        triggered = any(bool(v.get('triggered', False)) for v in self._metric_monitor.values())
        for v in self._metric_monitor.values():
            v['triggered'] = False
        return triggered

    def _metric_unit_and_reliability(self, key: str, row: Dict[str, float], task_name: Optional[str] = None) -> Tuple[float, float]:
        task = (task_name or '').strip().lower()

        if task in self.task_main_bounds and key in self.task_main_bounds[task]:
            lower, upper = self.task_main_bounds[task][key]
        else:
            lower, upper = self.shared_bounds[key]

        value = row.get(key)
        larger_better = (key != 'niqe')
        unit = self._to_unit_interval(value, lower, upper, larger_better=larger_better)
        reliability = self._reliability_from_bounds(value, lower, upper)
        clipped = bool(value is not None and (float(value) < lower or float(value) > upper))
        self._record_clip_event(key=key, value=value, clipped=clipped)
        return unit, reliability

    def _modulate_weights(self, base: List[float], reliability: List[float]) -> List[float]:
        eff = []
        for b, r in zip(base, reliability):
            # reliability in [0,1] -> modulation in [1-band, 1+band]
            factor = 1.0 + self.reliability_band * (2.0 * float(np.clip(r, 0.0, 1.0)) - 1.0)
            eff.append(float(max(0.0, b * factor)))
        s = float(sum(eff))
        if s <= 1e-12:
            n = len(base)
            return [1.0 / n for _ in range(n)]
        return [e / s for e in eff]

    def _alpha_for_task(self, task_name: Optional[str]) -> float:
        key = (task_name or '').strip().lower()
        if key in self.alpha_by_task:
            return float(self.alpha_by_task[key])
        return float(self.alpha_task)

    def _task_score(self, task_name: Optional[str], row: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        key = (task_name or '').strip().lower()
        if key not in {'derain', 'dehaze', 'desnow'}:
            return 0.0, {
                'main_util': 0.0,
                'lc_inv_util': 0.0,
                'ec_inv_util': 0.0,
                'main_weight': 0.0,
                'lc_weight': 0.0,
                'ec_weight': 0.0,
            }

        if key == 'derain':
            base = [self.w_rain_main, self.w_rain_lc, self.w_rain_ec]
        elif key == 'dehaze':
            base = [self.w_haze_main, self.w_haze_lc, self.w_haze_ec]
        else:
            base = [self.w_snow_main, self.w_snow_lc, self.w_snow_ec]

        main_key = self.TASK_MAIN_KEY[key]
        main_util, main_rel = self._metric_unit_and_reliability(main_key, row, task_name=key)
        lc_util, lc_rel = self._metric_unit_and_reliability('local_contrast', row)
        ec_util, ec_rel = self._metric_unit_and_reliability('edge_continuity', row)

        main_good = 1.0 - main_util
        rel = [main_rel, lc_rel, ec_rel]
        w_main, w_lc, w_ec = self._modulate_weights(base, rel)

        score = float(w_main * main_good + w_lc * lc_util + w_ec * ec_util)
        details = {
            'main_util': main_good,
            'lc_inv_util': lc_util,
            'ec_inv_util': ec_util,
            'main_weight': w_main,
            'lc_weight': w_lc,
            'ec_weight': w_ec,
            'main_rel': main_rel,
            'lc_rel': lc_rel,
            'ec_rel': ec_rel,
            'main_key': main_key,
        }
        return score, details

    def _general_score(self, row: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        keys = ['maniqa', 'clipiqa', 'topiq_nr', 'niqe']
        base = [0.25, 0.25, 0.25, 0.25]

        units: List[float] = []
        rels: List[float] = []
        for k in keys:
            u, r = self._metric_unit_and_reliability(k, row)
            units.append(u)
            rels.append(r)

        weights = self._modulate_weights(base, rels)
        score = float(sum(w * u for w, u in zip(weights, units)))

        details = {
            'maniqa_util': units[0],
            'clipiqa_util': units[1],
            'topiq_util': units[2],
            'niqe_util': units[3],
            'maniqa_weight': weights[0],
            'clipiqa_weight': weights[1],
            'topiq_weight': weights[2],
            'niqe_weight': weights[3],
            'maniqa_rel': rels[0],
            'clipiqa_rel': rels[1],
            'topiq_rel': rels[2],
            'niqe_rel': rels[3],
        }
        return score, details

    def evaluate(self, image_path, task_name: Optional[str] = None):
        """Return score for one image (used in step guards/logging)."""
        row = self._extract_features(image_path)
        task_score, _ = self._task_score(task_name, row)
        general_score, _ = self._general_score(row)

        key = (task_name or '').strip().lower()
        alpha = self._alpha_for_task(task_name)
        if key in {'derain', 'dehaze', 'desnow'}:
            return float(alpha * task_score + (1.0 - alpha) * general_score)
        return float(general_score)

    def select_best(self, candidate_paths, task_name: Optional[str] = None):
        """Return (best_path, best_score) among candidates."""
        rows: List[Dict[str, float]] = []
        for p in candidate_paths:
            feat = self._extract_features(p)
            rows.append({'path': p, **feat})

        scores = []
        details = []

        key = (task_name or '').strip().lower()
        for idx, row in enumerate(rows):
            task_score, task_details = self._task_score(task_name, row)
            general_score, general_details = self._general_score(row)

            if key in {'derain', 'dehaze', 'desnow'}:
                alpha = self._alpha_for_task(task_name)
                final_score = float(alpha * task_score + (1.0 - alpha) * general_score)
            else:
                alpha = self.alpha_task
                final_score = general_score

            scores.append(final_score)
            task_contrib = float(alpha * task_score)
            general_contrib = float((1.0 - alpha) * general_score)
            denom = max(final_score, 1e-12)

            details.append(
                {
                    'idx': idx,
                    'name': Path(row['path']).name,
                    'final': final_score,
                    'task': task_score,
                    'general': general_score,
                    'task_contrib_pct': float(np.clip(task_contrib / denom, 0.0, 10.0)),
                    'general_contrib_pct': float(np.clip(general_contrib / denom, 0.0, 10.0)),
                    'task_main_util': task_details['main_util'],
                    'task_main_key': task_details.get('main_key', ''),
                    'task_main_weight': task_details['main_weight'],
                    'task_lc_weight': task_details['lc_weight'],
                    'task_ec_weight': task_details['ec_weight'],
                    'task_main_rel': task_details['main_rel'],
                    'task_lc_rel': task_details['lc_rel'],
                    'task_ec_rel': task_details['ec_rel'],
                    'g_maniqa_util': general_details['maniqa_util'],
                    'g_clipiqa_util': general_details['clipiqa_util'],
                    'g_topiq_util': general_details['topiq_util'],
                    'g_niqe_util': general_details['niqe_util'],
                    'g_maniqa_weight': general_details['maniqa_weight'],
                    'g_clipiqa_weight': general_details['clipiqa_weight'],
                    'g_topiq_weight': general_details['topiq_weight'],
                    'g_niqe_weight': general_details['niqe_weight'],
                    'rain': float(row['rain_residual_score']),
                    'haze': float(row['fog_density_score']),
                    'snow': float(row['bright_artifact_ratio']),
                    'lc': float(row['local_contrast']),
                    'ec': float(row['edge_continuity']),
                }
            )

        order = sorted(range(len(details)), key=lambda i: details[i]['final'], reverse=True)
        best_idx = int(order[0])

        # Tie-break for weather tasks: final within epsilon -> compare task score, then task main util.
        if key in {'derain', 'dehaze', 'desnow'} and len(order) >= 2:
            top = details[order[0]]
            second = details[order[1]]
            if (top['final'] - second['final']) <= self.tie_epsilon:
                if second['task'] > top['task']:
                    best_idx = int(order[1])
                elif abs(second['task'] - top['task']) <= self.tie_epsilon:
                    if second['task_main_util'] > top['task_main_util']:
                        best_idx = int(order[1])

        if self.verbose_selection:
            print(
                f'[QE] Task={key or "generic"}, alpha={self._alpha_for_task(key):.2f}, '
                f'task_weight={self._alpha_for_task(key):.2f}, general_weight={1.0 - self._alpha_for_task(key):.2f}, '
                f'tie_eps={self.tie_epsilon:.3f}, reliability_band={self.reliability_band:.2f}'
            )
            for item in sorted(details, key=lambda x: x['final'], reverse=True):
                print(
                    '[QE] '
                    f"{item['name']}: final={item['final']:.6f}, task={item['task']:.6f}, general={item['general']:.6f}, "
                    f"task%={100.0 * item['task_contrib_pct']:.1f}, gen%={100.0 * item['general_contrib_pct']:.1f}, "
                    f"main={item['task_main_key']} util={item['task_main_util']:.4f} "
                    f"w(main/lc/ec)=({item['task_main_weight']:.3f}/{item['task_lc_weight']:.3f}/{item['task_ec_weight']:.3f}), "
                    f"gW(m/c/t/n)=({item['g_maniqa_weight']:.3f}/{item['g_clipiqa_weight']:.3f}/{item['g_topiq_weight']:.3f}/{item['g_niqe_weight']:.3f}), "
                    f"rain={item['rain']:.4f}, haze={item['haze']:.4f}, snow={item['snow']:.4f}, "
                    f"LC={item['lc']:.4f}, EC={item['ec']:.4f}"
                )

        return candidate_paths[best_idx], scores[best_idx]
