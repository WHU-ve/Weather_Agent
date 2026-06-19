import os
import subprocess
import time
from pathlib import Path
from typing import Optional


WEATHER_AGENT_ENV = os.environ.get("WEATHER_AGENT_ENV", "weather_agent")
WEATHER_AGENT_RIDCP_ENV = os.environ.get("WEATHER_AGENT_RIDCP_ENV", "weather_agent_ridcp")
WEATHER_AGENT_NAFNET_ENV = os.environ.get("WEATHER_AGENT_NAFNET_ENV", "weather_agent_nafnet")
WEATHER_AGENT_MAXIM_ENV = os.environ.get("WEATHER_AGENT_MAXIM_ENV", "weather_agent_maxim")
WEATHER_AGENT_DIFFPLUGIN_ENV = os.environ.get("WEATHER_AGENT_DIFFPLUGIN_ENV", "weather_agent_diffplugin")
WEATHER_AGENT_JSTASR_ENV = os.environ.get("WEATHER_AGENT_JSTASR_ENV", "weather_agent_jstasr")
WEATHER_AGENT_STARNET_ENV = os.environ.get("WEATHER_AGENT_STARNET_ENV", "weather_agent_starnet")
WEATHER_AGENT_DDMSNET_ENV = os.environ.get("WEATHER_AGENT_DDMSNET_ENV", "weather_agent_DDMSNet")

# Direct python paths to bypass conda-run routing issues.
_ANACONDA_ENVS = Path('/root/project/huangchao/anaconda3/envs')
_PYTHON_CACHE = {}
def _env_python(env_name: str) -> str:
    if env_name not in _PYTHON_CACHE:
        _PYTHON_CACHE[env_name] = str(_ANACONDA_ENVS / env_name / 'bin' / 'python')
    return _PYTHON_CACHE[env_name]
WEATHER_UTILS_DIR = os.environ.get("WEATHER_UTILS_DIR", "utils_new")


def _auto_select_best_gpu() -> Optional[int]:
    """Pick the least-used GPU based on memory usage.

    Priority:
    1) lower used memory
    2) lower utilization
    3) lower temperature
    """
    force = os.environ.get('WEATHER_FORCE_GPU_ID', '').strip()
    if force:
        try:
            return int(force)
        except ValueError:
            pass

    disable_auto = os.environ.get('WEATHER_DISABLE_AUTO_GPU_PICK', '0').strip().lower() in {'1', 'true', 'yes'}
    if disable_auto:
        return None

    try:
        out = subprocess.check_output(
            [
                'nvidia-smi',
                '--query-gpu=index,memory.used,utilization.gpu,temperature.gpu',
                '--format=csv,noheader,nounits',
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None

    candidates = []
    for raw in out.strip().splitlines():
        parts = [p.strip() for p in raw.split(',')]
        if len(parts) < 4:
            continue
        try:
            idx = int(parts[0])
            mem_used = int(parts[1])
            util = int(parts[2])
            temp = int(parts[3])
            candidates.append((mem_used, util, temp, idx))
        except ValueError:
            continue

    if not candidates:
        return None

    # Prefer GPUs with enough free memory to avoid OOM on heavy models.
    min_free_mb = int(os.environ.get('EXPERT_MIN_FREE_MB', '3000'))
    try:
        total_out = subprocess.check_output(
            ['nvidia-smi','--query-gpu=index,memory.total','--format=csv,noheader,nounits'],
            text=True, stderr=subprocess.DEVNULL)
        total_map = {}
        for line in total_out.strip().splitlines():
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 2: total_map[int(parts[0])] = int(parts[1])
        filtered = []
        for mem_used, util, temp, idx in candidates:
            total = total_map.get(idx, 24564)
            if total - mem_used >= min_free_mb:
                filtered.append((mem_used, util, temp, idx))
        if not filtered:
            # All GPUs full — wait and retry.
            time.sleep(10)
            return _auto_select_best_gpu()
        candidates = filtered
    except Exception:
        pass  # fall back to unfiltered

    candidates.sort()
    return candidates[0][3]


def _auto_select_top_k_gpus(k: int) -> list[int]:
    """Pick top-k GPUs by free memory (then util/temp)."""
    if k <= 0:
        return []

    try:
        out = subprocess.check_output(
            [
                'nvidia-smi',
                '--query-gpu=index,memory.free,utilization.gpu,temperature.gpu',
                '--format=csv,noheader,nounits',
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []

    rows = []
    for raw in out.strip().splitlines():
        parts = [p.strip() for p in raw.split(',')]
        if len(parts) < 4:
            continue
        try:
            idx = int(parts[0])
            free_mb = int(parts[1])
            util = int(parts[2])
            temp = int(parts[3])
            rows.append((free_mb, -util, -temp, idx))
        except ValueError:
            continue

    if not rows:
        return []

    rows.sort(reverse=True)
    return [idx for _free, _neg_util, _neg_temp, idx in rows[:k]]


def _respect_existing_cuda_visible_devices() -> bool:
    raw = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    if not raw:
        return False
    return raw not in {'-1', 'none', 'None'}


def _get_tool_gpu_ids_env(tool_name: str) -> str:
    if tool_name in {'maxim', 'diffplugin'}:
        topk_key = 'WEATHER_MAXIM_TOPK_GPUS' if tool_name == 'maxim' else 'WEATHER_DIFFPLUGIN_TOPK_GPUS'
        try:
            topk = int(os.environ.get(topk_key, os.environ.get('WEATHER_MULTI_GPU_TOPK', '4')).strip())
        except ValueError:
            topk = 4
        topk = max(1, topk)

        dynamic_ids = _auto_select_top_k_gpus(topk)
        if dynamic_ids:
            return ','.join(str(i) for i in dynamic_ids)

        fallback_key = 'WEATHER_MAXIM_GPU_IDS' if tool_name == 'maxim' else 'WEATHER_DIFFPLUGIN_GPU_IDS'
        return os.environ.get(fallback_key, '').strip()

    key_map = {
        'restormer': 'WEATHER_RESTORMER_GPU_IDS',
        'mprnet': 'WEATHER_MPRNET_GPU_IDS',
        'xrestormer': 'WEATHER_XRESTORMER_GPU_IDS',
        'nafnet': 'WEATHER_NAFNET_GPU_IDS',
        'ridcp': 'WEATHER_RIDCP_GPU_IDS',
        'dehazeformer': 'WEATHER_DEHAZEFORMER_GPU_IDS',
        'starnet': 'WEATHER_STARNET_GPU_IDS',
        'ddmsnet': 'WEATHER_DDMSNET_GPU_IDS',
    }
    key = key_map.get(tool_name)
    if not key:
        return ''
    return os.environ.get(key, '').strip()

class Tool:
    """Abstract class for a tool.

    Args:
        tool_name (str): Tool name, a valid identifier serving as the name of environment, configuration file, etc.
        subtask (str): Subtask name, serving as the name of the directory for the subtask.
        work_dir (str | None, optional): Basename of working directory. Defaults to None.
        script_rel_path (Path | str | None, optional): Path relative to the working directory of the script to run. Defaults to None.
    """

    def __init__(
        self,
        tool_name: str,
        subtask: str,
        work_dir: Optional[Path] = None,
        script_rel_path: Optional[Path | str] = None
    ):
        self.tool_name = tool_name
        self.subtask = subtask
        self.work_dir: Optional[Path] = None
        self.script_path: Optional[Path] = None
        if work_dir is not None:
            assert script_rel_path is not None, "If `work_dir` is provided, `script_rel_path` should also be provided."
            self.work_dir = Path().resolve() / WEATHER_UTILS_DIR / subtask / 'tools' / work_dir
            self.script_path = self.work_dir / script_rel_path

    def __call__(self, input_dir: Path, output_dir: Path, silent: bool = False, run_gpu_id: Optional[int] = None, *args) -> None:
        if not silent:
            print('-' * 100)
            print(f"Subtask\t: {self.subtask}")
            print(f"Tool\t: {self.tool_name}")
            print(f"Input\t: {list(input_dir.glob('*'))[0]}")

        start_time = time.time()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.run_gpu_id = run_gpu_id

        self._precheck()
        self._invoke(*args)
        self._postcheck()

        end_time = time.time()
        if not silent:
            print(f"Output\t: {list(output_dir.glob('*'))[0]}")
            print(f"Time\t: {round(end_time - start_time, 3)}s")

    def _precheck(self) -> None:
        assert len(os.listdir(self.input_dir)) == 1, "The input directory should contain the input only."
        assert os.listdir(self.output_dir) == [], "The output directory should be empty."

    def _postcheck(self) -> None:
        output = [file for file in self.output_dir.glob('*') if file.suffix in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']]
        assert len(output) == 1, "There are other files in the same directory as the output image."
        if output[0].name != 'output.png':
            output[0].replace(self.output_dir / 'output.png')

    def _invoke(self, *args) -> None:
        self._preprocess()

        cmd = self._get_cmd_in_weather_env()
        verbose_tool_error = os.environ.get('WEATHER_TOOL_VERBOSE_ERRORS', '0').strip().lower() in {'1', 'true', 'yes'}
        if verbose_tool_error:
            subprocess.run(cmd, cwd=self.work_dir, shell=True, check=True)
        else:
            subprocess.run(cmd, cwd=self.work_dir, shell=True, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._postprocess()

    def _get_cmd(self) -> str:
        opts = self._get_cmd_opts()
        if self.run_gpu_id is not None:
            cmd = f"CUDA_VISIBLE_DEVICES={self.run_gpu_id} python '{self.script_path}'"
        else:
            auto_gpu = _auto_select_best_gpu()
            if auto_gpu is not None:
                cmd = f"CUDA_VISIBLE_DEVICES={auto_gpu} python '{self.script_path}'"
            else:
                cmd = f"python '{self.script_path}'"
        for opt in opts:
            cmd += f" '{opt}'"
        return cmd
    
    def _get_cmd_in_weather_env(self) -> str:
        opts = self._get_cmd_opts()

        pythonpath_prefix = f"PYTHONPATH='{self.work_dir}':$PYTHONPATH"
        env_name = WEATHER_AGENT_ENV
        runtime_env_parts = []
        if self.tool_name == 'ridcp':
            env_name = WEATHER_AGENT_RIDCP_ENV
        elif self.tool_name == 'nafnet':
            env_name = WEATHER_AGENT_NAFNET_ENV
        elif self.tool_name == 'maxim':
            env_name = WEATHER_AGENT_MAXIM_ENV
            runtime_env_parts.extend([
                'TF_CPP_MIN_LOG_LEVEL=3',
            ])
        elif self.tool_name == 'diffplugin':
            env_name = WEATHER_AGENT_DIFFPLUGIN_ENV
        elif self.tool_name == 'jstasr':
            env_name = WEATHER_AGENT_JSTASR_ENV
        elif self.tool_name == 'starnet':
            env_name = WEATHER_AGENT_STARNET_ENV
        elif self.tool_name == 'ddmsnet':
            env_name = WEATHER_AGENT_DDMSNET_ENV

        runtime_env = ''
        if runtime_env_parts:
            runtime_env = ' ' + ' '.join(runtime_env_parts)

        selected_gpu = self.run_gpu_id
        requested_gpu_ids = _get_tool_gpu_ids_env(self.tool_name)

        # Use direct python for non-default envs; add PATH so subprocess `python` resolves.
        if env_name == WEATHER_AGENT_ENV:
            python_cmd = f"conda run -n {env_name} python"
            extra_env = ""
        else:
            env_dir = str(_ANACONDA_ENVS / env_name)
            python_cmd = f"{env_dir}/bin/python"
            extra_env = f" PATH={env_dir}/bin:$PATH"

        if selected_gpu is not None:
            cmd = f"{pythonpath_prefix}{extra_env}{runtime_env} CUDA_VISIBLE_DEVICES={selected_gpu} {python_cmd} '{self.script_path}'"
        elif requested_gpu_ids:
            cmd = f"{pythonpath_prefix}{extra_env}{runtime_env} CUDA_VISIBLE_DEVICES={requested_gpu_ids} {python_cmd} '{self.script_path}'"
        else:
            auto_gpu = None if _respect_existing_cuda_visible_devices() else _auto_select_best_gpu()
            if auto_gpu is not None:
                cmd = f"{pythonpath_prefix}{extra_env}{runtime_env} CUDA_VISIBLE_DEVICES={auto_gpu} {python_cmd} '{self.script_path}'"
            else:
                cmd = f"{pythonpath_prefix}{extra_env}{runtime_env} {python_cmd} '{self.script_path}'"

        for opt in opts:
            cmd += f" '{opt}'"
        return cmd

    def _get_cmd_opts(self, *args) -> list[str]:
        raise NotImplementedError

    def _preprocess(self) -> None:
        pass

    def _postprocess(self) -> None:
        pass
