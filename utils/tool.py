import os
import subprocess
import time
from pathlib import Path
from typing import Optional
import shutil


WEATHER_AGENT_ENV = "weather_agent"
WEATHER_AGENT_RIDCP_ENV = "weather_agent_ridcp"
WEATHER_AGENT_NAFNET_ENV = "weather_agent_nafnet"
WEATHER_AGENT_MAXIM_ENV = "weather_agent_maxim"
WEATHER_AGENT_DIFFPLUGIN_ENV = "weather_agent_diffplugin"
WEATHER_AGENT_JSTASR_ENV = "weather_agent_jstasr"
WEATHER_AGENT_STARNET_ENV = "weather_agent_starnet"
WEATHER_AGENT_DDMSNET_ENV = "weather_agent_DDMSNet"
WEATHER_AGENT_PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
        self.env_name_override: Optional[str] = None
        if work_dir is not None:
            assert script_rel_path is not None, "If `work_dir` is provided, `script_rel_path` should also be provided."
            self.work_dir = Path().resolve() / 'utils' / subtask / 'tools' / work_dir
            self.script_path = self.work_dir / script_rel_path

    def __call__(self, input_dir: Path, output_dir: Path, silent: bool = False, run_gpu_id: Optional[int] = None, *args) -> None:
        input_dir = Path(input_dir).resolve()
        output_dir = Path(output_dir).resolve()
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
        assert self.work_dir is not None and self.work_dir.exists(), \
            f"Tool work dir does not exist: {self.work_dir}"
        assert self.script_path is not None and self.script_path.exists(), \
            f"Tool script does not exist: {self.script_path}"

        # Keep all experiment I/O inside the current weather_agent project.
        assert self.input_dir.is_relative_to(WEATHER_AGENT_PROJECT_ROOT), \
            f"Input must be in project: {self.input_dir}"
        assert self.output_dir.is_relative_to(WEATHER_AGENT_PROJECT_ROOT), \
            f"Output must be in project: {self.output_dir}"

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
        try:
            if verbose_tool_error:
                subprocess.run(cmd, cwd=self.work_dir, shell=True, check=True)
            else:
                subprocess.run(cmd, cwd=self.work_dir, shell=True, check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            # Avoid carrying broken partial outputs to later experiments.
            for p in self.output_dir.glob('*'):
                if p.is_file() or p.is_symlink():
                    p.unlink(missing_ok=True)
                elif p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
            raise
        self._postprocess()

    def _get_cmd(self) -> str:
        opts = self._get_cmd_opts()
        if self.run_gpu_id is not None:
            cmd = f"CUDA_VISIBLE_DEVICES={self.run_gpu_id} python '{self.script_path}'"
        else:
            cmd = f"python '{self.script_path}'"
        for opt in opts:
            cmd += f" '{opt}'"
        return cmd
    
    def _get_cmd_in_weather_env(self) -> str:
        opts = self._get_cmd_opts()

        pythonpath_prefix = f"PYTHONPATH='{self.work_dir}':$PYTHONPATH"
        env_name = self.env_name_override or WEATHER_AGENT_ENV
        if self.env_name_override is None:
            if self.tool_name == 'ridcp':
                env_name = WEATHER_AGENT_RIDCP_ENV
            elif self.tool_name == 'nafnet':
                env_name = WEATHER_AGENT_NAFNET_ENV
            elif self.tool_name == 'maxim':
                env_name = WEATHER_AGENT_MAXIM_ENV
            elif self.tool_name == 'diffplugin':
                env_name = WEATHER_AGENT_DIFFPLUGIN_ENV
            elif self.tool_name == 'jstasr':
                env_name = WEATHER_AGENT_JSTASR_ENV
            elif self.tool_name == 'starnet':
                env_name = WEATHER_AGENT_STARNET_ENV
            elif self.tool_name == 'ddmsnet':
                env_name = WEATHER_AGENT_DDMSNET_ENV

        # Fail fast if the expected runtime environment is missing.
        env_check = subprocess.run(
            ["conda", "env", "list"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if env_check.returncode != 0:
            raise RuntimeError(f"Failed to list conda envs: {env_check.stderr.strip()}")
        env_found = any(line.split() and line.split()[0] == env_name for line in env_check.stdout.splitlines() if not line.startswith('#'))
        if not env_found:
            raise RuntimeError(f"Conda env not found: {env_name}")

        if self.run_gpu_id is not None:
            cmd = f"{pythonpath_prefix} CUDA_VISIBLE_DEVICES={self.run_gpu_id} conda run -n {env_name} python '{self.script_path}'"
        else:
            cmd = f"{pythonpath_prefix} conda run -n {env_name} python '{self.script_path}'"

        for opt in opts:
            cmd += f" '{opt}'"
        return cmd

    def _get_cmd_opts(self, *args) -> list[str]:
        raise NotImplementedError

    def _preprocess(self) -> None:
        pass

    def _postprocess(self) -> None:
        pass
