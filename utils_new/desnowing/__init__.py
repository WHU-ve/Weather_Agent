import os
from pathlib import Path

from ..tool import Tool
from ..multitask_tools import DiffPlugin


__all__ = ['desnowing_toolbox']

project_root = Path(__file__).resolve().parents[2]
CKPT_DIR_NAME = os.environ.get('WEATHER_CKPT_DIR', 'pretrained_ckpts_new')


class DesnowNetTool(Tool):
    def __init__(self):
        super().__init__(
            tool_name='desnownet',
            subtask='desnowing',
            work_dir='DesnowNet',
            script_rel_path='infer_desnownet_weather.py'
        )

    def _get_cmd_opts(self) -> list[str]:
        return [
            '--input_dir', self.input_dir,
            '--output_dir', self.output_dir,
            '--ckpt', str(project_root / f'{CKPT_DIR_NAME}/DesnowNet/model.pth'),
            '--device', 'cuda'
        ]


class JSTASRTool(Tool):
    def __init__(self):
        super().__init__(
            tool_name='jstasr',
            subtask='desnowing',
            work_dir='JSTASR',
            script_rel_path='infer_jstasr_weather.py'
        )

    def _get_cmd_opts(self) -> list[str]:
        return [
            '--input_dir', self.input_dir,
            '--output_dir', self.output_dir,
            '--model_param_dir', str(project_root / f'{CKPT_DIR_NAME}/JSTASR'),
            '--batch_size', '1'
        ]


class DDMSNetTool(Tool):
    def __init__(self):
        super().__init__(
            tool_name='ddmsnet',
            subtask='desnowing',
            work_dir='DDMSNet',
            script_rel_path='infer_ddmsnet_weather.py'
        )

    def _get_cmd_opts(self) -> list[str]:
        return [
            '--input_dir', self.input_dir,
            '--output_dir', self.output_dir,
            '--ckpt_root', str(project_root / f'{CKPT_DIR_NAME}/DDMSNet')
        ]


class StarNetTool(Tool):
    def __init__(self):
        super().__init__(
            tool_name='starnet',
            subtask='desnowing',
            work_dir='StarNet',
            script_rel_path='infer_starnet_weather.py'
        )

    def _get_cmd_opts(self) -> list[str]:
        return [
            '--input_dir', self.input_dir,
            '--output_dir', self.output_dir
        ]


class DiffPluginDesnow(DiffPlugin):
    def __init__(self):
        # Reuse the maintained Diff-Plugin runtime under deraining/tools/Diff-Plugin.
        super().__init__(subtask='deraining')
        self.subtask = 'desnowing'


desnowing_toolbox = [
    StarNetTool(),
    DiffPluginDesnow(),
    DesnowNetTool(),
    DDMSNetTool(),
]
