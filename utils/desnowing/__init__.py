from pathlib import Path

from ..tool import Tool


__all__ = ['desnowing_toolbox']

project_root = Path(__file__).resolve().parents[2]


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
            '--ckpt', str(project_root / 'pretrained_ckpts/DesnowNet/model.pth'),
            '--device', 'cuda',
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
            '--model_param_dir', str(project_root / 'pretrained_ckpts/JSTASR'),
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
            '--ckpt_root', str(project_root / 'pretrained_ckpts/DDMSNet'),
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
            '--output_dir', self.output_dir,
        ]


class DiffPluginDesnow(Tool):
    def __init__(self):
        super().__init__(
            tool_name='diffplugin',
            subtask='deraining',
            work_dir='Diff-Plugin',
            script_rel_path='infer_diffplugin_4kagent.py'
        )
        self.subtask = 'desnowing'

    def _get_cmd_opts(self) -> list[str]:
        ckpt_dir = str(project_root / 'pretrained_ckpts/Diff-Plugin/desnow')
        return [
            '--pretrained_model_name_or_path', 'CompVis/stable-diffusion-v1-4',
            '--clip_path', 'openai/clip-vit-large-patch14',
            '--num_inference_steps', '20',
            '--img_path_dir', self.input_dir,
            '--save_root', self.output_dir,
            '--ckpt_dir', ckpt_dir,
        ]


desnowing_toolbox = [
    DesnowNetTool(),
    JSTASRTool(),
    DDMSNetTool(),
    StarNetTool(),
    DiffPluginDesnow(),
]
