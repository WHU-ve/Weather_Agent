import os
import shutil
from pathlib import Path

from ..tool import Tool
from ..multitask_tools import *


__all__ = ['dehazing_toolbox']
project_root = Path(__file__).resolve().parents[2]
CKPT_DIR_NAME = os.environ.get('WEATHER_CKPT_DIR', 'pretrained_ckpts_new')


class DehazeFormer(Tool):
    """[Vision Transformers for Single Image Dehazing (TIP 2023)](https://doi.org/10.1109/TIP.2023.3256763)"""    

    def __init__(self):
        super().__init__(
            tool_name="dehazeformer",
            subtask="dehazing",
            work_dir="DehazeFormer",
            script_rel_path="inference.py"
        )

    def _get_cmd_opts(self) -> list[str]:
        return [
            "--data_dir", self.input_dir,
            "--result_dir", self.output_dir,
            "--save_dir",  str(project_root / f"{CKPT_DIR_NAME}/DehazeFormer"),
            "--tile_size", "1024",
            "--tile_overlap", "64"
        ]
    

class RIDCP(Tool):
    """[RIDCP: Revitalizing Real Image Dehazing via High-Quality Codebook Priors (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_RIDCP_Revitalizing_Real_Image_Dehazing_via_High-Quality_Codebook_Priors_CVPR_2023_paper.pdf)"""    

    def __init__(self):
        super().__init__(
            tool_name="ridcp",
            subtask="dehazing",
            work_dir="RIDCP_dehazing",
            script_rel_path="infer_ridcp_4kagent.py"
        )

    def _get_cmd_opts(self) -> list[str]:
        ridcp_max_size = os.environ.get('WEATHER_RIDCP_MAX_SIZE', '1000')
        return [
            "-i", self.input_dir,
            "-o", self.output_dir,
            "--weight", str(project_root / f"{CKPT_DIR_NAME}/RIDCP_dehazing/pretrained_RIDCP.pth"),
            "--matching_weight_path", str(project_root / f"{CKPT_DIR_NAME}/RIDCP_dehazing/weight_for_matching_dehazing_Flickr.pth"),
            "--use_weight",
            "--alpha", "-21.25",
            "--max_size", ridcp_max_size,
        ]
    
    
class MWFormer(Tool):
    def __init__(self, subtask: str):
        super().__init__(
            tool_name="MWFormer",
            subtask=subtask,
            work_dir="MWFormer",
            script_rel_path="infer_mwformer_4kagent.py"
        )

    def _get_cmd_opts(self) -> list[str]:
        # Prefer migrated ckpt location in pretrained_ckpts_new/pretrained_ckpts/MWFormer.
        ckpt_root = project_root / f"{CKPT_DIR_NAME}/pretrained_ckpts/MWFormer/MWFormer_L"
        if not ckpt_root.exists():
            ckpt_root = project_root / "MWFormer/MWFormer_L"
        return [
            "--val_data_dir", self.input_dir,
            "--result_dir", self.output_dir,
            "--restore-from-stylefilter", str(ckpt_root / "style_filter"),
            "--restore-from-backbone", str(ckpt_root / "backbone"),
        ]


class DiffPluginDehaze(DiffPlugin):
    def __init__(self):
        # Reuse maintained Diff-Plugin runtime under deraining/tools/Diff-Plugin.
        super().__init__(subtask='deraining')
        self.subtask = 'dehazing'
        

subtask = 'dehazing'
dehazing_toolbox = [
    DehazeFormer(),
    MAXIM(subtask=subtask),
    RIDCP(),
    XRestormer(subtask=subtask),
    DiffPluginDehaze(),
    # AutoDIR(subtask=subtask),
]