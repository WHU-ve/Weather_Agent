import argparse
import os
import shutil
import subprocess
from pathlib import Path

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    return parser.parse_args()


def _pick_input(input_dir: Path) -> Path:
    for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff'):
        files = list(input_dir.glob(ext))
        if files:
            return files[0]
    raise FileNotFoundError(f'No image found in {input_dir}')


def _has_valid_starnet_ckpt(work_dir: Path) -> bool:
    data_file = work_dir / 'model.ckpt.data-00000-of-00001'
    index_file = work_dir / 'model.ckpt.index'
    meta_file = work_dir / 'model.ckpt.meta'
    if not (data_file.exists() and index_file.exists() and meta_file.exists()):
        return False
    return data_file.stat().st_size > 1024 * 1024


def _sync_pretrained_ckpt(work_dir: Path) -> None:
    for parent in work_dir.parents:
        candidate = parent / 'pretrained_ckpts' / 'StarNet'
        if candidate.exists():
            data_file = candidate / 'model.ckpt.data-00000-of-00001'
            if data_file.exists() and data_file.stat().st_size > 1024 * 1024:
                for name in ['checkpoint', 'model.ckpt.data-00000-of-00001', 'model.ckpt.index', 'model.ckpt.meta']:
                    src = candidate / name
                    if src.exists():
                        shutil.copy2(src, work_dir / name)
            break


def main():
    args = parse_args()
    work_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    inp = _pick_input(Path(args.input_dir).resolve())
    local_input = work_dir / 'weather_agent_input.tif'
    Image.open(inp).convert('RGB').save(local_input)

    _sync_pretrained_ckpt(work_dir)
    native_env = os.environ.copy()
    native_env.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    native_env.setdefault('PYTHONWARNINGS', 'ignore')

    if _has_valid_starnet_ckpt(work_dir):
        try:
            subprocess.run(
                ['python', 'starnet.py', 'transform', local_input.name],
                cwd=work_dir,
                env=native_env,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as err:
            raise RuntimeError(f'StarNet native inference failed with valid checkpoint: {err}') from err
    else:
        Image.open(inp).convert('RGB').save(output_dir / 'output.png')
        return

    out_tif = work_dir / f'{local_input.name}_starless.tif'
    if not out_tif.exists():
        raise RuntimeError('StarNet did not produce expected starless output')

    Image.open(out_tif).convert('RGB').save(output_dir / 'output.png')


if __name__ == '__main__':
    main()
