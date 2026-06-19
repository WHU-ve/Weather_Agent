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
    parser.add_argument('--model_param_dir', required=True)
    parser.add_argument('--batch_size', default='1')
    return parser.parse_args()


def _pick_output(out_dir: Path) -> Path:
    for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff'):
        files = list(out_dir.glob(ext))
        if files:
            return files[0]
    raise RuntimeError(f'JSTASR produced no image in {out_dir}')


def _run_desnownet_fallback(work_dir: Path, input_dir: Path, output_dir: Path):
    project_root = work_dir.parents[3]
    desnownet_script = project_root / 'utils' / 'desnowing' / 'tools' / 'DesnowNet' / 'infer_desnownet_weather.py'
    desnownet_ckpt = project_root / 'pretrained_ckpts' / 'DesnowNet' / 'model.pth'
    if not desnownet_script.exists() or not desnownet_ckpt.exists():
        raise RuntimeError('JSTASR fallback failed: DesnowNet script/checkpoint not found')

    cmd = (
        f"conda run -n weather_agent python '{desnownet_script}' "
        f"--input_dir '{input_dir}' --output_dir '{output_dir}' "
        f"--ckpt '{desnownet_ckpt}' --device cuda"
    )
    subprocess.run(cmd, shell=True, check=True)


def main():
    args = parse_args()
    work_dir = Path(__file__).resolve().parent
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_param_dir = Path(args.model_param_dir).resolve()
    required_weights = ('A.h5', 'finalModel.h5', 'snowmodel.h5')
    if not model_param_dir.exists():
        print(f'JSTASR weights directory missing, fallback to DesnowNet: {model_param_dir}')
    if model_param_dir.exists():
        mp_target = work_dir / 'modelParam'
        mp_target.mkdir(exist_ok=True)
        for name in required_weights:
            src = model_param_dir / name
            if src.exists():
                dst = mp_target / name
                if not dst.exists() or src.stat().st_size != dst.stat().st_size:
                    shutil.copy2(src, dst)

    has_native_weights = all((work_dir / 'modelParam' / name).exists() for name in required_weights)

    native_cmd = [
        'python', 'predict.py',
        '-dataroot', str(input_dir),
        '-predictpath', str(output_dir),
        '-batch_size', str(args.batch_size),
    ]
    native_env = os.environ.copy()
    native_env.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    native_env.setdefault('PYTHONWARNINGS', 'ignore')

    if has_native_weights:
        try:
            subprocess.run(
                native_cmd,
                cwd=work_dir,
                env=native_env,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as err:
            print(f'JSTASR native run failed, fallback to DesnowNet: {err}')
            _run_desnownet_fallback(work_dir, input_dir, output_dir)
    else:
        print('JSTASR weights missing, fallback to DesnowNet')
        _run_desnownet_fallback(work_dir, input_dir, output_dir)

    first_out = _pick_output(output_dir)
    normalized_out = output_dir / 'output.png'
    Image.open(first_out).convert('RGB').save(normalized_out)

    for file_path in output_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}:
            if file_path.name != 'output.png':
                file_path.unlink()


if __name__ == '__main__':
    main()
