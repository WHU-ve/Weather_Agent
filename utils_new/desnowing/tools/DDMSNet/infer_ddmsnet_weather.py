import argparse
import shutil
import subprocess
from pathlib import Path
import os

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--ckpt_root', required=True)
    return parser.parse_args()


def _pick_input(input_dir: Path) -> Path:
    for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff'):
        files = list(input_dir.glob(ext))
        if files:
            return files[0]
    raise FileNotFoundError(f'No image found in {input_dir}')


def main():
    args = parse_args()
    work_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = work_dir / 'img' / 'raw'
    desnow_dir = work_dir / 'img' / 'desnow'
    raw_dir.mkdir(parents=True, exist_ok=True)
    desnow_dir.mkdir(parents=True, exist_ok=True)
    target_input = raw_dir / 'test_pic.png'
    stale_output = desnow_dir / 'desnow_test_pic.png'
    if target_input.exists():
        target_input.unlink()
    if stale_output.exists():
        stale_output.unlink()

    input_img = _pick_input(Path(args.input_dir).resolve())
    Image.open(input_img).convert('RGB').save(target_input)

    ckpt_root = Path(args.ckpt_root).resolve()
    required_any = ['kitti_best.pth', 'cityscapes_DDMSNet', 'kitti_DDMSNet', 'snow100k_DDMSNet']
    required_optional = ['kitti_eigen.pth']

    if not ckpt_root.exists():
        raise FileNotFoundError(f'DDMSNet ckpt_root does not exist: {ckpt_root}')

    has_main_weight = any((ckpt_root / name).exists() for name in required_any)
    if not has_main_weight:
        raise FileNotFoundError(
            'DDMSNet missing required checkpoints under '
            f'{ckpt_root}. Expected at least one of: {required_any}'
        )

    if not any((ckpt_root / name).exists() for name in required_optional):
        print(f'Warning: DDMSNet depth checkpoint not found in {ckpt_root}: {required_optional}')

    if ckpt_root.exists():
        sync_candidates = {
            'kitti_eigen.pth': ['kitti_eigen.pth', 'kitti_eigin.pth', 'kitti_eigen.path'],
            'kitti_best.pth': ['kitti_best.pth', 'cityscapes_best.pth'],
            'cityscapes_DDMSNet': ['cityscapes_DDMSNet', 'snow100k_DDMSNet'],
            'kitti_DDMSNet': ['kitti_DDMSNet'],
        }
        for dst_name, candidates in sync_candidates.items():
            src = next((ckpt_root / name for name in candidates if (ckpt_root / name).exists()), None)
            if src is None:
                continue
            dst = work_dir / dst_name
            if dst.exists():
                same_size = src.is_file() and dst.is_file() and src.stat().st_size == dst.stat().st_size
                if same_size:
                    continue
                if dst.is_dir():
                    shutil.rmtree(dst)
                else:
                    dst.unlink()
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

    env = os.environ.copy()
    env.setdefault('PYTHONWARNINGS', 'ignore::UserWarning')
    subprocess.run(['python', 'test_one.py'], cwd=work_dir, env=env, check=True)

    candidate = desnow_dir / 'desnow_test_pic.png'
    if not candidate.exists():
        raise RuntimeError('DDMSNet did not produce expected output image')
    Image.open(candidate).convert('RGB').save(output_dir / 'output.png')


if __name__ == '__main__':
    main()
