import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def _load_jstasr_model(model_param_dir: Path):
	# JSTASR codebase expects project root on sys.path and model weights under ./modelParam.
	root_dir = Path(__file__).resolve().parent
	if str(root_dir) not in sys.path:
		sys.path.insert(0, str(root_dir))

	from model.model import build_combine_model  # pylint: disable=import-outside-toplevel

	# Prefer direct modelParam directory from checkpoints; fallback to local modelParam if already provisioned.
	model_param_dir = model_param_dir.resolve()
	if model_param_dir.is_dir():
		cwd = os.getcwd()
		try:
			os.chdir(root_dir)
			local_dir = root_dir / 'modelParam'
			if local_dir.is_symlink():
				local_dir.unlink()
			if local_dir.exists() and local_dir.resolve() != model_param_dir:
				shutil.rmtree(local_dir)
			if not local_dir.exists():
				local_dir.symlink_to(model_param_dir, target_is_directory=True)
			return build_combine_model()
		finally:
			os.chdir(cwd)

	raise FileNotFoundError(f'JSTASR model_param_dir not found: {model_param_dir}')


def _first_image(input_dir: Path) -> Path:
	exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
	files = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])
	if not files:
		raise FileNotFoundError(f'No image found in {input_dir}')
	return files[0]


def run(input_dir: Path, output_dir: Path, model_param_dir: Path) -> None:
	src = _first_image(input_dir)
	img = Image.open(src).convert('RGB')
	orig_w, orig_h = img.size

	model = _load_jstasr_model(model_param_dir)

	# The official JSTASR graph is fixed to 640x480.
	model_input = img.resize((640, 480), Image.BICUBIC)
	model_input_np = np.asarray(model_input).astype(np.float32) / 255.0
	pred = model.predict(model_input_np[None, ...])
	if isinstance(pred, list):
		pred = pred[0]
	pred = np.clip(pred[0], 0.0, 1.0)

	out = Image.fromarray((pred * 255.0).astype(np.uint8)).resize((orig_w, orig_h), Image.BICUBIC)

	output_dir.mkdir(parents=True, exist_ok=True)
	out.save(output_dir / 'output.png')


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_dir', type=str, required=True)
	parser.add_argument('--output_dir', type=str, required=True)
	parser.add_argument('--model_param_dir', type=str, default='')
	parser.add_argument('--batch_size', type=int, default=1)
	return parser.parse_args()


if __name__ == '__main__':
	args = parse_args()
	run(Path(args.input_dir), Path(args.output_dir), Path(args.model_param_dir))

