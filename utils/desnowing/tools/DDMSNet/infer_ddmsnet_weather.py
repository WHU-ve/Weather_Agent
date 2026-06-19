import argparse
from collections import OrderedDict
from pathlib import Path
import sys

import cv2
import dill
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, ToTensor

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR / 'VNL_depth'))
sys.path.insert(0, str(THIS_DIR / 'semantic_seg'))

from model import DDMSNet
from VNL_depth.lib.models.metric_depth_model import MetricDepthModel
from semantic_seg.config import infer_cfg
import semantic_seg.network
from semantic_seg.datasets import kitti


def _first_image(input_dir: Path) -> Path:
	exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
	files = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])
	if not files:
		raise FileNotFoundError(f'No image found in {input_dir}')
	return files[0]


def _load_depth_model(device: torch.device, ckpt_root: Path) -> nn.Module:
	depth_extract_net = MetricDepthModel().to(device)
	depth_extract_net = nn.DataParallel(depth_extract_net)

	ckpt = torch.load(str(ckpt_root / 'kitti_eigen.pth'), map_location='cpu', pickle_module=dill)
	state_dict = ckpt['model_state_dict']
	new_state_dict = OrderedDict((f'module.{k}', v) for k, v in state_dict.items())
	depth_extract_net.load_state_dict(new_state_dict, strict=False)

	for param in depth_extract_net.parameters():
		param.requires_grad = False
	depth_extract_net.eval()
	return depth_extract_net


def _load_semantic_model(device: torch.device, ckpt_root: Path) -> nn.Module:
	infer_cfg(train_mode=False)
	arch = 'semantic_seg.network.deepv3.DeepWV3Plus'
	semantic_extract_net = semantic_seg.network.get_net(arch, kitti, criterion=None).to(device)

	ckpt = torch.load(str(ckpt_root / 'kitti_best.pth'), map_location='cpu', pickle_module=dill)
	state_dict = ckpt['state_dict']
	new_state_dict = OrderedDict()
	for k, v in state_dict.items():
		new_k = k[7:] if k.startswith('module.') else k
		new_state_dict[new_k] = v
	semantic_extract_net.load_state_dict(new_state_dict, strict=False)

	for param in semantic_extract_net.parameters():
		param.requires_grad = False
	semantic_extract_net.eval()
	return semantic_extract_net


def _pick_ddms_ckpt(ckpt_root: Path) -> Path:
	candidates = ['snow100k_DDMSNet', 'kitti_DDMSNet', 'cityscapes_DDMSNet']
	for name in candidates:
		p = ckpt_root / name
		if p.exists():
			return p
	raise FileNotFoundError(f'No DDMSNet checkpoint found under {ckpt_root}')


def run(input_dir: Path, output_dir: Path, ckpt_root: Path) -> None:
	src = _first_image(input_dir)

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	depth_extract_net = _load_depth_model(device, ckpt_root)
	semantic_extract_net = _load_semantic_model(device, ckpt_root)

	net = DDMSNet(depth_extract_model=depth_extract_net, semantic_extract_model=semantic_extract_net)
	net = net.to(device)
	net = nn.DataParallel(net)

	ddms_ckpt_path = _pick_ddms_ckpt(ckpt_root)
	ckpt = torch.load(str(ddms_ckpt_path), map_location='cpu')
	net.load_state_dict(ckpt['net'], strict=False)
	net.eval()

	transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	img_raw = Image.open(src).convert('RGB')
	x = transform(img_raw).unsqueeze(0).to(device)

	with torch.no_grad():
		out = net(x)

	out_np = out[0].detach().cpu().numpy()
	out_np = np.transpose(out_np, (1, 2, 0))
	out_np = np.clip((out_np + 1.0) * 127.5, 0, 255).astype(np.uint8)
	out_np = out_np[:, :, ::-1]

	output_dir.mkdir(parents=True, exist_ok=True)
	cv2.imwrite(str(output_dir / 'output.png'), out_np)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_dir', type=str, required=True)
	parser.add_argument('--output_dir', type=str, required=True)
	parser.add_argument('--ckpt_root', type=str, required=True)
	return parser.parse_args()


if __name__ == '__main__':
	args = parse_args()
	run(Path(args.input_dir), Path(args.output_dir), Path(args.ckpt_root))

