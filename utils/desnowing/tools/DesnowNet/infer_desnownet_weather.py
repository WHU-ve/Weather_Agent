import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
NETWORK_DIR = SCRIPT_DIR / 'network'
if str(NETWORK_DIR) not in sys.path:
	sys.path.insert(0, str(NETWORK_DIR))

from DesnowNet import DesnowNet


def _first_image(input_dir: Path) -> Path:
	exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
	files = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])
	if not files:
		raise FileNotFoundError(f'No image found in {input_dir}')
	return files[0]


def _remap_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
	remapped = {}
	for key, value in state_dict.items():
		new_key = key
		new_key = new_key.replace('descriptorT.iv4.', 'TR.D_t.backbone.')
		new_key = new_key.replace('descriptorT.dp.dilatedConv0.', 'TR.D_t.DP.block.0.')
		new_key = new_key.replace('descriptorT.dp.dilatedConv1.', 'TR.D_t.DP.block.1.')
		new_key = new_key.replace('descriptorT.dp.dilatedConv2.', 'TR.D_t.DP.block.2.')
		new_key = new_key.replace('descriptorT.dp.dilatedConv3.', 'TR.D_t.DP.block.3.')
		new_key = new_key.replace('descriptorT.dp.dilatedConv4.', 'TR.D_t.DP.block.4.')
		new_key = new_key.replace('descriptorR.iv4.', 'RG.D_r.backbone.')
		new_key = new_key.replace('descriptorR.dp.dilatedConv0.', 'RG.D_r.DP.block.0.')
		new_key = new_key.replace('descriptorR.dp.dilatedConv1.', 'RG.D_r.DP.block.1.')
		new_key = new_key.replace('descriptorR.dp.dilatedConv2.', 'RG.D_r.DP.block.2.')
		new_key = new_key.replace('descriptorR.dp.dilatedConv3.', 'RG.D_r.DP.block.3.')
		new_key = new_key.replace('descriptorR.dp.dilatedConv4.', 'RG.D_r.DP.block.4.')
		new_key = new_key.replace('snowExtractor.pyramidMaxout.conv1.', 'TR.R_t.SE.conv_module.0.')
		new_key = new_key.replace('snowExtractor.pyramidMaxout.conv3.', 'TR.R_t.SE.conv_module.1.')
		new_key = new_key.replace('snowExtractor.pyramidMaxout.conv5.', 'TR.R_t.SE.conv_module.2.')
		new_key = new_key.replace('snowExtractor.pyramidMaxout.conv7.', 'TR.R_t.SE.conv_module.3.')
		new_key = new_key.replace('snowExtractor.prelu.', 'TR.R_t.SE.activation.')
		new_key = new_key.replace('aberrationExtractor.pyramidMaxout.conv1.', 'TR.R_t.AE.conv_module.0.')
		new_key = new_key.replace('aberrationExtractor.pyramidMaxout.conv3.', 'TR.R_t.AE.conv_module.1.')
		new_key = new_key.replace('aberrationExtractor.pyramidMaxout.conv5.', 'TR.R_t.AE.conv_module.2.')
		new_key = new_key.replace('aberrationExtractor.pyramidMaxout.conv7.', 'TR.R_t.AE.conv_module.3.')
		new_key = new_key.replace('aberrationExtractor.prelu.', 'TR.R_t.AE.activation.')
		new_key = new_key.replace('recoveryR.pyramidSum.conv1.', 'RG.conv_module.0.')
		new_key = new_key.replace('recoveryR.pyramidSum.conv3.', 'RG.conv_module.1.')
		new_key = new_key.replace('recoveryR.pyramidSum.conv5.', 'RG.conv_module.2.')
		new_key = new_key.replace('recoveryR.pyramidSum.conv7.', 'RG.conv_module.3.')
		remapped[new_key] = value
	return remapped


def _resolve_checkpoint_path(ckpt_arg: str) -> Path:
	if ckpt_arg:
		return Path(ckpt_arg)
	return Path(__file__).resolve().parents[4] / 'pretrained_ckpts/DesnowNet/model.pth'


def _load_model(ckpt_path: Path, device: torch.device) -> DesnowNet:
	checkpoint = torch.load(ckpt_path, map_location='cpu')
	state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
	model = DesnowNet(mode='original').to(device).eval()
	missing, unexpected = model.load_state_dict(_remap_state_dict(state_dict), strict=False)
	if missing or unexpected:
		raise RuntimeError(
			f'DesnowNet checkpoint mismatch after remap: missing={len(missing)}, unexpected={len(unexpected)}; '
			f'first_missing={missing[:8]}, first_unexpected={unexpected[:8]}'
		)
	return model


def _to_uint8_image(tensor: torch.Tensor) -> Image.Image:
	image = tensor.detach().clamp(0.0, 1.0).cpu().permute(1, 2, 0).numpy()
	image = (image * 255.0).round().astype(np.uint8)
	return Image.fromarray(image)


def run(input_dir: Path, output_dir: Path, ckpt: Path, device_name: str) -> None:
	src = _first_image(input_dir)
	device = torch.device(device_name if device_name == 'cpu' or torch.cuda.is_available() else 'cpu')
	model = _load_model(ckpt, device)
	img = Image.open(src).convert('RGB')
	img_tensor = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
	with torch.no_grad():
		y_hat, _, _, _ = model(img_tensor)
	output_dir.mkdir(parents=True, exist_ok=True)
	_to_uint8_image(y_hat[0]).save(output_dir / 'output.png')


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_dir', type=str, required=True)
	parser.add_argument('--output_dir', type=str, required=True)
	parser.add_argument('--ckpt', type=str, default='')
	parser.add_argument('--device', type=str, default='cuda')
	return parser.parse_args()


if __name__ == '__main__':
	args = parse_args()
	run(Path(args.input_dir), Path(args.output_dir), _resolve_checkpoint_path(args.ckpt), args.device)

