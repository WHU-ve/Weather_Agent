import argparse
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))

from model_impl import DeSnowNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def _pick_image(input_dir: Path) -> Path:
    for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff'):
        files = list(input_dir.glob(ext))
        if files:
            return files[0]
    raise FileNotFoundError(f'No image found in {input_dir}')


def _infer_tiled(model: DeSnowNet, image_tensor: torch.Tensor, tile_size: int = 64) -> torch.Tensor:
    _, _, height, width = image_tensor.shape
    pad_h = (tile_size - (height % tile_size)) % tile_size
    pad_w = (tile_size - (width % tile_size)) % tile_size

    padded = F.pad(image_tensor, (0, pad_w, 0, pad_h), mode='reflect')
    _, _, padded_h, padded_w = padded.shape

    out = torch.zeros_like(padded)
    for top in range(0, padded_h, tile_size):
        for left in range(0, padded_w, tile_size):
            patch = padded[:, :, top:top + tile_size, left:left + tile_size]
            y_hat, _, _ = model(patch)
            out[:, :, top:top + tile_size, left:left + tile_size] = y_hat

    return out[:, :, :height, :width]


def main():
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = Path(args.ckpt).resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f'DesnowNet checkpoint not found: {ckpt}')

    image_path = _pick_image(input_dir)
    image = Image.open(image_path).convert('RGB')
    to_tensor = transforms.ToTensor()
    image_tensor = to_tensor(image).unsqueeze(0)

    device = args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu'
    image_tensor = image_tensor.to(device)

    model = DeSnowNet().to(device)
    checkpoint = torch.load(str(ckpt), map_location=device)
    state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    with torch.no_grad():
        y_hat = _infer_tiled(model, image_tensor)
        y_hat = torch.clamp(y_hat, 0, 1)

    output_img = transforms.ToPILImage()(y_hat.squeeze(0).cpu())
    output_img.save(output_dir / 'output.png')


if __name__ == '__main__':
    main()
