import os
import argparse
import warnings
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch

from transformers.utils import logging as transformers_logging
from diffusers.utils import logging as diffusers_logging

from transformers import CLIPVisionModel, AutoTokenizer, CLIPImageProcessor
from diffusers import AutoencoderKL, UNet2DConditionModel, UniPCMultistepScheduler
from diffusers.utils import load_image
from diffusers.image_processor import VaeImageProcessor

from modules import SCBNet
from modules import TPBNet
from utils import concat_imgs, import_model_class_from_model_name_or_path


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a Diff-Plugin inference script.")

    parser.add_argument("--pretrained_model_name_or_path", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--clip_path", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--inp_of_crossatt", type=str, default="clip", choices=["text", "clip"])
    parser.add_argument("--inp_of_unet_is_random_noise", action="store_true", default=False)
    parser.add_argument("--ckpt_dir", type=str, default="")
    parser.add_argument("--used_clip_vision_layers", type=int, default=24)
    parser.add_argument("--used_clip_vision_global", action="store_true", default=False)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--time_threshold", type=int, default=960)
    parser.add_argument("--save_root", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img_path", type=str)
    parser.add_argument("--img_path_dir", type=str)
    parser.add_argument("--hf_cache_dir", type=str, default="")
    parser.add_argument("--local_files_only", action="store_true", default=False)

    parser.add_argument("--enable_multi_gpu_tiling", action="store_true", default=False)
    parser.add_argument("--gpu_ids", type=str, default="")
    parser.add_argument("--tile_size", type=int, default=1536)
    parser.add_argument("--tile_overlap", type=int, default=128)
    parser.add_argument("--tile_trigger_long_side", type=int, default=3000)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def _ensure_local_backbone(args):
    if not args.local_files_only:
        return

    model_path = args.pretrained_model_name_or_path
    clip_path = args.clip_path

    model_ok = False
    if os.path.isdir(model_path):
        model_ok = os.path.isfile(os.path.join(model_path, 'model_index.json'))
    elif args.hf_cache_dir:
        model_ok = os.path.isfile(os.path.join(args.hf_cache_dir, 'stable-diffusion-v1-4', 'model_index.json'))

    clip_ok = False
    if os.path.isdir(clip_path):
        clip_ok = os.path.isfile(os.path.join(clip_path, 'config.json'))
    elif args.hf_cache_dir:
        clip_ok = os.path.isfile(os.path.join(args.hf_cache_dir, 'clip-vit-large-patch14', 'config.json'))

    if not model_ok or not clip_ok:
        raise FileNotFoundError(
            "Diff-Plugin local backbone cache missing. Please place:\n"
            "  1) stable-diffusion-v1-4 (with model_index.json)\n"
            "  2) clip-vit-large-patch14 (with config.json)\n"
            f"under cache dir: {args.hf_cache_dir or '[not set]'}"
        )


def _parse_gpu_ids(raw: str) -> list[int]:
    ids = []
    for token in (raw or '').split(','):
        token = token.strip()
        if not token:
            continue
        try:
            ids.append(int(token))
        except ValueError:
            continue
    return ids


def _pick_free_gpu_ids(min_needed: int) -> list[int]:
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,memory.free,utilization.gpu', '--format=csv,noheader,nounits'],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []

    rows = []
    for ln in out.strip().splitlines():
        parts = [x.strip() for x in ln.split(',')]
        if len(parts) != 3:
            continue
        try:
            idx, free_mb, util = int(parts[0]), int(parts[1]), int(parts[2])
            rows.append((free_mb, -util, idx))
        except ValueError:
            continue

    rows.sort(reverse=True)
    picked = [idx for _free, _neg_util, idx in rows[:max(1, min_needed)]]
    return picked


def _gen_tiles(width: int, height: int, tile_size: int, overlap: int):
    stride = max(64, tile_size - overlap)
    xs = list(range(0, max(1, width - tile_size + 1), stride))
    ys = list(range(0, max(1, height - tile_size + 1), stride))
    if not xs or xs[-1] != max(0, width - tile_size):
        xs.append(max(0, width - tile_size))
    if not ys or ys[-1] != max(0, height - tile_size):
        ys.append(max(0, height - tile_size))
    tiles = []
    for y in ys:
        for x in xs:
            x2 = min(width, x + tile_size)
            y2 = min(height, y + tile_size)
            tiles.append((x, y, x2, y2))
    return tiles


def _run_tiled_multi_gpu(args):
    from PIL import Image

    img = load_image(args.img_path)
    width, height = img.size
    tile_size = max(512, int(args.tile_size))
    overlap = max(0, int(args.tile_overlap))

    gpu_ids = _parse_gpu_ids(args.gpu_ids)
    if not gpu_ids:
        gpu_ids = _pick_free_gpu_ids(min_needed=2)
    if not gpu_ids:
        raise RuntimeError('No GPU ids available for tiled multi-GPU Diff-Plugin run.')

    tiles = _gen_tiles(width, height, tile_size=tile_size, overlap=overlap)
    os.makedirs(args.save_root, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix='diffplugin_tiles_') as td:
        tmp_root = Path(td)

        tasks_by_gpu = {g: [] for g in gpu_ids}
        for i, box in enumerate(tiles):
            x1, y1, x2, y2 = box
            tile_img = img.crop((x1, y1, x2, y2))
            in_dir = tmp_root / f'in_{i:04d}'
            out_dir = tmp_root / f'out_{i:04d}'
            in_img_dir = in_dir / 'input'
            in_img_dir.mkdir(parents=True, exist_ok=True)
            out_dir.mkdir(parents=True, exist_ok=True)
            tile_name = f'tile_{i:04d}.png'
            tile_path = in_img_dir / tile_name
            tile_img.save(tile_path)

            gpu = gpu_ids[i % len(gpu_ids)]
            tasks_by_gpu[gpu].append((i, box, out_dir, tile_name, tile_path, in_img_dir))

        def _run_one_gpu_queue(gpu: int, queue_tasks: list[tuple]):
            failed_local = []
            for i, box, out_dir, tile_name, tile_path, in_img_dir in queue_tasks:
                cmd = [
                    sys.executable,
                    __file__,
                    '--pretrained_model_name_or_path', args.pretrained_model_name_or_path,
                    '--clip_path', args.clip_path,
                    '--inp_of_crossatt', args.inp_of_crossatt,
                    '--ckpt_dir', args.ckpt_dir,
                    '--used_clip_vision_layers', str(args.used_clip_vision_layers),
                    '--num_inference_steps', str(args.num_inference_steps),
                    '--time_threshold', str(args.time_threshold),
                    '--save_root', str(out_dir),
                    '--seed', str(args.seed),
                    '--img_path', str(tile_path),
                    '--img_path_dir', str(in_img_dir),
                    '--hf_cache_dir', args.hf_cache_dir,
                ]
                if args.used_clip_vision_global:
                    cmd.append('--used_clip_vision_global')
                if args.inp_of_unet_is_random_noise:
                    cmd.append('--inp_of_unet_is_random_noise')
                if args.local_files_only:
                    cmd.append('--local_files_only')

                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = str(gpu)
                rc = subprocess.call(cmd, env=env)
                if rc != 0:
                    failed_local.append((i, rc, gpu))
                    continue
                expected = out_dir / tile_name
                if not expected.exists():
                    failed_local.append((i, -1, gpu))
            return failed_local

        failed = []
        max_workers = max(1, len(gpu_ids))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = []
            for g in gpu_ids:
                q = tasks_by_gpu.get(g, [])
                if not q:
                    continue
                futs.append(ex.submit(_run_one_gpu_queue, g, q))
            for fut in as_completed(futs):
                failed.extend(fut.result())

        if failed:
            raise RuntimeError(f'Diff-Plugin tiled multi-GPU failed on tiles: {failed[:8]}')

        import numpy as np
        acc = np.zeros((height, width, 3), dtype=np.float32)
        wacc = np.zeros((height, width, 1), dtype=np.float32)

        for i, (x1, y1, x2, y2) in enumerate(tiles):
            tile_name = f'tile_{i:04d}.png'
            pred = Image.open(tmp_root / f'out_{i:04d}' / tile_name).convert('RGB')
            arr = np.asarray(pred, dtype=np.float32)
            h, w = arr.shape[:2]

            yy = np.linspace(0, 1, h, dtype=np.float32)
            xx = np.linspace(0, 1, w, dtype=np.float32)
            wy = np.minimum(yy, 1 - yy)
            wx = np.minimum(xx, 1 - xx)
            wm = np.outer(wy, wx)
            wm = np.maximum(wm, 1e-3)[..., None]

            acc[y1:y2, x1:x2, :] += arr * wm
            wacc[y1:y2, x1:x2, :] += wm

        out = (acc / np.maximum(wacc, 1e-6)).clip(0, 255).astype('uint8')
        merged = Image.fromarray(out)
        save_path = os.path.join(args.save_root, os.path.basename(args.img_path))
        merged.save(save_path)
        print('---------done (tiled-multi-gpu)-----------')



if __name__ == "__main__":

    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    transformers_logging.set_verbosity_error()
    diffusers_logging.set_verbosity_error()

    args = parse_args()

    if args.hf_cache_dir:
        os.environ.setdefault('HF_HOME', args.hf_cache_dir)
        os.environ.setdefault('HUGGINGFACE_HUB_CACHE', args.hf_cache_dir)
        os.environ.setdefault('TRANSFORMERS_CACHE', args.hf_cache_dir)

    if args.local_files_only:
        os.environ.setdefault('HF_HUB_OFFLINE', '1')
        os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

    _ensure_local_backbone(args)

    # step-1: settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SCBNet_path = os.path.join(args.ckpt_dir, "scb") 
    TPBNet_path = os.path.join(args.ckpt_dir, "tpb.pt")
    print('--------loading SCB from: ', SCBNet_path, '   , TPB from:  ', TPBNet_path, '----------------------')
    
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.save_root, exist_ok=True)

    if not os.path.isdir(SCBNet_path):
        raise FileNotFoundError(f'Diff-Plugin SCB checkpoint directory not found: {SCBNet_path}')
    if not os.path.isfile(TPBNet_path):
        raise FileNotFoundError(f'Diff-Plugin TPB checkpoint not found: {TPBNet_path}')
    
    ## [TODO] download checkpoints
    subtask_name = os.path.basename(args.ckpt_dir)
    
    
    # Step-2: instantiate models and schedulers
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=None,
        cache_dir=args.hf_cache_dir if args.hf_cache_dir else None,
        local_files_only=args.local_files_only,
    ).to(device)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=None,
        cache_dir=args.hf_cache_dir if args.hf_cache_dir else None,
        local_files_only=args.local_files_only,
    ).to(device)
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, None)
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=None,
        cache_dir=args.hf_cache_dir if args.hf_cache_dir else None,
        local_files_only=args.local_files_only,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=None,
        use_fast=False,
        cache_dir=args.hf_cache_dir if args.hf_cache_dir else None,
        local_files_only=args.local_files_only,
    )
    clip_v = CLIPVisionModel.from_pretrained(
        args.clip_path,
        cache_dir=args.hf_cache_dir if args.hf_cache_dir else None,
        local_files_only=args.local_files_only,
    ).to(device)
    noise_scheduler = UniPCMultistepScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
        cache_dir=args.hf_cache_dir if args.hf_cache_dir else None,
        local_files_only=args.local_files_only,
    )

    clip_image_processor = CLIPImageProcessor()
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    vae_image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_convert_rgb=True, do_normalize=True)
    
    scb_net = SCBNet.from_pretrained(SCBNet_path).to(device)
    tpb_net = TPBNet().to(device) 
    try:
        tpb_net.load_state_dict(torch.load(TPBNet_path)['model'], strict=True)
    except:
        tpb_net = torch.nn.DataParallel(tpb_net)
        tpb_net.load_state_dict(torch.load(TPBNet_path)['model'], strict=True)
 
    scb_net.eval()
    tpb_net.eval()


    # Step-3: prepare data
    from glob import glob
    img_formats = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')
    img_files = []
    for fmt in img_formats:
        img_files.extend(glob(os.path.join(args.img_path_dir, fmt)))

    if img_files:
        args.img_path = img_files[0]  # 选择第一个找到的图片
    else:
        raise FileNotFoundError(f"No image found in {args.img_path_dir}")

    # For large images (e.g., 4K), optionally use tiled multi-GPU inference to avoid OOM.
    if args.enable_multi_gpu_tiling:
        probe = load_image(args.img_path)
        long_side = max(probe.size[0], probe.size[1])
        if long_side >= int(args.tile_trigger_long_side):
            _run_tiled_multi_gpu(args)
            sys.exit(0)

    image = load_image(args.img_path)
    pil_image = image.copy()
    

    with torch.no_grad():
        # TPB
        clip_visual_input = clip_image_processor(images=image, return_tensors="pt").pixel_values.to(device=vae.device)
        prompt_embeds = tpb_net(clip_vision_outputs=clip_v(clip_visual_input, output_attentions=True, output_hidden_states=True),
                                use_global=args.used_clip_vision_global,
                                layer_ids=args.used_clip_vision_layers,)

        # resolution adjustment (one can adjust this resolution also, as long as the short side is equal to or larger than 512)
        width, height = image.size
        if width < 512 or height < 512:
            if width < height:
                new_width = 512
                new_height = int((512 / width) * height)
            else:
                new_height = 512
                new_width = int((512 / height) * width)
            image = image.resize((new_width, new_height))
        else:
            new_height = height
            new_width = width

        # pre-process image
        image = vae_image_processor.preprocess(image, height=new_height, width=new_width).to(device=vae.device)  # image now is tensor in [-1,1]
        scb_cond = vae.config.scaling_factor * torch.chunk(vae.quant_conv(vae.encoder(image)), 2, dim=1)[0]
        b, c, h, w = scb_cond.size()

        # set/load random seed
        generator = torch.Generator()
        generator.manual_seed(args.seed) # one can also adjust this seed to get different results

        # set the noise or latents
        if args.inp_of_unet_is_random_noise:
            latents = torch.randn((1,4, h, w), generator=generator).to(device)
        else:
            noise = torch.randn((1,4, h, w), generator=generator).to(device)

        # set the time step
        noise_scheduler.set_timesteps(args.num_inference_steps, device=vae.device)
        timesteps = noise_scheduler.timesteps
        timesteps = timesteps.long()

        # feedforward
        for i, t in enumerate(timesteps):
            # add noise 
            if t >= args.time_threshold and not args.inp_of_unet_is_random_noise:
                latents = noise_scheduler.add_noise(scb_cond, noise, t, )

            # SCB
            down_block_res_samples = scb_net(
                latents,
                t,
                encoder_hidden_states=prompt_embeds,
                cond_img=scb_cond,
                return_dict=False,
            )

            # diffusion unet
            noise_pred = unet(latents,
                t,
                encoder_hidden_states=prompt_embeds,
                down_block_additional_residuals= down_block_res_samples, 
            ).sample

            # update the latents
            latents = noise_scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # post-process
        pred = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
        pred = vae_image_processor.postprocess(pred, output_type='pil')[0]

        # resize back
        pred = pred.resize((width, height))
    
    pred.save(os.path.join(args.save_root, os.path.basename(args.img_path)))
    print('---------done-----------')
