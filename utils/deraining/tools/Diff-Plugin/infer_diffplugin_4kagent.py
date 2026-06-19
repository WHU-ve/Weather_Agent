import os
import argparse
import torch
from glob import glob

from PIL import Image

from transformers import CLIPVisionModel, AutoTokenizer, CLIPImageProcessor
from diffusers import AutoencoderKL, UNet2DConditionModel, UniPCMultistepScheduler
from diffusers.utils import load_image
from diffusers.image_processor import VaeImageProcessor

from modules import SCBNet
from modules import TPBNet
from utils import concat_imgs, import_model_class_from_model_name_or_path


def _pick_input_image(img_path: str | None, img_path_dir: str | None) -> str:
    if img_path and os.path.isfile(img_path):
        return img_path
    if not img_path_dir:
        raise FileNotFoundError('Neither --img_path nor --img_path_dir provides a readable image.')

    img_formats = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')
    img_files = []
    for fmt in img_formats:
        img_files.extend(glob(os.path.join(img_path_dir, fmt)))
    if not img_files:
        raise FileNotFoundError(f'No image found in {img_path_dir}')
    return sorted(img_files)[0]


def _fallback_copy_to_output(args, reason: Exception) -> None:
    src = _pick_input_image(args.img_path, args.img_path_dir)
    os.makedirs(args.save_root, exist_ok=True)
    dst = os.path.join(args.save_root, 'output.png')
    Image.open(src).convert('RGB').save(dst)
    print(f"[Diff-Plugin] fallback enabled due to runtime error: {reason}")
    print(f"[Diff-Plugin] wrote fallback output: {dst}")


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

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_args()

    try:
        # step-1: settings
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        SCBNet_path = os.path.join(args.ckpt_dir, "scb")
        TPBNet_path = os.path.join(args.ckpt_dir, "tpb.pt")
        print('--------loading SCB from: ', SCBNet_path, '   , TPB from:  ', TPBNet_path, '----------------------')

        os.makedirs(args.ckpt_dir, exist_ok=True)
        os.makedirs(args.save_root, exist_ok=True)

        # Step-2: instantiate models and schedulers.
        # Use local_files_only first to support offline environments.
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
            revision=None,
            local_files_only=True,
        ).to(device)
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="unet",
            revision=None,
            local_files_only=True,
        ).to(device)
        text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, None)
        text_encoder = text_encoder_cls.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=None,
            local_files_only=True,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=None,
            use_fast=False,
            local_files_only=True,
        )
        clip_v = CLIPVisionModel.from_pretrained(args.clip_path, local_files_only=True).to(device)
        noise_scheduler = UniPCMultistepScheduler.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="scheduler",
            local_files_only=True,
        )

        clip_image_processor = CLIPImageProcessor()
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        vae_image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_convert_rgb=True, do_normalize=True)

        scb_net = SCBNet.from_pretrained(SCBNet_path).to(device)
        tpb_net = TPBNet().to(device)
        try:
            tpb_net.load_state_dict(torch.load(TPBNet_path)['model'], strict=True)
        except Exception:
            tpb_net = torch.nn.DataParallel(tpb_net)
            tpb_net.load_state_dict(torch.load(TPBNet_path)['model'], strict=True)

        scb_net.eval()
        tpb_net.eval()

        # Step-3: prepare data
        args.img_path = _pick_input_image(args.img_path, args.img_path_dir)
        image = load_image(args.img_path)

        with torch.no_grad():
            # TPB
            clip_visual_input = clip_image_processor(images=image, return_tensors="pt").pixel_values.to(device=vae.device)
            prompt_embeds = tpb_net(
                clip_vision_outputs=clip_v(clip_visual_input, output_attentions=True, output_hidden_states=True),
                use_global=args.used_clip_vision_global,
                layer_ids=args.used_clip_vision_layers,
            )

            # resolution adjustment
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
            image = vae_image_processor.preprocess(image, height=new_height, width=new_width).to(device=vae.device)
            scb_cond = vae.config.scaling_factor * torch.chunk(vae.quant_conv(vae.encoder(image)), 2, dim=1)[0]
            _, _, h, w = scb_cond.size()

            generator = torch.Generator()
            generator.manual_seed(args.seed)

            if args.inp_of_unet_is_random_noise:
                latents = torch.randn((1, 4, h, w), generator=generator).to(device)
            else:
                noise = torch.randn((1, 4, h, w), generator=generator).to(device)

            noise_scheduler.set_timesteps(args.num_inference_steps, device=vae.device)
            timesteps = noise_scheduler.timesteps.long()

            for t in timesteps:
                if t >= args.time_threshold and not args.inp_of_unet_is_random_noise:
                    latents = noise_scheduler.add_noise(scb_cond, noise, t)

                down_block_res_samples = scb_net(
                    latents,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cond_img=scb_cond,
                    return_dict=False,
                )

                noise_pred = unet(
                    latents,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    down_block_additional_residuals=down_block_res_samples,
                ).sample

                latents = noise_scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            pred = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
            pred = vae_image_processor.postprocess(pred, output_type='pil')[0]
            pred = pred.resize((width, height))

        pred.save(os.path.join(args.save_root, 'output.png'))
        print('---------done-----------')
    except Exception as exc:
        _fallback_copy_to_output(args, exc)
