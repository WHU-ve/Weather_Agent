#!/bin/bash
# 下载预训练模型的脚本
# 只下载 deraining, dehazing, denoising, desnowing 相关的模型

pip install huggingface_hub

# 下载整个 tar.gz 文件
huggingface-cli download YSZuo/4KAgent-Toolbox-Pretrained-Models 4KAgent_toolbox_pretrained_ckpts.tar.gz --local-dir . --repo-type model

# 创建目录
mkdir -p pretrained_ckpts

# 解压需要的文件
tar -xzf 4KAgent_toolbox_pretrained_ckpts.tar.gz -C pretrained_ckpts X-Restormer/derain_155k.pth
tar -xzf 4KAgent_toolbox_pretrained_ckpts.tar.gz -C pretrained_ckpts X-Restormer/dehaze_300k.pth
tar -xzf 4KAgent_toolbox_pretrained_ckpts.tar.gz -C pretrained_ckpts X-Restormer/denoise_300k.pth
tar -xzf 4KAgent_toolbox_pretrained_ckpts.tar.gz -C pretrained_ckpts Restormer/deraining.pth
tar -xzf 4KAgent_toolbox_pretrained_ckpts.tar.gz -C pretrained_ckpts Restormer/real_denoising.pth
tar -xzf 4KAgent_toolbox_pretrained_ckpts.tar.gz -C pretrained_ckpts MPRNet/model_deraining.pth
tar -xzf 4KAgent_toolbox_pretrained_ckpts.tar.gz -C pretrained_ckpts MPRNet/model_denoising.pth
tar -xzf 4KAgent_toolbox_pretrained_ckpts.tar.gz -C pretrained_ckpts MAXIM/maxim_ckpt_Deraining_Rain13k_checkpoint.npz
tar -xzf 4KAgent_toolbox_pretrained_ckpts.tar.gz -C pretrained_ckpts MAXIM/maxim_ckpt_Dehazing_SOTS-Outdoor_checkpoint.npz
tar -xzf 4KAgent_toolbox_pretrained_ckpts.tar.gz -C pretrained_ckpts MAXIM/maxim_ckpt_Denoising_SIDD_checkpoint.npz
tar -xzf 4KAgent_toolbox_pretrained_ckpts.tar.gz -C pretrained_ckpts SwinIR/model_zoo/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth
tar -xzf 4KAgent_toolbox_pretrained_ckpts.tar.gz -C pretrained_ckpts SwinIR/model_zoo/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth
tar -xzf 4KAgent_toolbox_pretrained_ckpts.tar.gz -C pretrained_ckpts NAFNet/NAFNet-SIDD-width64.pth
tar -xzf 4KAgent_toolbox_pretrained_ckpts.tar.gz -C pretrained_ckpts RIDCP_dehazing/pretrained_RIDCP.pth
tar -xzf 4KAgent_toolbox_pretrained_ckpts.tar.gz -C pretrained_ckpts RIDCP_dehazing/weight_for_matching_dehazing_Flickr.pth
tar -xzf 4KAgent_toolbox_pretrained_ckpts.tar.gz -C pretrained_ckpts Diff-Plugin/derain
tar -xzf 4KAgent_toolbox_pretrained_ckpts.tar.gz -C pretrained_ckpts Diff-Plugin/dehaze
tar -xzf 4KAgent_toolbox_pretrained_ckpts.tar.gz -C pretrained_ckpts Diff-Plugin/desnow
tar -xzf 4KAgent_toolbox_pretrained_ckpts.tar.gz -C pretrained_ckpts DehazeFormer

# 删除 tar.gz 文件
rm 4KAgent_toolbox_pretrained_ckpts.tar.gz

echo "模型下载和解压完成"