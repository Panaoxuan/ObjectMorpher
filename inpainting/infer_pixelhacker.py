import os
import os.path as osp

import sys
sys.path.append(".")

from copy import deepcopy
from typing import Union, Dict, List
from types import SimpleNamespace
from pathlib import Path
from tqdm import tqdm

from PIL import Image, ImageFilter
import cv2
import numpy as np

import torch
import torch.utils.data
import torch.utils
import torch.nn as nn
from torchvision import transforms as TF
from diffusers import DDIMScheduler, AutoencoderKL


from utils import load_cfg, build_model, build_vae
from dataset import SimpleInferDataset
from pipeline import PixelHacker_Pipeline

import argparse

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/home/xyhugo/2D-SpaceEdit/inpainting/config/PixelHacker_sdvae_f8d4.yaml")
    parser.add_argument("--weight", default="/home/xyhugo/2D-SpaceEdit/inpainting/weight/ft_places2/diffusion_pytorch_model.bin") # [FIX] ft_places -> ft_places2
    parser.add_argument("--image_dir", default="imgs")
    parser.add_argument("--mask_dir", default="masks")
    parser.add_argument("--output_dir", default="outputs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse()

    model_cfg = load_cfg(args.config)

    model = build_model(model_cfg, 20).to(device)
    # import os
    # print("当前工作目录:", os.getcwd())
    # import pdb;pdb.set_trace()
    print(model.load_state_dict(torch.load(args.weight, device)))
    # weight/pretrained/diffusion_pytorch_model.bin
    # weight/ft_places2/diffusion_pytorch_model.bin
    # weight/ft_celebahq/diffusion_pytorch_model.bin
    # weight/ft_ffhq/diffusion_pytorch_model.bin

    vae = build_vae(model_cfg).to(device)
    scheduler = DDIMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
        num_train_timesteps=1000, clip_sample=False)

    pipe = PixelHacker_Pipeline(
        model=model,
        vae=vae,
        scheduler=scheduler,
        device=device,
        dtype=torch.float)

    vae_ds_ratio = 2 ** (len(vae.config.block_out_channels) - 1)
    img_size = model.diff_model.config.sample_size * vae_ds_ratio
    assert img_size == model_cfg['data']['image_size']

    dataset = SimpleInferDataset(img_dir = args.image_dir, mask_dir = args.mask_dir)

    save_root = Path(args.output_dir)
    save_root.mkdir(parents=True, exist_ok=True)
    for idx, (image, mask, iname) in tqdm(enumerate(dataset)):
        image = image.resize((img_size,img_size))
        mask = mask.resize((img_size,img_size))

        out = pipe(
            image, mask,
            image_size=img_size,
            num_steps=20,
            strength=0.999,
            guidance_scale=4.5,
            noise_offset=0.0357,
            paste=False,
        )[0]

        out.save(save_root.joinpath(iname)) # [FIX] str(iname)+".png" -> iname
