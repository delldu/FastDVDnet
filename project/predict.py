"""Model predict."""
# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2020, All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 11月 02日 星期一 17:51:08 CST
# ***
# ************************************************************************************/
#
import argparse
import glob
import os
import pdb

import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from data import Video
from model import enable_amp, get_model, model_device, model_load

if __name__ == "__main__":
    """Predict."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default="models/VideoClean.pth", help="checkpint file")
    parser.add_argument('--sigma', type=float, default=10, help="Noise Sigma")
    parser.add_argument(
        '--input', type=str, default="dataset/predict/input", help="video input folder")
    parser.add_argument(
        '--output', type=str, default="dataset/predict/output", help="video output folder")

    args = parser.parse_args()

    model = get_model()
    model_load(model, args.checkpoint)
    device = model_device()
    model.to(device)
    model.eval()

    enable_amp(model)

    totensor = transforms.ToTensor()
    toimage = transforms.ToPILImage()

    video = Video()
    video.reset(args.input)
    progress_bar = tqdm(total=len(video))
    noise_std = torch.FloatTensor([args.sigma/255.0])

    for index in range(len(video)):
        progress_bar.update(1)

        # print(index, ":", video[index].size())
        input_tensor = video[index].unsqueeze(0)
        N, C, H, W = input_tensor.size()
        noise_tensor = noise_std.expand((N, 1, H, W))

        input_tensor = input_tensor.to(device)
        noise_tensor = noise_tensor.to(device)

        with torch.no_grad():
            output_tensor = model(
                input_tensor, noise_tensor).clamp(0, 1.0).squeeze()

        toimage(output_tensor.cpu()).save(
            "{}/{:06d}.png".format(args.output, index + 1))
