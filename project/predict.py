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
from model import get_model, model_load, model_setenv

if __name__ == "__main__":
    """Predict."""

    model_setenv()

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default="models/VideoClean.pth", help="checkpint file")
    parser.add_argument('--sigma', type=float, default=10, help="Noise Sigma")
    parser.add_argument(
        '--input', type=str, default="dataset/predict/input", help="video input folder")
    parser.add_argument(
        '--output', type=str, default="dataset/predict/output", help="video output folder")

    args = parser.parse_args()

    # CPU or GPU ?
    device = torch.device(os.environ["DEVICE"])

    model = get_model()
    model_load(model, args.checkpoint)
    model.to(device)
    model.eval()

    if os.environ["ENABLE_APEX"] == "YES":
        from apex import amp
        model = amp.initialize(model, opt_level="O1")

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
