"""
FastDVDnet denoising algorithm

@author: Matias Tassano <mtassano@parisdescartes.fr>
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm

import pdb


def temp_denoise(model, noisyframe, sigma_noise):
    '''Encapsulates call to denoising model and handles padding.
        Expects noisyframe to be normalized in [0., 1.]
    '''
    # make size a multiple of four (we have two scales in the denoiser)
    sh_im = noisyframe.size()
    expanded_h = sh_im[-2] % 4
    if expanded_h:
        expanded_h = 4 - expanded_h
    expanded_w = sh_im[-1] % 4
    if expanded_w:
        expanded_w = 4 - expanded_w
    padexp = (0, expanded_w, 0, expanded_h)
    noisyframe = F.pad(input=noisyframe, pad=padexp, mode='reflect')
    sigma_noise = F.pad(input=sigma_noise, pad=padexp, mode='reflect')

    # denoise
    # pdb.set_trace()
    # (Pdb) noisyframe.size()
    # torch.Size([1, 15, 540, 960])
    # (Pdb) sigma_noise.size()
    # torch.Size([1, 1, 540, 960])

    out = torch.clamp(model(noisyframe, sigma_noise), 0., 1.)

    if expanded_h:
        out = out[:, :, :-expanded_h, :]
    if expanded_w:
        out = out[:, :, :, :-expanded_w]

    return out


def denoise_seq_fastdvdnet(seq, noise_std, temp_psz, model_temporal):
    r"""Denoises a sequence of frames with FastDVDnet.

    Args:
        seq: Tensor. [numframes, 1, C, H, W] array containing the noisy input frames
        noise_std: Tensor. Standard deviation of the added noise
        temp_psz: size of the temporal patch
        model_temp: instance of the PyTorch model of the temporal denoiser
    Returns:
        denframes: Tensor, [numframes, C, H, W]
    """
    # init arrays to handle contiguous frames and related patches
    numframes, C, H, W = seq.shape
    ctrlfr_idx = int((temp_psz - 1) // 2)
    # ctrlfr_idx == 2
    inframes = list()
    denframes = torch.empty((numframes, C, H, W)).to(seq.device)

    # build noise map from noise std---assuming Gaussian noise
    noise_map = noise_std.expand((1, 1, H, W))
    # (Pdb) noise_std
    # tensor([0.0392], device='cuda:0')
    # (Pdb) noise_std.size()
    # torch.Size([1])

    # pdb.set_trace()
    # (Pdb) seq.shape
    # torch.Size([100, 3, 540, 960])
    # (Pdb) temp_psz 5
    # (Pdb) noise_map.size()
    # torch.Size([1, 1, 540, 960])
    # (Pdb) denframes.size()
    # torch.Size([100, 3, 540, 960])

    for fridx in tqdm(range(numframes)):
        # load input frames
        if not inframes:
        # if list not yet created, fill it with temp_patchsz frames
            for idx in range(temp_psz):
                relidx = abs(idx - ctrlfr_idx)  # handle border conditions, reflect
                inframes.append(seq[relidx])
        else:
            del inframes[0]
            relidx = min(fridx + ctrlfr_idx, -fridx + 2 * (numframes - 1) -
                         ctrlfr_idx)  # handle border conditions
            inframes.append(seq[relidx])

        inframes_t = torch.stack(inframes, dim=0).contiguous().view(
            (1, temp_psz * C, H, W)).to(seq.device)

        # append result to output list
        denframes[fridx] = temp_denoise(model_temporal, inframes_t, noise_map)

    # free memory up
    del inframes
    del inframes_t
    torch.cuda.empty_cache()

    # convert to appropiate type and return
    # pdb.set_trace()
    return denframes
