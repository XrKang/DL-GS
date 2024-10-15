#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2, mask=None):
    if mask is None:
        mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    else:
        mask_bin = (mask == 1.)
        mse = (((img1 - img2)[mask_bin]) ** 2).mean()
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def center_crop_shift(img, shift=24):
	_, h, w= img.shape
	img = img[:, h//4:h//4*3, w//4+shift:w//4*3+shift]
	return img


def center_crop_shift_vertical(img, shift):
	_, h, w= img.shape
	img = img[:, (h // 10 * 3)-shift: (h // 10 * 8)-shift, (w // 16 * 4): (w // 16 * 12)]
	return img