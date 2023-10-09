import torch
import numpy as np
import torch.nn.functional as F
import imageio
from math import exp
HUGE_NUMBER = 1e10
TINY_NUMBER = 1e-6

to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
mse2psnr = lambda x: -10.0 * np.log(x + TINY_NUMBER) / np.log(10.0)

def img2mse(x, y, mask=None):
    """
    :param x: img 1, [(...), 3]
    :param y: img 2, [(...), 3]
    :param mask: optional, [(...)]
    :return: mse score
    """
    if mask is None:
        return torch.mean((x - y) * (x - y))
    else:
        return torch.sum((x - y) * (x - y) * mask.unsqueeze(-1)) / (
            torch.sum(mask) * x.shape[-1] + TINY_NUMBER
        )

def img2psnr(x, y, mask=None):
    return mse2psnr(img2mse(x, y, mask).item())

def imwrite(path, image):
    imageio.imwrite(path, to8b(image))