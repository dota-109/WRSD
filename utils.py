import math
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from scipy.signal import convolve2d
import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import Counter

# ========================================KL_distance=========================
def KL(tensor_1, tensor_2):
    kl_loss = torch.nn.KLDivLoss(reduce=True)
    bce_loss = torch.nn.BCELoss(reduce=True)
    dis1 = calcu_distribution(tensor_1)
    dis2 = calcu_distribution(tensor_2)
    
    dis1_n = dis1.cpu().numpy()
    dis2_n = dis2.cpu().numpy()
    
    dis1_n = to_one_hot(dis1_n)
    dis2_n = to_one_hot(dis2_n)
    
    dis1 = F.log_softmax(dis1)
    dis2 = F.softmax(dis2)
    return kl_loss(dis1, dis2)+1*bce_loss(torch.from_numpy(dis1_n).cuda(),torch.from_numpy(dis2_n).cuda())


def calcu_distribution(tensor):
    image_distribution = np.empty([0, 256])
    base_line = np.linspace(-1, 1, 256)

    input_np = tensor.cpu().detach().numpy()
    input_np = (input_np/2+0.5)*255
    for idx in input_np:

        dis = np.zeros((1, 256))
        # dis = []
        idx_reshape = np.reshape(idx, [idx.shape[1], idx.shape[2]])
        static_dict = Counter(np.ceil(idx_reshape.flatten()))
        for k, v in static_dict.items():
                dis[:, k.astype("int")] = v
        image_distribution = np.append(image_distribution, dis, axis=0)
    image_distribution = torch.from_numpy(image_distribution)
    image_distribution = image_distribution.cuda()

    return image_distribution

def to_one_hot(array):
    i = 0
    for index in array:
        norm_index = np.floor(index/np.max(index))
        array[i] = norm_index
        i += 1
    return array


# =====================================================SSIM====================
def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)


def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):
    im1 = np.array(im1)
    im2 = np.array(im2)
    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    M, N = im1.shape
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    window = matlab_style_gauss2D(shape=(win_size, win_size), sigma=1.5)
    window = window / np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1 * im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2 * im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1 * im2, window, 'valid') - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigmal2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return np.mean(np.mean(ssim_map))

#============================================WDSR Residual Block=============================

class WDSR_RESBLOCK(nn.Module):
    def __init__(self):
        super(WDSR_RESBLOCK, self).__init__()
        WDSR_base = [nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=True),
                     nn.ReLU(),
                     nn.Conv2d(256, 48, kernel_size=1, stride=1, padding=0, bias=True),
                     nn.ReLU(),
                     nn.Conv2d(48, 48, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=True),
                     nn.Conv2d(48, 64, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=True)]
        self.body = nn.Sequential(*WDSR_base)

    def forward(self, x):
        y = self.body(x)
        return x + y
