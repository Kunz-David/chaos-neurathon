from typing import Dict, List
import math
import numpy as np

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from einops import einops
from torchvision.models import VGG19_Weights


def ddgauss(x, sigma):
    return ((x ** 2 / sigma ** 4) - 1 / sigma ** 2) * gauss(x, sigma)


# function D=dgauss(x,sigma)
#     D = - (x./sigma^2).*gauss(x,sigma);
def dgauss(x, sigma):
    return -(x / sigma ** 2) * gauss(x, sigma)


# function G = gauss(x,sigma)
#     G = 1/(sqrt(2*pi)*sigma) * exp(-x.^2/(2*sigma.^2));
def gauss(x, sigma):
    return 1 / (math.sqrt(2 * th.pi) * sigma) * th.exp(-x ** 2 / (2 * sigma ** 2))


def prepare_kernels(dev, config, channels_num=1):
    sigma = th.tensor(config['kernel_sigma'])

    sigma_space = th.arange(-th.ceil(sigma ** 3), th.ceil(sigma ** 3) + 1)

    g_kernel = gauss(sigma_space, sigma)[None]
    g_kernel = th.repeat_interleave(g_kernel, channels_num, dim=0)[:, None].to(dev)

    gd_kernel = dgauss(sigma_space, sigma)[None]
    gd_kernel = th.repeat_interleave(gd_kernel, channels_num, dim=0)[:, None].to(dev)

    gdd_kernel = ddgauss(sigma_space, sigma)[None]
    gdd_kernel = th.repeat_interleave(gdd_kernel, channels_num, dim=0)[:, None].to(dev)

    return g_kernel, gd_kernel, gdd_kernel, sigma_space


def compute_image_derivatives(frame_m1p1_d, kernel_x, kernel_y, sigma_space):
    b, c, h, w = frame_m1p1_d.shape

    frame_m1p1_d_w = einops.rearrange(frame_m1p1_d, 'b c h w -> (b h) c w')

    frame_m1p1_dx = torch.nn.functional.conv1d(frame_m1p1_d_w, kernel_x, groups=c, padding=sigma_space.shape[0] // 2)
    frame_m1p1_dx_h = einops.rearrange(frame_m1p1_dx, '(b h) c w -> (b w) c h', h=h, w=w, c=c)

    frame_m1p1_dxdy = torch.nn.functional.conv1d(frame_m1p1_dx_h, kernel_y, groups=c, padding=sigma_space.shape[0] // 2)
    frame_m1p1_dxdy = einops.rearrange(frame_m1p1_dxdy, '(b w) c h -> b c h w', h=h, w=w, c=c)

    return frame_m1p1_dxdy


def to_uint8(f_m1p1):
    f = (f_m1p1[0, 0].cpu().detach().numpy() * 127.5 + 127.5).astype(np.uint8)
    return f

def compute_gradient_loss(target_m1p1, output_m1p1, y_kernel, x_kernel, sigma_space):
    target_m1p1_dy = compute_image_derivatives(target_m1p1.detach(), y_kernel, x_kernel, sigma_space)
    target_m1p1_dx = compute_image_derivatives(target_m1p1.detach(), x_kernel, y_kernel, sigma_space)

    output_m1p1_dy = compute_image_derivatives(output_m1p1, y_kernel, x_kernel, sigma_space)
    output_m1p1_dx = compute_image_derivatives(output_m1p1, x_kernel, y_kernel, sigma_space)

    return ((target_m1p1_dy - output_m1p1_dy) ** 2).mean([-1, -2, -3]) + ((target_m1p1_dx - output_m1p1_dx) ** 2).mean([-1, -2, -3])


class GramMatrix(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        n, c, h, w = x.shape
        M = x.view(n, c, h * w)
        G = torch.bmm(M, M.transpose(1, 2))
        G.div_(h * w * c)
        return G


class Vgg19_Extractor(nn.Module):
    def __init__(self, capture_layers):
        super().__init__()
        self.vgg_layers = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        # Load the old model if requested
        self.vgg_layers = self.vgg_layers.features
        self.len_layers = 37  # len(self.vgg_layers)

        for param in self.parameters():
            param.requires_grad = False
        self.capture_layers = capture_layers

    def forward(self, x):
        feat = []
        if -1 in self.capture_layers:
            feat.append(x)
        i = 0
        for mod in self.vgg_layers:
            x = mod(x)
            i += 1
            if i in self.capture_layers:
                feat.append(x)
        return feat


class InnerProductLoss(nn.Module):
    def __init__(self, capture_layers, device):
        super().__init__()
        self.layers = capture_layers
        self.device = device
        self.vgg = Vgg19_Extractor(capture_layers).to(device)
        self.stored_mean = (torch.Tensor([0.485, 0.456, 0.406]).to(device).view(1, -1, 1, 1))
        self.stored_std = (torch.Tensor([0.229, 0.224, 0.225]).to(device).view(1, -1, 1, 1))
        self.gmm = GramMatrix()
        self.dist = nn.MSELoss()
        self.cache: Dict[float, List[torch.Tensor]] = {0.: [torch.empty((0))]}  # torch.Tensor

    def extractor(self, x):
        # remap x to vgg range
        x = (x + 1.) / 2.
        x = x - self.stored_mean
        x = x / self.stored_std
        return self.vgg(x)

    def run_scale(self, frame_y, pure_y, cache_y2: bool = False, scale: float = 1.):
        frame_y = F.interpolate(frame_y, scale_factor=float(scale), mode='bilinear', align_corners=False,
                                recompute_scale_factor=False)
        feat_frame_y = self.extractor(frame_y)
        if cache_y2:
            if scale not in self.cache:
                pure_y = F.interpolate(pure_y, scale_factor=scale, mode='bilinear', align_corners=False)
                self.cache[scale] = [self.gmm(l) for idx, l in enumerate(
                    self.extractor(pure_y))]
            gmm_pure_y = self.cache[scale]
            raise RuntimeError("Cache not implemented")
        else:
            pure_y = F.interpolate(pure_y, scale_factor=scale, mode='bilinear', align_corners=False)
            feat_pure_y = self.extractor(pure_y)
            gmm_pure_y = [self.gmm(l) for idx, l in enumerate(feat_pure_y)]

        # loss : List[torch.Tensor] = []
        loss = torch.empty((len(feat_frame_y),)).to(frame_y.device)
        for l in range(len(feat_frame_y)):
            gmm_frame_y = self.gmm(feat_frame_y[l])
            assert gmm_pure_y[l].shape[0] == gmm_frame_y.shape[0]
            # assert not (gmm_y2[l].requires_grad)
            dist = self.dist(gmm_pure_y[l].detach(), gmm_frame_y)
            loss[l] = dist
        return torch.sum(loss)

    def forward(self, frame_y, pure_y, cache_y2: bool = False):
        scale_1_loss = self.run_scale(frame_y, pure_y, cache_y2, scale=1.0)
        # scale_2_loss = self.run_scale(y1, y2, cache_y2, scale=0.5)
        return scale_1_loss

