import math

import torch
import torch.nn as nn


class SinusoidalPosEnc(nn.Module):
    def __init__(self, embed_dim: int, theta=10000):
        super().__init__()
        self.dim = embed_dim
        self.theta = theta

    def forward(self, x):
        half_dim = self.dim // 2
        enc = math.log(self.theta) / (half_dim - 1)
        enc = torch.exp(torch.arange(half_dim, device=x.device) * -enc)
        enc = x[:, None] * enc[None, :]
        enc = torch.cat((enc.sin(), enc.cos()), dim=-1)
        return enc
