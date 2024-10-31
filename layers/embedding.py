import torch
import torch.nn as nn


class ClassDropout(nn.Module):
    def __init__(self, p: float, num_classes: int):
        super(ClassDropout, self).__init__()
        self.p = p
        self.zero_class = num_classes - 1

    def forward(self, x):
        if self.training:
            x = x.float()
            y = torch.where(
                torch.rand_like(x) > self.p,
                x,
                self.zero_class * torch.ones_like(x)
            ).long()
            return y

        return x
