
import torch
import torch.nn as nn


class ChannelWiseLayerNorm(nn.Module):
    def __init__(self, channels: int, epsilon: float = 1e-5):
        super(ChannelWiseLayerNorm, self).__init__()

        self.scale = nn.Parameter(torch.ones((channels, )))
        self.shift = nn.Parameter(torch.ones((channels, )))
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor):
        _, channels, _, _ = x.shape

        scale = self.scale.view((1, channels, 1, 1))
        shift = self.shift.view((1, channels, 1, 1))

        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True)

        return (
            scale * ((x - mean) / torch.sqrt(var + self.epsilon)) +
            shift
        )
