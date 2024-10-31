
import torch
import torch.nn as nn
from einops import rearrange

from .attention import CrossAttention, SelfAttention
from .mlp import FeedForward
from .norm import ChannelWiseLayerNorm


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        head_dim: int,
        cond_dim: int
    ):
        super().__init__()
        # Self-attention layer and pre-norm layer
        self.mhsa = SelfAttention(embed_dim, num_heads, head_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        # Cross attention layer and pre-norm layer
        self.mhca = CrossAttention(
            num_heads,
            embed_dim,
            cond_dim,
            head_dim
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        # Feed-forward network and pre-norm layer
        self.ffn = FeedForward(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        # x: [batch_size, height * width, channels]
        # c: [batch_size, num_cond, cond_dim]
        # Self attention
        h = x
        h = self.mhsa(self.norm1(h)) + x
        y = h
        # Cross-attention with conditioning
        h = self.mhca(self.norm2(h), c) + y
        z = h
        # Feed-forward network
        h = self.ffn(self.norm3(h)) + z
        return h


class SpatialTransformer(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        num_layers: int,
        cond_dim: int
    ):
        super().__init__()
        self.norm = ChannelWiseLayerNorm(channels)
        self.proj_in = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.blocks = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    num_heads=num_heads,
                    embed_dim=channels,
                    head_dim=channels // num_heads,
                    cond_dim=cond_dim
                ) for _ in range(num_layers)
            ]
        )
        self.proj_out = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        # x: [batch_size, height * width, channels]
        # c: [batch_size, num_cond, cond_dim]
        # For residual connection
        _, _, h, w = x.shape
        y = x
        # Normalize
        x = self.norm(x)
        # Initial $1 \times 1$ convolution
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c', h=h, w=w)
        # Apply the transformer layers
        for block in self.blocks:
            x = block(x, c)
        # Reshape and transpose
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + y
