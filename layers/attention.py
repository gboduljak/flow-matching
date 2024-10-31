from functools import partial

import torch
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from torch import nn

from .norm import ChannelWiseLayerNorm


class LinearAttention(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_heads=4,
        num_mem_kv=4,
        head_dim=32,
    ):
        super().__init__()
        self.scale = head_dim ** -0.5
        self.num_heads = num_heads
        hidden_dim = head_dim * num_heads

        self.norm = ChannelWiseLayerNorm(in_channels)
        self.mem_kv = nn.Parameter(
            torch.randn(2, num_heads, head_dim, num_mem_kv)
        )
        self.to_qkv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_dim * 3,
            kernel_size=1,
            bias=False
        )
        self.to_out = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=in_channels,
                kernel_size=1
            ),
            ChannelWiseLayerNorm(in_channels)
        )

    def forward(self, x):
        b, _, h, w = x.shape

        x = self.norm(x)

        q, k, v = map(
            lambda t: rearrange(
                t,
                'b (h c) x y -> b h c (x y)',
                h=self.num_heads
            ),
            self.to_qkv(x).chunk(3, dim=1)
        )
        mk, mv = map(
            lambda t: repeat(
                t,
                'h c n -> b h c n',
                b=b
            ),
            self.mem_kv
        )
        k, v = map(
            partial(torch.cat, dim=-1),
            [(mk, k), (mv, v)]
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        ctx = einsum(
            k, v, 'b h d n, b h e n -> b h d e'
        )

        out = einsum(
            ctx, q, 'b h d e, b h d n -> b h e n'
        )
        out = rearrange(
            out,
            'b h c (x y) -> b (h c) x y',
            h=self.num_heads,
            x=h,
            y=w
        )
        return self.to_out(out)


class SelfAttention(nn.Module):

    def __init__(
        self,
        channels: int,
        num_heads: int,
        head_dim: int
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = head_dim
        attn_dim = head_dim * num_heads
        # Attention scaling factor
        self.scale = head_dim ** -0.5
        # Query, key and value mappings
        self.to_qkv = nn.Linear(channels, 3 * attn_dim, bias=False)
        # Final linear layer
        self.to_out = nn.Linear(attn_dim, channels)

    def forward(self, x: torch.Tensor):
        # x: [batch_size, height * width, channels]
        [q, k, v] = self.to_qkv(x).chunk(3, dim=-1)

        x = F.scaled_dot_product_attention(
            q, k, v,
            scale=self.scale
        )
        x = self.to_out(x)

        return x


class CrossAttention(nn.Module):

    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        cond_dim: int,
        head_dim: int
    ):

        super().__init__()

        self.n_heads = num_heads
        self.d_head = head_dim

        # Attention scaling factor
        self.scale = head_dim ** -0.5

        # Query, key and value mappings
        attn_dim = head_dim * num_heads
        self.to_q = nn.Linear(embed_dim, attn_dim, bias=False)
        self.to_k = nn.Linear(cond_dim, attn_dim, bias=False)
        self.to_v = nn.Linear(cond_dim, attn_dim, bias=False)

        # Final linear layer
        self.to_out = nn.Linear(attn_dim, embed_dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        # x: [batch_size, height * width, d_model]
        # c: [batch_size, n_cond, d_cond]

        # Get query, key and value vectors
        q = self.to_q(x)  # [batch_size, height * width, attn_dim]
        k = self.to_k(c)  # [batch_size, height * width, attn_dim]
        v = self.to_v(c)  # [batch_size, height * width, attn_dim]

        x = F.scaled_dot_product_attention(
            q, k, v,
            scale=self.scale
        )
        x = self.to_out(x)

        return x
