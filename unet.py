from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

from layers.attention import LinearAttention
from layers.embedding import ClassDropout
from layers.mlp import FeedForward
from layers.norm import ChannelWiseLayerNorm
from layers.positional_encoding import SinusoidalPosEnc
from layers.spatial_transformer import SpatialTransformer


class Upsample(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: Optional[int] = None):
        super(Upsample, self).__init__(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=(
                    out_channels if out_channels is not None else in_channels
                ),
                kernel_size=1
            )
        )


class Downsample(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: Optional[int] = None):
        super(Downsample, self).__init__(
            Rearrange(
                'b c (h p1) (w p2) -> b (c p1 p2) h w',
                p1=2,
                p2=2
            ),
            nn.Conv2d(
                in_channels=in_channels*4,
                out_channels=(
                    out_channels if out_channels is not None else in_channels
                ),
                kernel_size=1
            )
        )


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )
        self.norm = ChannelWiseLayerNorm(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x, scale=0, shift=0):
        x = self.conv(x)
        x = self.norm(x)
        x = x * (scale + 1) + shift
        x = self.act(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: Optional[int] = None
    ):
        super().__init__()

        if time_dim:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_dim, out_channels * 2)
            )

        self.conv1 = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels
        )
        self.conv2 = ConvBlock(
            in_channels=out_channels,
            out_channels=out_channels
        )

        if in_channels != out_channels:
            self.residual_proj = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1
            )

    def forward(self, x, t=None):

        if hasattr(self, "time_mlp") and t is not None:
            t = self.time_mlp(t)  # [B, 2 * C]
            t = rearrange(t, 'b c -> b c 1 1')
            scale, shift = t.chunk(2, dim=1)
        else:
            scale, shift = 0, 0

        if hasattr(self, "residual_proj"):
            y = self.residual_proj(x)
        else:
            y = 0

        x = self.conv1(x, scale, shift)
        x = self.conv2(x)

        return x + y


class Unet(nn.Module):
    def __init__(
        self,
        channels: int,
        channel_mults=(1, 2, 4, 8),
        in_channels=3,
        class_conditional=False,
        class_dropout_prob=0.1,
        num_attrs=10,
        num_classes=10,
        num_spatial_transformer_layers: int = 1,
        head_dim=32,
        num_heads=4,
        full_attn=None,  # defaults to full attention only for inner most layer
        **kwargs
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.num_attrs = num_attrs

        self.from_image = nn.Conv2d(
            in_channels=in_channels,
            out_channels=channels,
            kernel_size=7,
            padding=3
        )

        dims = [channels, *map(lambda m: channels * m, channel_mults)]

        in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings

        self.time_dim = channels * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPosEnc(embed_dim=self.time_dim),
            FeedForward(embed_dim=self.time_dim, mult=1)
        )

        # class conditioning
        self.class_conditional = class_conditional

        if self.class_conditional:
            self.class_dropout = ClassDropout(
                class_dropout_prob,
                num_classes + 1
            )
            self.class_emb = nn.Embedding(num_classes + 1, self.time_dim)
            self.class_mlp = FeedForward(embed_dim=self.time_dim, mult=1)

        # attention
        if full_attn:
            full_attn = ((True,) * (len(channel_mults)))
        else:
            full_attn = (*((False,) * (len(channel_mults) - 1)), True)

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        num_resolutions = len(in_out)

        for idx, ((dim_in, dim_out), layer_full_attn) in enumerate(
            zip(in_out, full_attn)
          ):
            is_last = idx >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList([
                    ResidualBlock(
                        in_channels=dim_in,
                        out_channels=dim_in,
                        time_dim=self.time_dim
                    ),
                    ResidualBlock(
                        in_channels=dim_in,
                        out_channels=dim_in,
                        time_dim=self.time_dim
                    ),
                    (
                        SpatialTransformer(
                            channels=dim_in,
                            num_heads=num_heads,
                            num_layers=num_spatial_transformer_layers,
                            cond_dim=self.time_dim,
                        ) if layer_full_attn else
                        LinearAttention(
                            in_channels=dim_in,
                            num_heads=num_heads,
                            head_dim=head_dim,
                        )
                    ),
                    (
                        Downsample(
                            in_channels=dim_in,
                            out_channels=dim_out
                        ) if not is_last else
                        nn.Conv2d(
                            in_channels=dim_in,
                            out_channels=dim_out,
                            kernel_size=3,
                            padding=1
                        )
                    )
                ])
            )

        mid_channels = dims[-1]

        self.mid = nn.ModuleList([
            ResidualBlock(
                in_channels=mid_channels,
                out_channels=mid_channels,
                time_dim=self.time_dim
            ),
            SpatialTransformer(
                channels=mid_channels,
                num_heads=num_heads,
                num_layers=num_spatial_transformer_layers,
                cond_dim=self.time_dim,
            ),
            ResidualBlock(
                in_channels=mid_channels,
                out_channels=mid_channels,
                time_dim=self.time_dim
            ),
        ])

        for idx, ((dim_in, dim_out), layer_full_attn) in enumerate(
            zip(*map(reversed, (in_out, full_attn)))
        ):
            is_last = idx == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList([
                    ResidualBlock(
                        in_channels=dim_out + dim_in,
                        out_channels=dim_out,
                        time_dim=self.time_dim
                    ),
                    ResidualBlock(
                        in_channels=dim_out + dim_in,
                        out_channels=dim_out,
                        time_dim=self.time_dim
                    ),
                    (
                        SpatialTransformer(
                            channels=dim_out,
                            num_heads=num_heads,
                            num_layers=num_spatial_transformer_layers,
                            cond_dim=self.time_dim,
                        ) if layer_full_attn else
                        LinearAttention(
                            in_channels=dim_out,
                            num_heads=num_heads,
                            head_dim=head_dim,
                        )
                    ),
                    (
                        Upsample(
                            in_channels=dim_out,
                            out_channels=dim_in
                        ) if not is_last else
                        nn.Conv2d(
                            in_channels=dim_out,
                            out_channels=dim_in,
                            kernel_size=3,
                            padding=1
                        )
                    )
                ])
            )

        self.final = nn.ModuleList([
            ResidualBlock(
                in_channels=2 * channels,
                out_channels=channels,
                time_dim=self.time_dim
            ),
            nn.Conv2d(
                in_channels=channels,
                out_channels=in_channels,
                kernel_size=1
            )
        ])

    def forward(self, x, time, c=None):
        batch_size, *_ = x.shape
        # Embed time
        t = self.time_mlp(time)
        # Embed classes
        if self.class_conditional:
            if c is None:
                c = self.class_dropout.zero_class * torch.ones(
                    (batch_size, ),
                    device=x.device,
                    dtype=torch.long
                )
            else:
                c = self.class_dropout(c)

            c = self.class_emb(c)
            c = self.class_mlp(c)
            c = c.view((batch_size, 1, self.time_dim))

        # Project
        x = self.from_image(x)
        y = x
        h = []
        # Down
        for block1, block2, attn, downsample in self.downs:
            # Push through block 1
            x = block1(x, t)
            h.append(x)
            # Push through block 2
            x = block2(x, t)
            # Push through attn
            if isinstance(attn, SpatialTransformer):
                x = attn(x, c) + x
            else:
                x = attn(x) + x
            h.append(x)
            # Downsample
            x = downsample(x)
        # Push through bottleneck
        [mid_left, mid_attn, mid_right] = self.mid
        x = mid_left(x, t)
        x = mid_attn(x, c) + x
        x = mid_right(x, t)
        # Up
        for block1, block2, attn, upsample in self.ups:
            # Push through block 1
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            # Push through block 2
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            # Push through attn
            if isinstance(attn, SpatialTransformer):
                x = attn(x, c) + x
            else:
                x = attn(x) + x
            # Upsample
            x = upsample(x)
        # Push through final
        x = torch.cat((x, y), dim=1)
        [final_block, to_noise] = self.final
        x = final_block(x, t)
        x = to_noise(x)
        return x
