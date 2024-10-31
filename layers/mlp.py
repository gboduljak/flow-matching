from torch import nn


class FeedForward(nn.Sequential):
    def __init__(self, embed_dim: int, mult: int = 4):
        super().__init__(
            nn.Linear(embed_dim, embed_dim * mult),
            nn.GELU(),
            nn.Linear(embed_dim * mult, embed_dim)
        )
