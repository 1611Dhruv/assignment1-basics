import numpy as np
import torch
import torch.nn as nn
from einops import einsum, rearrange


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        sigma = np.sqrt(2 / (in_features + out_features))
        self.weights = nn.Parameter(
            nn.init.trunc_normal_(
                torch.zeros(
                    out_features,
                    in_features,
                    device=device,
                    dtype=dtype,
                ),
                mean=0,
                std=sigma,
                a=-3 * sigma,
                b=3 * sigma,
            ),
        )
        self.device = device
        self.dtype = dtype
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for linear layer
        applies y = Wx
        """
        return einsum(x, self.weights, "... d_in, d_out d_in -> ... d_out")
