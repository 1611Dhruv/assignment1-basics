import torch
import torch.nn as nn
from einops import einsum, rearrange
from numpy import sqrt


def _get_mat(
    d1: int,
    d2: int,
    device: torch.device | None = None,
    dtype: torch.device | None = None,
) -> torch.Tensor:
    sigma = sqrt(2 / (d1 + d2))
    return nn.init.trunc_normal_(
        torch.zeros(
            d1,
            d2,
            device=device,
            dtype=dtype,
        ),
        mean=0,
        std=sigma,
        a=-3 * sigma,
        b=3 * sigma,
    )


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.device | None = None,
    ):
        super().__init__()
        self.weights = nn.Parameter(
            _get_mat(
                out_features,
                in_features,
                device,
                dtype,
            )
        )
        self.device = device
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for linear layer
        applies (... d_in) -> (... d_out)
        """
        return einsum(x, self.weights, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.device | None = None,
    ):
        super().__init__()
        self.embedding = nn.Parameter(
            nn.init.trunc_normal_(
                torch.zeros(
                    num_embeddings,
                    embedding_dim,
                    device=device,
                    dtype=dtype,
                ),
                mean=0,
                std=1,
                a=-3,
                b=3,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for embedding layer
        applies (... id) -> (... d_model)
        """
        return self.embedding[x]


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d_model))
        self.d_model = d_model
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_type = x.dtype
        x = x.to(torch.float32)
        rms = 1 / torch.sqrt(1 / self.d_model * einsum(x, x, "... dim_model, ... dim_model -> ...") + self.eps)
        norm = einsum(x, rms, self.g, "... dim_model, ..., dim_model -> ... dim_model")
        x = x.to(x_type)
        return norm


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.w1 = nn.Parameter(_get_mat(d_ff, d_model, device, dtype))
        self.w2 = nn.Parameter(_get_mat(d_model, d_ff, device, dtype))
        self.w3 = nn.Parameter(_get_mat(d_ff, d_model, device, dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        act1 = einsum(x, self.w1, "... d_model, d_ff d_model -> ... d_ff")
        act2 = einsum(x, self.w3, "... d_model, d_ff d_model -> ... d_ff")
        activation = act1 * torch.sigmoid(act1) * act2
        return einsum(activation, self.w2, "... d_ff, d_model d_ff -> ... d_model")
