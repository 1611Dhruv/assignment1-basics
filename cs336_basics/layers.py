from concurrent.futures import ProcessPoolExecutor, as_completed
from math import cos, sin

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


class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class NoPE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        return x


class RoPE(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        """
        theta: value for rope
        d_k: dimension of query key
        max_seq_len: Maximum sequence length for input
        device: Device to store data
        """
        super().__init__()
        assert d_k % 2 == 0, "Model dim must be even to make pair ropes encodable"
        self.register_buffer(
            "sin_buff",
            torch.tensor([[[sin(i / theta ** (2 * k / d_k))] for k in range(d_k // 2)] for i in range(max_seq_len)]),
            persistent=False,
        )
        self.register_buffer(
            "cos_buff",
            torch.tensor([[[cos(i / theta ** (2 * k / d_k))] for k in range(d_k // 2)] for i in range(max_seq_len)]),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:

        x_p = rearrange(x, "... (pair two) -> ... pair two", two=2)
        q1, q2 = x_p.chunk(2, dim=-1)
        rope_cos = x_p * self.cos_buff[token_positions]
        rope_sin = torch.cat([-q2, q1], dim=-1) * self.sin_buff[token_positions]
        roped = rearrange(rope_sin + rope_cos, "... pair two -> ... (pair two)")
        return roped


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    stable = x - torch.max(x, dim=dim, keepdim=True).values
    e = torch.exp(stable)
    return e / e.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor):
    projs = einsum(Q, K, "... seq_q d_k, ... seq_k d_k -> ... seq_q seq_k") / sqrt(Q.shape[-1])
    masked_projs = projs.masked_fill(mask == False, float("-inf"))
    probs = softmax(masked_projs, -1)
    return einsum(probs, V, "... seq_q seq_k, ... seq_k d_v -> ... seq_q d_v")


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        positional_encoding: nn.Module = None,
    ):
        super().__init__()
        d_k = d_model // num_heads
        d_v = d_model // num_heads
        self.h = num_heads

        self.wq = nn.Parameter(_get_mat(num_heads * d_k, d_model, device, dtype))
        self.wk = nn.Parameter(_get_mat(num_heads * d_k, d_model, device, dtype))
        self.wv = nn.Parameter(_get_mat(num_heads * d_k, d_model, device, dtype))
        self.wo = nn.Parameter(_get_mat(d_model, num_heads * d_v, device, dtype))

        if positional_encoding is None:
            positional_encoding = NoPE()

        self.pe = positional_encoding
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Qh = einsum(x, self.wq, "... d_model, nh_dk d_model -> ... nh_dk")
        Kh = einsum(x, self.wk, "... d_model, nh_dk d_model -> ... nh_dk")
        Vh = einsum(x, self.wv, "... d_model, nh_dv d_model -> ... nh_dv")

        # Apply positional_encoding
        seq_q = Qh.shape[-2]
        seq_k = Qh.shape[-2]
        pos_q = torch.arange(seq_q)
        pos_k = torch.arange(seq_k)

        Qs = self.pe(rearrange(Qh, "... seq (nh d_k) -> ... nh seq d_k", nh=self.h), pos_q)
        Ks = self.pe(rearrange(Kh, "... seq (nh d_k) -> ... nh seq d_k", nh=self.h), pos_k)

        Vs = rearrange(Vh, "... seq (nh d_v) -> ... nh seq d_v", nh=self.h)

        causal_mask = torch.tensor([[j <= i for j in range(seq_k)] for i in range(seq_q)], device=x.device)
        attention_out = scaled_dot_product_attention(Qs, Ks, Vs, causal_mask)

        attend = rearrange(attention_out, "... nh seq d_v -> ... seq (nh d_v)")
        return einsum(attend, self.wo, "... seq nh_dv, ... d_model nh_dv -> ... seq d_model")
