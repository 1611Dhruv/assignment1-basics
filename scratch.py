import torch
import torch.nn as nn

from cs336_basics.layers import RMSNorm, SwiGLU

batch = 5
seq = 4
d_model = 2

inp = torch.tensor(
    [[[i + i * j + i * j * k for k in range(d_model)] for j in range(seq)] for i in range(batch)], dtype=torch.float32
)
print(inp)
rms_norm = RMSNorm(d_model)
print(rms_norm.forward(inp))

swiglu = SwiGLU(d_model, 10)
print(swiglu.forward(inp))
