import torch
import torch.nn as nn

from cs336_basics.layers import RoPE, Transformer

"""
t = Transformer(50_257, 1_600, 48, 25, 4_288)

print(sum(p.numel() for p in t.parameters()))
print((50257 * 1600) + 48 * (25 * (1600 * 64 * 3) + 1600 * 1600 + 2 * 1600 + 3 * 1600 * 4288) + 1600 + 1600 * 50257)
"""

FILE = "data/TinyStoriesV2-GPT4-train.txt"
