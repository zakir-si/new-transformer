"""Reference MHA/GQA modules with native SDPA GQA support when available."""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    """Causal attention with MHA/GQA and selectable GQA implementation.

    gqa_mode="native" asks CUDA SDPA to handle grouped query attention without
    explicitly expanding K/V in Python. On non-CUDA devices it falls back to
    explicit expansion, because native GQA kernel support is backend-dependent.
    gqa_mode="expand" always expands K/V heads before SDPA/manual attention.
    """
    def __init__(
        self,
        dim: int,
        num_q_heads: int,
        num_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
        use_sdpa: bool = True,
        gqa_mode: str = "native",
    ) -> None:
        super().__init__()
        num_kv_heads = num_q_heads if num_kv_heads is None else num_kv_heads
        if min(dim, num_q_heads, num_kv_heads) <= 0:
            raise ValueError("dim and head counts must be positive")
        if dim % num_q_heads or num_q_heads % num_kv_heads:
            raise ValueError("dim must divide by query heads and query heads by KV heads")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if gqa_mode not in {"native", "expand"}:
            raise ValueError("gqa_mode must be 'native' or 'expand'")

        self.dim = dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_q_heads
        self.dropout = float(dropout)
        self.gqa_mode = gqa_mode
        self.use_sdpa = use_sdpa and hasattr(F, "scaled_dot_product_attention")

        self.q_proj = nn.Linear(dim, num_q_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.dim:
            raise ValueError(f"x must have shape [batch, sequence, {self.dim}]")
        b, n, _ = x.shape
        q = self.q_proj(x).view(b, n, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, n, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, n, self.num_kv_heads, self.head_dim).transpose(1, 2)
        p = self.dropout if self.training else 0.0

        native_gqa = (
            self.use_sdpa
            and self.gqa_mode == "native"
            and self.num_kv_heads != self.num_q_heads
            and q.is_cuda
        )
        if native_gqa:
            # PyTorch SDPA can dispatch to a backend optimized for GQA.
            y = F.scaled_dot_product_attention(
                q, k, v, dropout_p=p, is_causal=causal, enable_gqa=True
            )
        else:
            # Reference path: expand K/V to query-head count. This is also the
            # compatibility path for non-CUDA devices and manual attention.
            if self.num_kv_heads != self.num_q_heads:
                group = self.num_q_heads // self.num_kv_heads
                k = k.repeat_interleave(group, dim=1)
                v = v.repeat_interleave(group, dim=1)

            if self.use_sdpa:
                y = F.scaled_dot_product_attention(q, k, v, dropout_p=p, is_causal=causal)
            else:
                scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
                if causal:
                    mask = torch.ones(n, n, device=x.device, dtype=torch.bool).triu(1)
                    scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
                y = self.attn_dropout(scores.softmax(dim=-1)) @ v

        y = y.transpose(1, 2).contiguous().view(b, n, self.dim)
        return self.resid_dropout(self.out_proj(y))


class FeedForward(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0, expansion: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, expansion * dim),
            nn.GELU(),
            nn.Linear(expansion * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, dim, q_heads, kv_heads, dropout=0.0, use_sdpa=True):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, q_heads, kv_heads, dropout, use_sdpa)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x), causal=True)
        return x + self.ffn(self.ln2(x))


class TinyDecoderLM(nn.Module):
    """Small GPT-like model for controlled experiments; not a production LLM."""
    def __init__(
        self,
        vocab_size,
        max_seq_len=512,
        dim=128,
        num_layers=2,
        num_q_heads=4,
        num_kv_heads=None,
        dropout=0.0,
        use_sdpa=True,
    ):
        super().__init__()
        if vocab_size < 2 or max_seq_len < 1 or num_layers < 1:
            raise ValueError("invalid model dimensions")
        self.vocab_size, self.max_seq_len = vocab_size, max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_seq_len, dim)
        kv = num_q_heads if num_kv_heads is None else num_kv_heads
        self.blocks = nn.ModuleList([
            DecoderBlock(dim, num_q_heads, kv, dropout, use_sdpa)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        _, n = input_ids.shape
        if n > self.max_seq_len:
            raise ValueError(f"sequence length {n} exceeds max_seq_len {self.max_seq_len}")
        pos = torch.arange(n, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(pos)[None, :, :]
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x))

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=20):
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        self.eval()
        out = input_ids
        for _ in range(max_new_tokens):
            logits = self(out[:, -self.max_seq_len:])[:, -1, :]
            out = torch.cat((out, logits.argmax(dim=-1, keepdim=True)), dim=1)
        return out
