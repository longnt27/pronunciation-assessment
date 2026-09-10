"""Compact, self-contained GOPT models for the three-system benchmark.

The implementation follows Gong et al.'s released GOPT architecture: learned
position embeddings, a 40-way canonical-phone projection, 24-dimensional
tokens, three one-head Transformer blocks, and task-specific regression heads.
Keeping the classes here makes benchmark checkpoints loadable without a
third-party checkout under ``tmp/``.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    """GOPT's single/multi-head self-attention (qkv has no bias)."""

    def __init__(self, width: int = 24, heads: int = 1) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.heads = heads
        self.scale = (width // heads) ** -0.5
        self.qkv = nn.Linear(width, width * 3, bias=False)
        self.projection = nn.Linear(width, width)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, length, width = values.shape
        qkv = self.qkv(values).reshape(
            batch, length, 3, self.heads, width // self.heads
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        weights = (query @ key.transpose(-2, -1) * self.scale).softmax(dim=-1)
        attended = (weights @ value).transpose(1, 2).reshape(batch, length, width)
        return self.projection(attended)


class SelfAttentionBlock(nn.Module):
    def __init__(self, width: int = 24, heads: int = 1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = Attention(width, heads)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * 4),
            nn.GELU(),
            nn.Linear(width * 4, width),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = values + self.attention(self.norm1(values))
        return values + self.mlp(self.norm2(values))


class GOPTEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        cls_tokens: int,
        width: int = 24,
        depth: int = 3,
        heads: int = 1,
        max_phones: int = 50,
    ) -> None:
        super().__init__()
        self.cls_tokens = cls_tokens
        self.max_phones = max_phones
        self.input_projection = nn.Linear(input_dim, width)
        self.phone_projection = nn.Linear(40, width)
        self.cls = nn.Parameter(torch.zeros(1, cls_tokens, width))
        self.positions = nn.Parameter(torch.zeros(1, max_phones + cls_tokens, width))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.positions, std=0.02)
        self.blocks = nn.ModuleList(
            [SelfAttentionBlock(width, heads) for _ in range(depth)]
        )

    def forward(
        self, features: torch.Tensor, phone_ids: torch.Tensor
    ) -> torch.Tensor:
        # SpeechOcean phone ids are 0..38; -1 is padding. Adding one reserves
        # one-hot index zero for the padding token, exactly as official GOPT.
        phone_one_hot = F.one_hot(phone_ids.long() + 1, num_classes=40).float()
        values = self.input_projection(features) + self.phone_projection(phone_one_hot)
        if self.cls_tokens:
            values = torch.cat(
                (self.cls.expand(len(values), -1, -1), values), dim=1
            )
        values = values + self.positions[:, : values.shape[1]]
        for block in self.blocks:
            values = block(values)
        return values


class RegressionHead(nn.Module):
    def __init__(self, width: int = 24) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values).squeeze(-1)


class GOPTPhone(nn.Module):
    """The phone-only GOPT configuration used by Cao et al."""

    def __init__(self, input_dim: int, max_phones: int = 50) -> None:
        super().__init__()
        self.encoder = GOPTEncoder(
            input_dim, cls_tokens=5, max_phones=max_phones
        )
        self.phone_head = RegressionHead()

    def forward(
        self, features: torch.Tensor, phone_ids: torch.Tensor
    ) -> torch.Tensor:
        return self.phone_head(self.encoder(features, phone_ids)[:, 5:])


class GOPTJoint(nn.Module):
    """Gong et al.'s complete phone/word/utterance multi-task GOPT."""

    def __init__(self, input_dim: int, max_phones: int = 50) -> None:
        super().__init__()
        self.encoder = GOPTEncoder(
            input_dim, cls_tokens=5, max_phones=max_phones
        )
        self.phone_head = RegressionHead()
        self.word_heads = nn.ModuleList([RegressionHead() for _ in range(3)])
        self.utterance_heads = nn.ModuleList(
            [RegressionHead() for _ in range(5)]
        )

    def forward(
        self, features: torch.Tensor, phone_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        values = self.encoder(features, phone_ids)
        utterance_values = values[:, :5]
        phone_values = values[:, 5:]
        phone = self.phone_head(phone_values)
        word = torch.stack(
            [head(phone_values) for head in self.word_heads], dim=-1
        )
        utterance = torch.stack(
            [
                head(utterance_values[:, index])
                for index, head in enumerate(self.utterance_heads)
            ],
            dim=-1,
        )
        return phone, word, utterance


def masked_mse(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    selected = (prediction - target)[mask]
    if not selected.numel():
        return prediction.sum() * 0.0
    return torch.mean(selected.square())


def parameter_count(model: nn.Module) -> int:
    return sum(math.prod(parameter.shape) for parameter in model.parameters())
