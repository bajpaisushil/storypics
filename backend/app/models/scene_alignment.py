"""Model C — Scene–Image Alignment head (does this image depict this scene description?).

    inputs   img (B, 768)  L2-normalised SigLIP image embedding (frozen encoder)
             txt (B, 768)  L2-normalised SigLIP text embedding of the scene description (frozen encoder)
    output   logit (B,)    → sigmoid = predicted human image–text alignment rating, normalised to [0, 1]

Architecture: separate linear projections → interaction features [p_i, p_t, p_i ⊙ p_t, |p_i − p_t|, cos] →
2-layer MLP. Trained on ImageRewardDB alignment ratings with MSE + within-prompt pairwise ranking loss
(training/train_alignment_model.py). Zero-shot SigLIP cosine is the baseline it must beat.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class AlignmentHead(nn.Module):
    def __init__(self, dim: int = 768, hidden: int = 384, dropout: float = 0.2):
        super().__init__()
        self.config = dict(dim=dim, hidden=hidden, dropout=dropout)
        self.img_proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, hidden))
        self.txt_proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, hidden))
        self.mlp = nn.Sequential(
            nn.Linear(4 * hidden + 1, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    def forward(self, img: torch.Tensor, txt: torch.Tensor) -> torch.Tensor:
        cos = (img * txt).sum(-1, keepdim=True)
        pi, pt = self.img_proj(img), self.txt_proj(txt)
        feats = torch.cat([pi, pt, pi * pt, (pi - pt).abs(), cos * 10], dim=-1)
        return self.mlp(feats).squeeze(-1)

    @torch.no_grad()
    def score(self, img: torch.Tensor, txt: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(F.normalize(img.float(), dim=-1), F.normalize(txt.float(), dim=-1)))

    def save(self, path: Path, extra: dict | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": self.config, "state_dict": self.state_dict(), "extra": extra or {}}, path)

    @classmethod
    def load(cls, path: Path, map_location: str = "cpu") -> tuple[AlignmentHead, dict]:
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        return model.eval(), ckpt.get("extra", {})
