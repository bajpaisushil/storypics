"""Model D — Image Quality Evaluator (predicted human fidelity rating of a generated image).

    inputs   sig   (B, 768)  L2-normalised SigLIP image embedding (semantic / aesthetic content)
             dino  (B, 384)  L2-normalised DINOv2-small CLS embedding (structure, local detail)
             stats (B, 6)    OpenCV statistics: sharpness, contrast, colourfulness, entropy, exposure, clipping
    output   logit (B,)      → sigmoid = predicted fidelity rating normalised to [0, 1]

Architecture: per-source projections (LayerNorm + Linear + GELU) → concatenation → 2-layer MLP.
Trained on ImageRewardDB fidelity ratings with MSE + within-prompt pairwise ranking loss
(training/train_quality_model.py); compared against a statistics-only baseline.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

NUM_STATS = 6


class QualityHead(nn.Module):
    def __init__(
        self,
        sig_dim: int = 768,
        dino_dim: int = 384,
        stats_dim: int = NUM_STATS,
        hidden: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.config = dict(sig_dim=sig_dim, dino_dim=dino_dim, stats_dim=stats_dim, hidden=hidden, dropout=dropout)

        def branch(d: int) -> nn.Sequential:
            return nn.Sequential(nn.LayerNorm(d), nn.Dropout(dropout), nn.Linear(d, hidden), nn.GELU())

        self.sig, self.dino, self.stats = branch(sig_dim), branch(dino_dim), branch(stats_dim)
        self.mlp = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, sig: torch.Tensor, dino: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([self.sig(sig), self.dino(dino), self.stats(stats)], dim=-1)).squeeze(-1)

    @torch.no_grad()
    def score(self, sig: torch.Tensor, dino: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(
            self.forward(F.normalize(sig.float(), dim=-1), F.normalize(dino.float(), dim=-1), stats.float())
        )

    def save(self, path: Path, extra: dict | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": self.config, "state_dict": self.state_dict(), "extra": extra or {}}, path)

    @classmethod
    def load(cls, path: Path, map_location: str = "cpu") -> tuple[QualityHead, dict]:
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        return model.eval(), ckpt.get("extra", {})
