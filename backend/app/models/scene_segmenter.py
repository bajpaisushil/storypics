"""Model A — Scene Boundary Detector.

Given the event units u_1..u_n of a story, predict for every adjacent pair (u_{i-1}, u_i) the probability
that a new visual scene starts at u_i.

    inputs   emb   (B, N, 384)  frozen MiniLM sentence embeddings of each unit
             feats (B, N, F)    discourse features (temporal cues, place/character change, dialogue …)
             mask  (B, N)       True for real units
    output   logits (B, N-1)    boundary logit before unit i (i = 1..N-1)

Architecture: feature projection → learned positions → 2-layer pre-norm Transformer encoder over the
story → pairwise boundary head on [h_{i-1}, h_i, |h_i − h_{i-1}|, h_i ⊙ h_{i-1}, cos(e_{i-1}, e_i)].
Trained with (soft-label) binary cross-entropy; see training/train_scene_segmenter.py and docs/MODELS.md.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

# --------------------------------------------------------------------------------------------- features

STRONG_TIME_CUES = re.compile(
    r"^(once upon a time|one (day|morning|evening|night|afternoon)|the next (day|morning|evening|night)|"
    r"next (day|morning)|later( that| on)?|that (night|evening|morning|afternoon)|after (a while|that|lunch|"
    r"dinner|breakfast|school)|the following|many (days|years) later|some time later|in the morning|"
    r"in the evening|at night|at dawn|at sunset|when (they|he|she|it|the)\b.*\b(arrived|got|reached|woke)|"
    r"meanwhile|years later|soon after|the day after)\b",
    re.I,
)
WEAK_CUES = re.compile(r"^(then|suddenly|soon|all of a sudden|just then|at once|after|when|now)\b", re.I)
ENDING_CUES = re.compile(r"^(finally|at last|in the end|from that day|and they lived|the end|ever since)\b", re.I)
TIME_WORDS = {
    "morning",
    "evening",
    "night",
    "afternoon",
    "sunset",
    "sunrise",
    "dawn",
    "dusk",
    "midnight",
    "noon",
    "bedtime",
}
MOVEMENT = {
    "go",
    "walk",
    "run",
    "travel",
    "fly",
    "swim",
    "explore",
    "climb",
    "ride",
    "sail",
    "arrive",
    "leave",
    "return",
    "enter",
    "head",
    "visit",
    "journey",
    "wander",
    "hop",
    "march",
    "drive",
}
STATIVE = {"be", "feel", "seem", "look", "know", "think", "like", "love", "want", "have"}

FEATURE_NAMES = [
    "is_first",
    "rel_position",
    "log_len",
    "is_continuation",
    "strong_time_cue",
    "weak_cue",
    "ending_cue",
    "time_word",
    "mentions_place",
    "new_place",
    "new_character",
    "char_jaccard_prev",
    "no_characters",
    "is_dialogue",
    "prev_dialogue",
    "movement_verb",
    "stative_main_verb",
]
NUM_FEATURES = len(FEATURE_NAMES)


@dataclass
class UnitInfo:
    """Model-agnostic description of a unit (produced by the story parser or the training data builder)."""

    text: str
    is_continuation: bool
    characters: list[str]
    places: list[str]
    main_verb_lemma: str | None
    lemmas: set[str]


def unit_features(units: list[UnitInfo]) -> torch.Tensor:
    n = len(units)
    feats = torch.zeros(n, NUM_FEATURES)
    seen_chars: set[str] = set()
    current_place: str | None = None
    for i, u in enumerate(units):
        text = u.text.strip().lstrip('"').strip()
        lowered = text.lower()
        prev = units[i - 1] if i else None
        chars, prev_chars = set(u.characters), set(prev.characters) if prev else set()
        union = chars | prev_chars
        new_place = bool(u.places) and (current_place is None or u.places[-1] != current_place)
        f = [
            float(i == 0),
            i / max(n - 1, 1),
            math.log1p(len(text.split())) / 4.0,
            float(u.is_continuation),
            float(bool(STRONG_TIME_CUES.search(lowered))),
            float(bool(WEAK_CUES.search(lowered))),
            float(bool(ENDING_CUES.search(lowered))),
            float(bool(TIME_WORDS & u.lemmas)),
            float(bool(u.places)),
            float(new_place and i > 0),
            float(bool(chars - seen_chars) and i > 0),
            (len(chars & prev_chars) / len(union)) if union else 1.0,
            float(not chars),
            float('"' in u.text),
            float(prev is not None and '"' in prev.text),
            float(u.main_verb_lemma in MOVEMENT),
            float(u.main_verb_lemma in STATIVE),
        ]
        feats[i] = torch.tensor(f)
        seen_chars |= chars
        if u.places:
            current_place = u.places[-1]
    return feats


# --------------------------------------------------------------------------------------------- model


class SceneBoundaryModel(nn.Module):
    def __init__(
        self,
        emb_dim: int = 384,
        feat_dim: int = NUM_FEATURES,
        d_model: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_units: int = 256,
    ):
        super().__init__()
        self.config = dict(
            emb_dim=emb_dim,
            feat_dim=feat_dim,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            max_units=max_units,
        )
        self.in_proj = nn.Sequential(
            nn.Linear(emb_dim + feat_dim, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout)
        )
        self.pos = nn.Embedding(max_units, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=2 * d_model, dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(
            nn.Linear(4 * d_model + 1, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 1)
        )

    def forward(self, emb: torch.Tensor, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        _b, n, _ = emb.shape
        positions = torch.arange(n, device=emb.device).clamp(max=self.pos.num_embeddings - 1)
        x = self.in_proj(torch.cat([emb, feats], dim=-1)) + self.pos(positions)[None]
        h = self.encoder(x, src_key_padding_mask=~mask)
        prev, cur = h[:, :-1], h[:, 1:]
        cos = F.cosine_similarity(emb[:, :-1], emb[:, 1:], dim=-1).unsqueeze(-1)
        pair = torch.cat([prev, cur, (cur - prev).abs(), cur * prev, cos], dim=-1)
        return self.head(pair).squeeze(-1)

    def save(self, path: Path, extra: dict | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": self.config, "state_dict": self.state_dict(), "extra": extra or {}}, path)

    @classmethod
    def load(cls, path: Path, map_location: str = "cpu") -> tuple[SceneBoundaryModel, dict]:
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        return model.eval(), ckpt.get("extra", {})


# --------------------------------------------------------------------------------------------- inference


class BoundaryScorer:
    """Common interface: probabilities that a scene starts before each unit (len n-1)."""

    name = "base"

    def boundary_probs(self, units: list[UnitInfo], embeddings: torch.Tensor) -> list[float]:
        raise NotImplementedError


class LearnedBoundaryScorer(BoundaryScorer):
    def __init__(self, checkpoint: Path):
        self.model, extra = SceneBoundaryModel.load(checkpoint)
        self.threshold = float(extra.get("threshold", 0.5))
        self.name = f"sbd-transformer ({checkpoint.name})"

    @torch.inference_mode()
    def boundary_probs(self, units: list[UnitInfo], embeddings: torch.Tensor) -> list[float]:
        if len(units) < 2:
            return []
        feats = unit_features(units)
        mask = torch.ones(1, len(units), dtype=torch.bool)
        logits = self.model(embeddings[None].float(), feats[None], mask)[0]
        return torch.sigmoid(logits).tolist()


class HeuristicBoundaryScorer(BoundaryScorer):
    """Labelling-function baseline: cue phrases + place/character change + embedding shift.

    Used as the fallback when no trained checkpoint exists, and reported as a baseline in docs/MODELS.md.
    """

    name = "heuristic-baseline"
    threshold = 0.5

    def boundary_probs(self, units: list[UnitInfo], embeddings: torch.Tensor) -> list[float]:
        if len(units) < 2:
            return []
        feats = unit_features(units)
        idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
        sims = F.cosine_similarity(embeddings[:-1], embeddings[1:], dim=-1)
        probs = []
        for i in range(1, len(units)):
            f = feats[i]
            score = (
                -1.2
                + 2.2 * f[idx["strong_time_cue"]]
                + 0.8 * f[idx["weak_cue"]]
                + 1.0 * f[idx["new_place"]]
                + 0.9 * f[idx["new_character"]]
                + 1.2 * f[idx["ending_cue"]]
                + 0.6 * f[idx["time_word"]]
                + 0.5 * f[idx["movement_verb"]]
                - 0.9 * f[idx["stative_main_verb"]]
                - 0.9 * (f[idx["is_dialogue"]] * f[idx["prev_dialogue"]])
                - 0.4 * f[idx["is_continuation"]]
                + 2.0 * (0.55 - float(sims[i - 1]))
            )
            probs.append(float(torch.sigmoid(torch.as_tensor(score, dtype=torch.float32))))
        return probs


def decode_boundaries(
    probs: list[float], requested: int | str, word_count: int, threshold: float = 0.5, max_scenes: int = 12
) -> list[int]:
    """Return indices of units that start a new scene (excluding unit 0).

    * fixed N: the N-1 most probable boundaries (exactly optimal for a factorised boundary model);
    * auto: boundaries above the calibrated threshold, clamped to a sane range for the story length.
    """
    n_units = len(probs) + 1
    order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    if requested != "auto":
        k = max(0, min(int(requested), n_units, max_scenes) - 1)
        return sorted(i + 1 for i in order[:k])
    chosen = [i for i in order if probs[i] >= threshold]
    lo = min(n_units, max(2, math.ceil(word_count / 140)))
    hi = min(n_units, max_scenes, 10, max(3, math.ceil(word_count / 22)))
    if len(chosen) + 1 < lo:
        chosen = order[: lo - 1]
    if len(chosen) + 1 > hi:
        chosen = order[: hi - 1]
    return sorted(i + 1 for i in chosen)
