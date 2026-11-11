"""Frozen pretrained text encoder shared by Models A and B.

all-MiniLM-L6-v2 (Apache-2.0, 22 M parameters, 384-d) is used as-is: mean-pooled sentence embeddings for
the scene-boundary model and first-sub-token word states for the mention tagger. Only the task-specific
layers on top are trained by this project.
"""

from __future__ import annotations

import threading

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


class TextEncoder:
    def __init__(self, name: str = DEFAULT_ENCODER, device: str = "cpu", local_files_only: bool = True):
        self.name = name
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=local_files_only)
        self.model = AutoModel.from_pretrained(name, local_files_only=local_files_only).to(self.device).eval()
        self.dim = int(self.model.config.hidden_size)
        self._lock = threading.Lock()

    @torch.inference_mode()
    def encode(self, texts: list[str], batch_size: int = 64) -> torch.Tensor:
        """(N, 384) L2-normalised mean-pooled sentence embeddings."""
        if not texts:
            return torch.zeros(0, self.dim)
        out = []
        with self._lock:
            for i in range(0, len(texts), batch_size):
                batch = self.tokenizer(
                    texts[i : i + batch_size], padding=True, truncation=True, max_length=128, return_tensors="pt"
                ).to(self.device)
                hidden = self.model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
                out.append(F.normalize(pooled, dim=-1).cpu())
        return torch.cat(out)

    @torch.inference_mode()
    def word_states(self, sentences: list[list[str]], max_length: int = 256) -> list[torch.Tensor]:
        """Per-word vectors (first sub-token of each word) for pre-tokenised sentences → list of (n_words, 384)."""
        results: list[torch.Tensor] = []
        with self._lock:
            for words in sentences:
                if not words:
                    results.append(torch.zeros(0, self.dim))
                    continue
                enc = self.tokenizer(
                    words, is_split_into_words=True, truncation=True, max_length=max_length, return_tensors="pt"
                )
                hidden = self.model(**{k: v.to(self.device) for k, v in enc.items()}).last_hidden_state[0].cpu()
                word_ids = enc.word_ids(0)
                first: dict[int, int] = {}
                for pos, wid in enumerate(word_ids):
                    if wid is not None and wid not in first:
                        first[wid] = pos
                vecs = torch.zeros(len(words), self.dim)
                for wid, pos in first.items():
                    vecs[wid] = hidden[pos]
                results.append(vecs)
        return results


_ENCODERS: dict[str, TextEncoder] = {}
_LOCK = threading.Lock()


def get_text_encoder(name: str = DEFAULT_ENCODER, local_files_only: bool = True) -> TextEncoder:
    with _LOCK:
        if name not in _ENCODERS:
            _ENCODERS[name] = TextEncoder(name, local_files_only=local_files_only)
        return _ENCODERS[name]
