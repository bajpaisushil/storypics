"""Model B — Character & Place mention tagger (BiLSTM-CRF over frozen MiniLM word states).

    inputs   words (B, T, 384)  first-sub-token MiniLM states for each word
             case  (B, T)       casing class id   (lower / Title / UPPER / digit / punct / other)
             pos   (B, T)       spaCy universal POS id
             mask  (B, T)       True for real words
    output   emissions (B, T, 5) over tags O, B-CHAR, I-CHAR, B-PLACE, I-PLACE; Viterbi-decoded by the CRF

The encoder is uncased, so an explicit casing embedding restores the capitalisation signal that matters
for proper names. The CRF (implemented here) has hard BIO constraints (no O→I-X, no B-X→I-Y).
Loss: CRF negative log-likelihood. See training/train_character_model.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

TAGS = ["O", "B-CHAR", "I-CHAR", "B-PLACE", "I-PLACE"]
TAG2ID = {t: i for i, t in enumerate(TAGS)}
UPOS = [
    "ADJ",
    "ADP",
    "ADV",
    "AUX",
    "CCONJ",
    "DET",
    "INTJ",
    "NOUN",
    "NUM",
    "PART",
    "PRON",
    "PROPN",
    "PUNCT",
    "SCONJ",
    "SYM",
    "VERB",
    "X",
    "SPACE",
]
POS2ID = {p: i + 1 for i, p in enumerate(UPOS)}  # 0 = padding / unknown
NUM_CASE = 7


def case_id(word: str) -> int:
    if not word:
        return 0
    if word.isdigit():
        return 4
    if all(not c.isalnum() for c in word):
        return 5
    if word.isupper() and len(word) > 1:
        return 3
    if word[0].isupper():
        return 2
    if word.islower():
        return 1
    return 6


class CRF(nn.Module):
    def __init__(self, num_tags: int, tags: list[str]):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.zeros(num_tags, num_tags))  # [from, to]
        self.start = nn.Parameter(torch.zeros(num_tags))
        self.end = nn.Parameter(torch.zeros(num_tags))
        forbid = torch.zeros(num_tags, num_tags)
        start_forbid = torch.zeros(num_tags)
        for j, to in enumerate(tags):
            if to.startswith("I-"):
                start_forbid[j] = -1e4
                for i, frm in enumerate(tags):
                    if frm == "O" or (frm[2:] != to[2:]):
                        forbid[i, j] = -1e4
        self.register_buffer("forbid", forbid)
        self.register_buffer("start_forbid", start_forbid)

    def _trans(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.transitions + self.forbid, self.start + self.start_forbid

    def nll(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean negative log-likelihood. emissions (B,T,K), tags (B,T), mask (B,T) with mask[:,0] all True."""
        trans, start = self._trans()
        _b, t, _ = emissions.shape
        maskf = mask.float()
        # score of the gold path
        score = start[tags[:, 0]] + emissions[:, 0].gather(1, tags[:, :1]).squeeze(1)
        for i in range(1, t):
            step = trans[tags[:, i - 1], tags[:, i]] + emissions[:, i].gather(1, tags[:, i : i + 1]).squeeze(1)
            score = score + step * maskf[:, i]
        lengths = mask.long().sum(1) - 1
        last_tags = tags.gather(1, lengths.unsqueeze(1)).squeeze(1)
        score = score + self.end[last_tags]
        # partition function (forward algorithm)
        alpha = start[None] + emissions[:, 0]
        for i in range(1, t):
            nxt = torch.logsumexp(alpha.unsqueeze(2) + trans[None] + emissions[:, i].unsqueeze(1), dim=1)
            alpha = torch.where(mask[:, i : i + 1], nxt, alpha)
        log_z = torch.logsumexp(alpha + self.end[None], dim=1)
        return (log_z - score).mean()

    @torch.no_grad()
    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        trans, start = self._trans()
        b, t, k = emissions.shape
        score = start[None] + emissions[:, 0]
        history = []
        for i in range(1, t):
            total = score.unsqueeze(2) + trans[None]
            best, idx = total.max(dim=1)
            nxt = best + emissions[:, i]
            score = torch.where(mask[:, i : i + 1], nxt, score)
            history.append(torch.where(mask[:, i : i + 1], idx, torch.arange(k, device=idx.device)[None].expand(b, k)))
        score = score + self.end[None]
        paths = []
        lengths = mask.long().sum(1)
        for n in range(b):
            best = int(score[n].argmax())
            path = [best]
            for i in range(int(lengths[n]) - 2, -1, -1):
                best = int(history[i][n, best])
                path.append(best)
            paths.append(path[::-1])
        return paths


class MentionTagger(nn.Module):
    def __init__(
        self,
        emb_dim: int = 384,
        case_dim: int = 16,
        pos_dim: int = 24,
        hidden: int = 192,
        dropout: float = 0.3,
        num_tags: int = len(TAGS),
    ):
        super().__init__()
        self.config = dict(
            emb_dim=emb_dim, case_dim=case_dim, pos_dim=pos_dim, hidden=hidden, dropout=dropout, num_tags=num_tags
        )
        self.case_emb = nn.Embedding(NUM_CASE, case_dim)
        self.pos_emb = nn.Embedding(len(UPOS) + 1, pos_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb_dim + case_dim + pos_dim, hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(2 * hidden, num_tags)
        self.crf = CRF(num_tags, TAGS)

    def emissions(self, words: torch.Tensor, case: torch.Tensor, pos: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = torch.cat([words, self.case_emb(case), self.pos_emb(pos)], dim=-1)
        x = self.dropout(x)
        lengths = mask.long().sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
        return self.proj(self.dropout(out))

    def loss(self, words, case, pos, mask, tags) -> torch.Tensor:
        return self.crf.nll(self.emissions(words, case, pos, mask), tags, mask)

    @torch.no_grad()
    def decode(self, words, case, pos, mask) -> list[list[int]]:
        return self.crf.decode(self.emissions(words, case, pos, mask), mask)

    def save(self, path: Path, extra: dict | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": self.config, "state_dict": self.state_dict(), "extra": extra or {}}, path)

    @classmethod
    def load(cls, path: Path, map_location: str = "cpu") -> tuple[MentionTagger, dict]:
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        return model.eval(), ckpt.get("extra", {})


@dataclass(frozen=True)
class TaggedSpan:
    start: int  # word index (inclusive)
    end: int  # word index (exclusive)
    label: str  # CHAR / PLACE


def bio_to_spans(tag_ids: list[int]) -> list[TaggedSpan]:
    spans: list[TaggedSpan] = []
    start, label = None, None
    for i, tid in enumerate([*tag_ids, 0]):
        tag = TAGS[tid]
        if tag.startswith("B-") or tag == "O" or (tag.startswith("I-") and tag[2:] != label):
            if start is not None:
                spans.append(TaggedSpan(start, i, label))  # type: ignore[arg-type]
                start, label = None, None
            if tag.startswith("B-") or tag.startswith("I-"):
                start, label = i, tag[2:]
    return spans


def spans_to_bio(n_words: int, spans: list[TaggedSpan]) -> list[int]:
    tags = [TAG2ID["O"]] * n_words
    for s in spans:
        tags[s.start] = TAG2ID[f"B-{s.label}"]
        for i in range(s.start + 1, s.end):
            tags[i] = TAG2ID[f"I-{s.label}"]
    return tags
