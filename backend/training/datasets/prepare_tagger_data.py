"""Build training data for Model B (character / place mention tagger).

Sources
  * LitBank (CC BY 4.0) — gold entity annotations over 100 public-domain novels. PER → CHAR, LOC/FAC/GPE → PLACE.
    Nested layers are reduced to the outermost mention, then trimmed to its base noun phrase (determiner +
    pre-modifiers + head) with spaCy, matching how mentions are used downstream. Pronoun-only spans are dropped
    (pronouns are resolved by the coreference step, not tagged).
  * TinyStories (CDLA-Sharing-1.0) — silver labels from the rule-based detector, for children's-story style.

Splits: LitBank by book (80/10/10, deterministic), TinyStories by story. Each sentence stores words, casing ids,
UPOS ids, BIO tag ids and frozen MiniLM word states (float16).

    python -m training.datasets.prepare_tagger_data --silver 3000
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
from spacy.tokens import Doc
from tqdm import tqdm

from app.models.character_extractor import POS2ID, TAG2ID, TaggedSpan, case_id, spans_to_bio
from app.models.encoders import get_text_encoder
from app.services.character_manager import RuleMentionDetector
from app.services.text_units import get_nlp, normalize_text, split_units

ROOT = Path(__file__).resolve().parents[1] / "data"
LITBANK = ROOT / "raw" / "litbank"
TINY = ROOT / "raw" / "tinystories" / "TinyStoriesV2-GPT4-train.head60mb.txt"
OUT = ROOT / "processed"
LABEL_MAP = {"PER": "CHAR", "LOC": "PLACE", "FAC": "PLACE", "GPE": "PLACE"}
MAX_WORDS = 120


def litbank_split(name: str) -> str:
    h = int(hashlib.md5(name.encode()).hexdigest(), 16) % 10
    return "test" if h == 0 else "dev" if h == 1 else "train"


def read_litbank_file(path: Path) -> list[tuple[list[str], list[tuple[int, int, str]]]]:
    sentences, words, layers = [], [], []
    for line in [*path.read_text(encoding="utf-8").splitlines(), ""]:
        if not line.strip():
            if words:
                sentences.append((words, layers))
            words, layers = [], []
            continue
        cols = line.split("\t")
        words.append(cols[0])
        layers.append([c for c in cols[1:] if c])
    out = []
    for words, tags in sentences:
        spans: list[tuple[int, int, str]] = []
        n_layers = max(len(t) for t in tags)
        for layer in range(n_layers):
            start, label = None, None
            for i in range(len(words) + 1):
                tag = tags[i][layer] if i < len(words) and layer < len(tags[i]) else "O"
                if tag.startswith("B-") or tag == "O" or (tag.startswith("I-") and tag[2:] != label):
                    if start is not None and label in LABEL_MAP:
                        spans.append((start, i, LABEL_MAP[label]))
                    start, label = (i, tag[2:]) if tag.startswith(("B-", "I-")) else (None, None)
        # keep outermost spans only
        spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
        outer: list[tuple[int, int, str]] = []
        for s in spans:
            if not any(o[0] <= s[0] and s[1] <= o[1] for o in outer):
                outer.append(s)
        out.append((words, outer))
    return out


def base_np(doc: Doc, start: int, end: int) -> tuple[int, int] | None:
    span = doc[start:end]
    root = span.root
    if root.pos_ == "PRON" and end - start == 1:
        return None
    if all(t.pos_ in ("PRON", "DET") for t in span):
        return None
    return start, root.i + 1 if start <= root.i < end else end


def encode_sentences(records: list[dict], desc: str) -> None:
    encoder = get_text_encoder()
    for i in tqdm(range(0, len(records), 64), desc=f"encode {desc}"):
        batch = records[i : i + 64]
        states = encoder.word_states([r["words"] for r in batch])
        for r, s in zip(batch, states, strict=False):
            r["states"] = s.half()


def make_record(doc: Doc, spans: list[TaggedSpan], source: str) -> dict:
    words = [t.text for t in doc]
    return {
        "words": words,
        "case": torch.tensor([case_id(w) for w in words], dtype=torch.int8),
        "pos": torch.tensor([POS2ID.get(t.pos_, 0) for t in doc], dtype=torch.int8),
        "tags": torch.tensor(spans_to_bio(len(words), spans), dtype=torch.int8),
        "source": source,
    }


def prepare_litbank() -> dict[str, list[dict]]:
    nlp = get_nlp()
    splits: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    for path in tqdm(sorted(LITBANK.glob("*.tsv")), desc="litbank"):
        split = litbank_split(path.name)
        for words, spans in read_litbank_file(path):
            for w0 in range(0, len(words), MAX_WORDS):
                chunk = words[w0 : w0 + MAX_WORDS]
                doc = Doc(nlp.vocab, words=chunk)
                for _, proc in nlp.pipeline:
                    doc = proc(doc)
                tagged = []
                for s, e, label in spans:
                    if s >= w0 and e <= w0 + MAX_WORDS:
                        trimmed = base_np(doc, s - w0, e - w0)
                        if trimmed:
                            tagged.append(TaggedSpan(trimmed[0], trimmed[1], label))
                splits[split].append(make_record(doc, tagged, f"litbank:{path.stem}"))
    return splits


def prepare_silver(n: int) -> dict[str, list[dict]]:
    nlp = get_nlp()
    detector = RuleMentionDetector()
    raw = TINY.read_text(encoding="utf-8", errors="ignore").split("<|endoftext|>")[50_000 : 50_000 + n]
    texts = [normalize_text(" ".join(p for p in s.strip().split("\n") if p.strip())) for s in raw]
    splits: dict[str, list[dict]] = {"train": [], "dev": []}
    for k, doc in enumerate(tqdm(nlp.pipe(texts, batch_size=64, n_process=4), total=len(texts), desc="silver")):
        story = split_units("", doc=doc)
        dets = detector.detect(story)
        split = "dev" if k % 10 == 0 else "train"
        for sent in doc.sents:
            tagged = []
            for d in dets:
                if sent.start_char <= d.start and d.end <= sent.end_char:
                    span = doc.char_span(d.start, d.end, alignment_mode="expand")
                    if span is not None:
                        tagged.append(TaggedSpan(span.start - sent.start, span.end - sent.start, d.label))
            sdoc = sent.as_doc()
            splits[split].append(make_record(sdoc, tagged, f"tinystories-silver:{k}"))
    return splits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--silver", type=int, default=3000, help="number of TinyStories stories for silver labels")
    args = ap.parse_args()
    torch.set_num_threads(4)
    data = {f"litbank_{k}": v for k, v in prepare_litbank().items()}
    data.update({f"silver_{k}": v for k, v in prepare_silver(args.silver).items()})
    for name, records in data.items():
        encode_sentences(records, name)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(data, OUT / "tagger_data.pt")
    for name, records in data.items():
        n_tok = sum(len(r["words"]) for r in records)
        n_char = sum(int((r["tags"] == TAG2ID["B-CHAR"]).sum()) for r in records)
        n_place = sum(int((r["tags"] == TAG2ID["B-PLACE"]).sum()) for r in records)
        print(f"{name:14s} sentences={len(records):6d} tokens={n_tok:7d} CHAR={n_char:6d} PLACE={n_place:5d}")


if __name__ == "__main__":
    main()
