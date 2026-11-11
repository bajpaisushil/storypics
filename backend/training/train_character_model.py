"""Train Model B — the BiLSTM-CRF character/place mention tagger.

    python -m training.train_character_model --epochs 12

Each epoch mixes all LitBank gold training sentences with a fresh random sample of TinyStories silver
sentences (so the much larger silver set cannot drown out the gold labels). The encoder is frozen and its
word states are pre-computed, so only the casing/POS embeddings, BiLSTM, emission layer and CRF are trained.
Model selection: mean span-F1 over LitBank dev (gold) and TinyStories silver dev.
Outputs: checkpoints/character_tagger.pt and checkpoints/character_tagger.metrics.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from app.models.character_extractor import TAGS, MentionTagger, bio_to_spans

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "training" / "data" / "processed" / "tagger_data.pt"
CKPT = ROOT / "checkpoints"


def collate(batch: list[dict]):
    n = max(len(r["words"]) for r in batch)
    b = len(batch)
    words = torch.zeros(b, n, batch[0]["states"].shape[1])
    case = torch.zeros(b, n, dtype=torch.long)
    pos = torch.zeros(b, n, dtype=torch.long)
    tags = torch.zeros(b, n, dtype=torch.long)
    mask = torch.zeros(b, n, dtype=torch.bool)
    for i, r in enumerate(batch):
        k = len(r["words"])
        words[i, :k] = r["states"].float()
        case[i, :k] = r["case"].long()
        pos[i, :k] = r["pos"].long()
        tags[i, :k] = r["tags"].long()
        mask[i, :k] = True
    return words, case, pos, tags, mask


def span_f1(model: MentionTagger, records: list[dict], bs: int = 128) -> dict:
    model.eval()
    tp = n_pred = n_gold = 0
    per = {lab: [0, 0, 0] for lab in ("CHAR", "PLACE")}
    for i in range(0, len(records), bs):
        batch = records[i : i + bs]
        words, case, pos, tags, mask = collate(batch)
        paths = model.decode(words, case, pos, mask)
        for j, path in enumerate(paths):
            gold = {(s.start, s.end, s.label) for s in bio_to_spans(tags[j, : len(path)].tolist())}
            pred = {(s.start, s.end, s.label) for s in bio_to_spans(path)}
            tp += len(gold & pred)
            n_pred += len(pred)
            n_gold += len(gold)
            for lab in per:
                g = {x for x in gold if x[2] == lab}
                p = {x for x in pred if x[2] == lab}
                per[lab][0] += len(g & p)
                per[lab][1] += len(p)
                per[lab][2] += len(g)

    def f(t, p, g):
        prec, rec = (t / p if p else 0.0), (t / g if g else 0.0)
        return round(2 * prec * rec / (prec + rec) if prec + rec else 0.0, 4)

    return {"f1": f(tp, n_pred, n_gold), **{f"f1_{lab}": f(*v) for lab, v in per.items()}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--silver-per-epoch", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", type=Path, default=CKPT / "character_tagger.pt")
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(6)

    data = torch.load(DATA, weights_only=False)
    lit_train, silver_train = data["litbank_train"], data["silver_train"]
    print(f"litbank train {len(lit_train)} · silver pool {len(silver_train)} · tags {TAGS}")
    model = MentionTagger()
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best, history = {"score": -1.0}, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_data = lit_train + random.sample(silver_train, min(args.silver_per_epoch, len(silver_train)))
        random.shuffle(epoch_data)
        # bucket by length for speed
        epoch_data.sort(key=lambda r: len(r["words"]) + random.random() * 6)
        batches = [epoch_data[i : i + args.batch_size] for i in range(0, len(epoch_data), args.batch_size)]
        random.shuffle(batches)
        t0, total = time.time(), 0.0
        for batch in batches:
            words, case, pos, tags, mask = collate(batch)
            loss = model.loss(words, case, pos, mask, tags)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item()
        sched.step()
        lit_dev = span_f1(model, data["litbank_dev"])
        sil_dev = span_f1(model, data["silver_dev"])
        score = (lit_dev["f1"] + sil_dev["f1"]) / 2
        row = {
            "epoch": epoch,
            "loss": round(total / len(batches), 4),
            "litbank_dev": lit_dev,
            "silver_dev": sil_dev,
            "seconds": round(time.time() - t0, 1),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if score > best["score"]:
            best = {"score": round(score, 4), "epoch": epoch, "litbank_dev": lit_dev, "silver_dev": sil_dev}
            model.save(args.out, extra={"encoder": "sentence-transformers/all-MiniLM-L6-v2", "tags": TAGS, "dev": best})
    model, _ = MentionTagger.load(args.out)
    test = span_f1(model, data["litbank_test"])
    metrics = {
        "model": "MentionTagger (BiLSTM-CRF)",
        "parameters": n_params,
        "best": best,
        "litbank_test": test,
        "history": history,
        "args": {k: str(v) for k, v in vars(args).items()},
    }
    args.out.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"best epoch {best['epoch']} · LitBank test {test} → {args.out}")


if __name__ == "__main__":
    main()
