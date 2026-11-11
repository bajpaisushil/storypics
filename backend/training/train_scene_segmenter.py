"""Train Model A — the scene boundary detector.

    python -m training.train_scene_segmenter --epochs 8

Data: training/data/processed/scene_{train,dev}.pt (built by training/datasets/prepare_scene_data.py).
Loss: binary cross-entropy against soft weak labels. Discourse features are randomly dropped during
training (feature dropout) so the model cannot simply copy the labelling functions and has to use the
sentence embeddings. The decision threshold for "auto" mode is calibrated on the dev split.
Outputs: checkpoints/scene_segmenter.pt and checkpoints/scene_segmenter.metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from app.models.scene_segmenter import FEATURE_NAMES, SceneBoundaryModel

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "training" / "data" / "processed"
CKPT = ROOT / "checkpoints"


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


def refine_labels(records: list[dict]) -> int:
    """Labelling-function refinement: TinyStories starts a new paragraph for most dialogue lines, which is a
    formatting convention rather than a scene change. Down-weight paragraph boundaries that open a quoted
    line without a temporal cue. Returns the number of adjusted transitions."""
    from app.models.scene_segmenter import STRONG_TIME_CUES

    changed = 0
    for r in records:
        for i, text in enumerate(r["texts"][1:]):
            t = text.lstrip()
            if r["hard"][i] and t.startswith('"') and not STRONG_TIME_CUES.search(t.lstrip('"').lower()):
                r["soft"][i] = 0.4
                changed += 1
    return changed


def collate(batch: list[dict], device: torch.device, feature_dropout: float = 0.0):
    n = max(len(r["texts"]) for r in batch)
    b = len(batch)
    emb = torch.zeros(b, n, batch[0]["embeddings"].shape[1])
    feats = torch.zeros(b, n, len(FEATURE_NAMES))
    mask = torch.zeros(b, n, dtype=torch.bool)
    soft = torch.zeros(b, n - 1)
    hard = torch.zeros(b, n - 1)
    pair_mask = torch.zeros(b, n - 1, dtype=torch.bool)
    for i, r in enumerate(batch):
        k = len(r["texts"])
        emb[i, :k] = r["embeddings"].float()
        f = r["features"].float()
        if feature_dropout:
            f = f * (torch.rand(f.shape[1]) > feature_dropout).float()  # drop whole feature columns per story
        feats[i, :k] = f
        mask[i, :k] = True
        soft[i, : k - 1] = r["soft"].float()
        hard[i, : k - 1] = r["hard"].float()
        pair_mask[i, : k - 1] = True
    return [t.to(device) for t in (emb, feats, mask, soft, hard, pair_mask)]


@torch.no_grad()
def predict(model: SceneBoundaryModel, records: list[dict], device: torch.device, bs: int = 64):
    model.eval()
    probs, golds = [], []
    for i in range(0, len(records), bs):
        emb, feats, mask, _, hard, pm = collate(records[i : i + bs], device)
        p = torch.sigmoid(model(emb, feats, mask))
        probs.append(p[pm].cpu())
        golds.append(hard[pm].cpu())
    return torch.cat(probs), torch.cat(golds)


def f1_at(probs: torch.Tensor, gold: torch.Tensor, t: float) -> tuple[float, float, float]:
    pred = probs >= t
    tp = (pred & (gold > 0.5)).sum().item()
    p = tp / max(pred.sum().item(), 1)
    r = tp / max((gold > 0.5).sum().item(), 1)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--feature-dropout", type=float, default=0.15)
    ap.add_argument("--max-train", type=int, default=0, help="limit training stories (0 = all)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", type=Path, default=CKPT / "scene_segmenter.pt")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    train = torch.load(DATA / "scene_train.pt", weights_only=False)
    dev = torch.load(DATA / "scene_dev.pt", weights_only=False)
    if args.max_train:
        train = train[: args.max_train]
    print(f"refined {refine_labels(train)} dialogue-paragraph labels in train")
    print(f"train {len(train)} stories · dev {len(dev)} stories · device {device}")

    model = SceneBoundaryModel().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * math.ceil(len(train) / args.batch_size)
    warmup = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))
    )

    best = {"f1": -1.0}
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train)
        t0, total = time.time(), 0.0
        for i in range(0, len(train), args.batch_size):
            emb, feats, mask, soft, _, pm = collate(train[i : i + args.batch_size], device, args.feature_dropout)
            logits = model(emb, feats, mask)
            loss = F.binary_cross_entropy_with_logits(logits[pm], soft[pm])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item() * len(train[i : i + args.batch_size])
        probs, gold = predict(model, dev, device)
        dev_loss = F.binary_cross_entropy(probs.clamp(1e-6, 1 - 1e-6), gold).item()
        thresholds = [t / 100 for t in range(10, 91, 2)]
        t_best = max(thresholds, key=lambda t: f1_at(probs, gold, t)[2])
        p, r, f1 = f1_at(probs, gold, t_best)
        row = {
            "epoch": epoch,
            "train_loss": round(total / len(train), 4),
            "dev_bce_vs_paragraphs": round(dev_loss, 4),
            "dev_f1": round(f1, 4),
            "dev_precision": round(p, 4),
            "dev_recall": round(r, 4),
            "threshold": t_best,
            "seconds": round(time.time() - t0, 1),
        }
        history.append(row)
        print(json.dumps(row))
        if f1 > best["f1"]:
            best = {"f1": f1, "epoch": epoch, "threshold": t_best, "precision": p, "recall": r}
            model.save(
                args.out,
                extra={
                    "threshold": t_best,
                    "features": FEATURE_NAMES,
                    "encoder": "sentence-transformers/all-MiniLM-L6-v2",
                    "dev": {k: round(v, 4) if isinstance(v, float) else v for k, v in best.items()},
                },
            )
    metrics = {
        "model": "SceneBoundaryModel",
        "parameters": n_params,
        "train_stories": len(train),
        "dev_stories": len(dev),
        "best": best,
        "history": history,
        "args": {k: str(v) for k, v in vars(args).items()},
        "note": "dev metrics are against weak paragraph labels; see training/evaluate.py for the gold set",
    }
    args.out.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"best epoch {best['epoch']} dev F1 {best['f1']:.3f} @ t={best['threshold']} → {args.out}")


if __name__ == "__main__":
    main()
