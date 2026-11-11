"""Train Model C (scene–image alignment) and Model D (image quality) on ImageRewardDB human ratings.

    python -m training.train_image_models extract      # frozen SigLIP / DINOv2 / OpenCV features → .pt cache
    python -m training.train_image_models train        # trains both heads, writes checkpoints + metrics
    python -m training.evaluate alignment|quality      # test-set evaluation vs baselines

Data: ImageRewardDB (Apache-2.0) metadata + downloaded image zips (read in place, never extracted). Each image has
1–7 human ratings for image–text alignment and fidelity, grouped by prompt. Splits follow the dataset's own
train / validation / test folders (prompt-disjoint). Ratings are normalised to [0, 1].

Loss = MSE(sigmoid(logit), rating) + λ · Bradley–Terry pairwise loss over images of the same prompt whose ratings
differ. Model C additionally sees mismatched (image, other prompt) pairs labelled 0 so it learns to flag images
that depict the wrong content — the failure StoryPix needs to catch.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "training" / "data" / "raw" / "imagereward"
FEATS = ROOT / "training" / "data" / "processed" / "imagereward_features.pt"
CKPT = ROOT / "checkpoints"
SPLIT_OF_FOLDER = {"train": "train", "validation": "dev", "test": "test"}


# ------------------------------------------------------------------------------------------------ extraction


def extract(device: str) -> None:
    import pyarrow.parquet as pq
    from PIL import Image
    from tqdm import tqdm
    from transformers import AutoImageProcessor, AutoModel, AutoProcessor

    from app.services.image_evaluator import DINO, SIGLIP, image_statistics

    rows = []
    for split in ("train", "validation", "test"):
        rows += pq.read_table(RAW / f"metadata-{split}.parquet").to_pylist()
    by_name = {Path(r["image_path"]).name: r for r in rows}  # zips store bare file names
    sig = AutoModel.from_pretrained(SIGLIP).eval().to(device)
    sig_proc = AutoProcessor.from_pretrained(SIGLIP)
    dino = AutoModel.from_pretrained(DINO).eval().to(device)
    dino_proc = AutoImageProcessor.from_pretrained(DINO)
    records, text_cache = [], {}
    zips = sorted(p for p in RAW.glob("*.zip"))
    if not zips:
        raise SystemExit("no ImageRewardDB zips found — run: python -m training.datasets.download_data images")
    for zpath in zips:
        with zipfile.ZipFile(zpath) as zf:
            names = [n for n in zf.namelist() if n.endswith((".webp", ".png", ".jpg"))]
            for i in tqdm(range(0, len(names), 32), desc=zpath.name):
                batch_names = names[i : i + 32]
                metas, images = [], []
                for n in batch_names:
                    meta = by_name.get(Path(n).name)
                    if meta is None:
                        continue
                    try:
                        images.append(Image.open(io.BytesIO(zf.read(n))).convert("RGB"))
                    except OSError:
                        continue
                    metas.append(meta)
                if not images:
                    continue
                with torch.inference_mode():
                    s_img = sig.get_image_features(**sig_proc(images=images, return_tensors="pt").to(device))
                    s_img = F.normalize(getattr(s_img, "pooler_output", s_img), dim=-1).cpu()
                    d = dino(**dino_proc(images=images, return_tensors="pt").to(device)).pooler_output
                    d = F.normalize(d, dim=-1).cpu()
                    prompts = [m["prompt"] for m in metas if m["prompt"] not in text_cache]
                    if prompts:
                        t_in = sig_proc(
                            text=prompts, padding="max_length", truncation=True, max_length=64, return_tensors="pt"
                        ).to(device)
                        t = sig.get_text_features(**t_in)
                        t = F.normalize(getattr(t, "pooler_output", t), dim=-1).cpu()
                        for p, v in zip(prompts, t, strict=False):
                            text_cache[p] = v.half()
                for k, m in enumerate(metas):
                    folder = m["image_path"].split("/")[1]
                    records.append(
                        {
                            "split": SPLIT_OF_FOLDER[folder],
                            "prompt_id": m["prompt_id"],
                            "prompt": m["prompt"],
                            "img": s_img[k].half(),
                            "dino": d[k].half(),
                            "txt": text_cache[m["prompt"]],
                            "stats": torch.from_numpy(image_statistics(images[k])),
                            "alignment": (m["image_text_alignment_rating"] - 1) / 6,
                            "fidelity": (m["fidelity_rating"] - 1) / 6,
                            "overall": (m["overall_rating"] - 1) / 6,
                        }
                    )
    FEATS.parent.mkdir(parents=True, exist_ok=True)
    torch.save(records, FEATS)
    counts = defaultdict(int)
    for r in records:
        counts[r["split"]] += 1
    print(f"saved {len(records)} feature records {dict(counts)} → {FEATS}")


# ------------------------------------------------------------------------------------------------ training


def stack(records: list[dict], key: str) -> torch.Tensor:
    return torch.stack([r[key].float() if torch.is_tensor(r[key]) else torch.tensor(float(r[key])) for r in records])


def pair_indices(records: list[dict], target: str) -> list[tuple[int, int]]:
    """(i, j) with the same prompt and rating_i > rating_j."""
    groups = defaultdict(list)
    for i, r in enumerate(records):
        groups[r["prompt_id"]].append(i)
    pairs = []
    for idx in groups.values():
        for a in idx:
            for b in idx:
                if records[a][target] > records[b][target]:
                    pairs.append((a, b))
    return pairs


def spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    def rank(v: torch.Tensor) -> torch.Tensor:
        return v.argsort().argsort().float()

    rx, ry = rank(x), rank(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx * ry).sum() / (rx.norm() * ry.norm() + 1e-9))


def pairwise_accuracy(pred: torch.Tensor, records: list[dict], target: str) -> float:
    pairs = pair_indices(records, target)
    if not pairs:
        return float("nan")
    return sum(float(pred[a] > pred[b]) for a, b in pairs) / len(pairs)


def scores_for(task: str, model, records: list[dict]) -> torch.Tensor:
    with torch.no_grad():
        if task == "alignment":
            return model.score(stack(records, "img"), stack(records, "txt"))
        return model.score(stack(records, "img"), stack(records, "dino"), stack(records, "stats"))


def splits(data: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Dataset splits (prompt-disjoint). If a split was not downloaded, carve it deterministically out of the
    available records by prompt id (70/15/15) so the pipeline still runs on a partial download."""
    train = [r for r in data if r["split"] == "train"]
    dev = [r for r in data if r["split"] == "dev"]
    test = [r for r in data if r["split"] == "test"]
    if train and dev and test:
        return train, dev, test
    pool = sorted({r["prompt_id"] for r in data})
    rng = random.Random(0)
    rng.shuffle(pool)
    n = len(pool)
    part = {pid: ("train" if i < 0.7 * n else "dev" if i < 0.85 * n else "test") for i, pid in enumerate(pool)}
    print(
        f"note: incomplete download (train={len(train)}, dev={len(dev)}, test={len(test)}); "
        f"using a prompt-disjoint 70/15/15 split of the {len(data)} available images",
        flush=True,
    )
    by = {k: [r for r in data if part[r["prompt_id"]] == k] for k in ("train", "dev", "test")}
    return by["train"], by["dev"], by["test"]


def train_head(task: str, data: list[dict], epochs: int, lr: float, rank_weight: float, seed: int) -> dict:
    from app.models.quality_evaluator import QualityHead
    from app.models.scene_alignment import AlignmentHead

    random.seed(seed)
    torch.manual_seed(seed)
    target = "alignment" if task == "alignment" else "fidelity"
    train, dev, _ = splits(data)
    model = AlignmentHead() if task == "alignment" else QualityHead()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    img, txt, dino, stats = (stack(train, k) for k in ("img", "txt", "dino", "stats"))
    y = stack(train, target)
    pairs = torch.tensor(pair_indices(train, target) or [(0, 0)])
    best, history = {"spearman": -2.0}, []
    out = CKPT / ("scene_alignment.pt" if task == "alignment" else "quality_evaluator.pt")
    n = len(train)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, 128):
            b = perm[i : i + 128]
            if task == "alignment":
                logits = model(img[b], txt[b])
                # mismatched prompts are negatives: same image, a random other prompt → alignment 0
                shuffled = txt[b][torch.randperm(len(b))]
                neg_logits = model(img[b], shuffled)
                same = (shuffled * txt[b]).sum(-1) > 0.95  # skip accidental matches
                loss = F.mse_loss(torch.sigmoid(logits), y[b])
                if (~same).any():
                    loss = loss + 0.5 * F.binary_cross_entropy_with_logits(
                        neg_logits[~same], torch.zeros(int((~same).sum()))
                    )
            else:
                logits = model(img[b], dino[b], stats[b])
                loss = F.mse_loss(torch.sigmoid(logits), y[b])
            pb = pairs[torch.randint(len(pairs), (128,))]
            if task == "alignment":
                s_hi, s_lo = model(img[pb[:, 0]], txt[pb[:, 0]]), model(img[pb[:, 1]], txt[pb[:, 1]])
            else:
                s_hi = model(img[pb[:, 0]], dino[pb[:, 0]], stats[pb[:, 0]])
                s_lo = model(img[pb[:, 1]], dino[pb[:, 1]], stats[pb[:, 1]])
            loss = loss + rank_weight * F.softplus(-(s_hi - s_lo)).mean()  # Bradley–Terry
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        model.eval()
        pred = scores_for(task, model, dev)
        rho = spearman(pred, stack(dev, target))
        acc = pairwise_accuracy(pred, dev, target)
        row = {
            "epoch": epoch,
            "loss": round(total / n, 4),
            "dev_spearman": round(rho, 4),
            "dev_pairwise_acc": round(acc, 4),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if rho > best["spearman"]:
            best = {"spearman": round(rho, 4), "pairwise_acc": round(acc, 4), "epoch": epoch}
            model.save(out, extra={"target": target, "dev": best, "train_images": len(train)})
    return {
        "task": task,
        "target": target,
        "train_images": len(train),
        "dev_images": len(dev),
        "parameters": sum(p.numel() for p in model.parameters()),
        "best": best,
        "history": history,
    }


def evaluate_image_model(task: str, checkpoints: Path) -> dict:
    from app.models.quality_evaluator import QualityHead
    from app.models.scene_alignment import AlignmentHead

    data = torch.load(FEATS, weights_only=False)
    train, _, test = splits(data)
    target = "alignment" if task == "alignment" else "fidelity"
    y = stack(test, target)
    results = {}
    if task == "alignment":
        model, _ = AlignmentHead.load(checkpoints / "scene_alignment.pt")
        zero = (stack(test, "img") * stack(test, "txt")).sum(-1)
        results["zero-shot SigLIP cosine"] = zero
        results["Model C (trained head)"] = scores_for(task, model, test)
        # mismatch detection: can the score tell the right prompt from a random other prompt?
        txt = stack(test, "txt")
        perm = torch.randperm(len(test), generator=torch.Generator().manual_seed(0))
        wrong = model.score(stack(test, "img"), txt[perm])
        right = results["Model C (trained head)"]
        zero_wrong = (stack(test, "img") * txt[perm]).sum(-1)
        mismatch = {
            "Model C (trained head)": float((right > wrong).float().mean()),
            "zero-shot SigLIP cosine": float((zero > zero_wrong).float().mean()),
        }
    else:
        model, _ = QualityHead.load(checkpoints / "quality_evaluator.pt")
        results["Model D (trained head)"] = scores_for(task, model, test)
        xs = torch.cat([stack(train, "stats"), torch.ones(len(train), 1)], dim=1)
        w = torch.linalg.lstsq(xs, stack(train, target)[:, None]).solution
        results["OpenCV statistics (linear baseline)"] = (
            torch.cat([stack(test, "stats"), torch.ones(len(test), 1)], 1) @ w
        )[:, 0]
        mismatch = None
    table = {
        name: {
            "spearman": round(spearman(p, y), 4),
            "pearson": round(float(np.corrcoef(p, y)[0, 1]), 4),
            "pairwise_acc_within_prompt": round(pairwise_accuracy(p, test, target), 4),
        }
        for name, p in results.items()
    }
    for name, r in table.items():
        print(
            f"{task:9s} | {name:38s} ρ={r['spearman']:.3f}  r={r['pearson']:.3f}  "
            f"pairwise={r['pairwise_acc_within_prompt']:.3f}"
        )
    out = {"test_images": len(test), "target": f"ImageRewardDB {target} rating", "results": table}
    if mismatch:
        out["correct_vs_random_prompt_accuracy"] = {k: round(v, 4) for k, v in mismatch.items()}
        for k, v in mismatch.items():
            print(f"alignment | right prompt scored above a random prompt: {k}: {v:.3f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["extract", "train"])
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rank-weight", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    if args.step == "extract":
        extract(args.device)
        return
    data = torch.load(FEATS, weights_only=False)
    t0 = time.time()
    for task in ("alignment", "quality"):
        metrics = train_head(task, data, args.epochs, args.lr, args.rank_weight, args.seed)
        name = "scene_alignment" if task == "alignment" else "quality_evaluator"
        (CKPT / f"{name}.metrics.json").write_text(json.dumps(metrics, indent=2))
        print(f"{task}: best {metrics['best']} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
