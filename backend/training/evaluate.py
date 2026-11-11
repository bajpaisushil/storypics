"""Evaluate StoryPix's custom models against gold data and baselines.

    python -m training.evaluate scene      # Model A vs baselines on training/annotations/scene_gold.jsonl
    python -m training.evaluate tagger     # Model B vs rules on LitBank test + children's-story gold mentions
    python -m training.evaluate alignment  # Model C vs zero-shot SigLIP on ImageRewardDB test
    python -m training.evaluate quality    # Model D vs statistics-only baseline on ImageRewardDB test

Results are printed and written to checkpoints/eval_<task>.json.
"""

from __future__ import annotations

import argparse
import json
import statistics
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT / "training" / "annotations"
CKPT = ROOT / "checkpoints"


# ------------------------------------------------------------------------------------------- segmentation metrics


def seg_ids(starts: set[int], n: int) -> list[int]:
    ids, cur = [], 0
    for i in range(n):
        if i in starts:
            cur += 1
        ids.append(cur)
    return ids


def pk_windowdiff(ref: set[int], hyp: set[int], n: int, k: int) -> tuple[float, float]:
    r, h = seg_ids(ref, n), seg_ids(hyp, n)
    if n - k <= 0:
        return 0.0, 0.0
    pk = sum((r[i] == r[i + k]) != (h[i] == h[i + k]) for i in range(n - k)) / (n - k)
    wd = sum((r[i + k] - r[i]) != (h[i + k] - h[i]) for i in range(n - k)) / (n - k)
    return pk, wd


def prf(tp: int, n_pred: int, n_gold: int) -> dict:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / (p + r) if p + r else 0.0, 3)}


def eval_scene(args) -> dict:
    from app.config import Settings
    from app.models.scene_segmenter import HeuristicBoundaryScorer, LearnedBoundaryScorer, decode_boundaries
    from app.services.character_manager import CharacterResolver
    from app.services.story_parser import StoryAnalyzer, unit_infos
    from app.services.text_units import split_units

    analyzer = StoryAnalyzer(Settings(checkpoints_dir=args.checkpoints))
    systems: dict[str, object] = {"heuristic (labelling functions)": HeuristicBoundaryScorer()}
    if (args.checkpoints / "scene_segmenter.pt").exists():
        systems["Model A (trained)"] = LearnedBoundaryScorer(args.checkpoints / "scene_segmenter.pt")
    gold = [json.loads(line) for line in (ANN / "scene_gold.jsonl").read_text().splitlines() if line.strip()]
    acc: dict[str, dict] = {}

    def record(name: str, mode: str, ref: set[int], hyp: set[int], n: int, k: int, source: str) -> None:
        a = acc.setdefault(
            f"{name} | {mode}", {"tp": 0, "tp1": 0, "pred": 0, "gold": 0, "pk": [], "wd": [], "by_source": {}}
        )
        tp = len(ref & hyp)
        tp1 = sum(1 for b in hyp if {b - 1, b, b + 1} & ref)
        a["tp"] += tp
        a["tp1"] += tp1
        a["pred"] += len(hyp)
        a["gold"] += len(ref)
        pk, wd = pk_windowdiff(ref, hyp, n, k)
        a["pk"].append(pk)
        a["wd"].append(wd)
        s = a["by_source"].setdefault(source, {"tp": 0, "pred": 0, "gold": 0})
        s["tp"] += tp
        s["pred"] += len(hyp)
        s["gold"] += len(ref)

    misaligned = 0
    for story in gold:
        processed = split_units(story["text"])
        units = processed.units
        n = len(units)
        ref = set()
        for off in story["scene_starts"]:
            best = min(range(n), key=lambda i: abs(units[i].start_char - off))
            if abs(units[best].start_char - off) > 3:
                misaligned += 1
            if best > 0:
                ref.add(best)
        n_gold = len(ref) + 1
        k = max(2, round(n / n_gold / 2))
        source = "tinystories" if story["id"].startswith("tinystories") else "written+fables"
        resolution = CharacterResolver(analyzer.detector).resolve(processed)
        infos = unit_infos(processed, resolution)
        emb = analyzer.encoder.encode([u.text for u in units])
        # baselines that need no model
        sent_starts = {i for i in range(1, n) if units[i].sentence_index != units[i - 1].sentence_index}
        every2 = {i for i in sent_starts if units[i].sentence_index % 2 == 0}
        record("every 2 sentences", "auto", ref, every2, n, k, source)
        uniform = {round(j * n / n_gold) for j in range(1, n_gold)} - {0}
        record("uniform spacing", "oracle-N", ref, uniform, n, k, source)
        sims = torch.nn.functional.cosine_similarity(emb[:-1], emb[1:], dim=-1)
        tiling = [float(1 - s) for s in sims]
        mean, sd = statistics.mean(tiling), statistics.pstdev(tiling) if len(tiling) > 1 else 0.0
        record(
            "TextTiling (MiniLM)",
            "auto",
            ref,
            {i + 1 for i, d in enumerate(tiling) if d > mean + 0.5 * sd},
            n,
            k,
            source,
        )
        record("TextTiling (MiniLM)", "oracle-N", ref, set(decode_boundaries(tiling, n_gold, 0)), n, k, source)
        for name, scorer in systems.items():
            probs = scorer.boundary_probs(infos, emb)
            t = getattr(scorer, "threshold", 0.5)
            words = processed.word_count
            record(name, "auto", ref, set(decode_boundaries(probs, "auto", words, threshold=t)), n, k, source)
            record(name, "oracle-N", ref, set(decode_boundaries(probs, n_gold, words)), n, k, source)

    results = {}
    for key, a in acc.items():
        results[key] = {
            **prf(a["tp"], a["pred"], a["gold"]),
            "f1_tol1": prf(a["tp1"], a["pred"], a["gold"])["f1"],
            "pk": round(statistics.mean(a["pk"]), 3),
            "windowdiff": round(statistics.mean(a["wd"]), 3),
            "by_source_f1": {s: prf(v["tp"], v["pred"], v["gold"])["f1"] for s, v in a["by_source"].items()},
        }
    out = {
        "stories": len(gold),
        "gold_boundaries": sum(len(s["scene_starts"]) for s in gold),
        "misaligned_gold_offsets": misaligned,
        "results": results,
        "notes": "auto = system chooses the number of scenes; oracle-N = given the gold scene count. "
        "Pk/WindowDiff: lower is better.",
    }
    print(f"{'system | mode':48s} {'P':>6s} {'R':>6s} {'F1':>6s} {'F1±1':>6s} {'Pk':>6s} {'WD':>6s}")
    for key, r in sorted(results.items(), key=lambda kv: (kv[0].split(" | ")[1], -kv[1]["f1"])):
        print(
            f"{key:48s} {r['precision']:6.3f} {r['recall']:6.3f} {r['f1']:6.3f} {r['f1_tol1']:6.3f} "
            f"{r['pk']:6.3f} {r['windowdiff']:6.3f}"
        )
    return out


# ------------------------------------------------------------------------------------------- tagger


def span_prf(pred: set, gold: set, overlap: bool = False) -> dict:
    if not overlap:
        tp = len(pred & gold)
    else:
        used, tp = set(), 0
        for ps, pe, pl in pred:
            for g in gold:
                if g not in used and g[2] == pl and ps < g[1] and g[0] < pe:
                    used.add(g)
                    tp += 1
                    break
    return prf(tp, len(pred), len(gold))


def eval_tagger(args) -> dict:
    from app.models.character_extractor import MentionTagger, bio_to_spans
    from app.models.encoders import get_text_encoder
    from app.services.character_manager import RuleMentionDetector, TaggerMentionDetector
    from app.services.text_units import split_units

    results: dict = {}
    ckpt = args.checkpoints / "character_tagger.pt"
    # (1) LitBank held-out books (gold, literary domain)
    data_path = ROOT / "training" / "data" / "processed" / "tagger_data.pt"
    if ckpt.exists() and data_path.exists():
        data = torch.load(data_path, weights_only=False)
        model, _ = MentionTagger.load(ckpt)
        pred, gold = set(), set()
        for si, r in enumerate(data["litbank_test"]):
            n = len(r["words"])
            mask = torch.ones(1, n, dtype=torch.bool)
            path = model.decode(r["states"].float()[None], r["case"].long()[None], r["pos"].long()[None], mask)[0]
            pred |= {(si, s.start, s.end, s.label) for s in bio_to_spans(path)}
            gold |= {(si, s.start, s.end, s.label) for s in bio_to_spans(r["tags"].long().tolist())}
        for label in ("CHAR", "PLACE"):
            results[f"litbank_test | Model B | {label}"] = span_prf(
                {p for p in pred if p[3] == label}, {g for g in gold if g[3] == label}
            )
        results["litbank_test | Model B | all"] = span_prf(pred, gold)
    # (2) children's stories (gold mentions written for this project)
    stories = {
        json.loads(line)["id"]: json.loads(line)["text"]
        for line in (ANN / "scene_gold.jsonl").read_text().splitlines()
        if line.strip()
    }
    ann = json.loads((ANN / "mention_gold.json").read_text())
    detectors = {"rules": RuleMentionDetector()}
    if ckpt.exists():
        enc = get_text_encoder()
        detectors["Model B"] = TaggerMentionDetector(ckpt, enc, lexicon_backstop=False)
        detectors["Model B + lexicon (production)"] = TaggerMentionDetector(ckpt, enc, lexicon_backstop=True)
    gold_spans: set = set()
    preds: dict[str, set] = {k: set() for k in detectors}
    for sid, mentions in ann.items():
        if sid.startswith("_"):
            continue
        processed = split_units(stories[sid])
        text, pos = processed.text, 0
        for label, mention in mentions:
            start = text.find(mention, pos)
            if start < 0:
                raise ValueError(f"{sid}: mention {mention!r} not found after offset {pos}")
            gold_spans.add((sid, start, start + len(mention), label))
            pos = start + len(mention)
        for name, det in detectors.items():
            preds[name] |= {(sid, d.start, d.end, d.label) for d in det.detect(processed)}
    for name, pred in preds.items():
        results[f"children_gold | {name} | exact"] = span_prf(pred, gold_spans)
        results[f"children_gold | {name} | overlap"] = span_prf(
            {(p[1] + hash(p[0]) * 10_000, p[2] + hash(p[0]) * 10_000, p[3]) for p in pred},
            {(g[1] + hash(g[0]) * 10_000, g[2] + hash(g[0]) * 10_000, g[3]) for g in gold_spans},
            overlap=True,
        )
    for key, r in results.items():
        print(f"{key:55s} P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}")
    return {"results": results, "gold_mentions": len(gold_spans)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=["scene", "tagger", "alignment", "quality"])
    ap.add_argument("--checkpoints", type=Path, default=CKPT)
    args = ap.parse_args()
    if args.task in ("alignment", "quality"):
        from training.train_image_models import evaluate_image_model

        out = evaluate_image_model(args.task, args.checkpoints)
    else:
        out = {"scene": eval_scene, "tagger": eval_tagger}[args.task](args)
    (args.checkpoints / f"eval_{args.task}.json").write_text(json.dumps(out, indent=2))
    print(f"→ {args.checkpoints / f'eval_{args.task}.json'}")


if __name__ == "__main__":
    main()
