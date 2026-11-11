"""Tensor-level tests for the custom PyTorch models (random weights; no training data needed)."""

import itertools

import torch

from app.models.character_extractor import CRF, TAGS, MentionTagger, TaggedSpan, bio_to_spans, spans_to_bio
from app.models.quality_evaluator import QualityHead
from app.models.scene_alignment import AlignmentHead
from app.models.scene_segmenter import NUM_FEATURES, SceneBoundaryModel, decode_boundaries
from app.services.image_generator import chunk_token_ids, count_chunks


def test_scene_boundary_model_shapes_and_roundtrip(tmp_path):
    model = SceneBoundaryModel().eval()
    emb, feats = torch.randn(2, 7, 384), torch.randn(2, 7, NUM_FEATURES)
    mask = torch.ones(2, 7, dtype=torch.bool)
    mask[1, 5:] = False
    out = model(emb, feats, mask)
    assert out.shape == (2, 6)
    model.save(tmp_path / "m.pt", extra={"threshold": 0.4})
    loaded, extra = SceneBoundaryModel.load(tmp_path / "m.pt")
    assert extra["threshold"] == 0.4 and torch.allclose(loaded(emb, feats, mask), out, atol=1e-6)


def test_decode_boundaries_fixed_and_auto():
    probs = [0.9, 0.1, 0.2, 0.8, 0.05]
    assert decode_boundaries(probs, 3, word_count=60) == [1, 4]
    assert decode_boundaries(probs, 10, word_count=60) == [1, 2, 3, 4, 5]  # capped at one unit per scene
    assert decode_boundaries(probs, "auto", word_count=60, threshold=0.5) == [1, 4]


def _brute_force_best(crf: CRF, em: torch.Tensor) -> list[int]:
    trans, start = crf._trans()
    best, best_path = -1e9, None
    for path in itertools.product(range(len(TAGS)), repeat=em.shape[0]):
        s = start[path[0]] + em[0, path[0]] + crf.end[path[-1]]
        s += sum(trans[path[i - 1], path[i]] + em[i, path[i]] for i in range(1, len(path)))
        if s > best:
            best, best_path = s, list(path)
    return best_path


def test_crf_viterbi_matches_brute_force_and_respects_bio():
    torch.manual_seed(0)
    crf = CRF(len(TAGS), TAGS)
    with torch.no_grad():
        crf.transitions.normal_()
    em = torch.randn(1, 4, len(TAGS))
    path = crf.decode(em, torch.ones(1, 4, dtype=torch.bool))[0]
    assert path == _brute_force_best(crf, em[0])
    for a, b in itertools.pairwise(path):
        assert not (TAGS[b].startswith("I-") and TAGS[a] in ("O",))


def test_crf_nll_matches_brute_force_partition():
    torch.manual_seed(1)
    crf = CRF(len(TAGS), TAGS)
    em = torch.randn(1, 3, len(TAGS))
    tags = torch.tensor([[1, 2, 0]])  # B-CHAR I-CHAR O
    nll = crf.nll(em, tags, torch.ones(1, 3, dtype=torch.bool))
    trans, start = crf._trans()
    scores = []
    for path in itertools.product(range(len(TAGS)), repeat=3):
        s = start[path[0]] + em[0, 0, path[0]] + crf.end[path[-1]]
        s += sum(trans[path[i - 1], path[i]] + em[0, i, path[i]] for i in range(1, 3))
        scores.append(s)
    gold = start[1] + em[0, 0, 1] + trans[1, 2] + em[0, 1, 2] + trans[2, 0] + em[0, 2, 0] + crf.end[0]
    expected = torch.logsumexp(torch.stack(scores), 0) - gold
    assert torch.allclose(nll, expected, atol=1e-4)


def test_tagger_loss_and_decode_with_padding():
    model = MentionTagger()
    words, case, pos = torch.randn(2, 6, 384), torch.ones(2, 6, dtype=torch.long), torch.ones(2, 6, dtype=torch.long)
    mask = torch.tensor([[True] * 6, [True] * 4 + [False] * 2])
    tags = torch.zeros(2, 6, dtype=torch.long)
    loss = model.loss(words, case, pos, mask, tags)
    loss.backward()
    assert loss.item() > 0
    paths = model.decode(words, case, pos, mask)
    assert [len(p) for p in paths] == [6, 4]


def test_bio_roundtrip():
    spans = [TaggedSpan(0, 2, "CHAR"), TaggedSpan(4, 5, "PLACE")]
    assert bio_to_spans(spans_to_bio(6, spans)) == spans


def test_evaluator_heads_output_probabilities():
    a = AlignmentHead().eval()
    q = QualityHead().eval()
    assert a.score(torch.randn(3, 768), torch.randn(3, 768)).shape == (3,)
    s = q.score(torch.randn(3, 768), torch.randn(3, 384), torch.randn(3, 6))
    assert s.shape == (3,) and ((s >= 0) & (s <= 1)).all()


def test_long_prompt_chunking():
    ids = list(range(1, 161))  # 160 tokens → 3 windows
    n = count_chunks(len(ids))
    rows = chunk_token_ids(ids, n, bos=500, eos=501, pad=0)
    assert n == 3 and all(len(r) == 77 for r in rows)
    assert rows[0][0] == 500 and rows[0][76] == 501 and rows[2][1:11] == list(range(151, 161))
    assert rows[2][11] == 501 and rows[2][-1] == 0
