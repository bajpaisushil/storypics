"""Stage 1 — text normalisation and event-unit splitting.

Scenes are built from *event units*, not sentences: a sentence such as "They became good friends and
explored the farm together." holds two depictable moments. Units are produced by splitting sentences at
coordinated verb phrases found by the dependency parse (spaCy, MIT-licensed `en_core_web_sm`). The
scene-boundary model then decides which units start a new scene.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span, Token

from app.schemas.story import EventUnit

_NLP: Language | None = None
_NLP_LOCK = threading.Lock()

MIN_CLAUSE_TOKENS = 3
QUOTE_CHARS = {'"', "“", "”"}


def get_nlp(model: str = "en_core_web_sm") -> Language:
    global _NLP
    with _NLP_LOCK:
        if _NLP is None:
            _NLP = spacy.load(model)
        return _NLP


def normalize_text(text: str) -> str:
    """Normalise quotes/whitespace while keeping paragraph breaks (blank lines) intact."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@dataclass
class Unit:
    """Internal unit with access to the parse; converted to the EventUnit schema for output."""

    index: int
    span: Span
    sentence_index: int
    paragraph_index: int
    is_continuation: bool
    inherited_subject: Span | None = None
    tokens: list[Token] = field(default_factory=list)

    @property
    def text(self) -> str:
        return self.span.text.strip()

    @property
    def start_char(self) -> int:
        return self.span.start_char

    @property
    def end_char(self) -> int:
        return self.span.end_char

    def schema(self) -> EventUnit:
        return EventUnit(
            index=self.index,
            text=self.text,
            start=self.start_char,
            end=self.end_char,
            sentence_index=self.sentence_index,
            paragraph_index=self.paragraph_index,
            is_continuation=self.is_continuation,
        )


@dataclass
class ProcessedStory:
    text: str
    doc: Doc
    units: list[Unit]
    paragraph_starts: list[int]  # char offsets where paragraphs start

    @property
    def word_count(self) -> int:
        return sum(1 for t in self.doc if not t.is_punct and not t.is_space)


def _quote_ranges(text: str) -> list[tuple[int, int]]:
    ranges, start = [], None
    for i, ch in enumerate(text):
        if ch in QUOTE_CHARS:
            if start is None:
                start = i
            else:
                ranges.append((start, i))
                start = None
    return ranges


def _inside(ranges: list[tuple[int, int]], pos: int) -> bool:
    return any(a <= pos <= b for a, b in ranges)


def _content_len(tokens: list[Token]) -> int:
    return sum(1 for t in tokens if not t.is_punct and not t.is_space)


def _subject_of(verb: Token) -> Span | None:
    for child in verb.children:
        if child.dep_ in ("nsubj", "nsubjpass", "expl"):
            return verb.doc[child.left_edge.i : child.right_edge.i + 1]
    return None


def _split_points(sent: Span, quotes: list[tuple[int, int]]) -> list[tuple[int, Token]]:
    """Token indices where a coordinated verb phrase starts, with the verb it attaches to."""
    points: list[tuple[int, Token]] = []
    root = sent.root
    # walk the chain ROOT → conj → conj ...
    frontier = [root]
    while frontier:
        head = frontier.pop()
        for child in head.children:
            if child.dep_ != "conj" or child.pos_ not in ("VERB", "AUX"):
                continue
            frontier.append(child)
            if _inside(quotes, child.idx):
                continue
            # clause starts at the coordinating conjunction / comma just before the conjunct's subtree
            start = child.left_edge.i
            j = start - 1
            while j > sent.start and (
                sent.doc[j].dep_ == "cc" or sent.doc[j].is_punct or sent.doc[j].lower_ in ("then", "and", "but", "so")
            ):
                j -= 1
            start = j + 1
            if start > sent.start:
                points.append((start, child))
    return sorted(points, key=lambda p: p[0])


def split_units(text: str, nlp: Language | None = None, doc: Doc | None = None) -> ProcessedStory:
    """Parse ``text`` (or reuse a pre-parsed ``doc`` of the *normalised* text) and split it into event units."""
    if doc is None:
        text = normalize_text(text)
        doc = (nlp or get_nlp())(text)
    else:
        text = doc.text
    paragraph_starts = [0] + [m.end() for m in re.finditer(r"\n\s*\n", text)]
    quotes = _quote_ranges(text)

    def paragraph_of(char: int) -> int:
        idx = 0
        for i, start in enumerate(paragraph_starts):
            if char >= start:
                idx = i
        return idx

    units: list[Unit] = []
    for s_idx, sent in enumerate(s for s in doc.sents if s.text.strip()):
        points = _split_points(sent, quotes)
        boundaries = [sent.start]
        verbs: list[Token | None] = [None]
        for start, verb in points:
            prev = boundaries[-1]
            if (
                _content_len(list(doc[prev:start])) >= MIN_CLAUSE_TOKENS
                and _content_len(list(doc[start : sent.end])) >= MIN_CLAUSE_TOKENS
            ):
                boundaries.append(start)
                verbs.append(verb)
        boundaries.append(sent.end)
        root_subject = _subject_of(sent.root)
        for k in range(len(boundaries) - 1):
            span = doc[boundaries[k] : boundaries[k + 1]]
            while len(span) and span[-1].is_space:
                span = doc[span.start : span.end - 1]
            if not span.text.strip():
                continue
            verb = verbs[k]
            inherited = None
            if verb is not None and _subject_of(verb) is None:
                inherited = root_subject
            units.append(
                Unit(
                    index=len(units),
                    span=span,
                    sentence_index=s_idx,
                    paragraph_index=paragraph_of(span.start_char),
                    is_continuation=k > 0,
                    inherited_subject=inherited,
                    tokens=list(span),
                )
            )
    return ProcessedStory(text=text, doc=doc, units=units, paragraph_starts=paragraph_starts)
