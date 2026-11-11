"""Stages 2–4 — character mentions → character clusters → canonical character profiles.

Mention detection is pluggable: the trained BiLSTM-CRF tagger (Model B) when its checkpoint exists,
otherwise a rule-based detector (which is also the silver-label source for training). Clustering and
pronoun resolution are deterministic and explainable; profiles merge explicit story details with
curated appearance priors (see ``appearance.py``).
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal, Protocol

from spacy.tokens import Span, Token

from app.schemas.story import CharacterCategory, CharacterMention, CharacterProfile, Gender
from app.services import appearance as kb
from app.services.text_units import ProcessedStory, Unit

Label = Literal["CHAR", "PLACE"]

PRONOUNS: dict[str, tuple[str, str]] = {  # pronoun → (gender, number)
    "he": ("male", "sg"),
    "him": ("male", "sg"),
    "his": ("male", "sg"),
    "himself": ("male", "sg"),
    "she": ("female", "sg"),
    "her": ("female", "sg"),
    "hers": ("female", "sg"),
    "herself": ("female", "sg"),
    "it": ("unknown", "sg"),
    "its": ("unknown", "sg"),
    "itself": ("unknown", "sg"),
    "they": ("unknown", "pl"),
    "them": ("unknown", "pl"),
    "their": ("unknown", "pl"),
    "themselves": ("unknown", "pl"),
    "theirs": ("unknown", "pl"),
}
GROUP_NOUNS = {"friend", "animal", "everyone", "everybody", "family", "group", "team", "creature", "pal", "buddy"}
SPEECH_VERBS = {
    "say",
    "ask",
    "reply",
    "shout",
    "whisper",
    "cry",
    "laugh",
    "smile",
    "think",
    "decide",
    "want",
    "feel",
    "giggle",
    "nod",
    "sigh",
    "yell",
    "answer",
    "exclaim",
    "wonder",
    "hope",
    "promise",
    "agree",
    "thank",
    "hug",
    "wave",
}
NON_CHARACTER_HEADS = {
    "day",
    "time",
    "way",
    "thing",
    "lot",
    "home",
    "place",
    "top",
    "side",
    "end",
    "morning",
    "evening",
    "night",
    "afternoon",
    "fun",
    "idea",
    "name",
    "story",
    "life",
    "world",
    "help",
}
PLACE_PREPS = {
    "in",
    "at",
    "on",
    "to",
    "into",
    "through",
    "across",
    "near",
    "around",
    "under",
    "inside",
    "toward",
    "towards",
    "onto",
    "by",
    "behind",
    "beside",
    "over",
    "from",
    "along",
    "past",
}


# --------------------------------------------------------------------------------------------- detection


@dataclass(frozen=True)
class DetectedSpan:
    start: int  # char offsets in the normalised story text
    end: int
    label: Label
    source: Literal["tagger", "rules"]


class MentionDetector(Protocol):
    name: str

    def detect(self, story: ProcessedStory) -> list[DetectedSpan]: ...


def _species_key(tokens: list[Token]) -> str | None:
    """Map a noun phrase to a species/role key, handling two-word keys ("teddy bear") and aliases."""
    words = [t.lemma_.lower() for t in tokens if t.pos_ in ("NOUN", "PROPN", "ADJ")]
    raw = [t.lower_ for t in tokens if t.pos_ in ("NOUN", "PROPN", "ADJ")]
    for seq in (raw, words):
        if len(seq) >= 2 and " ".join(seq[-2:]) in kb.SPECIES:
            return " ".join(seq[-2:])
    for w in (raw[-1:] + words[-1:]) if raw else []:
        w = kb.ALIASES.get(w, w)
        if w in kb.SPECIES or w in kb.HUMANS:
            return w
    return None


class RuleMentionDetector:
    """High-precision rules over the dependency parse; also used to produce silver training labels."""

    name = "rules-v1"

    def detect(self, story: ProcessedStory) -> list[DetectedSpan]:
        doc = story.doc
        spans: list[DetectedSpan] = []
        taken: set[int] = set()

        def add(span: Span, label: Label) -> None:
            if any(i in taken for i in range(span.start, span.end)):
                return
            taken.update(range(span.start, span.end))
            spans.append(DetectedSpan(span.start_char, span.end_char, label, "rules"))

        for ent in doc.ents:
            if ent.label_ == "PERSON":
                add(ent, "CHAR")
        for chunk in doc.noun_chunks:
            head = chunk.root
            if head.pos_ == "PRON":
                continue
            # strip leading determiners/possessives that belong to another referent ("its home")
            key = _species_key(list(chunk))
            if key is not None:
                add(chunk, "CHAR")
                continue
            lemma = head.lemma_.lower()
            if lemma in kb.PLACES and (
                (head.dep_ == "pobj" and head.head.lower_ in PLACE_PREPS) or head.dep_ in ("dobj", "nsubj", "attr")
            ):
                add(chunk, "PLACE")
                continue
            if (
                head.pos_ == "PROPN"
                and head.dep_ in ("nsubj", "dobj", "pobj", "appos", "conj", "attr")
                and lemma not in kb.PLACES
            ):
                add(chunk, "CHAR")
                continue
            if (
                head.dep_ == "nsubj"
                and head.head.lemma_ in SPEECH_VERBS
                and lemma not in NON_CHARACTER_HEADS
                and lemma not in kb.PLACES
                and head.tag_ in ("NN", "NNP")
            ):
                add(chunk, "CHAR")
        for ent in doc.ents:
            if ent.label_ in ("GPE", "LOC", "FAC"):
                add(ent, "PLACE")
        return sorted(spans, key=lambda s: s.start)


# --------------------------------------------------------------------------------------------- clustering


@dataclass
class Cluster:
    id: str
    key: str  # species/role key or name-derived key
    name: str | None = None
    species: str | None = None
    plural: bool = False
    gender: Gender = "unknown"
    mentions: list[CharacterMention] = field(default_factory=list)
    traits: list[str] = field(default_factory=list)
    colors: list[str] = field(default_factory=list)
    with_phrases: list[str] = field(default_factory=list)
    units: set[int] = field(default_factory=set)
    surface: str | None = None  # head word as written, e.g. "Grandma" for canonical species "grandmother"

    @property
    def display_name(self) -> str:
        if self.name:
            return self.name
        if self.surface and self.surface[:1].isupper():
            return self.surface
        return (self.species or self.key).title()


@dataclass
class CharacterResolution:
    clusters: list[Cluster]
    unit_characters: list[list[str]]  # unit index → ordered cluster ids present
    unit_places: list[list[str]]  # unit index → place keys mentioned
    place_spans: list[DetectedSpan]
    pronoun_links: dict[int, list[str]]  # token index → cluster ids (for summaries)
    subject_links: dict[int, list[str]]  # unit index → cluster ids of its grammatical subject
    detector: str
    mention_tokens: dict[int, str] = field(default_factory=dict)  # head token index → cluster id


def _span_for(story: ProcessedStory, det: DetectedSpan) -> Span | None:
    return story.doc.char_span(det.start, det.end, alignment_mode="expand")


def _unit_of(story: ProcessedStory, char: int) -> int:
    for u in story.units:
        if u.start_char <= char < max(u.end_char, u.start_char + 1):
            return u.index
    # fall back to the nearest preceding unit
    prev = 0
    for u in story.units:
        if u.start_char <= char:
            prev = u.index
    return prev


def _name_link(head: Token) -> str | None:
    """'a hen named Henny' / 'a cat called Tom' / 'Henny, a hen' → 'Henny'."""
    for child in head.children:
        if child.dep_ == "acl" and child.lemma_ in ("name", "call"):
            for gc in child.children:
                if gc.pos_ == "PROPN" or gc.dep_ in ("oprd", "attr", "dobj"):
                    tokens = [t for t in gc.subtree if t.pos_ == "PROPN"] or [gc]
                    return " ".join(t.text for t in tokens)
        if child.dep_ == "appos" and child.pos_ == "PROPN":
            return child.text
    if head.dep_ == "appos" and head.head.pos_ == "PROPN":
        return head.head.text
    return None


def _modifiers(span: Span) -> tuple[list[str], list[str], list[str]]:
    """Adjective traits, colour words and 'with …' phrases attached to a mention."""
    traits, colors, withs = [], [], []
    head = span.root
    for tok in span:
        if tok.dep_ in ("amod", "compound", "nmod") and tok.head == head and tok.pos_ in ("ADJ", "NOUN", "VERB"):
            word = tok.lower_
            if word in kb.COLOR_WORDS:
                colors.append(word)
            elif tok.pos_ in ("ADJ", "VERB"):
                traits.append(word)
    for child in head.children:
        if child.dep_ == "prep" and child.lower_ == "with":
            phrase = " ".join(t.text for t in child.subtree)
            if len(phrase.split()) <= 7:
                withs.append(phrase)
    # copular descriptions elsewhere: "The cow was big and brown."
    if head.dep_ == "nsubj" and head.head.lemma_ in ("be", "look", "seem"):
        for child in head.head.children:
            if child.dep_ == "acomp":
                for t in [child] + [c for c in child.children if c.dep_ == "conj"]:
                    (colors if t.lower_ in kb.COLOR_WORDS else traits).append(t.lower_)
    return traits, colors, withs


class CharacterResolver:
    def __init__(self, detector: MentionDetector):
        self.detector = detector

    def resolve(self, story: ProcessedStory) -> CharacterResolution:
        detected = self.detector.detect(story)
        char_spans = [d for d in detected if d.label == "CHAR"]
        place_spans = [d for d in detected if d.label == "PLACE"]
        clusters: list[Cluster] = []
        by_name: dict[str, Cluster] = {}
        by_species: dict[str, Cluster] = {}
        token_cluster: dict[int, str] = {}  # head token index → cluster id
        absent: set[int] = set()  # mentions that do not put the character in the scene ("a letter from X")

        def cluster_by_index(cid: str) -> Cluster:
            return next(c for c in clusters if c.id == cid)

        def new_cluster(key: str) -> Cluster:
            c = Cluster(id=f"c{len(clusters) + 1}", key=key)
            clusters.append(c)
            return c

        # 1) nominal + named mentions, in text order
        for det in char_spans:
            span = _span_for(story, det)
            if span is None:
                continue
            head = span.root
            unit = _unit_of(story, det.start)
            lemma = head.lemma_.lower()
            if lemma in GROUP_NOUNS and head.tag_ == "NNS":
                continue  # "the friends" — handled like "they"
            verb = head.head
            if (
                head.dep_ == "attr"
                and verb.lemma_ in ("be", "become")
                and not any(c.dep_ == "expl" for c in verb.children)
                and _copula_subject(head) is None
            ):
                continue  # "when she was a girl" describes an existing referent, it is not a new character
            if head.dep_ == "pobj" and verb.lower_ == "as":
                continue  # "crowned Arin as the protector" — a role, not a new character
            species = _species_key(list(span))
            plural = head.tag_ in ("NNS", "NNPS")
            if plural and species and species in by_species and not by_species[species].plural:
                continue  # "all dragons" is generic, not the dragon of this story
            is_name = head.pos_ == "PROPN" and species is None
            name = span.text if is_name else _name_link(head)
            if name:
                name = re.sub(r"^(the|a|an)\s+", "", name, flags=re.I).strip()
            cluster = None
            copula_subject = _copula_subject(head)
            if copula_subject is not None and copula_subject.i in token_cluster:
                cluster = cluster_by_index(token_cluster[copula_subject.i])
            elif name and name.lower() in by_name:
                cluster = by_name[name.lower()]
            elif species and species in by_species and not name:
                cluster = by_species[species]
            elif name and species and species in by_species and by_species[species].name is None:
                cluster = by_species[species]  # "a hen ... the hen, Henny"
            if cluster is None:
                cluster = new_cluster(species or (name or span.text).lower())
                cluster.surface = None if name else head.text
            if name and not cluster.name:
                cluster.name = name
                by_name[name.lower()] = cluster
            if species:
                cluster.species = cluster.species or species
                by_species.setdefault(species, cluster)
            if plural and species and len(cluster.mentions) == 0:
                cluster.plural = True
            traits, colors, withs = _modifiers(span)
            cluster.traits += [t for t in traits if t not in cluster.traits]
            cluster.colors += [c for c in colors if c not in cluster.colors]
            cluster.with_phrases += [w for w in withs if w not in cluster.with_phrases]
            cluster.mentions.append(
                CharacterMention(
                    text=span.text,
                    start=det.start,
                    end=det.end,
                    unit_index=unit,
                    kind="name" if is_name else "nominal",
                    source=det.source,
                )
            )
            token_cluster[head.i] = cluster.id
            if _is_participant(head):
                cluster.units.add(unit)
            else:
                absent.add(head.i)

        # 1b) name propagation: once "Storm" is known to be a character, every "Storm" is a mention of it
        for tok in story.doc:
            if tok.i in token_cluster or tok.pos_ not in ("PROPN", "NOUN") or not tok.text[:1].isupper():
                continue
            cluster = by_name.get(tok.text.lower())
            if cluster is not None:
                unit = _unit_of(story, tok.idx)
                cluster.mentions.append(
                    CharacterMention(
                        text=tok.text,
                        start=tok.idx,
                        end=tok.idx + len(tok.text),
                        unit_index=unit,
                        kind="name",
                        source="resolver",
                    )
                )
                token_cluster[tok.i] = cluster.id
                if _is_participant(tok):
                    cluster.units.add(unit)
                else:
                    absent.add(tok.i)

        for c in clusters:
            prior = kb.SPECIES.get(c.species or "")
            human = kb.HUMANS.get(c.species or "")
            if prior and prior.gender != "unknown":
                c.gender = prior.gender
            elif human and human.gender != "unknown":
                c.gender = human.gender

        # 2) pronouns + group nouns, with recency-based salience
        quote_ranges = _quotes(story.text)
        first_person = [
            t
            for t in story.doc
            if t.lower_ in FIRST_SINGULAR | FIRST_PLURAL and not any(a <= t.idx <= b for a, b in quote_ranges)
        ]
        narrator: Cluster | None = None
        if any(t.lower_ in FIRST_SINGULAR for t in first_person) or len(first_person) >= 2:
            narrator = new_cluster("narrator")  # first-person storyteller ("I visited my grandmother")
            narrator.name, narrator.species = NARRATOR_NAME, "child"
        cluster_by_id = {c.id: c for c in clusters}
        last_seen: dict[str, int] = {}
        pronoun_links: dict[int, list[str]] = {}
        unit_chars: list[list[str]] = [[] for _ in story.units]

        def note(unit_idx: int, cids: list[str], pos: int) -> None:
            for cid in cids:
                last_seen[cid] = pos
                if cid not in unit_chars[unit_idx]:
                    unit_chars[unit_idx].append(cid)
                cluster_by_id[cid].units.add(unit_idx)

        for unit in story.units:
            for tok in unit.tokens:
                if tok.i in token_cluster:
                    if tok.i in absent:
                        last_seen[token_cluster[tok.i]] = tok.i  # still salient for pronouns
                    else:
                        note(unit.index, [token_cluster[tok.i]], tok.i)
                    continue
                low = tok.lower_
                in_quote = any(a <= tok.idx <= b for a, b in quote_ranges)
                if in_quote:
                    continue
                if narrator is not None and low in FIRST_SINGULAR | FIRST_PLURAL:
                    group = [narrator.id]
                    if low in FIRST_PLURAL:
                        group += [c for c in _recent_group(story, unit.index, unit_chars, []) if c != narrator.id][:3]
                    pronoun_links[tok.i] = group
                    narrator.mentions.append(
                        CharacterMention(
                            text=tok.text,
                            start=tok.idx,
                            end=tok.idx + len(tok.text),
                            unit_index=unit.index,
                            kind="pronoun",
                            source="resolver",
                        )
                    )
                    note(unit.index, group, tok.i)
                    continue
                is_group_noun = tok.lemma_.lower() in GROUP_NOUNS and tok.tag_ == "NNS" and tok.pos_ == "NOUN"
                if low not in PRONOUNS and not is_group_noun:
                    continue
                gender, number = PRONOUNS.get(low, ("unknown", "pl"))
                ranked = sorted(last_seen, key=lambda cid: last_seen[cid], reverse=True)
                target: list[str] = []
                if number == "pl":
                    plural = [cid for cid in ranked if cluster_by_id[cid].plural]
                    recent = _recent_group(story, unit.index, unit_chars, ranked)
                    if len(recent) >= 2:
                        target = recent
                    elif plural:
                        target = plural[:1]
                    else:
                        target = ranked[:2]
                else:
                    excluded = _clause_subject_ids(tok, token_cluster, pronoun_links)
                    recent_window = {cid for u in (unit.index, unit.index - 1) if u >= 0 for cid in unit_chars[u]}
                    if gender in ("male", "female"):  # prefer a recent character whose gender is already known
                        exact = [
                            cid
                            for cid in ranked
                            if cid in recent_window
                            and cid not in excluded
                            and cluster_by_id[cid].gender == gender
                            and not cluster_by_id[cid].plural
                        ]
                        ranked = exact + [cid for cid in ranked if cid not in exact]
                    for cid in ranked:
                        c = cluster_by_id[cid]
                        if c.plural or cid in excluded:
                            continue
                        if gender == "unknown":  # it/its: a recent animal/object of unknown gender, else unresolved
                            recent = any(cid in unit_chars[u] for u in (unit.index, unit.index - 1) if u >= 0)
                            if _category(c) in ("animal", "creature", "object") and c.gender == "unknown" and recent:
                                target = [cid]
                                break
                        elif c.gender in (gender, "unknown"):
                            target = [cid]
                            break
                if target:
                    if gender in ("male", "female") and len(target) == 1:
                        c = cluster_by_id[target[0]]
                        if c.gender == "unknown":
                            c.gender = gender  # pronoun evidence fixes gender
                    pronoun_links[tok.i] = target
                    for cid in target:
                        cluster_by_id[cid].mentions.append(
                            CharacterMention(
                                text=tok.text,
                                start=tok.idx,
                                end=tok.idx + len(tok.text),
                                unit_index=unit.index,
                                kind="pronoun",
                                source="resolver",
                            )
                        )
                    note(unit.index, target, tok.i)

        # 2b) "his sister" / "her brother" → the one named character of that gender (other than the possessor)
        for c in list(clusters):
            want = SIBLINGS.get(c.species or "")
            if not want or c.name or not c.mentions:
                continue
            possessors = {
                cid
                for m in c.mentions
                for tok in story.doc.char_span(m.start, m.end, alignment_mode="expand") or []
                for cid in pronoun_links.get(tok.i, [])
                if tok.dep_ == "poss"
            }
            candidates = [
                o
                for o in clusters
                if o is not c
                and o.name
                and o.name != NARRATOR_NAME
                and o.gender == want
                and o.id not in possessors
                and _category(o) == "human"
            ]
            if len(candidates) == 1:
                _merge(c, candidates[0], token_cluster, pronoun_links, unit_chars)

        # 3) continuation units inherit the subject of the clause they were split from
        subject_links: dict[int, list[str]] = {}
        for unit in story.units:
            subj = _unit_subject(unit)
            ids = _ids_for(subj, token_cluster, pronoun_links) if subj is not None else []
            if not ids and unit.inherited_subject is not None:
                ids = _ids_for(unit.inherited_subject, token_cluster, pronoun_links)
                others = [cid for cid in unit_chars[unit.index] if cid not in ids]
                unit_chars[unit.index] = ids + others
                for cid in ids:
                    cluster_by_id[cid].units.add(unit.index)
            subject_links[unit.index] = ids

        unit_places: list[list[str]] = [[] for _ in story.units]
        for det in place_spans:
            span = _span_for(story, det)
            if span is None:
                continue
            key = span.root.lemma_.lower()
            u = _unit_of(story, det.start)
            if key not in unit_places[u]:
                unit_places[u].append(key)

        return CharacterResolution(
            clusters=[c for c in clusters if c.mentions],
            unit_characters=unit_chars,
            unit_places=unit_places,
            place_spans=place_spans,
            pronoun_links=pronoun_links,
            subject_links=subject_links,
            detector=self.detector.name,
            mention_tokens=token_cluster,
        )


NON_PARTICIPANT_PREPS = {"from", "about", "of", "like", "for"}
FIRST_SINGULAR = {"i", "me", "my", "mine", "myself"}
FIRST_PLURAL = {"we", "us", "our", "ours", "ourselves"}
NARRATOR_NAME = "Narrator"
SIBLINGS = {"sister": "female", "brother": "male"}


def _is_participant(head: Token) -> bool:
    """Mentions that do not put a character in the picture: 'a letter from a kind old fisherman',
    'my grandmother's cottage'."""
    if head.dep_ == "poss":
        return False
    return not (head.dep_ == "pobj" and head.head.lower_ in NON_PARTICIPANT_PREPS and head.head.head.pos_ == "NOUN")


def _merge(
    src: Cluster,
    dst: Cluster,
    token_cluster: dict[int, str],
    pronoun_links: dict[int, list[str]],
    unit_chars: list[list[str]],
) -> None:
    """Fold cluster ``src`` into ``dst`` and rewrite every reference to it."""
    dst.mentions += src.mentions
    dst.units |= src.units
    dst.traits += [t for t in src.traits if t not in dst.traits]
    dst.colors += [t for t in src.colors if t not in dst.colors]
    dst.with_phrases += [t for t in src.with_phrases if t not in dst.with_phrases]
    for k, v in token_cluster.items():
        if v == src.id:
            token_cluster[k] = dst.id
    for k, ids in pronoun_links.items():
        pronoun_links[k] = list(dict.fromkeys(dst.id if i == src.id else i for i in ids))
    for i, ids in enumerate(unit_chars):
        unit_chars[i] = list(dict.fromkeys(dst.id if c == src.id else c for c in ids))
    src.mentions = []


def _copula_subject(head: Token) -> Token | None:
    """For 'Lily was a little girl', return the token 'Lily' when called on 'girl'."""
    if head.dep_ == "attr" and head.head.lemma_ == "be":
        for child in head.head.children:
            if child.dep_ in ("nsubj", "nsubjpass") and child.pos_ != "PRON":
                return child
    return None


SLOT_WORDS = {
    "Hair": {"hair", "braid", "braids", "pigtails", "ponytail", "curls", "bun", "beard", "mustache"},
    "Eyes": {"eyes", "eye"},
    "Outfit": {
        "dress",
        "shirt",
        "coat",
        "jacket",
        "sweater",
        "overalls",
        "gown",
        "skirt",
        "apron",
        "cloak",
        "cape",
        "robe",
        "suit",
        "uniform",
        "trousers",
        "jeans",
        "shorts",
        "boots",
        "shoes",
        "raincoat",
    },
    "Accessories": {
        "hat",
        "glasses",
        "scarf",
        "crown",
        "bow",
        "ribbon",
        "backpack",
        "bag",
        "necklace",
        "collar",
        "bell",
        "umbrella",
        "wand",
        "basket",
    },
    "Tail": {"tail"},
    "Ears": {"ears"},
    "Wings": {"wings"},
    "Horns": {"horns", "horn"},
    "Mane": {"mane"},
    "Markings": {"spots", "stripes", "patches", "spot", "stripe"},
}


def slot_for_phrase(phrase: str) -> str | None:
    words = set(re.findall(r"[a-z]+", phrase.lower()))
    for slot, keys in SLOT_WORDS.items():
        if words & keys:
            return slot
    return None


def _quotes(text: str) -> list[tuple[int, int]]:
    out, start = [], None
    for i, ch in enumerate(text):
        if ch == '"':
            if start is None:
                start = i
            else:
                out.append((start, i))
                start = None
    return out


OBJECT_DEPS = {"dobj", "dative", "pobj", "iobj"}


def _clause_subject_ids(tok: Token, token_cluster: dict[int, str], pronoun_links: dict[int, list[str]]) -> set[str]:
    """Binding constraint: in 'the dragon thanked him', 'him' cannot be the dragon (non-reflexive object)."""
    if tok.dep_ not in OBJECT_DEPS or tok.lower_.endswith(("self", "selves")):
        return set()
    verb = tok.head.head if tok.dep_ == "pobj" else tok.head
    ids: set[str] = set()
    for child in verb.children:
        if child.dep_ in ("nsubj", "nsubjpass"):
            for t in child.subtree:
                if t.i in token_cluster:
                    ids.add(token_cluster[t.i])
                ids.update(pronoun_links.get(t.i, []))
    return ids


def _recent_group(story: ProcessedStory, unit_idx: int, unit_chars: list[list[str]], ranked: list[str]) -> list[str]:
    """Characters active in the current or previous unit — the usual referent of 'they'."""
    group: list[str] = []
    for u in (unit_idx, unit_idx - 1, unit_idx - 2):
        if u < 0:
            continue
        for cid in unit_chars[u]:
            if cid not in group:
                group.append(cid)
        if len(group) >= 2:
            break
    return group


def _unit_subject(unit: Unit) -> Span | None:
    for tok in unit.tokens:
        if tok.dep_ in ("nsubj", "nsubjpass") and tok.head in unit.tokens:
            return tok.doc[tok.left_edge.i : tok.right_edge.i + 1]
    return None


def _ids_for(span: Span, token_cluster: dict[int, str], pronoun_links: dict[int, list[str]]) -> list[str]:
    ids: list[str] = []
    for tok in span:
        for cid in ([token_cluster[tok.i]] if tok.i in token_cluster else []) + pronoun_links.get(tok.i, []):
            if cid not in ids:
                ids.append(cid)
    return ids


def _category(c: Cluster) -> CharacterCategory:
    if c.species in kb.HUMANS:
        return "human"
    prior = kb.SPECIES.get(c.species or "")
    if prior:
        return prior.category
    if c.name and not c.species:
        return "human"  # a bare name ("Lily") is assumed to be a person
    return "creature"


# --------------------------------------------------------------------------------------------- profiles


def build_profile(cluster: Cluster, story_seed: str, scene_indices: list[int]) -> CharacterProfile:
    category = "group" if cluster.plural else _category(cluster)
    species = cluster.species or ("person" if category == "human" else cluster.key)
    traits = [t for t in cluster.traits if t not in kb.SIZE_WORDS and t not in kb.AGE_WORDS]
    sizes = [kb.SIZE_WORDS[t] for t in cluster.traits if t in kb.SIZE_WORDS]
    ages = [kb.AGE_WORDS[t] for t in cluster.traits if t in kb.AGE_WORDS]
    attrs: dict[str, str] = {}
    noun = species

    if category == "human" or species in kb.HUMANS:
        default_role = {"female": "girl", "male": "boy"}.get(cluster.gender, "child")
        role = kb.HUMANS.get(species) or kb.HUMANS[default_role]
        gender = cluster.gender if cluster.gender != "unknown" else role.gender
        stage = "elder" if ("old" in ages or "elderly" in ages) else role.stage
        seed_key = (story_seed, cluster.id, cluster.display_name)
        attrs["Age"] = (
            "elderly " if stage == "elder" and "elder" not in role.age and "old" not in role.age else ""
        ) + role.age
        hair_pool = kb.ELDER_HAIR[gender] if stage == "elder" else kb.HAIR_POOLS[gender]
        attrs["Hair"] = role.hair or kb.seeded_choice(hair_pool, *seed_key, "hair")
        attrs["Skin"] = kb.seeded_choice(kb.SKIN_POOL, *seed_key, "skin")
        outfit = role.outfit or kb.seeded_choice(kb.OUTFIT_POOLS[(stage, gender)], *seed_key, "outfit")
        if cluster.colors and not role.outfit:
            outfit = f"{cluster.colors[0]} {outfit.split(' ', 1)[-1]}"
        attrs["Outfit"] = outfit
        attrs.update(role.extras)
        noun = species if species in kb.HUMANS else default_role
    else:
        prior = kb.SPECIES.get(species)
        if prior:
            attrs.update(prior.attributes)
            noun = prior.noun or species
            if cluster.colors and prior.color_slot and prior.color_slot in attrs:
                base = attrs[prior.color_slot]
                texture = base.split()[-1]  # "feathers", "fur", "coat" …
                attrs[prior.color_slot] = f"{' and '.join(cluster.colors[:2])} {texture}"
        else:
            color = " and ".join(cluster.colors[:2]) or kb.seeded_choice(
                ["soft blue", "warm orange", "leafy green", "lavender purple", "sunny yellow"], story_seed, cluster.id
            )
            attrs["Color"] = f"{color} coloring"
    for phrase in cluster.with_phrases[:3]:
        text = re.sub(r"^with\s+(a |an |the )?", "", phrase).strip()
        slot = slot_for_phrase(text)
        if slot is not None:
            attrs[slot] = text
        else:
            attrs[f"Detail {len(attrs) + 1}"] = text
    if sizes:
        attrs["Size"] = sizes[0]
    expressions = [kb.TRAIT_EXPRESSIONS[t] for t in traits if t in kb.TRAIT_EXPRESSIONS]
    descriptor = compose_descriptor(noun, attrs, sizes, cluster.name, category)
    return CharacterProfile(
        id=cluster.id,
        name=cluster.display_name,
        species=species,
        category=category,
        gender=cluster.gender,
        attributes=attrs,
        explicit_traits=sorted(set(cluster.traits + cluster.colors)),
        personality=expressions,
        descriptor=descriptor,
        mention_count=len(cluster.mentions),
        mentions=cluster.mentions,
        scene_indices=scene_indices,
        is_recurring=len(scene_indices) >= 2,
    )


def compose_descriptor(
    noun: str, attrs: dict[str, str], sizes: list[str], name: str | None, category: str, max_details: int = 3
) -> str:
    """Compact canonical phrase, e.g. 'a small plump hen with warm orange-brown feathers, a red comb …'."""
    size = (sizes[0] + " ") if sizes else ""
    if category == "human" or "Age" in attrs:
        details = [attrs[k] for k in ("Hair", "Skin", "Outfit") if k in attrs]
        accessories = attrs.get("Accessories")
        core = attrs.get("Age", noun)
        text = f"{core} with {details[0]}" if details else core
        if len(details) > 1:
            text += f", {details[1]}"
        if len(details) > 2:
            text += f", wearing {details[2]}"
        if accessories:
            text += f" and {accessories}"
    else:
        details = [v for k, v in attrs.items() if k not in ("Size", "Build")][:max_details]
        build = attrs.get("Build")
        lead = f"{size}{noun}" if not build else f"{noun} with a {build}"
        text = lead if not details else f"{lead} with " + _join([_article(d) for d in details])
        if build and details:
            text = f"{size}{noun} with " + _join([_article(d) for d in details] + [f"a {build}"])
    text = _article(text)
    return f"{name}, {text}" if name else text


UNCOUNTABLE = {
    "fur",
    "wool",
    "hair",
    "skin",
    "down",
    "plumage",
    "metal",
    "paint",
    "wood",
    "light",
    "glow",
    "coloring",
    "clothing",
    "armor",
    "armour",
    "plush",
    "lineart",
}


def _article(phrase: str) -> str:
    """Add a/an unless the head noun is plural or uncountable ('warm orange feathers', 'soft fur')."""
    words = phrase.split()
    if not words:
        return phrase
    first = words[0].lower()
    head_words = phrase.split(" with ")[0].split(",")[0].split()
    head = head_words[-1].lower() if head_words else first
    if first in ("a", "an", "the", "two", "three", "one", "some"):
        return phrase
    if head in UNCOUNTABLE or (head.endswith("s") and not head.endswith(("ss", "us"))):
        return phrase
    return f"{'an' if first[0] in 'aeiou' else 'a'} {phrase}"


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def group_mentions_by_unit(resolution: CharacterResolution) -> dict[int, list[CharacterMention]]:
    out: dict[int, list[CharacterMention]] = defaultdict(list)
    for c in resolution.clusters:
        for m in c.mentions:
            out[m.unit_index].append(m)
    return out


class TaggerMentionDetector:
    """Model B inference: BiLSTM-CRF over MiniLM word states, sentence by sentence.

    Known species / roles that the tagger misses are added back from the curated lexicon (a precision-safe
    backstop); evaluation in docs/MODELS.md reports the tagger alone and the combination separately.
    """

    def __init__(self, checkpoint, encoder, lexicon_backstop: bool = True):
        import torch  # local import keeps the rules-only path light

        from app.models.character_extractor import POS2ID, MentionTagger, bio_to_spans, case_id

        self._torch = torch
        self._case_id, self._pos2id, self._bio_to_spans = case_id, POS2ID, bio_to_spans
        self.model, _ = MentionTagger.load(checkpoint)
        self.encoder = encoder
        self.backstop = RuleMentionDetector() if lexicon_backstop else None
        self.name = f"bilstm-crf ({checkpoint.name})" + (" + lexicon" if lexicon_backstop else "")

    def detect(self, story: ProcessedStory) -> list[DetectedSpan]:
        torch = self._torch
        sents = [s for s in story.doc.sents if s.text.strip()]
        words = [[t.text for t in s] for s in sents]
        states = self.encoder.word_states(words)
        out: list[DetectedSpan] = []
        for sent, vecs in zip(sents, states, strict=False):
            n = len(sent)
            case = torch.tensor([[self._case_id(t.text) for t in sent]])
            pos = torch.tensor([[self._pos2id.get(t.pos_, 0) for t in sent]])
            mask = torch.ones(1, n, dtype=torch.bool)
            path = self.model.decode(vecs[None], case, pos, mask)[0]
            for span in self._bio_to_spans(path):
                for toks in _split_coordination(sent[span.start : span.end]):
                    if span.label == "CHAR" and all(t.pos_ in ("PRON", "DET") for t in toks):
                        continue  # pronouns are handled by the resolver
                    out.append(DetectedSpan(toks.start_char, toks.end_char, span.label, "tagger"))  # type: ignore[arg-type]
        if self.backstop is not None:
            covered = [(d.start, d.end) for d in out]
            for d in self.backstop.detect(story):
                overlaps = any(not (d.end <= a or d.start >= b) for a, b in covered)
                if not overlaps:
                    span = story.doc.char_span(d.start, d.end, alignment_mode="expand")
                    if span is not None and (_species_key(list(span)) is not None or d.label == "PLACE"):
                        out.append(d)
        return sorted(out, key=lambda s: s.start)


def _split_coordination(span: Span) -> list[Span]:
    """'Arin and Storm' → ['Arin', 'Storm']; spans without coordination are returned unchanged."""
    parts, start = [], span.start
    for tok in span:
        if tok.dep_ == "cc" or (tok.text == "," and tok.i + 1 < span.end):
            if tok.i > start:
                parts.append(span.doc[start : tok.i])
            start = tok.i + 1
    if start < span.end:
        parts.append(span.doc[start : span.end])
    parts = [p for p in parts if any(t.pos_ in ("NOUN", "PROPN") for t in p)]
    return parts if len(parts) > 1 else [span]
