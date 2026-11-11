"""Stage 6 — turn a group of event units into a structured, depictable Scene.

Everything here is derived from the dependency parse plus small curated lexicons, so each field of the
output can be traced back to words in the story.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from lemminflect import getInflection
from spacy.tokens import Span, Token

from app.schemas.story import CharacterProfile, Scene, SceneAction, SceneLocation
from app.services import appearance as kb
from app.services.character_manager import NARRATOR_NAME, PRONOUNS, CharacterResolution
from app.services.text_units import ProcessedStory, Unit

EMOTION_LEXICON: dict[str, set[str]] = {
    "kindness": {
        "help",
        "hug",
        "kind",
        "share",
        "care",
        "thank",
        "together",
        "rescue",
        "save",
        "comfort",
        "gift",
        "give",
        "friend",
    },
    "joy": {
        "happy",
        "glad",
        "smile",
        "laugh",
        "play",
        "fun",
        "joy",
        "cheer",
        "giggle",
        "dance",
        "celebrate",
        "delighted",
        "excited",
        "enjoy",
        "love",
        "sing",
        "hooray",
        "wonderful",
        "yay",
        "grin",
        "merry",
    },
    "curiosity": {
        "explore",
        "wonder",
        "curious",
        "discover",
        "search",
        "adventure",
        "journey",
        "investigate",
        "peek",
        "map",
    },
    "surprise": {"surprise", "suddenly", "amazed", "shock", "gasp", "astonish", "wow", "unexpected"},
    "fear": {
        "scared",
        "afraid",
        "fear",
        "frighten",
        "terrify",
        "storm",
        "shiver",
        "tremble",
        "danger",
        "monster",
        "scream",
        "nervous",
        "worry",
        "worried",
        "scary",
    },
    "sadness": {"sad", "cry", "tear", "lonely", "miss", "sob", "unhappy", "upset", "gloomy", "weep"},
    "anger": {"angry", "mad", "shout", "yell", "grumpy", "furious", "stomp", "argue", "fight"},
    "calm": {"quiet", "peaceful", "sleep", "rest", "gentle", "calm", "relax", "dream", "cozy", "snuggle", "nap"},
}
MOOD = {
    "joy": "cheerful joyful atmosphere",
    "sadness": "soft melancholy mood",
    "fear": "tense but child-friendly mood",
    "anger": "dramatic tense mood",
    "surprise": "dynamic surprised moment",
    "calm": "peaceful serene atmosphere",
    "curiosity": "sense of wonder and discovery",
    "kindness": "warm heartwarming atmosphere",
    "neutral": "pleasant atmosphere",
}
EXPRESSION = {
    "joy": "smiling happily",
    "sadness": "looking sad",
    "fear": "looking worried",
    "anger": "frowning",
    "surprise": "looking surprised",
    "calm": "looking calm",
    "curiosity": "looking curious",
    "kindness": "smiling warmly",
    "neutral": "",
}
MOVEMENT_VERBS = {
    "explore",
    "run",
    "walk",
    "travel",
    "fly",
    "swim",
    "journey",
    "wander",
    "race",
    "climb",
    "march",
    "ride",
    "sail",
    "chase",
    "hike",
    "roam",
    "search",
    "go",
    "jump",
    "hop",
    "follow",
}
ENCOUNTER_VERBS = {"meet", "see", "find", "spot", "notice", "discover", "encounter"}
STATIVE_VERBS = {"be", "have", "feel", "seem", "know", "think", "want", "like", "love", "need", "become"}
SPEECH_VERBS = {"say", "ask", "bark", "shout", "whisper", "reply", "tell", "answer", "exclaim", "call", "cry"}
ON_PLACES = {
    "farm",
    "farmyard",
    "island",
    "beach",
    "hill",
    "mountain",
    "ship",
    "moon",
    "sea",
    "ocean",
    "street",
    "road",
    "path",
    "bridge",
    "playground",
    "field",
    "meadow",
    "lake",
    "river",
    "pond",
}
# abstract predicates → something a picture can show
ABSTRACT_VISUALS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"\b(became|become|becoming|being|were|are) (good |best |great )?friends\b"),
        "smiling together side by side as new best friends",
    ),
    (re.compile(r"\b(loved|love|loving|liked|liking) each other\b"), "sharing a warm hug"),
    (re.compile(r"\blived happily\b"), "smiling happily together"),
    (
        re.compile(r"\b(learned|learnt|learning|realized|realizing|understood|understanding)\b"),
        "having a thoughtful moment together",
    ),
    (re.compile(r"\b(had|having) (a lot of |so much |great )?fun\b"), "having fun and laughing together"),
]
TEMPORAL_RESETS = {"day", "morning", "next", "later", "afternoon"}


@dataclass
class SceneDraft:
    units: list[Unit]
    boundary_confidence: float | None


# --------------------------------------------------------------------------------------------- rendering


def to_gerund(verb: Token) -> str:
    lemma = verb.lemma_.lower()
    if lemma == "be":
        return "being"
    forms = getInflection(lemma, tag="VBG")
    return forms[0] if forms else lemma + "ing"


def _names(cids: list[str], names: dict[str, str]) -> str:
    labels = [names.get(cid, "someone") for cid in cids]
    if len(labels) <= 1:
        return "".join(labels)
    return ", ".join(labels[:-1]) + " and " + labels[-1]


def _render(
    tokens: list[Token], resolution: CharacterResolution, names: dict[str, str], gerunds: set[int] | None = None
) -> str:
    """Tokens → text; pronouns become names unless the referent was just named; conjunct verbs → gerunds."""
    gerunds = gerunds or set()
    out: list[str] = []
    seen: set[str] = set()
    for tok in tokens:
        if tok.i in resolution.mention_tokens:
            seen.add(resolution.mention_tokens[tok.i])
        cids = resolution.pronoun_links.get(tok.i)
        if tok.i in gerunds:
            out.append(to_gerund(tok) + tok.whitespace_)
        elif cids and tok.lower_ in PRONOUNS and not set(cids) <= seen:
            label = _names(cids, names)
            if tok.lower_ in ("his", "her", "its", "their") and tok.dep_ == "poss":
                label = label + "'s" if not label.endswith("s") else label + "'"
            out.append(label + tok.whitespace_)
            seen.update(cids)
        else:
            out.append(tok.text_with_ws)
    text = "".join(out).strip()
    return re.sub(r"\s+([,.!?;:])", r"\1", text)


def _main_verb(unit: Unit) -> Token | None:
    tokens = unit.tokens
    candidates = [t for t in tokens if t.pos_ in ("VERB", "AUX") and (t.head not in tokens or t.head == t)]
    if not candidates:
        candidates = [t for t in tokens if t.pos_ == "VERB"]
    return candidates[0] if candidates else None


def _chunk(tok: Token) -> list[Token]:
    """Pre-modifiers + head of a noun phrase (drops relative clauses and trailing PPs)."""
    return [t for t in tok.subtree if tok.left_edge.i <= t.i <= tok.i]


def _is_passive(verb: Token) -> bool:
    return any(c.dep_ in ("auxpass", "nsubjpass") for c in verb.children)


def _verb_form(verb: Token) -> str:
    return verb.text.lower() if _is_passive(verb) else to_gerund(verb)


def _short_phrase(verb: Token, resolution: CharacterResolution, names: dict[str, str]) -> str:
    """Verb + its first object or prepositional phrase: 'reading the letter', 'walking to the beach'."""
    parts: list[Token] = []
    for child in verb.children:
        if child.i < verb.i:
            continue
        if child.dep_ == "prt":
            parts.append(child)
        elif child.dep_ in ("dobj", "attr", "oprd", "acomp", "dative"):
            parts.extend(_chunk(child))
            break
        elif child.dep_ == "prep":
            pobj = next((c for c in child.children if c.dep_ == "pobj"), None)
            parts.append(child)
            if pobj is not None:
                parts.extend(_chunk(pobj))
            break
        elif child.dep_ in ("xcomp", "ccomp"):
            obj = next((c for c in child.children if c.dep_ in ("dobj", "attr")), None)
            end = obj.i if obj is not None else child.i
            parts.extend(t for t in child.subtree if (t.i <= end and t.dep_ != "amod") or t.i == end)
            break
    head = "" if verb.lemma_ == "be" else _verb_form(verb)
    return f"{head} {_render(parts, resolution, names)}".strip()


def _object_phrase(tokens: list[Token]) -> str | None:
    """Most recent non-character object noun phrase ('a wooden chest') — the usual referent of 'it'."""
    for tok in reversed(tokens):
        if tok.dep_ in ("dobj", "pobj") and tok.pos_ == "NOUN" and tok.lemma_.lower() not in kb.PLACES:
            words = [t.text for t in _chunk(tok)]
            if words and words[0].lower() in ("a", "an"):
                words[0] = "the"
            return " ".join(words)
    return None


def action_for(
    unit: Unit, resolution: CharacterResolution, names: dict[str, str], prior_tokens: list[Token]
) -> SceneAction | None:
    verb = _main_verb(unit)
    if verb is None:
        return None
    actors = resolution.subject_links.get(unit.index, [])
    span_tokens = [t for t in unit.tokens if not t.is_space]
    targets = [cid for cid in resolution.unit_characters[unit.index] if cid not in actors]
    is_existential = any(t.dep_ == "expl" for t in span_tokens) and verb.lemma_ == "be"
    attr = next((c for c in verb.children if c.dep_ in ("attr", "nsubj" if is_existential else "attr")), None)
    # introductions: "there was a hen who lived on a farm" / "Lily was a girl who lived in a village"
    if verb.lemma_ == "be" and attr is not None and (is_existential or actors):
        rel = next((c for c in attr.children if c.dep_ == "relcl"), None)
        phrase = ""
        if rel is not None:
            rest = [t for t in rel.subtree if t.i > rel.i]
            phrase = f"{to_gerund(rel)} {_render(rest, resolution, names)}".strip()
        ids = actors or resolution.unit_characters[unit.index][:1]
        return SceneAction(
            actor_ids=ids, verb="be", phrase=phrase or "standing happily", short_phrase="", target_ids=[]
        )
    subj = next((c for c in verb.children if c.dep_ in ("nsubj", "nsubjpass", "expl")), None)
    excluded = {t.i for t in subj.subtree} if subj is not None else set()
    subtree = [t for t in verb.subtree if t in span_tokens] if verb in span_tokens else span_tokens
    tail = [t for t in subtree if t.i not in excluded and t.i >= verb.i]
    while tail and tail[-1].is_punct:
        tail.pop()
    rest = tail[1:] if tail and tail[0].i == verb.i else tail
    gerunds = {c.i for c in verb.children if c.dep_ == "conj" and c.pos_ == "VERB"}
    body = _render(rest, resolution, names, gerunds)
    phrase = body if verb.lemma_ == "be" else f"{_verb_form(verb)} {body}".strip()
    subject_text = None
    if not actors and subj is not None:
        if subj.lower_ == "it":
            subject_text = _object_phrase(prior_tokens)
        elif subj.pos_ != "PRON":
            subject_text = " ".join(t.text for t in _chunk(subj))
    return SceneAction(
        actor_ids=actors,
        verb=verb.lemma_.lower(),
        phrase=phrase.strip(" ,.;"),
        short_phrase=_short_phrase(verb, resolution, names),
        target_ids=targets,
        subject_text=subject_text,
    )


def summarize_unit(unit: Unit, resolution: CharacterResolution, names: dict[str, str]) -> str:
    text = _render(unit.tokens, resolution, names)
    if unit.is_continuation:
        text = re.sub(r"^(and|but|so|then|,)\s+", "", text, flags=re.I)
        subj = _names(resolution.subject_links.get(unit.index, []), names)
        if unit.inherited_subject is not None and subj and not text.lower().startswith(subj.lower()):
            text = f"{subj} {text}"
    text = text.strip()
    if text and text[-1] not in ".!?\"'":
        text += "."
    return text[:1].upper() + text[1:]


# --------------------------------------------------------------------------------------------- scene fields


def _emotion(tokens: list[Token], mention_tokens: dict[int, str] | None = None) -> str:
    scores: dict[str, int] = {k: 0 for k in EMOTION_LEXICON}
    for tok in tokens:
        if tok.pos_ == "PROPN" or (mention_tokens and tok.i in mention_tokens):
            continue
        lemma, low = tok.lemma_.lower(), tok.lower_
        for emo, words in EMOTION_LEXICON.items():
            if lemma in words or low in words:
                scores[emo] += 1
    best = max(scores, key=lambda k: scores[k])  # dict order breaks ties
    return best if scores[best] > 0 else "neutral"


def _time_and_weather(tokens: list[Token]) -> tuple[str | None, str | None, bool]:
    time = weather = None
    reset = False
    for tok in tokens:
        low = tok.lower_
        if low in kb.TIME_OF_DAY and time is None:
            time = low
        if low in kb.WEATHER and weather is None:
            weather = low
        if low in TEMPORAL_RESETS:
            reset = True
    return time, weather, reset


def _composition(index: int, n_chars: int, verbs: list[str], emotion: str, intro: bool, prev: str | None) -> str:
    if intro or index == 1:
        shot = "wide establishing shot"
    elif any(v in MOVEMENT_VERBS for v in verbs):
        shot = "wide shot showing the surroundings"
    elif n_chars >= 3:
        shot = "wide shot with the whole group"
    elif emotion in ("sadness", "fear", "surprise", "joy") and n_chars == 1:
        shot = "close-up"
    elif n_chars == 2:
        shot = "medium shot of the two characters together"
    else:
        shot = "medium shot"
    angle = "eye-level view"
    if prev and prev.split(",")[0] == shot:
        angle = "slightly low angle view" if "low" not in prev else "slightly high angle view"
    return f"{shot}, {angle}"


def _salience(a: SceneAction) -> int:
    score = 2 * bool(a.actor_ids) + bool(a.target_ids) + (a.verb not in STATIVE_VERBS)
    score += a.verb in MOVEMENT_VERBS or a.verb in ENCOUNTER_VERBS
    score -= 2 * (a.verb in SPEECH_VERBS)
    return score


def _title(action: SceneAction | None, names: dict[str, str], location: SceneLocation | None) -> str:
    if action is None:
        return "A New Moment"
    actor = _names(action.actor_ids, names) if action.actor_ids else ""
    if action.verb == "be":
        who = actor or "Our Story Begins"
        if location and location.name not in ("setting", "home"):
            prep = "on" if location.name in ON_PLACES else "in"
            title = f"{who} {prep} the {location.name}"
        else:
            title = who
    elif action.verb == "meet" and action.target_ids:
        title = f"{actor} Meets {_names(action.target_ids[:2], names)}"
    elif action.verb in SPEECH_VERBS and actor:
        title = f"{actor} Tells a Story" if action.verb == "tell" else f"{actor} Speaks Up"
    else:
        title = action.short_phrase or action.phrase
        words = title.split()
        title = " ".join(words[:8])
    small = {"a", "an", "the", "and", "of", "on", "in", "to", "with", "at", "for", "by"}
    words = re.sub(r"\s+", " ", title).strip(" ,.").split()
    return " ".join(w if (i and w.lower() in small) else w[:1].upper() + w[1:] for i, w in enumerate(words)) or "Scene"


def _place_description(story: ProcessedStory, resolution: CharacterResolution, unit: Unit, key: str) -> str:
    base = kb.PLACES.get(key, f"a {key}")
    span: Span | None = None
    for det in resolution.place_spans:
        if unit.start_char <= det.start < unit.end_char:
            cand = story.doc.char_span(det.start, det.end, alignment_mode="expand")
            if cand is not None and cand.root.lemma_.lower() == key:
                span = cand
                break
    if span is None:
        return base
    head = span.root
    adjs = [t.lower_ for t in span if t.dep_ in ("amod", "compound") and t.head == head and t.pos_ in ("ADJ", "NOUN")]
    pps = []
    for child in head.children:
        if (
            child.dep_ == "prep"
            and child.lower_ in ("by", "near", "beside", "on", "under")
            and len(list(child.subtree)) <= 4
        ):
            pps.append(" ".join(t.text for t in child.subtree))
    if not adjs and not pps:
        return base
    details = base.split(" with ", 1)[1] if " with " in base else ""
    core = " ".join(["a", *adjs, head.lower_, *pps])
    return f"{core} with {details}" if details else core


DESTINATION_PREPS = {"to", "into", "in", "inside", "at", "on", "onto", "toward", "towards"}
PATH_PREPS = {"through", "across", "along", "past", "by", "near", "from", "under", "behind", "beside", "over", "around"}


def _unit_place_spans(story: ProcessedStory, resolution: CharacterResolution, unit: Unit) -> list[Span]:
    spans = []
    for det in resolution.place_spans:
        if unit.start_char <= det.start < unit.end_char:
            span = story.doc.char_span(det.start, det.end, alignment_mode="expand")
            if span is not None:
                spans.append(span)
    return spans


def _place_score(span: Span) -> int:
    root = span.root
    if root.lemma_.lower() in kb.GENERIC_PLACES:
        return 0
    prep = root.head.lower_ if root.dep_ == "pobj" else None
    if prep in DESTINATION_PREPS:
        return 3
    if prep in PATH_PREPS:
        return 1
    return 2


def _home_owner(
    tokens: list[Token], resolution: CharacterResolution, profiles: dict[str, CharacterProfile]
) -> str | None:
    for tok in tokens:
        if tok.lemma_.lower() == "home":
            for child in tok.children:
                if child.dep_ == "poss":
                    cids = resolution.pronoun_links.get(child.i) or (
                        [resolution.mention_tokens[child.i]] if child.i in resolution.mention_tokens else []
                    )
                    for cid in cids:
                        prior = kb.SPECIES.get(profiles[cid].species) if cid in profiles else None
                        if prior and prior.home:
                            return cid
    return None


def depictable(phrase: str) -> str:
    """Replace abstract predicates ('becoming good friends') with something a picture can show."""
    for pattern, visual in ABSTRACT_VISUALS:
        if pattern.search(phrase.lower()):
            return visual
    return phrase


def _visual_description(actions: list[SceneAction], names: dict[str, str], char_ids: list[str], emotion: str) -> str:
    ranked = sorted(range(len(actions)), key=lambda i: _salience(actions[i]), reverse=True)[:2]
    parts: list[str] = []
    for i in sorted(ranked):
        a = actions[i]
        phrase = depictable(a.phrase)
        subject = _names(a.actor_ids, names) if a.actor_ids else (a.subject_text or "")
        parts.append(f"{subject} {phrase}".strip())
    if not parts and char_ids:
        parts.append(_names(char_ids, names))
    text = "; ".join(dict.fromkeys(p for p in parts if p))
    expr = EXPRESSION.get(emotion, "")
    if expr and expr not in text and char_ids:
        text = f"{text}, {expr}" if text else expr
    return text


def build_scenes(
    story: ProcessedStory,
    drafts: list[SceneDraft],
    resolution: CharacterResolution,
    profiles: dict[str, CharacterProfile],
) -> list[Scene]:
    names = {
        cid: (
            "the child narrator"
            if p.name == NARRATOR_NAME
            else p.name
            if p.name.lower() != p.species
            else f"the {p.species}"
        )
        for cid, p in profiles.items()
    }
    title_names = {cid: (f"the {p.name}" if p.name.lower() == p.species else p.name) for cid, p in profiles.items()}
    scenes: list[Scene] = []
    location: SceneLocation | None = None
    time: str | None = None
    prev_comp: str | None = None
    for i, draft in enumerate(drafts, start=1):
        tokens = [t for u in draft.units for t in u.tokens]
        char_ids: list[str] = []
        for u in draft.units:
            for cid in resolution.unit_characters[u.index]:
                if cid not in char_ids and cid in profiles:
                    char_ids.append(cid)
        actions: list[SceneAction] = []
        for u in draft.units:
            prior = [t for t in story.doc[: u.span.start]][-60:]
            a = action_for(u, resolution, names, prior)
            if a is not None:
                actions.append(a)

        # location: explicit mention in this scene > the owner's home ("find its home") > carried over
        new_location: SceneLocation | None = None
        for u in reversed(draft.units):
            candidates = _unit_place_spans(story, resolution, u)
            if candidates:
                span = max(candidates, key=lambda sp: (_place_score(sp), -sp.start))
                key = span.root.lemma_.lower()
                new_location = SceneLocation(
                    name=key, description=_place_description(story, resolution, u, key), carried_over=False
                )
                break
        if any(t.lemma_.lower() == "home" for t in tokens):
            owner = _home_owner(tokens, resolution, profiles)
            if owner:
                home = kb.SPECIES[profiles[owner].species].home or "a cozy home"
                new_location = SceneLocation(name="home", description=home, carried_over=False)
        if new_location is None and location is not None:
            new_location = location.model_copy(update={"carried_over": True})
        if new_location is None:
            first = next((profiles[c] for c in char_ids if kb.SPECIES.get(profiles[c].species)), None)
            home = kb.SPECIES[first.species].home if first else None
            new_location = SceneLocation(
                name="setting", description=home or "a pleasant storybook setting", carried_over=True
            )
        location = new_location

        t, weather, reset = _time_and_weather(tokens)
        if t is not None:
            time = t
        elif reset:
            time = None
        emotion = _emotion(tokens, resolution.mention_tokens)
        verbs = [a.verb for a in actions]
        intro = i == 1 and bool(actions) and actions[0].verb == "be"
        composition = _composition(i, len(char_ids), verbs, emotion, intro, prev_comp)
        prev_comp = composition
        if intro:
            main = actions[0]
        else:
            ranked = sorted(actions, key=_salience, reverse=True)
            main = ranked[0] if ranked else None
        scenes.append(
            Scene(
                index=i,
                title=_title(main, title_names, location),
                text=story.text[draft.units[0].start_char : draft.units[-1].end_char].strip(),
                summary=" ".join(summarize_unit(u, resolution, names) for u in draft.units),
                unit_indices=[u.index for u in draft.units],
                character_ids=char_ids,
                actions=actions,
                location=location,
                time_of_day=time,
                weather=weather,
                emotion=emotion,
                mood=MOOD[emotion],
                composition=composition,
                visual_description=_visual_description(actions, names, char_ids, emotion),
                previous_context=(f"Previous scene: {scenes[-1].title}. {scenes[-1].summary}" if scenes else None),
                boundary_confidence=draft.boundary_confidence,
            )
        )
    return scenes
