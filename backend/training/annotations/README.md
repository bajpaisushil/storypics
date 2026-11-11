# Gold annotations

Hand-annotated evaluation data used by `training/evaluate.py`. Never used for training.

| File | Task | Size |
|---|---|---|
| `scene_gold.jsonl` | Scene boundaries (Model A) | 38 stories, 124 boundaries |
| `mention_gold.json` | Character / place mentions (Model B) | 6 stories, 75 mentions |
| `build_scene_gold.py` | Rebuilds `scene_gold.jsonl` from the annotation tables (needs the TinyStories download) | — |

## Guidelines

**Scenes.** A new scene starts where a picture book would need a new illustration: a change of time
("One day", "That night"), a change of place, a character arriving, or a major new visible event. Dialogue,
reactions and reflections that continue the same moment stay in the current scene. Boundaries are stored as
character offsets into the (normalised) story text, so they do not depend on the clause splitter.

**Mentions.** Base noun phrases (determiner + pre-modifiers + head) of characters (`CHAR`: people, animals,
creatures that act in the story) and places (`PLACE`: settings and locations). Pronouns and predicate nominals
("they became *good friends*") are excluded.

## Provenance and honesty notes

* Single annotator, AI-assisted (annotated during development with Claude), not adjudicated by a second person.
  Scene segmentation is inherently subjective, so treat absolute numbers as indicative; comparisons between
  systems on the same data are the meaningful part.
* After the first Model A evaluation, one labelling function was refined (dialogue lines that open a new paragraph
  are no longer treated as strong scene boundaries) because that error type showed up on this set. The gold labels
  themselves were not changed.

## Sources and licences

* 24 stories from the **TinyStories V2 validation split** (indices ≥ 6000, held out from all training/dev data).
  TinyStories is licensed **CDLA-Sharing-1.0**; the TinyStories-derived records in `scene_gold.jsonl` are shared
  under the same licence.
* 10 stories written for StoryPix and 4 retellings of public-domain Aesop fables (`fable-*`), written for this project.
