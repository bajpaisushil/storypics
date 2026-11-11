"""Stage-level tests for story understanding and prompt building (no image model needed)."""

from app.services.prompt_builder import PromptBuilder, image_size, rewrite_instruction
from app.services.styles import STYLES, get_style
from app.services.text_units import split_units
from tests.conftest import HEN_STORY

LILY = (
    "Lily was a curious little girl with red hair who lived in a small village by the sea. Every morning she "
    'walked to the beach with her dog, Max.\n\nOne day, Max found an old bottle in the sand. "Look, Lily!" he '
    "barked. Inside the bottle was a map."
)


def test_event_units_split_coordinated_clauses():
    units = split_units(HEN_STORY).units
    assert [u.text for u in units][2:4] == ["They became good friends", "and explored the farm together."]
    assert units[3].is_continuation and units[3].inherited_subject is not None


def test_paragraphs_are_tracked():
    units = split_units(LILY).units
    assert units[0].paragraph_index == 0 and units[2].paragraph_index == 1


def test_hen_story_matches_spec_example(analyzer):
    a = analyzer.analyze(HEN_STORY, 5)
    assert [s.title for s in a.scenes] == [
        "The Hen on the Farm",
        "The Hen Meets the Cow",
        "Becoming Good Friends",
        "Exploring the Farm",
        "Helping a Rabbit Find Its Home",
    ]
    hen, cow, rabbit = (next(c for c in a.characters if c.species == k) for k in ("hen", "cow", "rabbit"))
    assert hen.gender == "female" and "feathers" in hen.descriptor and hen.is_recurring
    assert cow.scene_indices == [2, 3, 4, 5]
    assert rabbit.scene_indices == [5] and "lost" in rabbit.explicit_traits
    last = a.scenes[-1]
    assert set(last.character_ids) == {hen.id, cow.id, rabbit.id}
    assert last.time_of_day == "evening" and "burrow" in last.location.description  # the rabbit's home


def test_auto_mode_short_story_illustrates_every_event(analyzer):
    assert len(analyzer.analyze(HEN_STORY, "auto").scenes) == 5


def test_copular_identity_and_explicit_attributes(analyzer):
    a = analyzer.analyze(LILY, "auto")
    lily = next(c for c in a.characters if c.name == "Lily")
    assert lily.species == "girl" and lily.attributes["Hair"] == "red hair"
    max_ = next(c for c in a.characters if c.name == "Max")
    assert max_.species == "dog" and max_.gender == "male"
    assert not any(c.species == "girl" and c.name != "Lily" for c in a.characters)


def test_requesting_more_scenes_than_events_warns(analyzer):
    a = analyzer.analyze(HEN_STORY, 10)
    assert len(a.scenes) == 5 and a.warnings


def test_style_changes_prompt_and_negative(analyzer):
    a = analyzer.analyze(HEN_STORY, 5)
    scene = a.scenes[1]
    prompts = {sid: PromptBuilder(get_style(sid)).scene_prompt(scene, a.characters, set()) for sid in STYLES}
    assert len({p.prompt for p in prompts.values()}) == len(STYLES)
    assert prompts["realistic"].prompt.startswith("photorealistic photograph")
    assert "cartoon" in prompts["realistic"].negative_prompt
    hen = next(c for c in a.characters if c.species == "hen")
    assert hen.descriptor in prompts["watercolor"].prompt  # canonical descriptor reused verbatim


def test_multi_character_layout_regions(analyzer):
    a = analyzer.analyze(HEN_STORY, 5)
    scene = a.scenes[1]
    built = PromptBuilder(get_style("storybook")).scene_prompt(scene, a.characters, set(scene.character_ids))
    assert len(built.slots) == 2 and built.slots[0].region.x1 < built.slots[1].region.x1
    assert "on the left" in built.prompt and "on the right" in built.prompt


def test_instruction_rewriting():
    assert rewrite_instruction("Make the hen happier.") == "the hen very happy"
    assert rewrite_instruction("Show the cow standing near the tree") == "the cow standing near the tree"


def test_image_sizes_use_sdxl_buckets():
    assert image_size("16:9") == (1344, 768) and image_size("1:1") == (1024, 1024)
    w, h = image_size("4:3", 768)
    assert w % 64 == 0 and h % 64 == 0 and w > h


FIRST_PERSON = (
    "Last summer I visited my grandmother's cottage in the mountains. Every morning we fed the goats. "
    "One rainy afternoon, my little brother Sam and I found a secret door behind the bookshelf. "
    "That night, Grandma told us the toys had belonged to her when she was a girl."
)
FANTASY = (
    "A brave knight named Arin rode his white horse, Storm, through the Whispering Woods. A tiny dragon with "
    "purple scales was stuck in a hunter's net. Arin cut the ropes with his sword, and the dragon thanked him "
    "with a puff of warm smoke. The next morning, the dragon guided Arin and Storm to a hidden castle. The queen "
    "crowned Arin as the protector of all dragons."
)
TWINS = (
    "Mia and Leo were twins who lived next to a big park. Mia used her long paintbrush to pull the ball out of "
    "the pond. Leo hugged his sister and they painted the park bench bright yellow together."
)


def test_first_person_narrator_and_family_roles(analyzer):
    a = analyzer.analyze(FIRST_PERSON, "auto")
    narrator = next(c for c in a.characters if c.name == "Narrator")
    assert narrator.category == "human" and narrator.scene_indices == list(range(1, len(a.scenes) + 1))
    grandma = next(c for c in a.characters if c.species == "grandmother")
    assert grandma.name == "Grandma" and "elderly" in grandma.descriptor
    assert not any(c.species == "girl" for c in a.characters)  # "when she was a girl" is not a new character
    sam = next(c for c in a.characters if c.name == "Sam")
    assert sam.gender == "male" and "pigtails" not in sam.descriptor


def test_names_binding_and_generic_plurals(analyzer):
    a = analyzer.analyze(FANTASY, "auto")
    by_name = {c.name: c for c in a.characters}
    assert {"Arin", "Storm"} <= set(by_name) and by_name["Storm"].species == "horse"
    dragon = next(c for c in a.characters if c.species == "dragon")
    assert dragon.category != "group"  # "all dragons" did not turn the dragon into a group
    assert not any("protector" in c.name.lower() for c in a.characters)
    thank = next(s for s in a.scenes if any(x.verb == "thank" for x in s.actions))
    action = next(x for x in thank.actions if x.verb == "thank")
    assert action.target_ids == [by_name["Arin"].id] or by_name["Arin"].id in thank.character_ids
    assert "Arin" in action.phrase  # "him" is Arin, not the dragon itself
    guide = next(s for s in a.scenes if any(x.verb == "guide" for x in s.actions))
    assert by_name["Storm"].id in guide.character_ids  # every later "Storm" is a mention


def test_sibling_role_links_to_named_character(analyzer):
    a = analyzer.analyze(TWINS, "auto")
    assert sorted(c.name for c in a.characters if c.category == "human") == ["Leo", "Mia"]
