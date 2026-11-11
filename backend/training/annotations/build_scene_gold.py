"""Builds scene_gold.jsonl — the hand-annotated scene-boundary evaluation set (run once; output is committed).

Annotation guideline (single annotator, AI-assisted, see README.md in this folder):
a new scene starts where a picture book would need a new illustration — a change of time ("One day",
"That night"), a change of place, a character arriving, or a major new visible event. Dialogue, reactions
and reflections that continue the same moment stay in the current scene.
"""

import json
from pathlib import Path

from app.services.text_units import normalize_text, split_units

HERE = Path(__file__).parent
RAW = HERE.parent / "data" / "raw" / "tinystories" / "TinyStoriesV2-GPT4-valid.txt"

# TinyStories V2 validation stories (held out: index ≥ 6000, never used in training/dev);
# values = indices of the event units (splitter at annotation time) that start a new scene.
TINYSTORIES = {
    11094: [4, 8, 12, 15],
    9645: [4, 9, 12],
    20367: [4, 8, 11],
    23793: [3, 7, 11, 14, 16],
    27120: [2, 4, 9],
    12942: [4, 8, 11],
    17972: [2, 12],
    21450: [2, 6, 11],
    7225: [2, 5, 7, 12, 18],
    7146: [5, 10, 12],
    27627: [2, 3, 10],
    17797: [3, 9],
    26191: [5, 14, 16],
    10611: [3, 10],
    25278: [3, 6, 12],
    11221: [3, 9, 11],
    15426: [4, 9, 11, 14, 19],
    16000: [3, 8, 12, 17],
    13257: [1, 6],
    15043: [5, 8, 11],
    19327: [4, 7, 11],
    19502: [3, 8, 11, 13],
    12052: [1, 4, 8, 13],
    18119: [4, 8],
}

# Original stories written for this project and retellings of public-domain fables; "|" marks a scene start.
WRITTEN = {
    "spec-example": "Once upon a time, there was a hen who lived on a farm. | One day, she met a cow. | They became "
    "good friends | and explored the farm together. | One evening, they helped a lost little rabbit find its home.",
    "robot-snow": "Maya the little robot woke up in a quiet workshop full of tools. She blinked her blue eyes and "
    "rolled to the window. | Outside, the city was covered in snow. Maya had never seen snow before, so she rolled "
    "out the door and into the street. | In the park, a group of children were building a snowman. They waved at "
    "Maya and asked her to help. | Maya used her strong arms to lift the biggest snowball onto the top. | When the "
    "sun went down, the children gave Maya a red scarf to keep her warm. | That night, Maya rolled back to the "
    "workshop, wearing her scarf and humming a happy tune.",
    "fox-river": "A small fox named Ruby lived at the edge of a deep green forest. Every morning she hunted for "
    "berries near the river. | One morning, the river was flooded and Ruby could not cross it. She sat on a rock "
    'and looked at the rushing water. | A wise old turtle swam up to her. "Climb on my shell," said the turtle, '
    '"and I will carry you across." | Ruby held on tightly while the turtle paddled through the waves. | On the '
    "other side, they found a bush full of the sweetest berries Ruby had ever tasted. They shared the berries and "
    "laughed in the warm sun.",
    "fishing": "Leo and his grandpa went fishing on a calm blue lake. They sat in a little wooden boat and waited. "
    "| Suddenly, something pulled hard on Leo's line. Leo pulled and pulled until a huge silver fish jumped out of "
    "the water. | The fish looked at Leo with sad eyes. Leo decided to let it go, and the fish splashed back into "
    "the lake. | When they rowed home at sunset, Grandpa told Leo he was proud of him.",
    "dragon": "There was once a lonely dragon who lived in a cave high on a mountain. The villagers below were "
    "afraid of him, and nobody ever visited. | One snowy winter, a little girl named Nia got lost in the storm and "
    "climbed up to the cave to stay warm. The dragon gently breathed a small fire so she would not be cold. | In "
    "the morning, the dragon flew Nia back to her village on his back. | The villagers saw that the dragon was "
    "kind, and they invited him to their winter festival. | That evening, the dragon lit the festival lanterns, "
    "and everyone danced together under the stars.",
    "muddy-ball": 'Sam the puppy found a muddy ball in the garden. "This is the best ball ever!" he barked. '
    '"It\'s so dirty," said Mia the cat. "You should wash it first." Sam shook his head and kept playing. | Then '
    "it started to rain. The rain washed the ball until it was bright red again. | Sam brought the shiny ball to "
    'Mia. "Look, the rain washed it for me!" Mia laughed, and they played catch until the sun came out.',
    "paper-boat": "Ben built a paper boat. | He sailed it down the stream behind his house. | The boat floated all "
    "the way to the pond, where a frog jumped on board.",
    "baker": "In a busy city lived an old baker named Rosa. Every day she baked bread in her tiny shop. | One "
    "afternoon, a hungry boy stood outside her window, looking at the cakes. Rosa invited him in and gave him a "
    "warm roll. | The boy came back the next day and offered to sweep the floor. | Soon the boy was helping Rosa "
    "bake, and together they made the most beautiful birthday cake for the whole street. | At the street party "
    "that night, everyone cheered for Rosa and her new helper.",
    "moon": "Captain Zoe flew her silver rocket to the moon. She bounced across the dusty craters in her space "
    "suit. | Behind a big rock, she discovered a tiny green alien crying. The alien had lost its spaceship. | Zoe "
    "and the alien searched the moon together and finally found the little spaceship stuck in a crater. | Zoe "
    "pulled it free with a rope, and the alien hugged her before flying home to its planet. | Back on Earth, Zoe "
    "looked up at the stars and smiled, knowing she had a friend far away.",
    "octopus": "Under the sea, a shy octopus named Ollie hid inside a shell all day. | One day, a school of colorful "
    "fish swam by and invited him to play hide and seek. Ollie was nervous but he joined them. | Ollie changed his "
    "colors to match the coral and won every round. | The fish cheered for Ollie, and from then on he was never "
    "shy again.",
    "fable-tortoise-hare": "A proud hare laughed at a slow tortoise and challenged him to a race. | The race began "
    "at the old oak tree. The hare dashed far ahead while the tortoise plodded along the path. | Halfway to the "
    "finish, the hare felt so sure of winning that he lay down under a shady tree and fell asleep. | The tortoise "
    "kept walking, slowly and steadily, past the sleeping hare. | When the hare woke up, he ran as fast as he "
    "could, but the tortoise was already crossing the finish line while the other animals cheered.",
    "fable-lion-mouse": "A mighty lion was sleeping in the forest when a little mouse ran across his nose. The lion "
    'woke up and caught the mouse in his huge paw. "Please let me go," squeaked the mouse, "and one day I will '
    'help you." The lion laughed, but he let the mouse go. | A few days later, hunters caught the lion in a strong '
    "net. The lion roared and roared, but he could not escape. | The little mouse heard him and came running. She "
    "chewed through the ropes until the lion was free. | From that day on, the lion and the mouse were the best of "
    "friends.",
    "fable-crow-pitcher": "On a hot summer day, a thirsty crow flew over the fields looking for water. | At last she "
    "found a tall pitcher in a garden, but the water at the bottom was too low for her beak to reach. | The clever "
    "crow picked up pebbles one by one and dropped them into the pitcher. | Slowly the water rose higher and "
    "higher, until the crow could finally drink.",
    "fable-ant-grasshopper": "All summer long, a grasshopper played his fiddle and sang in the sunny meadow, while a "
    'busy ant carried grains of wheat to her nest. "Why work so hard?" asked the grasshopper. "Come and sing '
    'with me!" But the ant kept working. | When winter came, the meadow was covered in snow and the grasshopper '
    "had nothing to eat. | Cold and hungry, he knocked on the ant's door. The kind ant let him in and shared her "
    "food by the warm fire.",
}


def main() -> None:
    raw = RAW.read_text(encoding="utf-8").split("<|endoftext|>")
    out = []
    for idx, starts in TINYSTORIES.items():
        paragraphs = [normalize_text(p) for p in raw[idx].strip().split("\n") if p.strip()]
        text = " ".join(paragraphs)
        units = split_units(text).units
        out.append(
            {
                "id": f"tinystories-valid-{idx}",
                "source": "TinyStories V2 validation (CDLA-Sharing-1.0)",
                "text": text,
                "scene_starts": [units[i].start_char for i in starts],
            }
        )
    for key, marked in WRITTEN.items():
        pieces = [p.strip() for p in marked.split("|")]
        text, starts = "", []
        for i, piece in enumerate(pieces):
            if i:
                text += " "
                starts.append(len(text))
            text += piece
        source = "public-domain fable, retold for StoryPix" if key.startswith("fable") else "written for StoryPix"
        out.append({"id": key, "source": source, "text": normalize_text(text), "scene_starts": starts})
    path = HERE / "scene_gold.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n")
    print(f"wrote {len(out)} stories, {sum(len(r['scene_starts']) for r in out)} boundaries → {path}")


if __name__ == "__main__":
    main()
