"""Curated appearance priors for common story characters, places and visual vocabulary.

Stories rarely say what a character looks like ("a hen"), but every image needs a concrete,
*repeatable* look. Priors give each species a canonical appearance; explicit details in the
story ("a white hen", "a girl with red hair") override them. Human details that the story leaves
open are picked deterministically from a seeded pool so the same story always yields the same cast.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from app.schemas.story import CharacterCategory, Gender


@dataclass(frozen=True)
class SpeciesPrior:
    category: CharacterCategory
    attributes: dict[str, str]  # display key → phrase, in descriptor priority order
    gender: Gender = "unknown"
    color_slot: str | None = "Color"  # which attribute an explicit colour word replaces
    noun: str | None = None  # how to name it in prompts if different from the key
    home: str | None = None  # where this character "lives" (used for "find its home")
    young: bool = False


def _animal(
    color: str,
    *extra: tuple[str, str],
    gender: Gender = "unknown",
    home: str | None = None,
    noun: str | None = None,
    young: bool = False,
) -> SpeciesPrior:
    attrs = {"Color": color}
    attrs.update(dict(extra))
    return SpeciesPrior("animal", attrs, gender=gender, home=home, noun=noun, young=young)


SPECIES: dict[str, SpeciesPrior] = {
    # --- farm -------------------------------------------------------------------------------
    "hen": _animal(
        "warm orange-brown feathers",
        ("Comb", "small red comb"),
        ("Beak", "short yellow beak"),
        ("Eyes", "round shiny eyes"),
        ("Build", "small plump body"),
        gender="female",
        home="a cozy wooden henhouse",
    ),
    "chicken": _animal(
        "white feathers",
        ("Comb", "red comb"),
        ("Beak", "yellow beak"),
        ("Build", "plump body"),
        home="a wooden henhouse",
    ),
    "rooster": _animal(
        "glossy red and golden feathers",
        ("Tail", "tall dark green tail feathers"),
        ("Comb", "big red comb"),
        ("Beak", "yellow beak"),
        gender="male",
        home="a farmyard fence",
    ),
    "chick": _animal(
        "fluffy bright yellow down",
        ("Beak", "tiny orange beak"),
        ("Eyes", "small dark eyes"),
        young=True,
        home="a straw nest",
    ),
    "cow": _animal(
        "brown and white patched coat",
        ("Eyes", "large gentle brown eyes"),
        ("Horns", "small curved cream horns"),
        ("Muzzle", "pink muzzle"),
        ("Accessories", "brass bell on a red collar"),
        gender="female",
        home="a red barn",
    ),
    "bull": _animal("dark brown coat", ("Horns", "big curved horns"), ("Build", "strong broad body"), gender="male"),
    "calf": _animal("light brown coat with white spots", ("Eyes", "big curious eyes"), young=True),
    "pig": _animal(
        "pink skin", ("Snout", "round snout"), ("Tail", "curly tail"), ("Ears", "floppy ears"), home="a muddy pigsty"
    ),
    "piglet": _animal(
        "light pink skin", ("Snout", "tiny round snout"), ("Tail", "curly tail"), young=True, home="a straw pigsty"
    ),
    "sheep": _animal("fluffy white wool", ("Face", "black face"), ("Legs", "thin black legs")),
    "lamb": _animal("soft white curly wool", ("Face", "small cream face"), young=True),
    "goat": _animal("white and grey coat", ("Horns", "small curved horns"), ("Beard", "little chin beard")),
    "horse": _animal(
        "chestnut brown coat",
        ("Mane", "flowing dark mane"),
        ("Markings", "white blaze on the face"),
        home="a wooden stable",
    ),
    "pony": _animal(
        "dappled grey coat", ("Mane", "fluffy white mane"), ("Build", "small sturdy body"), home="a little stable"
    ),
    "donkey": _animal("soft grey coat", ("Ears", "long upright ears"), ("Muzzle", "pale muzzle")),
    "duck": _animal("white feathers", ("Beak", "orange bill"), ("Feet", "orange webbed feet"), home="a reedy pond"),
    "duckling": _animal("fluffy yellow down", ("Beak", "small orange bill"), young=True, home="a reedy pond"),
    "goose": _animal("white feathers", ("Beak", "orange beak"), ("Neck", "long neck"), home="a farm pond"),
    "turkey": _animal("bronze-brown feathers", ("Tail", "fanned striped tail"), ("Wattle", "red wattle")),
    # --- pets -------------------------------------------------------------------------------
    "dog": _animal(
        "golden-brown fur",
        ("Ears", "floppy ears"),
        ("Tail", "wagging tail"),
        ("Accessories", "blue collar"),
        home="a cozy dog house",
    ),
    "puppy": _animal(
        "fluffy golden fur",
        ("Ears", "floppy ears"),
        ("Eyes", "big shiny eyes"),
        ("Accessories", "red collar"),
        noun="puppy",
        young=True,
        home="a cozy dog bed",
    ),
    "cat": _animal(
        "ginger striped fur",
        ("Eyes", "bright green eyes"),
        ("Tail", "long fluffy tail"),
        ("Accessories", "small bell collar"),
        home="a cozy cottage",
    ),
    "kitten": _animal(
        "fluffy grey fur", ("Eyes", "big blue eyes"), ("Paws", "white paws"), young=True, home="a basket by the fire"
    ),
    "hamster": _animal("golden fur", ("Cheeks", "round cheeks"), ("Build", "tiny round body")),
    "parrot": _animal("bright green feathers", ("Wings", "red and blue wing tips"), ("Beak", "curved beak")),
    "goldfish": _animal("shiny orange scales", ("Fins", "flowing fins"), home="a round fishbowl"),
    # --- woodland & wild ----------------------------------------------------------------------
    "rabbit": _animal(
        "soft light-brown fur",
        ("Ears", "long upright ears"),
        ("Tail", "fluffy white tail"),
        ("Eyes", "big dark eyes"),
        home="a cozy burrow under a grassy hill",
    ),
    "bunny": _animal(
        "soft white fur",
        ("Ears", "long floppy ears"),
        ("Nose", "pink twitching nose"),
        ("Tail", "cotton tail"),
        noun="bunny",
        home="a cozy burrow under a grassy hill",
    ),
    "hare": _animal("sandy brown fur", ("Ears", "very long black-tipped ears"), ("Legs", "long hind legs")),
    "fox": _animal(
        "bright orange fur",
        ("Tail", "bushy white-tipped tail"),
        ("Chest", "white chest"),
        ("Ears", "pointed black-tipped ears"),
        home="a den under tree roots",
    ),
    "wolf": _animal(
        "thick grey fur", ("Eyes", "amber eyes"), ("Ears", "pointed ears"), ("Tail", "bushy tail"), home="a rocky den"
    ),
    "bear": _animal(
        "thick brown fur", ("Build", "big round body"), ("Ears", "small round ears"), home="a cave in the woods"
    ),
    "squirrel": _animal(
        "reddish-brown fur", ("Tail", "big bushy tail"), ("Ears", "tufted ears"), home="a hollow in an oak tree"
    ),
    "hedgehog": _animal(
        "brown spines", ("Face", "small pointed face"), ("Nose", "black button nose"), home="a nest of leaves"
    ),
    "mouse": _animal(
        "soft grey fur",
        ("Ears", "big round pink ears"),
        ("Tail", "long thin tail"),
        ("Build", "tiny body"),
        home="a tiny hole in the wall",
    ),
    "deer": _animal(
        "tan coat with white spots", ("Eyes", "large dark eyes"), ("Legs", "slender legs"), home="a quiet forest glade"
    ),
    "owl": _animal(
        "speckled brown feathers",
        ("Eyes", "big round amber eyes"),
        ("Ears", "feathery ear tufts"),
        home="a hollow in an old tree",
    ),
    "bird": _animal(
        "bright blue feathers",
        ("Breast", "orange breast"),
        ("Beak", "small pointed beak"),
        home="a twig nest in a tree",
    ),
    "robin": _animal("brown feathers", ("Breast", "red breast"), ("Beak", "small beak"), home="a nest in a hedge"),
    "frog": _animal(
        "bright green skin", ("Eyes", "big bulging eyes"), ("Legs", "long legs"), home="a lily pad on a pond"
    ),
    "turtle": _animal("green skin", ("Shell", "patterned brown shell"), ("Eyes", "kind eyes"), home="a sunny pond"),
    "tortoise": _animal("olive skin", ("Shell", "high domed shell"), ("Eyes", "wise old eyes")),
    "snail": _animal("soft grey body", ("Shell", "spiral brown shell"), ("Eyes", "eye stalks")),
    "butterfly": _animal("blue and orange wings", ("Body", "slender black body"), ("Antennae", "thin antennae")),
    "bee": _animal(
        "yellow and black stripes",
        ("Wings", "small translucent wings"),
        ("Build", "fuzzy round body"),
        home="a golden beehive",
    ),
    "ant": _animal("shiny red-brown body", ("Antennae", "bent antennae"), home="an ant hill"),
    "beaver": _animal("brown fur", ("Tail", "flat paddle tail"), ("Teeth", "big front teeth"), home="a stick lodge"),
    "otter": _animal("sleek brown fur", ("Whiskers", "long whiskers"), home="a riverbank"),
    "raccoon": _animal("grey fur", ("Face", "black eye mask"), ("Tail", "ringed tail")),
    # --- exotic --------------------------------------------------------------------------------
    "lion": _animal("golden fur", ("Mane", "big shaggy mane"), ("Tail", "tufted tail"), gender="male"),
    "lioness": _animal("sandy golden fur", ("Build", "sleek strong body"), gender="female"),
    "tiger": _animal("orange fur with black stripes", ("Eyes", "amber eyes"), ("Chest", "white chest")),
    "elephant": _animal(
        "wrinkly grey skin", ("Ears", "huge flapping ears"), ("Trunk", "long trunk"), ("Tusks", "small ivory tusks")
    ),
    "giraffe": _animal("yellow coat with brown patches", ("Neck", "very long neck"), ("Horns", "small ossicones")),
    "zebra": _animal("black and white stripes", ("Mane", "short striped mane")),
    "monkey": _animal(
        "brown fur", ("Face", "peach-colored face"), ("Tail", "long curly tail"), home="a tall jungle tree"
    ),
    "panda": _animal("black and white fur", ("Eyes", "black eye patches"), ("Build", "round fluffy body")),
    "penguin": _animal(
        "black and white feathers", ("Beak", "orange beak"), ("Build", "round waddling body"), home="an icy shore"
    ),
    "polar bear": _animal("thick white fur", ("Nose", "black nose"), ("Build", "big body")),
    "kangaroo": _animal("reddish-brown fur", ("Tail", "long strong tail"), ("Pouch", "front pouch")),
    "koala": _animal("fluffy grey fur", ("Ears", "big fuzzy ears"), ("Nose", "large black nose")),
    "crocodile": _animal("bumpy green scales", ("Mouth", "long toothy snout")),
    "snake": _animal("green scales with yellow diamonds", ("Eyes", "round friendly eyes")),
    "whale": _animal("blue-grey skin", ("Belly", "pale grooved belly"), ("Tail", "huge tail fluke")),
    "dolphin": _animal("smooth grey skin", ("Snout", "long smiling snout"), ("Fin", "curved dorsal fin")),
    "fish": _animal("shiny silver-blue scales", ("Fins", "flowing fins"), home="a colorful coral reef"),
    "octopus": _animal("purple skin", ("Arms", "eight curling arms"), ("Eyes", "big curious eyes")),
    "crab": _animal("bright red shell", ("Claws", "big pincers"), ("Eyes", "stalk eyes")),
    "dinosaur": _animal("green scaly skin", ("Back", "orange back plates"), ("Tail", "long thick tail")),
    # --- fantasy -------------------------------------------------------------------------------
    "dragon": SpeciesPrior(
        "creature",
        {
            "Color": "emerald green scales",
            "Wings": "leathery wings",
            "Belly": "golden belly",
            "Horns": "two small curved horns",
        },
        home="a mountain cave",
    ),
    "unicorn": SpeciesPrior(
        "creature", {"Color": "white coat", "Horn": "spiral golden horn", "Mane": "rainbow pastel mane"}
    ),
    "fairy": SpeciesPrior(
        "creature",
        {"Wings": "shimmering translucent wings", "Hair": "golden hair", "Outfit": "petal dress", "Size": "tiny"},
        gender="female",
        color_slot="Outfit",
    ),
    "elf": SpeciesPrior(
        "creature", {"Ears": "pointed ears", "Outfit": "green tunic and pointed hat"}, color_slot="Outfit"
    ),
    "giant": SpeciesPrior(
        "creature", {"Size": "enormous", "Beard": "bushy beard", "Outfit": "patched brown clothes"}, color_slot="Outfit"
    ),
    "troll": SpeciesPrior(
        "creature", {"Skin": "mossy green skin", "Nose": "big round nose", "Hair": "shaggy hair"}, color_slot="Skin"
    ),
    "monster": SpeciesPrior(
        "creature", {"Color": "fluffy purple fur", "Eyes": "one big friendly eye", "Horns": "two little horns"}
    ),
    "robot": SpeciesPrior(
        "object",
        {
            "Color": "shiny silver metal body",
            "Eyes": "glowing round blue eyes",
            "Head": "antenna on its head",
            "Build": "boxy friendly shape",
        },
    ),
    "mermaid": SpeciesPrior(
        "creature",
        {"Tail": "shimmering teal fish tail", "Hair": "long wavy red hair", "Accessories": "shell necklace"},
        gender="female",
        color_slot="Tail",
    ),
    "ghost": SpeciesPrior("creature", {"Color": "soft glowing white sheet-like body", "Eyes": "round dark eyes"}),
    "alien": SpeciesPrior("creature", {"Color": "light green skin", "Eyes": "big black eyes", "Head": "two antennae"}),
    "snowman": SpeciesPrior(
        "object",
        {"Body": "three round snowballs", "Nose": "carrot nose", "Accessories": "red scarf and black top hat"},
        color_slot="Accessories",
    ),
    "teddy bear": SpeciesPrior(
        "object", {"Color": "soft brown plush fur", "Accessories": "red ribbon bow", "Eyes": "black button eyes"}
    ),
    "doll": SpeciesPrior(
        "object", {"Hair": "yarn hair", "Outfit": "patchwork dress", "Eyes": "button eyes"}, color_slot="Outfit"
    ),
}

# Words that name the same kind of character.
ALIASES: dict[str, str] = {
    "bunnies": "bunny",
    "hens": "hen",
    "kitty": "kitten",
    "pup": "puppy",
    "doggy": "dog",
    "doggie": "dog",
    "piggy": "pig",
    "birdie": "bird",
    "froggy": "frog",
    "teddy": "teddy bear",
    "fishy": "fish",
    "chickens": "chicken",
    "cows": "cow",
    "dino": "dinosaur",
    "t-rex": "dinosaur",
    "lamb": "lamb",
    # family roles have many names for the same person
    "grandma": "grandmother",
    "granny": "grandmother",
    "nana": "grandmother",
    "grandpa": "grandfather",
    "granddad": "grandfather",
    "mom": "mother",
    "mum": "mother",
    "mommy": "mother",
    "mama": "mother",
    "dad": "father",
    "daddy": "father",
    "papa": "father",
}

# --- humans ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class HumanRole:
    gender: Gender
    age: str  # descriptor of apparent age
    outfit: str | None = None  # fixed outfit for archetypal roles; None → seeded pool
    hair: str | None = None  # fixed hair; None → seeded pool
    extras: dict[str, str] = field(default_factory=dict)
    stage: str = "adult"  # kid | adult | elder — selects age-appropriate hair/outfit pools


HUMANS: dict[str, HumanRole] = {
    "girl": HumanRole("female", "young girl about seven years old", stage="kid"),
    "boy": HumanRole("male", "young boy about seven years old", stage="kid"),
    "child": HumanRole("unknown", "young child", stage="kid"),
    "kid": HumanRole("unknown", "young child", stage="kid"),
    "baby": HumanRole("unknown", "chubby baby", outfit="soft white onesie", hair="wispy hair", stage="kid"),
    "toddler": HumanRole("unknown", "small toddler", stage="kid"),
    "teenager": HumanRole("unknown", "teenager"),
    "woman": HumanRole("female", "young woman"),
    "lady": HumanRole("female", "kind lady"),
    "man": HumanRole("male", "young man"),
    "mother": HumanRole("female", "mother in her thirties"),
    "mom": HumanRole("female", "mother in her thirties"),
    "mum": HumanRole("female", "mother in her thirties"),
    "mommy": HumanRole("female", "mother in her thirties"),
    "father": HumanRole("male", "father in his thirties"),
    "dad": HumanRole("male", "father in his thirties"),
    "daddy": HumanRole("male", "father in his thirties"),
    "grandmother": HumanRole(
        "female",
        "kind elderly woman",
        hair="silver hair in a neat bun",
        extras={"Accessories": "round glasses"},
        stage="elder",
    ),
    "grandma": HumanRole(
        "female",
        "kind elderly woman",
        hair="silver hair in a neat bun",
        extras={"Accessories": "round glasses"},
        stage="elder",
    ),
    "granny": HumanRole(
        "female",
        "kind elderly woman",
        hair="silver hair in a neat bun",
        extras={"Accessories": "round glasses"},
        stage="elder",
    ),
    "grandfather": HumanRole("male", "cheerful elderly man", hair="white hair and a white mustache", stage="elder"),
    "grandpa": HumanRole("male", "cheerful elderly man", hair="white hair and a white mustache", stage="elder"),
    "sister": HumanRole("female", "young girl", stage="kid"),
    "brother": HumanRole("male", "young boy", stage="kid"),
    "friend": HumanRole("unknown", "young child", stage="kid"),
    "farmer": HumanRole("unknown", "friendly farmer", outfit="straw hat, plaid shirt and blue denim overalls"),
    "king": HumanRole(
        "male", "kindly middle-aged king", outfit="golden crown and long red velvet robe", hair="neat grey beard"
    ),
    "queen": HumanRole("female", "graceful queen", outfit="jeweled crown and flowing royal blue gown"),
    "prince": HumanRole("male", "young prince", outfit="small gold circlet and blue tunic with a short cape"),
    "princess": HumanRole("female", "young princess", outfit="sparkly tiara and flowing lavender gown"),
    "knight": HumanRole("unknown", "brave knight", outfit="shining silver armor with a red plume"),
    "wizard": HumanRole(
        "male", "old wizard", outfit="tall pointed starry blue hat and robe", hair="long white beard", stage="elder"
    ),
    "witch": HumanRole("female", "friendly witch", outfit="pointed black hat and purple cloak"),
    "pirate": HumanRole("unknown", "jolly pirate", outfit="tricorn hat, striped shirt and red sash"),
    "astronaut": HumanRole("unknown", "astronaut", outfit="white space suit with a round helmet"),
    "teacher": HumanRole("unknown", "kind teacher", outfit="cozy cardigan", extras={"Accessories": "glasses"}),
    "doctor": HumanRole("unknown", "friendly doctor", outfit="white coat with a stethoscope"),
    "chef": HumanRole("unknown", "jolly chef", outfit="white chef hat and apron"),
    "fisherman": HumanRole("male", "weathered fisherman", outfit="yellow raincoat and rubber boots"),
    "sailor": HumanRole("unknown", "sailor", outfit="navy blue sailor uniform"),
    "shepherd": HumanRole("unknown", "young shepherd", outfit="simple tunic and wooden crook"),
    "soldier": HumanRole("unknown", "toy-like soldier", outfit="red coat and tall hat"),
    "woodcutter": HumanRole("male", "strong woodcutter", outfit="red flannel shirt", hair="bushy brown beard"),
    "baker": HumanRole("unknown", "cheerful baker", outfit="white apron dusted with flour"),
    "gardener": HumanRole("unknown", "gardener", outfit="sun hat and green apron"),
}

HAIR_POOLS = {
    "female": [
        "curly auburn hair",
        "long black hair in a ponytail",
        "wavy blonde hair",
        "red hair in two braids",
        "light brown hair in pigtails",
        "dark curly hair with a yellow headband",
    ],
    "male": [
        "short brown hair",
        "messy black hair",
        "short curly dark hair",
        "short blonde hair",
        "spiky red hair",
        "short wavy brown hair",
    ],
    "unknown": ["short curly auburn hair", "short black hair", "wavy brown hair", "light brown hair"],
}
ELDER_HAIR = {
    "female": ["silver hair in a neat bun", "short curly white hair"],
    "male": ["white hair and a white mustache", "grey hair and a neat grey beard"],
    "unknown": ["short silver hair"],
}
SKIN_POOL = ["fair skin", "light tan skin", "olive skin", "warm brown skin", "deep brown skin", "golden tan skin"]
OUTFIT_POOLS = {
    ("kid", "female"): [
        "yellow raincoat and red boots",
        "red dress with white dots",
        "teal dress and white sneakers",
        "purple sweater and denim skirt",
        "pink t-shirt and blue overalls",
    ],
    ("kid", "male"): [
        "green hoodie and jeans",
        "blue striped shirt and brown shorts",
        "orange t-shirt and denim overalls",
        "red jacket and blue jeans",
        "yellow raincoat and green boots",
    ],
    ("kid", "unknown"): [
        "green hoodie and jeans",
        "yellow raincoat and red boots",
        "orange t-shirt and denim overalls",
    ],
    ("adult", "female"): [
        "long flowery skirt and white blouse",
        "blue dress with a cozy cardigan",
        "green sweater and dark jeans",
    ],
    ("adult", "male"): [
        "checked shirt and brown trousers",
        "navy sweater and grey trousers",
        "blue denim shirt and khaki trousers",
    ],
    ("adult", "unknown"): ["warm red sweater and dark jeans", "cozy green cardigan and brown trousers"],
    ("elder", "female"): ["lavender cardigan and long skirt", "flowery dress with a knitted shawl"],
    ("elder", "male"): ["brown tweed vest and trousers", "cozy cardigan and a flat cap"],
    ("elder", "unknown"): ["cozy knitted cardigan and soft trousers"],
}


def seeded_choice(options: list[str], *keys: str) -> str:
    digest = hashlib.sha256("|".join(keys).encode()).digest()
    return options[int.from_bytes(digest[:4], "big") % len(options)]


# --- vocabulary --------------------------------------------------------------------------------

COLOR_WORDS = {
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "violet",
    "pink",
    "brown",
    "black",
    "white",
    "grey",
    "gray",
    "golden",
    "gold",
    "silver",
    "ginger",
    "tan",
    "cream",
    "beige",
    "spotted",
    "striped",
    "spotty",
    "rainbow",
    "teal",
    "turquoise",
    "lilac",
    "crimson",
    "scarlet",
    "copper",
    "bronze",
    "speckled",
    "fluffy",
    "shiny",
    "sparkly",
    "dark",
    "light",
}
SIZE_WORDS = {
    "little": "small",
    "small": "small",
    "tiny": "tiny",
    "big": "big",
    "large": "big",
    "huge": "huge",
    "giant": "giant",
    "enormous": "enormous",
    "tall": "tall",
    "short": "short",
    "fat": "plump",
    "plump": "plump",
    "chubby": "chubby",
    "round": "round",
    "thin": "thin",
    "skinny": "thin",
    "long": "long",
}
AGE_WORDS = {"old": "old", "elderly": "elderly", "young": "young", "baby": "baby", "ancient": "ancient"}
# trait adjective → facial expression / body language shown in images
TRAIT_EXPRESSIONS = {
    "friendly": "friendly smile",
    "kind": "kind gentle smile",
    "happy": "happy smile",
    "cheerful": "cheerful smile",
    "brave": "brave confident look",
    "shy": "shy bashful look",
    "grumpy": "grumpy frown",
    "sad": "sad eyes",
    "lonely": "lonely sad look",
    "lost": "worried lost look",
    "scared": "frightened look",
    "frightened": "frightened look",
    "afraid": "frightened look",
    "curious": "curious wide-eyed look",
    "clever": "clever bright eyes",
    "wise": "wise calm expression",
    "silly": "silly goofy grin",
    "naughty": "mischievous grin",
    "mischievous": "mischievous grin",
    "lazy": "sleepy relaxed look",
    "sleepy": "sleepy eyes",
    "tired": "tired eyes",
    "hungry": "hungry look",
    "proud": "proud posture",
    "gentle": "gentle expression",
    "angry": "angry frown",
    "excited": "excited expression",
    "playful": "playful expression",
    "nice": "warm smile",
    "good": "warm smile",
    "helpful": "helpful smile",
    "beautiful": "graceful look",
    "pretty": "graceful look",
    "cute": "cute look",
    "fierce": "fierce look",
}

# --- places ---------------------------------------------------------------------------------------

PLACES: dict[str, str] = {
    "farm": "a sunny farmyard with a red barn, a wooden fence and green meadows",
    "farmyard": "a sunny farmyard with a red barn and a wooden fence",
    "barn": "inside a rustic wooden barn with hay bales",
    "field": "an open green field under a wide sky",
    "meadow": "a flowery meadow with tall grass",
    "pond": "the edge of a calm pond with lily pads and reeds",
    "lake": "the shore of a sparkling blue lake",
    "river": "the bank of a gently flowing river",
    "stream": "a babbling stream with mossy stones",
    "forest": "a lush green forest with tall trees and dappled sunlight",
    "woods": "a lush green forest with tall trees and dappled sunlight",
    "wood": "a leafy wood with tall trees",
    "jungle": "a dense tropical jungle with giant leaves and vines",
    "garden": "a colorful flower garden",
    "park": "a green park with trees, a path and flower beds",
    "playground": "a bright playground with swings and a slide",
    "house": "a cozy house",
    "home": "a cozy home",
    "cottage": "a cozy thatched cottage",
    "kitchen": "a warm cozy kitchen",
    "bedroom": "a cozy bedroom with a soft bed",
    "room": "a cozy room",
    "school": "a cheerful classroom",
    "classroom": "a cheerful classroom with desks and drawings on the walls",
    "library": "a quiet library with tall bookshelves",
    "village": "a quaint village with little houses and a cobblestone street",
    "town": "a small friendly town square",
    "city": "a bustling city street",
    "street": "a quiet street lined with houses",
    "market": "a busy colorful market with fruit stalls",
    "castle": "a grand stone castle with tall towers and flags",
    "palace": "a grand palace hall",
    "tower": "a tall stone tower",
    "beach": "a sandy beach with gentle waves",
    "sea": "the open sea with rolling waves",
    "ocean": "the deep blue ocean",
    "island": "a small tropical island with palm trees",
    "mountain": "a tall mountain with rocky slopes and snowy peaks",
    "mountains": "majestic mountains with snowy peaks",
    "hill": "a rolling green hill",
    "valley": "a green valley between hills",
    "cave": "a mysterious cave entrance with glowing crystals",
    "desert": "a sunny desert with sand dunes",
    "snow": "a snowy landscape",
    "sky": "the open sky among fluffy clouds",
    "space": "outer space with stars and colorful planets",
    "moon": "the surface of the moon with craters",
    "zoo": "a sunny zoo",
    "nest": "a cozy twig nest high in a tree",
    "burrow": "a cozy burrow under a grassy hill",
    "den": "a snug den under tree roots",
    "henhouse": "a cozy wooden henhouse",
    "stable": "a wooden stable with straw",
    "shop": "a little cozy shop",
    "bakery": "a warm bakery full of bread",
    "ship": "the deck of a wooden sailing ship",
    "boat": "a little boat on the water",
    "bridge": "an old stone bridge",
    "road": "a winding country road",
    "path": "a winding path",
    "orchard": "an apple orchard with fruit-laden trees",
    "swamp": "a misty swamp",
    "underwater": "an underwater coral reef",
    "reef": "a colorful coral reef",
    "cellar": "a dim cellar with wooden shelves and old boxes",
    "attic": "a dusty attic with old trunks and a round window",
    "staircase": "a narrow old staircase",
    "hallway": "a long hallway",
    "yard": "a backyard with green grass and a wooden fence",
    "museum": "a bright museum hall with exhibits",
    "hotel": "a pretty hotel with many rooms",
    "crater": "a dusty moon crater",
    "planet": "a colorful alien planet",
}
# Frame-setting places ("in a kingdom far away") are poor illustration settings when a more specific
# place is mentioned in the same clause.
GENERIC_PLACES = {"kingdom", "land", "world", "country", "place", "universe", "realm"}

TIME_OF_DAY = {
    "morning": "soft golden morning light",
    "sunrise": "pink and golden sunrise light",
    "dawn": "pale dawn light",
    "noon": "bright midday sunlight",
    "afternoon": "warm afternoon sunlight",
    "day": "bright sunny daylight",
    "evening": "warm orange sunset light with long shadows",
    "sunset": "glowing orange sunset light",
    "dusk": "purple dusk light",
    "night": "moonlit night with twinkling stars and soft blue light",
    "midnight": "dark moonlit night",
    "bedtime": "cozy lamplight at night",
}

WEATHER = {
    "rain": "gentle rain falling",
    "raining": "gentle rain falling",
    "rainy": "rainy weather",
    "storm": "stormy sky with dark clouds",
    "stormy": "stormy sky with dark clouds",
    "thunder": "stormy sky",
    "snow": "snowflakes falling",
    "snowing": "snowflakes falling",
    "snowy": "snow-covered ground",
    "wind": "windy weather with blowing leaves",
    "windy": "windy weather with blowing leaves",
    "fog": "soft misty fog",
    "foggy": "soft misty fog",
    "sunny": "bright sunshine",
    "rainbow": "a rainbow in the sky",
}
