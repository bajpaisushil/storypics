"""Download the pretrained weights StoryPix uses (explicit, opt-in; nothing is fetched at request time).

Usage:
    python scripts/download_models.py               # everything needed for the default profile
    python scripts/download_models.py --only nlp    # just the small NLP/evaluation encoders
    python scripts/download_models.py --list        # show what would be downloaded
    python scripts/download_models.py --only extras # optional 4-step Lightning LoRA (STORYPIX_LIGHTNING_STEPS=4)

All files land in the standard Hugging Face cache (HF_HOME / ~/.cache/huggingface).
Licenses are listed next to each entry; see docs/ARCHITECTURE.md for the full review.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

from huggingface_hub import snapshot_download


@dataclass(frozen=True)
class WeightSpec:
    group: str
    repo_id: str
    license: str
    approx_gb: float
    allow_patterns: list[str] = field(default_factory=list)


SPECS: list[WeightSpec] = [
    WeightSpec(
        "image",
        "stabilityai/stable-diffusion-xl-base-1.0",
        "CreativeML OpenRAIL++-M",
        6.8,
        [
            "model_index.json",
            "scheduler/*",
            "tokenizer/*",
            "tokenizer_2/*",
            "text_encoder/config.json",
            "text_encoder/model.fp16.safetensors",
            "text_encoder_2/config.json",
            "text_encoder_2/model.fp16.safetensors",
            "unet/config.json",
            "unet/diffusion_pytorch_model.fp16.safetensors",
            "vae/config.json",
        ],
    ),
    WeightSpec(
        "image", "madebyollin/sdxl-vae-fp16-fix", "MIT", 0.33, ["config.json", "diffusion_pytorch_model.safetensors"]
    ),
    WeightSpec("image", "ByteDance/SDXL-Lightning", "OpenRAIL++", 0.39, ["sdxl_lightning_8step_lora.safetensors"]),
    WeightSpec(
        "image",
        "h94/IP-Adapter",
        "Apache-2.0",
        3.4,
        ["sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors", "models/image_encoder/*"],
    ),
    WeightSpec("extras", "ByteDance/SDXL-Lightning", "OpenRAIL++", 0.39, ["sdxl_lightning_4step_lora.safetensors"]),
    WeightSpec(
        "nlp",
        "sentence-transformers/all-MiniLM-L6-v2",
        "Apache-2.0",
        0.09,
        [
            "config.json",
            "model.safetensors",
            "tokenizer*",
            "vocab.txt",
            "special_tokens_map.json",
            "modules.json",
            "sentence_bert_config.json",
            "1_Pooling/*",
        ],
    ),
    WeightSpec(
        "eval",
        "google/siglip-base-patch16-224",
        "Apache-2.0",
        0.81,
        [
            "config.json",
            "model.safetensors",
            "preprocessor_config.json",
            "tokenizer*",
            "spiece.model",
            "special_tokens_map.json",
        ],
    ),
    WeightSpec(
        "eval",
        "facebook/dinov2-small",
        "Apache-2.0",
        0.09,
        ["config.json", "model.safetensors", "preprocessor_config.json"],
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=["image", "nlp", "eval", "extras"], help="download one group only")
    parser.add_argument("--list", action="store_true", help="print the plan without downloading")
    args = parser.parse_args()

    selected = [s for s in SPECS if (s.group == args.only if args.only else s.group != "extras")]
    total = sum(s.approx_gb for s in selected)
    print(f"{len(selected)} repositories, ~{total:.1f} GB")
    for spec in selected:
        print(f"  [{spec.group:5s}] {spec.repo_id:45s} {spec.approx_gb:5.2f} GB  ({spec.license})")
    if args.list:
        return
    for spec in selected:
        print(f"\n→ {spec.repo_id}")
        path = snapshot_download(spec.repo_id, allow_patterns=spec.allow_patterns)
        print(f"  cached at {path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
