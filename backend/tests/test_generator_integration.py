"""Integration test of LocalDiffusersGenerator against the real Diffusers SDXL pipeline code, using the official
tiny random-weight test pipeline (~25 MB download). Opt-in: STORYPIX_RUN_INTEGRATION=1 pytest -k integration.
It checks the plumbing (loading, scheduler, long-prompt encoding, CFG, callbacks, cancellation), not image quality."""

import os
import threading

import pytest

from app.config import Settings
from app.schemas.generation import GenerationRequest
from app.services.image_generator import GenerationCancelled, LocalDiffusersGenerator, count_chunks, encode_long_prompt

pytestmark = pytest.mark.skipif(
    os.environ.get("STORYPIX_RUN_INTEGRATION") != "1",
    reason="set STORYPIX_RUN_INTEGRATION=1 (downloads a 25 MB tiny SDXL test pipeline)",
)
TINY = "hf-internal-testing/tiny-stable-diffusion-xl-pipe"


@pytest.fixture(scope="module")
def generator():
    settings = Settings(
        base_model=TINY,
        base_variant=None,
        vae_model=None,
        profile="quality",
        device="cpu",
        quality_steps=2,
        allow_downloads=True,
        enable_ip_adapter=False,  # the real IP-Adapter does not fit the tiny test UNet
        safety_checker=False,  # plumbing test: no 1.2 GB classifier download
    )
    gen = LocalDiffusersGenerator(settings)
    gen.load()
    return gen


def test_generates_image_with_long_prompt_and_cfg(generator):
    steps = []
    long_prompt = ", ".join(["a small plump hen with warm orange-brown feathers and a red comb"] * 12)  # > 77 tokens
    image, info = generator.generate(
        GenerationRequest(
            kind="scene",
            prompt=long_prompt,
            negative_prompt="blurry, lowres",
            width=256,
            height=256,
            seed=7,
            guidance_scale=5.0,
        ),
        progress=lambda s, t: steps.append((s, t)),
    )
    assert image.size == (256, 256) and info.steps == 2 and info.guidance_scale == 5.0
    assert steps == [(1, 2), (2, 2)] and info.reference_mode == "none"


def test_long_prompt_is_chunked_not_truncated(generator):
    pipe = generator._pipe
    long_prompt = " ".join(["hen"] * 200)
    n_tokens = max(len(t(long_prompt, add_special_tokens=False).input_ids) for t in (pipe.tokenizer, pipe.tokenizer_2))
    emb, pooled, neg, _neg_pooled = encode_long_prompt(pipe, long_prompt, "blurry", pipe.device, True)
    assert n_tokens > 77  # would have been truncated by the standard encoder
    assert emb.shape[1] == 77 * count_chunks(n_tokens) and neg.shape == emb.shape and pooled.shape[0] == 1


def test_same_seed_is_reproducible(generator):
    req = GenerationRequest(kind="scene", prompt="a cow on a farm", width=256, height=256, seed=3, guidance_scale=0)
    a, _ = generator.generate(req)
    b, _ = generator.generate(req)
    assert list(a.getdata()) == list(b.getdata())


def test_cancellation_stops_generation(generator):
    flag = threading.Event()
    flag.set()
    with pytest.raises(GenerationCancelled):
        generator.generate(GenerationRequest(kind="scene", prompt="a cow", width=256, height=256, seed=1), cancel=flag)


@pytest.fixture(scope="module")
def lcm_generator(tmp_path_factory):
    """A tiny LatentConsistencyModelPipeline (SD1.5 family) assembled from cached tiny test components."""
    from diffusers import AutoencoderKL, LatentConsistencyModelPipeline, LCMScheduler, UNet2DConditionModel
    from huggingface_hub import snapshot_download
    from transformers import CLIPTextModel, CLIPTokenizer

    tiny = snapshot_download(
        TINY,
        allow_patterns=[
            "tokenizer/*",
            "text_encoder/config.json",
            "text_encoder/pytorch_model.bin",
            "vae/config.json",
            "vae/diffusion_pytorch_model.bin",
        ],
    )
    unet = UNet2DConditionModel(
        block_out_channels=(32, 64),
        layers_per_block=1,
        sample_size=32,
        in_channels=4,
        out_channels=4,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim=32,
        time_cond_proj_dim=32,
        norm_num_groups=32,
    )
    pipe = LatentConsistencyModelPipeline(
        vae=AutoencoderKL.from_pretrained(tiny, subfolder="vae"),
        text_encoder=CLIPTextModel.from_pretrained(tiny, subfolder="text_encoder"),
        tokenizer=CLIPTokenizer.from_pretrained(tiny, subfolder="tokenizer"),
        unet=unet,
        scheduler=LCMScheduler(),
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    path = tmp_path_factory.mktemp("tiny_lcm")
    pipe.save_pretrained(path)
    settings = Settings(
        profile="lcm", lcm_model=str(path), device="cpu", lcm_steps=2, enable_ip_adapter=False, safety_checker=False
    )
    gen = LocalDiffusersGenerator(settings)
    assert gen.status().available and gen.family == "sd15"
    gen.load()
    return gen


def test_lcm_profile_generates_with_long_prompt(lcm_generator):
    steps = []
    prompt = ", ".join(["a hen with warm orange-brown feathers meeting a cow on a sunny farm"] * 10)
    image, info = lcm_generator.generate(
        GenerationRequest(kind="scene", prompt=prompt, width=256, height=256, seed=5),
        progress=lambda s, t: steps.append((s, t)),
    )
    assert image.size == (256, 256) and info.profile == "lcm" and info.steps == 2 and steps[-1] == (2, 2)
    pipe = lcm_generator._pipe
    emb, pooled, _, _ = encode_long_prompt(pipe, prompt, "", pipe.device, False)
    assert pooled is None and emb.shape[1] % 77 == 0 and emb.shape[1] > 77  # chunked, SD1.5 has no pooled output
