"""Stage 9 — image generation behind a small interface, with a local Diffusers backend and a remote client.

LocalDiffusersGenerator
    SDXL base 1.0 (OpenRAIL++) + madebyollin fp16-fix VAE (MIT)
    profile "lightning": SDXL-Lightning LoRA (OpenRAIL++) fused into the UNet, Euler "trailing", CFG 0, 4/8 steps
    profile "quality"  : DPM-Solver++ (Karras), 28 steps, CFG from the style, negative prompts
    IP-Adapter Plus SDXL ViT-H (Apache-2.0) for character reference conditioning; several references are
    confined to their own canvas regions with IP-Adapter attention masks.
RemoteGenerator
    Sends the same GenerationRequest (references inlined as base64) to worker/server.py on a GPU machine.

If weights are missing the generator reports itself unavailable with the exact fix. It never substitutes
placeholder images.
"""

from __future__ import annotations

import base64
import io
import logging
import math
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from PIL import Image

from app.config import Settings
from app.schemas.generation import GenerationInfo, GenerationRequest, GeneratorStatus, ReferenceImage

log = logging.getLogger(__name__)

ProgressFn = Callable[[int, int], None]


class GenerationCancelled(Exception):
    pass


class GeneratorUnavailable(RuntimeError):
    pass


class ImageGenerator(Protocol):
    def status(self) -> GeneratorStatus: ...

    def load(self) -> None: ...

    def generate(
        self, request: GenerationRequest, progress: ProgressFn | None = None, cancel: threading.Event | None = None
    ) -> tuple[Image.Image, GenerationInfo]: ...


def pick_device(preference: str) -> str:
    import torch

    if preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_reference(ref: ReferenceImage) -> Image.Image:
    if ref.path:
        return Image.open(ref.path).convert("RGB")
    if ref.data_b64:
        return Image.open(io.BytesIO(base64.b64decode(ref.data_b64))).convert("RGB")
    raise ValueError(f"reference for {ref.character_id} has no image data")


# ------------------------------------------------------------------------------------- long prompts

CHUNK = 75  # CLIP context is 77 tokens = BOS + 75 + EOS


def chunk_token_ids(ids: list[int], n_chunks: int, bos: int, eos: int, pad: int) -> list[list[int]]:
    """Split token ids into ``n_chunks`` 77-token windows, each wrapped in BOS/EOS and padded."""
    rows = []
    for c in range(n_chunks):
        piece = ids[c * CHUNK : (c + 1) * CHUNK]
        row = [bos, *piece, eos]
        rows.append(row + [pad] * (CHUNK + 2 - len(row)))
    return rows


def count_chunks(n_tokens: int) -> int:
    return max(1, math.ceil(n_tokens / CHUNK))


def encode_long_prompt(pipe, prompt: str, negative: str, device, with_negative: bool):
    """Prompt encoding without CLIP's 77-token truncation, for SDXL (two encoders) and SD1.5-family models.

    Every encoder sees the prompt in 77-token windows whose hidden states are concatenated along the sequence
    axis. SDXL uses penultimate hidden states plus the pooled embedding of text_encoder_2's first window (its
    standard conditioning); SD1.5 uses the final hidden states and has no pooled embedding.
    """
    import torch

    sdxl = getattr(pipe, "text_encoder_2", None) is not None
    tokenizers = [pipe.tokenizer, *([pipe.tokenizer_2] if sdxl else [])]
    encoders = [pipe.text_encoder, *([pipe.text_encoder_2] if sdxl else [])]

    def ids(tok, text: str) -> list[int]:
        return tok(text, add_special_tokens=False, truncation=False).input_ids

    pos_ids = [ids(t, prompt) for t in tokenizers]
    neg_ids = [ids(t, negative) for t in tokenizers] if with_negative else []
    n = max(count_chunks(len(x)) for x in pos_ids + neg_ids)

    def run(text_ids: list[list[int]]):
        hidden, pooled = [], None
        for tok, enc, tid in zip(tokenizers, encoders, text_ids, strict=True):
            pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
            rows = chunk_token_ids(tid, n, tok.bos_token_id, tok.eos_token_id, pad)
            batch = torch.tensor(rows, device=device)
            out = enc(batch, output_hidden_states=True)
            states = out.hidden_states[-2] if sdxl else out[0]  # (n, 77, d)
            hidden.append(states.reshape(1, n * (CHUNK + 2), -1))
            if sdxl and enc is pipe.text_encoder_2:
                pooled = out[0][:1]  # text_embeds of the first window
        return torch.cat(hidden, dim=-1), pooled

    with torch.no_grad():
        prompt_embeds, pooled = run(pos_ids)
        if not with_negative:
            return prompt_embeds, pooled, None, None
        if negative.strip():
            neg_embeds, neg_pooled = run(neg_ids)
        else:
            neg_embeds = torch.zeros_like(prompt_embeds)
            neg_pooled = torch.zeros_like(pooled) if pooled is not None else None
    return prompt_embeds, pooled, neg_embeds, neg_pooled


# ------------------------------------------------------------------------------------- local backend

PROFILE_FAMILY = {"lightning": "sdxl", "quality": "sdxl", "lcm": "sd15"}
# (repository template, file) pairs per model family; templates are filled from Settings
REQUIRED_FILES = {
    "sdxl": [
        ("{base}", "unet/diffusion_pytorch_model.fp16.safetensors"),
        ("{base}", "text_encoder/model.fp16.safetensors"),
        ("{base}", "text_encoder_2/model.fp16.safetensors"),
        ("{vae}", "diffusion_pytorch_model.safetensors"),
    ],
    "sd15": [
        ("{lcm}", "unet/diffusion_pytorch_model.safetensors"),
        ("{lcm}", "text_encoder/model.safetensors"),
        ("{lcm}", "vae/diffusion_pytorch_model.safetensors"),
    ],
}
# Character reference conditioning. Optional: without it the generator still works (text-only consistency)
# and /api/health reports reference_conditioning=false with the fix.
IP_ADAPTER_FILES = {
    "sdxl": [("{ip}", "sdxl_models/{ip_weight}"), ("{ip}", "models/image_encoder/model.safetensors")],
    "sd15": [("{ip}", "models/{ip_weight_sd15}"), ("{ip}", "models/image_encoder/model.safetensors")],
}
# Exact files to fetch when STORYPIX_ALLOW_DOWNLOADS=true (e.g. on a server); avoids pulling duplicate ONNX weights.
SD15_PATTERNS = [
    "model_index.json",
    "scheduler/*",
    "tokenizer/*",
    "feature_extractor/*",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "unet/config.json",
    "unet/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
SAFETY_PATTERNS = ["safety_checker/config.json", "safety_checker/model.safetensors"]
SDXL_PATTERNS = [
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
]


class GenerationBlocked(RuntimeError):
    """The safety checker flagged the image; it is discarded rather than shown."""


class LocalDiffusersGenerator:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._pipe = None
        self._lock = threading.Lock()
        self._load_error: str | None = None
        self.device: str | None = None
        self.profile: str | None = None
        self._ip_loaded = False
        self._gray = Image.new("RGB", (224, 224), (128, 128, 128))

    # -- configuration ------------------------------------------------------------------------------
    def _resolve_profile(self) -> str:
        if self.settings.profile != "auto":
            return self.settings.profile
        device = pick_device(self.settings.device)
        return {"cuda": "quality", "mps": "lightning"}.get(device, "lcm")

    @property
    def family(self) -> str:
        return PROFILE_FAMILY[self.profile or self._resolve_profile()]

    @property
    def model_name(self) -> str:
        return self.settings.lcm_model if self.family == "sd15" else self.settings.base_model

    def _lightning_file(self) -> str:
        return f"sdxl_lightning_{self.settings.lightning_steps}step_lora.safetensors"

    def _files(self, spec: list[tuple[str, str]]) -> list[tuple[str, str]]:
        s = self.settings
        fill = {"base": s.base_model, "vae": s.vae_model or s.base_model, "ip": s.ip_adapter_repo, "lcm": s.lcm_model}
        names = {"ip_weight": s.ip_adapter_weight, "ip_weight_sd15": s.ip_adapter_sd15_weight}
        return [(r.format(**fill), f.format(**names)) for r, f in spec]

    @staticmethod
    def _present(repo: str, filename: str) -> bool:
        from huggingface_hub import try_to_load_from_cache

        if Path(repo).is_dir():  # a local pipeline folder
            return (Path(repo) / filename).exists()
        return isinstance(try_to_load_from_cache(repo, filename), str)

    def missing_weights(self) -> list[str]:
        files = self._files(REQUIRED_FILES[self.family])
        if self._resolve_profile() == "lightning":
            files.append((self.settings.lightning_repo, self._lightning_file()))
        if self.family == "sd15" and self.settings.safety_checker:
            files.append((self.settings.lcm_model, "safety_checker/model.safetensors"))
        return [f"{r}/{f}" for r, f in files if not self._present(r, f)]

    def ip_adapter_missing(self) -> list[str]:
        return [f"{r}/{f}" for r, f in self._files(IP_ADAPTER_FILES[self.family]) if not self._present(r, f)]

    def status(self) -> GeneratorStatus:
        device = self.device or pick_device(self.settings.device)
        profile = self.profile or self._resolve_profile()
        reason = self._load_error
        missing = [] if self._pipe is not None else self.missing_weights()
        if missing and not self.settings.allow_downloads:
            size = "~8 GB" if self.family == "sd15" else "~12 GB"
            reason = (
                f"Image model weights are not downloaded ({len(missing)} files missing, e.g. {missing[0]}). "
                f"Run `python scripts/download_models.py` in backend/ ({size})."
            )
        refs = (
            self._ip_loaded
            if self._pipe is not None
            else (self.settings.enable_ip_adapter and (self.settings.allow_downloads or not self.ip_adapter_missing()))
        )
        notice = (
            None
            if refs
            else (
                "IP-Adapter weights are not downloaded, so character reference images are "
                "not used (text-only consistency). Run `python scripts/download_models.py`."
            )
        )
        return GeneratorStatus(
            backend="local",
            available=reason is None,
            loaded=self._pipe is not None,
            device=device,
            profile=profile,
            model=self.model_name,
            reason=reason,
            reference_conditioning=refs,
            notice=notice,
        )

    # -- loading ----------------------------------------------------------------------------------
    def load(self) -> None:
        with self._lock:
            if self._pipe is not None:
                return
            status = self.status()
            if not status.available:
                raise GeneratorUnavailable(status.reason)
            try:
                self._pipe = self._build()
                self._load_error = None
            except Exception as exc:  # surface the real reason in /api/health
                self._load_error = f"Failed to load the image model: {exc}"
                log.exception("generator load failed")
                raise GeneratorUnavailable(self._load_error) from exc

    def _ensure_weights(self) -> None:
        """Fetch exactly the files this profile needs (only when downloads are allowed, e.g. on a server)."""
        from huggingface_hub import snapshot_download

        s = self.settings
        if self.family == "sd15":
            if not Path(s.lcm_model).is_dir():
                snapshot_download(
                    s.lcm_model, allow_patterns=SD15_PATTERNS + (SAFETY_PATTERNS if s.safety_checker else [])
                )
            ip = [f"models/{s.ip_adapter_sd15_weight}", "models/image_encoder/*"]
        else:
            snapshot_download(s.base_model, allow_patterns=SDXL_PATTERNS)
            if s.vae_model:
                snapshot_download(s.vae_model, allow_patterns=["config.json", "diffusion_pytorch_model.safetensors"])
            if self._resolve_profile() == "lightning":
                snapshot_download(s.lightning_repo, allow_patterns=[self._lightning_file()])
            ip = [f"sdxl_models/{s.ip_adapter_weight}", "models/image_encoder/*"]
        if s.enable_ip_adapter:
            snapshot_download(s.ip_adapter_repo, allow_patterns=ip)

    def _build(self):
        import torch

        s = self.settings
        self.device = pick_device(s.device)
        self.profile = self._resolve_profile()
        if self.device == "cpu" and s.torch_threads:
            torch.set_num_threads(s.torch_threads)
        dtype = torch.float32 if self.device == "cpu" else torch.float16
        t0 = time.perf_counter()
        if s.allow_downloads:
            self._ensure_weights()
        pipe = self._build_sd15(dtype) if self.family == "sd15" else self._build_sdxl(dtype)
        ip_ok = s.enable_ip_adapter and not self.ip_adapter_missing()
        if ip_ok:
            sd15 = self.family == "sd15"
            pipe.load_ip_adapter(
                s.ip_adapter_repo,
                subfolder="models" if sd15 else "sdxl_models",
                weight_name=s.ip_adapter_sd15_weight if sd15 else s.ip_adapter_weight,
                image_encoder_folder="image_encoder" if sd15 else "models/image_encoder",
                local_files_only=True,
            )
            self._ip_loaded = True
        else:
            log.warning("IP-Adapter weights missing — generating without character reference conditioning")
        pipe.to(self.device)
        if s.attention_slicing:
            pipe.enable_attention_slicing()
        pipe.set_progress_bar_config(disable=True)
        log.info(
            "%s pipeline (%s) ready on %s in %.1fs",
            self.model_name,
            self.profile,
            self.device,
            time.perf_counter() - t0,
        )
        return pipe

    def _build_sd15(self, dtype):
        from diffusers import DiffusionPipeline, LCMScheduler

        s = self.settings
        extra = {} if s.safety_checker else {"safety_checker": None, "requires_safety_checker": False}
        pipe = DiffusionPipeline.from_pretrained(s.lcm_model, dtype=dtype, local_files_only=True, **extra)
        if not isinstance(pipe.scheduler, LCMScheduler):
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        return pipe

    def _build_sdxl(self, dtype):
        from diffusers import (
            AutoencoderKL,
            DPMSolverMultistepScheduler,
            EulerDiscreteScheduler,
            StableDiffusionXLPipeline,
        )
        from huggingface_hub import hf_hub_download

        s = self.settings
        components = {}
        if s.vae_model:
            components["vae"] = AutoencoderKL.from_pretrained(s.vae_model, dtype=dtype, local_files_only=True)
        pipe = StableDiffusionXLPipeline.from_pretrained(
            s.base_model, dtype=dtype, variant=s.base_variant, local_files_only=True, **components
        )
        if self.profile == "lightning":
            lora = hf_hub_download(s.lightning_repo, self._lightning_file(), local_files_only=True)
            pipe.load_lora_weights(lora, adapter_name="lightning")
            pipe.fuse_lora(lora_scale=1.0)
            pipe.unload_lora_weights()  # keeps the fused weights, frees the LoRA layers
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        else:
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++"
            )
        return pipe

    # -- generation -------------------------------------------------------------------------------
    def generate(
        self, request: GenerationRequest, progress: ProgressFn | None = None, cancel: threading.Event | None = None
    ) -> tuple[Image.Image, GenerationInfo]:
        import torch
        from diffusers.image_processor import IPAdapterMaskProcessor

        self.load()
        pipe = self._pipe
        assert pipe is not None
        s = self.settings
        if self.family == "sd15":  # latent-consistency sampling: few steps, guidance is embedded (no negatives)
            steps = request.steps or s.lcm_steps
            cfg = s.lcm_guidance
            with_negative = False
        elif self.profile == "lightning":
            steps, cfg, with_negative = request.steps or s.lightning_steps, 0.0, False
        else:
            steps = request.steps or s.quality_steps
            cfg = request.guidance_scale if request.guidance_scale is not None else 6.0
            with_negative = cfg > 1.0

        refs = [r for r in request.references if r.path or r.data_b64] if self._ip_loaded else []
        kwargs: dict = {}
        if not self._ip_loaded:
            mode = "none"
        elif not refs:
            mode = "none"
            pipe.set_ip_adapter_scale(0.0)  # a zero scale skips the IP-Adapter attention entirely
            kwargs["ip_adapter_image"] = self._gray
        elif len(refs) == 1 or any(r.region is None for r in refs):
            mode = "single" if len(refs) == 1 else "unmasked"
            images = [load_reference(r) for r in refs]
            # unmasked: the images' tokens are concatenated and share one float scale (diffusers multiplies the
            # attention output by `scale` directly; per-image scale lists are only valid with masks)
            pipe.set_ip_adapter_scale(request.ip_adapter_scale)
            kwargs["ip_adapter_image"] = [images] if len(images) > 1 else images[0]
        else:
            mode = "masked"
            images = [load_reference(r) for r in refs]
            masks = []
            for r in refs:
                m = Image.new("L", (request.width, request.height), 0)
                box = (
                    int(r.region.x0 * request.width),
                    int(r.region.y0 * request.height),
                    int(r.region.x1 * request.width),
                    int(r.region.y1 * request.height),
                )
                m.paste(255, box)
                masks.append(m)
            processed = IPAdapterMaskProcessor().preprocess(masks, height=request.height, width=request.width)
            processed = processed.reshape(1, processed.shape[0], processed.shape[2], processed.shape[3])
            pipe.set_ip_adapter_scale([[request.ip_adapter_scale] * len(images)])
            kwargs["ip_adapter_image"] = [images]
            kwargs["cross_attention_kwargs"] = {"ip_adapter_masks": [processed]}

        prompt = request.prompt + (f", {request.prompt_2}" if request.prompt_2 else "")
        prompt_embeds, pooled, neg, neg_pooled = encode_long_prompt(
            pipe, prompt, request.negative_prompt, pipe.device, with_negative
        )
        kwargs["prompt_embeds"] = prompt_embeds
        if pooled is not None:
            kwargs["pooled_prompt_embeds"] = pooled
        if with_negative:
            kwargs["negative_prompt_embeds"] = neg
            if neg_pooled is not None:
                kwargs["negative_pooled_prompt_embeds"] = neg_pooled

        def on_step(_pipe, step: int, _t, cb_kwargs):
            if cancel is not None and cancel.is_set():
                raise GenerationCancelled()
            if progress:
                progress(step + 1, steps)
            return cb_kwargs

        gen_device = "cpu" if self.device == "mps" else self.device
        generator = torch.Generator(device=gen_device).manual_seed(request.seed)
        t0 = time.perf_counter()
        with self._lock:
            result = pipe(
                num_inference_steps=steps,
                guidance_scale=cfg,
                width=request.width,
                height=request.height,
                generator=generator,
                callback_on_step_end=on_step,
                **kwargs,
            )
        flagged = getattr(result, "nsfw_content_detected", None)
        if flagged and flagged[0]:
            raise GenerationBlocked("The safety filter blocked this image. Try regenerating, or reword the scene.")
        image = result.images[0]
        duration = time.perf_counter() - t0
        peak = None
        if self.device == "mps":
            peak = round(torch.mps.driver_allocated_memory() / 1e9, 2)
            torch.mps.empty_cache()
        elif self.device == "cuda":
            peak = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        info = GenerationInfo(
            seed=request.seed,
            steps=steps,
            guidance_scale=cfg,
            width=request.width,
            height=request.height,
            profile=self.profile or "",
            device=self.device or "",
            model=self.model_name,
            reference_mode=mode,
            duration_s=round(duration, 2),
            peak_memory_gb=peak,
        )
        return image, info


# ------------------------------------------------------------------------------------- remote backend


class RemoteGenerator:
    """Client for worker/server.py running on a GPU machine you control."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.url = (settings.remote_generator_url or "").rstrip("/")

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.settings.worker_token}"} if self.settings.worker_token else {}

    def status(self) -> GeneratorStatus:
        import httpx

        if not self.url:
            return GeneratorStatus(
                backend="remote", available=False, loaded=False, reason="STORYPIX_REMOTE_GENERATOR_URL is not set."
            )
        try:
            r = httpx.get(f"{self.url}/v1/health", headers=self._headers(), timeout=5)
            r.raise_for_status()
            data = r.json()
            return GeneratorStatus(
                backend="remote",
                available=bool(data.get("available")),
                loaded=bool(data.get("loaded")),
                device=data.get("device"),
                profile=data.get("profile"),
                model=data.get("model"),
                reason=data.get("reason"),
            )
        except Exception as exc:  # network errors are reported, not raised
            return GeneratorStatus(
                backend="remote",
                available=False,
                loaded=False,
                reason=f"Remote worker unreachable at {self.url}: {exc}",
            )

    def load(self) -> None:
        status = self.status()
        if not status.available:
            raise GeneratorUnavailable(status.reason)

    def generate(
        self, request: GenerationRequest, progress: ProgressFn | None = None, cancel: threading.Event | None = None
    ) -> tuple[Image.Image, GenerationInfo]:
        import httpx

        payload = request.model_copy(deep=True)
        for ref in payload.references:
            if ref.path and not ref.data_b64:
                ref.data_b64 = base64.b64encode(Path(ref.path).read_bytes()).decode()
                ref.path = None
        r = httpx.post(f"{self.url}/v1/generate", json=payload.model_dump(), headers=self._headers(), timeout=900)
        if r.status_code != 200:
            raise RuntimeError(f"worker error {r.status_code}: {r.text[:300]}")
        data = r.json()
        image = Image.open(io.BytesIO(base64.b64decode(data["image_b64"]))).convert("RGB")
        if progress:
            progress(1, 1)
        return image, GenerationInfo(**data["info"])


class DisabledGenerator:
    def status(self) -> GeneratorStatus:
        return GeneratorStatus(
            backend="disabled",
            available=False,
            loaded=False,
            reason="Image generation is disabled (STORYPIX_GENERATOR=disabled).",
        )

    def load(self) -> None:
        raise GeneratorUnavailable(self.status().reason)

    def generate(self, request, progress=None, cancel=None):
        raise GeneratorUnavailable(self.status().reason)


def create_generator(settings: Settings) -> ImageGenerator:
    if settings.generator == "remote":
        return RemoteGenerator(settings)
    if settings.generator == "disabled":
        return DisabledGenerator()
    return LocalDiffusersGenerator(settings)
