# StoryPix — Architecture & Implementation Plan

This document is the design record for StoryPix. It covers the environment the project
was built in, the architecture, model and dataset choices (with verified licenses),
hardware requirements, the MVP scope, and the implementation plan.

Measured numbers (latency, memory, model metrics) live in [MODELS.md](MODELS.md) and
[PERFORMANCE.md](PERFORMANCE.md); this document only states design-time estimates and
says so where it does.

---

## 1. Environment inspected

| Item | Value |
|---|---|
| Machine | Apple M4 (10-core CPU, 10-core GPU, Metal 4), **16 GB unified memory** |
| OS | macOS 27.2 |
| Free disk | ~140 GB |
| Python | Homebrew 3.15 (too new for PyTorch wheels) → project uses **Python 3.12 via `uv`** |
| Node | v24.9 (npm 11, pnpm, bun available) |
| CUDA GPU | none locally → local inference uses **PyTorch MPS** |

Library versions at time of writing: torch 2.15, diffusers 0.42, transformers 5.22,
FastAPI 0.162, Pydantic 2.14, Next.js 16.4, React 19.3, Tailwind 4.3, shadcn CLI 4.x.

---

## 2. System overview

```
┌──────────────────────────┐      REST + SSE      ┌──────────────────────────────────────┐
│ Next.js 16 frontend       │ ───────────────────▶ │ FastAPI backend                       │
│ (App Router, shadcn/ui)   │ ◀─────────────────── │  api/  → job_manager (async queue)    │
└──────────────────────────┘   progress events    │  story pipeline (CPU, < 1 s)          │
                                                   │  image pipeline ──┐                   │
                                                   └───────────────────┼──────────────────┘
                                                                       │ ImageGenerator interface
                                       ┌───────────────────────────────┴───────────────┐
                                       │ LocalDiffusersGenerator (MPS / CUDA / CPU)      │
                                       │ RemoteGenerator → worker/ (same code, GPU box)  │
                                       └───────────────────────────────────────────────┘
```

Every stage exchanges **Pydantic models serialised as JSON**, so each stage can be run and
tested on its own (`python -m app.cli analyze story.txt` prints the full analysis JSON).

### 2.1 Pipeline stages

| # | Stage | Module | Kind |
|---|---|---|---|
| 1 | Text normalisation, sentence + clause ("event unit") splitting | `services/text_units.py` | spaCy parse + rules |
| 2 | Character / place mention tagging | `models/character_extractor.py` | **custom PyTorch model B** (BiLSTM-CRF over frozen MiniLM) |
| 3 | Mention clustering + pronoun resolution | `services/character_manager.py` | deterministic resolver |
| 4 | Character profile creation (appearance) | `services/character_manager.py`, `services/appearance.py` | explicit text attributes + curated appearance priors |
| 5 | Scene boundary detection + segmentation decoding | `models/scene_segmenter.py`, `services/story_parser.py` | **custom PyTorch model A** + exact top-k / DP decoding |
| 6 | Scene understanding (actions, setting, time, emotion, composition, title) | `services/scene_builder.py` | dependency parse + lexicons + rules |
| 7 | Scene prompt generation + style conditioning | `services/prompt_builder.py`, `services/styles.py` | token-budgeted prompt assembly |
| 8 | Reference conditioning (character anchors, uploads) | `services/image_generator.py` | IP-Adapter Plus (masked multi-reference) |
| 9 | Image generation (sequential, seeded) | `services/image_generator.py` | SDXL (+ SDXL-Lightning LoRA) via Diffusers |
| 10 | Evaluation + automatic retry/rerank | `services/image_evaluator.py`, `models/scene_alignment.py`, `models/quality_evaluator.py` | **custom PyTorch models C, D** + DINOv2 identity similarity |
| 11 | Ordered storybook output | `services/exporter.py` | PDF (reportlab) + ZIP + manifest |

### 2.2 Why not an LLM for story understanding?

A local instruction-tuned LLM (e.g. Qwen3-1.7B, Apache-2.0) could write richer scene
descriptions, but on a 16 GB Mac it competes with SDXL for memory, adds 30–60 s per
story, and is not "our" model. The MVP therefore uses **small trained models plus explicit
linguistic processing**: analysis runs in well under a second on CPU, is deterministic,
and every decision is inspectable. An optional LLM enrichment stage is listed under
advanced features.

---

## 3. Image generation model selection

Requirements: open weights, a license that permits commercial use, Diffusers support,
mature **image-reference conditioning**, and the ability to run on a 16 GB Apple-silicon Mac
and on a free 16 GB T4.

| Model | License (verified on HF) | Params | Ref. conditioning | Fits 16 GB Mac? | Decision |
|---|---|---|---|---|---|
| **SDXL base 1.0** | CreativeML **OpenRAIL++-M** (commercial OK, use restrictions) | 3.5 B | IP-Adapter (mature), ControlNet | Yes (fp16 ≈ 7 GB) | **Chosen base** |
| **SDXL-Lightning** (ByteDance) 4/8-step LoRA | **OpenRAIL++** | LoRA | inherits SDXL | Yes | **Chosen for fast mode** |
| **IP-Adapter** (h94) SDXL Plus, ViT-H | **Apache-2.0** | ~0.85 B incl. encoder | — | Yes | **Chosen** |
| sdxl-vae-fp16-fix | **MIT** | 84 M | — | Yes | **Chosen** (avoids NaNs/black images in fp16) |
| SDXL-Turbo | sai-nc-community (**non-commercial**) | 3.5 B | — | Yes | Rejected (license) |
| SD 3.5 Medium | Stability Community License (gated, revenue cap) | 2.5 B + T5 | limited | Tight | Rejected (license terms, ecosystem) |
| FLUX.1-schnell | Apache-2.0 (gated: auto) | 12 B | community adapters for *dev* (non-commercial) | No | Rejected (size) |
| Z-Image-Turbo | Apache-2.0 | 6 B + 4 B text encoder | none mature | No (without heavy quantisation) | Documented alternative for big GPUs |
| SD 1.5 | CreativeML OpenRAIL-M | 1 B | IP-Adapter | Yes (fast) | Fallback profile for ≤ 8 GB machines |

### 3.1 Character consistency strategy

Repeating text descriptions does **not** guarantee the same-looking character, so StoryPix
layers four mechanisms:

1. **Canonical character profiles.** Each recurring character gets one fixed descriptor
   (species, colours, features, clothing) that is injected verbatim into every scene prompt.
   Explicit attributes from the story override curated priors.
2. **Character anchors (reference sheets).** Before any scene is drawn, each recurring
   character gets a portrait in the chosen style (or a stylised version of the user's uploaded
   reference). Anchors are visible and regenerable in the UI.
3. **IP-Adapter Plus reference conditioning.** Scenes are conditioned on the anchors of the
   characters present. For multi-character scenes, **IP-Adapter attention masks** restrict each
   reference to its own region (a left/right or thirds layout that is also stated in the prompt),
   which reduces cross-character attribute bleeding. An ablation against the unmasked variant
   is documented in MODELS.md.
4. **Measure, then retry.** After generation, a DINOv2 identity similarity between each anchor
   and the generated scene, plus the trained alignment/quality models, decide whether to
   automatically retry with a new seed (best-of-k on GPU).

Seeds are derived deterministically per story and scene, so any image can be reproduced, and
"keep composition" regeneration reuses the seed with an edited prompt.

ControlNet (pose/depth) was considered for composition control. It needs a pose/depth source
per scene, which we do not have for arbitrary animals and objects, so it is an advanced feature.
InstantID/PhotoMaker were rejected for the MVP: they target human faces and InstantID depends on
insightface models with a non-commercial license.

---

## 4. Custom PyTorch models

All four models use **frozen pretrained encoders** (permissively licensed) and **task-specific
layers we design, train and evaluate ourselves**. Full specs (tensor shapes, losses, metrics,
results) are in [MODELS.md](MODELS.md).

| Model | Task | Pretrained encoder (frozen) | Our layers | Training data |
|---|---|---|---|---|
| **A. Scene Boundary Detector** | P(new visual scene starts at event unit *i*) | all-MiniLM-L6-v2 (Apache-2.0) sentence embeddings | feature projection → 2-layer Transformer encoder over the story → pairwise boundary MLP | TinyStories (CDLA-Sharing-1.0) with weak supervision (paragraph structure + labelling functions); gold eval set annotated for this project |
| **B. Character & Place Tagger** | BIO tags: CHAR / PLACE spans | all-MiniLM-L6-v2 token states | casing embedding → BiLSTM → linear emissions → **CRF** (own implementation) | LitBank (CC BY 4.0) entity layer + TinyStories silver labels |
| **C. Scene–Image Alignment** | How well an image depicts a text description | SigLIP 2 base (Apache-2.0) text + image embeddings | interaction features → MLP regressor | ImageRewardDB (Apache-2.0) human alignment ratings |
| **D. Image Quality Evaluator** | Perceived fidelity / artifact-free quality | SigLIP 2 + DINOv2-small (Apache-2.0) image embeddings + OpenCV statistics | fusion MLP regressor | ImageRewardDB human fidelity ratings |

Losses: A — class-weighted binary cross-entropy; B — CRF negative log-likelihood; C and D —
MSE on normalised ratings + within-prompt pairwise ranking (Bradley–Terry) loss.

Character **identity consistency** uses DINOv2 cosine similarity between the anchor and the
character's region of the scene. This is a zero-shot metric, not a trained model, and is
labelled that way everywhere.

### 4.1 Datasets and licenses

| Dataset | License | Used for | Notes |
|---|---|---|---|
| TinyStories (roneneldan/TinyStories) | CDLA-Sharing-1.0 | Model A (weak labels), Model B (silver labels) | Synthetic children's stories, a good domain match. Trained weights are "Results" under CDLA and not share-alike |
| LitBank (dbamman/litbank) | CC BY 4.0 | Model B | Public-domain literature with nominal + named entity mentions. Attribution in MODELS.md |
| ImageRewardDB (zai-org/ImageRewardDB) | Apache-2.0 | Models C, D | Subset used (≈ 5 GB of 23.7 GB) |
| Aesop's Fables (Project Gutenberg) | Public domain (US) | Model A gold evaluation | Gutenberg header/footer stripped |
| StoryPix gold scene set | Project-owned | Model A evaluation | Annotated for this project with AI assistance, documented in MODELS.md |

Rejected for licensing: ROCStories (request-only terms), VIST (research terms), COCO/Flickr30k
images (mixed or non-commercial image licenses), AVA, JourneyDB (research only),
Pick-a-Pic (gated), NRC emotion lexicon (non-commercial).

---

## 5. Hardware and memory

Design-time estimates (fp16). Measured values replace these in PERFORMANCE.md.

| Component | Weights in memory |
|---|---|
| SDXL UNet | ~5.1 GB |
| SDXL text encoders (CLIP-L + OpenCLIP bigG) | ~1.6 GB |
| VAE fp16-fix | ~0.2 GB |
| IP-Adapter Plus + ViT-H image encoder | ~2.0 GB |
| Evaluators (SigLIP 2 base, DINOv2-small) + MiniLM | ~0.9 GB |
| **Total** | **≈ 9.8 GB** plus 1–3 GB activations at ~1 MP |

| Target | Feasible? | Profile | Expected speed (estimate) |
|---|---|---|---|
| Apple silicon, 16 GB | Yes, primary dev target | `lightning` (4 steps, CFG 0) at ~1 MP | tens of seconds per image |
| Apple silicon, 8 GB | Marginal | `sd15` profile (512–768 px) | — |
| NVIDIA ≥ 12 GB VRAM | Yes | `quality` (25–30 steps, CFG ~6, best-of-k) or `lightning` | seconds per image |
| Free Colab / Kaggle T4 (16 GB) | Yes, notebook batch mode | `lightning` or `quality` | seconds per image; sessions are time-limited and GPU availability is not guaranteed |
| CPU only | Technically, not recommended | `lightning` | many minutes per image |

**Local vs GPU.** Story analysis, the four custom models (training and inference), the API and the
frontend all run locally on CPU/MPS. Image generation runs locally on Apple silicon (slower)
or on any CUDA GPU. Free notebooks run the same pipeline in **batch mode** through the CLI. We do
not tunnel a public API out of free notebooks, because that can conflict with the free tiers'
usage policies. The `RemoteGenerator` is intended for a GPU machine you control.

---

## 6. MVP vs advanced

**MVP (this implementation)**
- Story input with limits, 8 styles with real generated previews, scene count (Auto/3/5/8/10),
  aspect ratio, optional per-character reference uploads.
- Story analysis: event units → trained boundary model → scenes; trained tagger → characters →
  profiles; scene details and titles.
- SDXL + Lightning generation on MPS/CUDA, character anchors, masked IP-Adapter conditioning,
  sequential generation with continuity, deterministic seeds.
- Async jobs (SQLite persistence), SSE progress, cancel, retry of failed scenes.
- Scene regeneration with free-text instructions ("keep composition" or "new variation"),
  version history with before/after comparison.
- Evaluation scores (alignment, quality, identity consistency) with automatic retry.
- Downloads: single image, ZIP (images + manifest with prompts, seeds and scores), PDF storybook.
- Training, evaluation and data-prep scripts; trained checkpoints; tests; README, API and model docs;
  CLI batch mode; free-GPU notebook.

**Advanced (documented, not in MVP)**
- InstantStyle-style second adapter for scene-to-scene palette continuity.
- ControlNet pose/depth or regional prompting for precise multi-character layout.
- Per-story character LoRA fine-tuning from anchors (GPU).
- Optional local LLM enrichment of scene descriptions.
- Learned coreference model, emotion classifier, NSFW image classifier.
- Multi-user accounts, object storage, horizontal workers.

---

## 7. API surface (summary)

Full reference: [API.md](API.md) and the live OpenAPI docs at `/docs`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | device, generator availability, model checkpoints loaded |
| GET | `/api/styles` | style catalogue |
| POST | `/api/story/analyze` | synchronous analysis → scenes + characters (no images) |
| POST | `/api/uploads` | upload a reference image → `upload_id` |
| POST | `/api/jobs` | create a generation job (202, returns job) |
| GET | `/api/jobs`, `/api/jobs/{id}` | list / get job state |
| GET | `/api/jobs/{id}/events` | Server-Sent Events progress stream |
| POST | `/api/jobs/{id}/cancel`, `/resume` | control |
| POST | `/api/jobs/{id}/scenes/{n}/regenerate` | regenerate one scene with optional instructions |
| POST | `/api/jobs/{id}/scenes/{n}/select` | choose which version goes into the storybook |
| PATCH | `/api/jobs/{id}/characters/{cid}` | edit a character's appearance |
| POST | `/api/jobs/{id}/characters/{cid}/regenerate` | regenerate a character anchor |
| GET | `/api/jobs/{id}/export.pdf`, `/export.zip` | storybook downloads |

---

## 8. Implementation plan

1. **Backend skeleton:** settings, schemas, SQLite job store, event bus, API routes, CLI.
2. **Story pipeline:** spaCy units, character resolver + appearance priors, scene builder, prompt
   builder, with heuristic fallbacks so the pipeline runs before any training.
3. **Image pipeline:** model download script, SDXL + Lightning + IP-Adapter on MPS, anchors, masked
   multi-reference, regeneration; benchmark latency and memory on the M4.
4. **Job orchestration:** sequential GPU worker, progress events, cancel/resume/retry, exports.
5. **Frontend:** Next.js + shadcn/ui two-panel app, SSE hook, viewer, regeneration dialog, comparison,
   downloads; style previews generated by the real pipeline.
6. **Models A–D:** data preparation, training, evaluation against baselines, checkpoints, model
   cards; wire them into the pipeline (replacing the heuristic fallbacks when checkpoints exist).
7. **Quality pass:** tests, docs, end-to-end run on the example story with visual inspection.

---

## 9. Revised deployment decision

The original plan ran SDXL on the developer's Apple-silicon Mac. Two constraints from the product owner changed that:
the app must be **deployable for free**, and nothing large (≈ 12 GB of SDXL weights) should be stored on the
developer's machine. Free hosting has no GPU, so the design now has two image-generation families behind the same
`ImageGenerator` interface:

| Profile | Model family | Where | Speed | Output |
|---|---|---|---|---|
| `lightning` / `quality` | SDXL + IP-Adapter Plus (SDXL) | Apple silicon / NVIDIA GPU you own | seconds per image | ~1 MP |
| `lcm` | LCM-DreamShaper v7 (MIT) + IP-Adapter Plus (SD1.5) | any CPU, e.g. a free Hugging Face Space | tens of seconds per image | 512 px class |

Deployment: **Vercel** serves the static Next.js app (built from GitHub); a **free Hugging Face Space** (Docker,
CPU basic: 2 vCPU, 16 GB RAM) runs the FastAPI backend, the four custom models and the `lcm` image profile. Weights
download inside the Space at first start. Public-deployment settings: the Stable Diffusion safety checker is on,
automatic re-draws are off (they would double the wait on a CPU) and at most two storybooks run at once.

Alternatives considered: HF ZeroGPU (fast, but needs a paid PRO account), free notebook GPUs (sessions expire and
serving a public API conflicts with their terms), external image APIs (paid, or not our own PyTorch pipeline), and
other free container hosts (512 MB–1 GB RAM, too small for PyTorch).
