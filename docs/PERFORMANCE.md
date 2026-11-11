# StoryPix — Measured Performance

Story understanding and training were measured on an Apple M4 (16 GB, macOS 27.2, PyTorch 2.15); image generation on
the live Modal deployment. Every number on this page was measured; nothing is estimated except where stated.

## Story understanding (CPU)

All four custom models and the full analysis pipeline run on CPU; the GPU stays free for image generation.

| Metric | Value |
|---|---|
| Analyzer start-up (spaCy + MiniLM + Models A and B) | 1.1 s |
| Analysis latency per story (38 gold stories, mean 114 words) | median **84 ms**, p95 110 ms, max 115 ms |
| Peak resident memory of the analysis process | 0.54 GB |

## Training (MPS / CPU)

| Model | Device | Time |
|---|---|---|
| A: scene boundary detector (29.5k stories, 8 epochs) | MPS | ≈ 35 s/epoch → under 5 min |
| B: mention tagger (12 epochs) | CPU | ≈ 11 s/epoch → about 2.5 min |
| C + D: evaluator heads (precomputed features) | CPU | seconds |
| Feature extraction for C/D (SigLIP + DINOv2 + OpenCV statistics) | MPS | about 12 images/s (1,018 images in 87 s) |

## Image generation — live deployment (Modal, CPU)

Measured on the deployed backend (`STORYPIX_MODAL_GPU=none`: 8 vCPU, 16 GiB, LCM-DreamShaper v7 + IP-Adapter Plus
SD1.5 + safety checker, 4 steps; 512×512 portraits, 640×480 scenes), example story, 2 portraits + 5 scenes:

| Metric | Value |
|---|---|
| Container cold start → first `/api/health` answer | ≈ 14 s |
| Image model ready on first ever start (≈ 8 GB weights fetched into a Modal Volume, then loaded) | ≈ 60 s |
| Time per image (diffusion + safety check + evaluator scoring) | 16–21 s |
| Whole storybook, model already loaded | 152 s |
| Whole storybook including the first model load | 209 s |
| Compute price while the container runs (8 vCPU + 16 GiB at Modal list prices) | ≈ $0.50 / hour |
| Idle cost | $0 (scales to zero after 4 idle minutes) |
| Month to date after building, experiments and several storybooks (`/api/usage`) | $0.43 of the $30 free credit |

## Image generation — local SDXL (Apple silicon)

Not measured: SDXL weights (≈ 12 GB) were not downloaded to the development machine, at the project owner's request.
`python -m app.cli benchmark` reports load time, seconds per image and peak memory on any machine that has them.
