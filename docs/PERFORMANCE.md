# StoryPix — Measured Performance

Reference machine: Apple M4 (10-core CPU / 10-core GPU), 16 GB unified memory, macOS 27.2, PyTorch 2.15 (MPS),
Diffusers 0.42. Every number on this page was measured on that machine; nothing is estimated.

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

## Image generation (MPS)

_Pending: the SDXL weights are still downloading. This section is filled in by `python -m app.cli benchmark` once they are available._
