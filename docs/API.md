# StoryPix API

Base URL: `http://localhost:8000`. The interactive OpenAPI docs are at `/docs` (Swagger) and `/redoc`.
All bodies are JSON unless noted. Errors return `{"detail": "<human-readable message>"}`.

| Status | Meaning |
|---|---|
| 400 | invalid request (e.g. unknown scene, regenerating a scene that is already running) |
| 401 | an access key is configured and `X-StoryPix-Key` is missing or wrong (optional feature) |
| 404 | job not found |
| 409 | nothing to export yet |
| 422 | validation error (story too short, unknown style, scene count out of range …) |
| 429 | `STORYPIX_MAX_ACTIVE_JOBS` reached |
| 503 | image generator unavailable; `detail` says why (e.g. weights not downloaded) and how to fix it |

---

## System

### `GET /api/health`
```json
{
  "status": "ok",
  "generator": {"backend": "local", "available": true, "loaded": false, "device": "mps",
                "profile": "lightning", "model": "stabilityai/stable-diffusion-xl-base-1.0", "reason": null},
  "analyzer": {"segmenter": "sbd-transformer (scene_segmenter.pt)", "tagger": "bilstm-crf (character_tagger.pt) + lexicon"},
  "evaluator": {"enabled": true, "loaded": false, "alignment": "model-c", "quality": "model-d",
                "consistency": "dinov2-zero-shot", "error": null},
  "queue_size": 0, "requires_key": false, "max_active_jobs": 0
}
```

### `GET /api/styles`
`[{"id": "watercolor", "name": "Watercolor", "tagline": "Soft washes on textured paper", "preview_url": "/styles/watercolor.webp"}, …]`
Style ids: `cartoon`, `realistic`, `anime`, `watercolor`, `3d`, `storybook`, `digital_painting`, `comic`.

---

## Story analysis (no images)

### `POST /api/story/analyze`
```json
{"story": "Once upon a time, there was a hen who lived on a farm. …", "scene_count": 5}
```
`scene_count` is `"auto"` or 1–12. The response is a `StoryAnalysis`:
```json
{
  "title": "The Hen and the Cow",
  "scenes": [{
    "index": 2, "title": "The Hen Meets the Cow",
    "text": "One day, she met a cow.", "summary": "One day, the hen met a cow.",
    "unit_indices": [1], "character_ids": ["c1", "c2"],
    "actions": [{"actor_ids": ["c1"], "verb": "meet", "phrase": "meeting a cow", "short_phrase": "meeting a cow",
                 "target_ids": ["c2"], "subject_text": null}],
    "location": {"name": "farm", "description": "a sunny farmyard with a red barn, a wooden fence and green meadows",
                 "carried_over": true},
    "time_of_day": "day", "weather": null, "emotion": "neutral", "mood": "pleasant atmosphere",
    "composition": "medium shot of the two characters together, eye-level view",
    "visual_description": "the hen meeting a cow",
    "previous_context": "Previous scene: The Hen on the Farm. …", "boundary_confidence": 0.74
  }],
  "characters": [{
    "id": "c1", "name": "Hen", "species": "hen", "category": "animal", "gender": "female",
    "attributes": {"Color": "warm orange-brown feathers", "Comb": "small red comb", "Beak": "short yellow beak",
                   "Eyes": "round shiny eyes", "Build": "small plump body"},
    "explicit_traits": [], "personality": [],
    "descriptor": "a hen with warm orange-brown feathers, a small red comb, a short yellow beak and a small plump body",
    "mention_count": 5, "mentions": [{"text": "a hen", "start": 27, "end": 32, "unit_index": 0, "kind": "nominal", "source": "tagger"}],
    "scene_indices": [1, 2, 3, 4, 5], "is_recurring": true, "reference_upload_id": null
  }],
  "units": [{"index": 0, "text": "Once upon a time, there was a hen who lived on a farm.", "start": 0, "end": 55,
             "sentence_index": 0, "paragraph_index": 0, "is_continuation": false}],
  "boundary_probabilities": [0.74, 0.19, 0.09, 0.89],
  "warnings": [],
  "meta": {"segmenter": "…", "tagger": "…", "unit_count": 5, "word_count": 51, "requested_scenes": 5,
           "timings_ms": {"parse_ms": 3.3, "characters_ms": 4.1, "segmentation_ms": 4.6, "scenes_ms": 0.4}}
}
```

### `POST /api/uploads` (multipart, field `file`)
Character reference image: PNG, JPEG or WebP, at most 10 MB and at least 64 px. It is normalised to RGB and at most 1024 px.
→ `{"upload_id": "3f2a…", "url": "/uploads/3f2a….png", "width": 1024, "height": 768}`

---

## Jobs (asynchronous generation)

### `POST /api/jobs` → `202 Accepted`
```json
{
  "story": "…", "style": "watercolor", "scene_count": "auto", "aspect_ratio": "4:3",
  "references": [{"upload_id": "3f2a…", "character_name": "Hen"}],
  "seed": 42, "title": null
}
```
Analysis runs right away; the response is the full `Job` with every task `queued`. A single background worker then
draws, in order: one **anchor portrait** per recurring character (stylised from the upload if one was given), then
each scene. Without a usable generator the call returns **503** and no job is created.

### `GET /api/jobs/{id}`: the `Job`
```json
{
  "id": "a1b2c3d4e5f60718", "status": "generating", "progress": 0.43, "title": "The Hen and the Cow",
  "story": "…", "settings": {"style": "watercolor", "aspect_ratio": "4:3", "scene_count": "auto", "seed": 42,
                            "width": 1152, "height": 896, "profile": "lightning"},
  "current": {"kind": "scene", "target": "2", "label": "Illustrating scene 2", "step": 3, "total_steps": 8, "attempt": 1},
  "characters": [{"…CharacterProfile fields…": "", "anchor_status": "completed", "selected_anchor": "v1",
                  "anchor_versions": [{"id": "v1", "url": "/media/a1b2…/char_c1_v1.png", "thumb_url": "…_thumb.webp", "…": ""}]}],
  "scenes": [{"…Scene fields…": "", "status": "completed", "selected_version": "v1", "error": null,
              "versions": [{"id": "v1", "url": "/media/a1b2…/scene_01_v1.png", "thumb_url": "/media/a1b2…/scene_01_v1_thumb.webp",
                            "mode": "initial", "seed": 49713, "prompt": "watercolor painting, the hen living on a farm, …",
                            "negative_prompt": "…", "instructions": null, "duration_s": 31.2, "attempts": 1,
                            "scores": {"alignment": 0.71, "quality": 0.64, "consistency": {"c1": 0.83}, "overall": 0.73},
                            "generation": {"steps": 8, "guidance_scale": 0, "reference_mode": "single", "peak_memory_gb": 11.2, "…": ""}}]}],
  "warnings": [], "error": null, "analysis": {"…AnalysisMeta…": ""}
}
```
Job `status`: `queued · generating · completed · partial · failed · cancelled · interrupted` (interrupted = the server
restarted mid-job; use *resume*). Task status: `pending · queued · generating · completed · failed · cancelled · skipped`.

### `GET /api/jobs`
Recent jobs: `[{"id", "title", "status", "style", "scene_count", "created_at", "cover_url"}]`.

### `GET /api/jobs/{id}/events`: Server-Sent Events
```
event: job        data: {"type": "job", "job": {…full Job…}}             ← on connect and after every state change
event: progress   data: {"type": "progress", "current": {…TaskProgress…}} ← every diffusion step
: keep-alive                                                              ← every 15 s
```

### Control
| Method & path | Body | Effect |
|---|---|---|
| `POST /api/jobs/{id}/cancel` | — | stops the running diffusion at the next step; queued work is cancelled |
| `POST /api/jobs/{id}/resume` | — | re-queues everything that failed / was cancelled / was interrupted |
| `DELETE /api/jobs/{id}` | — | deletes the job and its images |

### Scenes and characters
| Method & path | Body | Effect |
|---|---|---|
| `POST /api/jobs/{id}/scenes/{n}/regenerate` | `{"instructions": "Make the hen happier", "mode": "keep_composition"}` | new version of one scene; `keep_composition` reuses the seed, `new_variation` draws a new one; previous versions are kept |
| `POST /api/jobs/{id}/scenes/{n}/select` | `{"version_id": "v1"}` | chooses the version used in downloads |
| `PATCH /api/jobs/{id}/characters/{cid}` | `{"name"?, "descriptor"?, "attributes"?, "regenerate": true}` | edits a character; with `regenerate`, redraws its anchor and then every scene it appears in (same seeds) |
| `POST /api/jobs/{id}/characters/{cid}/regenerate` | — | redraws the anchor portrait only |

### Downloads
| Path | Returns |
|---|---|
| `GET /api/jobs/{id}/scenes/{n}/download[?version=v2]` | one scene image (PNG, `Content-Disposition: attachment`) |
| `GET /api/jobs/{id}/characters/{cid}/download` | a character reference portrait (PNG) |
| `GET /api/jobs/{id}/export.zip` | all selected scene images + character portraits + `story.txt` + `storybook.json` (prompts, seeds, scores, model settings) |
| `GET /api/jobs/{id}/export.pdf` | the storybook: cover, one page per scene with its text, cast page |
| `GET /media/...` | static image files (originals and WebP thumbnails) |

---

## Remote generation worker (optional)

`worker/server.py` exposes the generator on another machine you own (e.g. a desktop with an NVIDIA GPU):
`GET /v1/health`, `POST /v1/generate` (a `GenerationRequest` with references inlined as base64 → `{"image_b64", "info"}`),
protected by `Authorization: Bearer $STORYPIX_WORKER_TOKEN`. Point the web backend at it with
`STORYPIX_GENERATOR=remote` and `STORYPIX_REMOTE_GENERATOR_URL`.
