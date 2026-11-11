"""Download the raw training data for StoryPix's custom models (stdlib only, resumable, licence-checked).

    python -m training.datasets.download_data text      # TinyStories subset + LitBank  (~85 MB)
    python -m training.datasets.download_data images [N] # ImageRewardDB metadata + N zips (~1.7 GB for 3)

Licences (verified 2026-11 on the dataset pages):
  * TinyStories (roneneldan/TinyStories)  — CDLA-Sharing-1.0
  * LitBank (dbamman/litbank)             — CC BY 4.0
  * ImageRewardDB (zai-org/ImageRewardDB) — Apache-2.0
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
HF = "https://huggingface.co/datasets"

TINYSTORIES = [
    # (url, local name, max bytes or None for the full file)
    (f"{HF}/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt", "TinyStoriesV2-GPT4-valid.txt", None),
    # The train file is 2.2 GB; the first 60 MB (~70k stories) is plenty for weak supervision.
    (
        f"{HF}/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt",
        "TinyStoriesV2-GPT4-train.head60mb.txt",
        60_000_000,
    ),
]
LITBANK_API = "https://api.github.com/repos/dbamman/litbank/contents/entities/tsv"
IMAGEREWARD = [
    f"{HF}/zai-org/ImageRewardDB/resolve/main/metadata-train.parquet",
    f"{HF}/zai-org/ImageRewardDB/resolve/main/metadata-validation.parquet",
    f"{HF}/zai-org/ImageRewardDB/resolve/main/metadata-test.parquet",
    f"{HF}/zai-org/ImageRewardDB/resolve/main/images/validation/validation_2.zip",
    f"{HF}/zai-org/ImageRewardDB/resolve/main/images/train/train_1.zip",
    f"{HF}/zai-org/ImageRewardDB/resolve/main/images/test/test_2.zip",
]


def fetch(url: str, dest: Path, max_bytes: int | None = None, attempts: int = 30) -> None:
    """Resumable download: keeps a .part file and continues with HTTP Range requests after network stalls."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (max_bytes is None or dest.stat().st_size >= max_bytes):
        print(f"  ✓ {dest.name} (cached)")
        return
    part = dest.with_suffix(dest.suffix + ".part")
    expected = max_bytes
    for attempt in range(1, attempts + 1):
        have = part.stat().st_size if part.exists() else 0
        if expected is not None and have >= expected:
            break
        end = "" if max_bytes is None else str(max_bytes - 1)
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-{end}", "User-Agent": "storypix-data/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp, open(part, "ab") as fh:
                if have and resp.status != 206:  # server ignored the range → start over
                    fh.truncate(0)
                    have = 0
                content_range = resp.headers.get("Content-Range", "")
                if max_bytes is None and "/" in content_range:
                    expected = int(content_range.rsplit("/", 1)[1])
                elif max_bytes is None and resp.headers.get("Content-Length"):
                    expected = have + int(resp.headers["Content-Length"])
                while chunk := resp.read(1 << 20):
                    fh.write(chunk)
            # A dropped connection can end the stream early without raising — only stop once complete.
            if expected is None or part.stat().st_size >= expected:
                break
            print(
                f"  … {dest.name}: stream ended at {part.stat().st_size / 1e6:.0f}/{expected / 1e6:.0f} MB, resuming",
                flush=True,
            )
        except urllib.error.HTTPError as exc:
            if exc.code == 416:  # range not satisfiable → already complete
                break
            raise
        except (TimeoutError, OSError) as exc:
            wait = min(60, 2 * attempt)
            print(
                f"  … {dest.name}: {exc.__class__.__name__} at {have / 1e6:.0f} MB, retry {attempt} in {wait}s",
                flush=True,
            )
            time.sleep(wait)
    else:
        raise RuntimeError(f"giving up on {url} after {attempts} attempts")
    part.rename(dest)
    print(f"  ✓ {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)", flush=True)


def download_text() -> None:
    print("TinyStories (CDLA-Sharing-1.0)")
    for url, name, limit in TINYSTORIES:
        fetch(url, RAW / "tinystories" / name, limit)
    print("LitBank entities (CC BY 4.0)")
    with urllib.request.urlopen(urllib.request.Request(LITBANK_API, headers={"User-Agent": "storypix"})) as r:
        listing = json.load(r)
    for item in listing:
        if item["name"].endswith(".tsv"):
            fetch(item["download_url"], RAW / "litbank" / item["name"])


def download_images(max_zips: int = 3) -> None:
    print("ImageRewardDB subset (Apache-2.0)")
    zips = [u for u in IMAGEREWARD if u.endswith(".zip")][:max_zips]
    for url in [u for u in IMAGEREWARD if not u.endswith(".zip")] + zips:
        fetch(url, RAW / "imagereward" / url.rsplit("/", 1)[-1])


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "text"
    if which == "images":
        download_images(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    else:
        download_text()
