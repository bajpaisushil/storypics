"""Deploy the StoryPix backend to a free Hugging Face Space (Docker SDK, CPU basic).

    hf auth login                                  # once, with a token that has write access
    python scripts/deploy_space.py [--name storypix] [--cors-regex 'https://storypics.*\\.vercel\\.app']

Uploads only what the server needs (app code, the small trained checkpoints, Dockerfile); model weights are
downloaded by the container itself on Hugging Face's infrastructure. Prints the public API URL to use as the
frontend's NEXT_PUBLIC_API_URL.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

BACKEND = Path(__file__).resolve().parents[1]
SPACE_README = """---
title: StoryPix API
emoji: 📖
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Story-to-illustrated-storybook backend (FastAPI + PyTorch)
---

# StoryPix API

Backend of [StoryPix](https://github.com/bajpaisushil/storypics): story understanding with four custom PyTorch
models and illustration with LCM-DreamShaper v7 + IP-Adapter on CPU. Interactive API docs: `/docs`.

This free CPU Space generates real images but slowly (tens of seconds per image) and sleeps when idle; the first
request after a sleep waits for the container to start and the model to load.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="storypix")
    ap.add_argument("--cors-regex", default=None, help="allowed frontend origins, e.g. your Vercel domain(s)")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    api = HfApi()
    user = api.whoami()["name"]
    repo_id = f"{user}/{args.name}"
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True)
    if args.cors_regex:
        api.add_space_variable(repo_id, "STORYPIX_CORS_ORIGIN_REGEX", args.cors_regex)
    api.upload_file(
        path_or_fileobj=SPACE_README.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="space",
        commit_message="Space card",
    )
    api.upload_folder(
        folder_path=BACKEND,
        repo_id=repo_id,
        repo_type="space",
        commit_message="Deploy StoryPix backend",
        allow_patterns=[
            "Dockerfile",
            ".dockerignore",
            "requirements.txt",
            "app/**",
            "checkpoints/**",
            "worker/**",
            "scripts/**",
        ],
        ignore_patterns=["**/__pycache__/**", "*.pyc"],
    )
    host = repo_id.replace("/", "-").replace("_", "-").lower()
    print(f"Space:   https://huggingface.co/spaces/{repo_id}")
    print(f"API URL: https://{host}.hf.space   (set NEXT_PUBLIC_API_URL to this in Vercel)")


if __name__ == "__main__":
    main()
