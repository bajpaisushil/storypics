# StoryPix backend — CPU image for a free Hugging Face Space (Docker SDK) or any Docker host.
# Model weights are downloaded inside the container at first start (never on the developer's machine).
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/user/.cache/huggingface
WORKDIR /home/user/app

# CPU build of PyTorch first (avoids pulling multi-GB CUDA wheels), then the rest.
COPY --chown=user requirements.txt .
RUN pip install --user torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --user -r requirements.txt

COPY --chown=user app ./app
COPY --chown=user checkpoints ./checkpoints
COPY --chown=user worker ./worker
COPY --chown=user scripts ./scripts

# Server profile: LCM-DreamShaper (SD1.5 family) on CPU, weights fetched on first start, one storybook at a time
# per CPU, safety checker on (public demo), no automatic re-draws (they double the wait on a CPU).
ENV STORYPIX_PROFILE=lcm \
    STORYPIX_DEVICE=cpu \
    STORYPIX_ALLOW_DOWNLOADS=true \
    STORYPIX_PRELOAD_MODELS=true \
    STORYPIX_SAFETY_CHECKER=true \
    STORYPIX_AUTO_RETRY=false \
    STORYPIX_MAX_ACTIVE_JOBS=2 \
    STORYPIX_DATA_DIR=/home/user/app/data

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
