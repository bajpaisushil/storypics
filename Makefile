# StoryPix developer commands (macOS/Linux). Run `make help`.
PY := backend/.venv/bin/python

help:
	@grep -E '^[a-z-]+:.*?## ' Makefile | awk 'BEGIN{FS=":.*?## "}{printf "  %-16s %s\n", $$1, $$2}'

setup: ## create the Python 3.12 venv and install backend + frontend dependencies
	cd backend && uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements-dev.txt
	cd frontend && npm install

models: ## download pretrained weights (~12 GB, once)
	cd backend && .venv/bin/python scripts/download_models.py

backend: ## run the API on http://localhost:8000
	cd backend && .venv/bin/uvicorn app.main:app --port 8000

frontend: ## run the web app on http://localhost:3000
	cd frontend && npm run dev

test: ## backend tests + frontend lint/build
	cd backend && .venv/bin/python -m pytest -q
	cd frontend && npm run lint && npm run build

data: ## download + prepare training data for all four models
	cd backend && .venv/bin/python -m training.datasets.download_data text && \
	  .venv/bin/python -m training.datasets.download_data images && \
	  .venv/bin/python -m training.datasets.prepare_scene_data && \
	  .venv/bin/python -m training.datasets.prepare_tagger_data && \
	  .venv/bin/python -m training.train_image_models extract

train: ## train Models A–D (minutes on an M-series Mac)
	cd backend && .venv/bin/python -m training.train_scene_segmenter && \
	  .venv/bin/python -m training.train_character_model && \
	  .venv/bin/python -m training.train_image_models train

evaluate: ## evaluate all models against gold data and baselines
	cd backend && for t in scene tagger alignment quality; do .venv/bin/python -m training.evaluate $$t; done

.PHONY: help setup models backend frontend test data train evaluate
