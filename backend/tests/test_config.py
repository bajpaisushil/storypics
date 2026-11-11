"""Settings must load from environment variables exactly as a deployment sets them (all values are strings)."""

import pytest

from app.config import Settings

DEPLOYMENT_ENV = {
    "STORYPIX_DATA_DIR": "/tmp/storypix-test-data",
    "STORYPIX_DEVICE": "cuda",
    "STORYPIX_PROFILE": "lightning",
    "STORYPIX_LIGHTNING_STEPS": "8",
    "STORYPIX_LCM_STEPS": "4",
    "STORYPIX_ALLOW_DOWNLOADS": "true",
    "STORYPIX_PRELOAD_MODELS": "true",
    "STORYPIX_SAFETY_CHECKER": "true",
    "STORYPIX_MAX_ACTIVE_JOBS": "2",
    "STORYPIX_CORS_ORIGINS": '["http://localhost:3000","http://127.0.0.1:3000"]',
    "STORYPIX_CORS_ORIGIN_REGEX": "https://storypics[a-z0-9-]*[.]vercel[.]app",
}


def test_deployment_environment_validates(monkeypatch):
    for key, value in DEPLOYMENT_ENV.items():
        monkeypatch.setenv(key, value)
    s = Settings(_env_file=None)
    assert s.lightning_steps == 8 and s.max_active_jobs == 2 and s.allow_downloads is True
    assert s.cors_origins == ["http://localhost:3000", "http://127.0.0.1:3000"]


def test_invalid_lightning_steps_rejected(monkeypatch):
    monkeypatch.setenv("STORYPIX_LIGHTNING_STEPS", "6")
    with pytest.raises(ValueError):
        Settings(_env_file=None)
