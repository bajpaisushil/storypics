from fastapi import Request

from app.config import Settings
from app.services.job_manager import JobManager


def get_manager(request: Request) -> JobManager:
    return request.app.state.manager


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings
