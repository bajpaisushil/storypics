"""Monthly compute usage of the hosted backend (Modal), for the "free compute this month" meter in the web app.

Inside a Modal container the workspace billing summary is readable with the container's own credentials, so no
account token is stored in the server. Results are cached for a few minutes. Anywhere else (local development) the
endpoint reports that usage tracking is not available.
"""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime
from decimal import Decimal

from pydantic import BaseModel

CACHE_SECONDS = 300


class UsageReport(BaseModel):
    provider: str  # "modal" | "none"
    available: bool
    month: str | None = None
    used_usd: float | None = None  # metered cost this month (before credits)
    billed_usd: float | None = None  # what Modal would actually invoice
    free_credit_usd: float | None = None  # monthly free compute credit of the plan
    remaining_usd: float | None = None
    exhausted: bool = False
    resets_on: str | None = None
    updated_at: str | None = None
    note: str | None = None


class UsageTracker:
    def __init__(self, monthly_credit_usd: float):
        self.monthly_credit = monthly_credit_usd
        self._cache: tuple[float, UsageReport] | None = None
        self._lock = threading.Lock()

    @staticmethod
    def on_modal() -> bool:
        try:
            import modal
        except ImportError:
            return False
        return not modal.is_local()

    def report(self) -> UsageReport:
        if not self.on_modal():
            return UsageReport(provider="none", available=False, note="Usage tracking is only available on Modal.")
        with self._lock:
            if self._cache and time.time() - self._cache[0] < CACHE_SECONDS:
                return self._cache[1]
            report = self._fetch()
            self._cache = (time.time(), report)
            return report

    def _fetch(self) -> UsageReport:
        now = datetime.now(UTC)
        nxt = datetime(now.year + (now.month == 12), now.month % 12 + 1, 1, tzinfo=UTC)
        try:
            import modal  # provided by the Modal runtime inside the container

            summary = modal.Workspace.from_context().billing.summary()
        except Exception as exc:  # never break the app over a usage meter
            return UsageReport(provider="modal", available=False, note=f"Usage unavailable: {exc.__class__.__name__}")
        used = float(Decimal(summary.metered_cost))
        billed = float(Decimal(summary.billed_cost))
        remaining = max(0.0, self.monthly_credit - used)
        return UsageReport(
            provider="modal",
            available=True,
            month=now.strftime("%Y-%m"),
            used_usd=round(used, 2),
            billed_usd=round(billed, 2),
            free_credit_usd=self.monthly_credit,
            remaining_usd=round(remaining, 2),
            exhausted=remaining <= 0 or billed > 0,
            resets_on=nxt.date().isoformat(),
            updated_at=now.isoformat(timespec="seconds"),
        )
