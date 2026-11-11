"use client";

import { useCallback, useSyncExternalStore } from "react";

import { ApiError, api, errorMessage } from "@/lib/api";
import type { Health } from "@/types/api";

/** Re-check while the backend can't be reached (waking up, or not started yet). */
const RETRY_UNREACHABLE_MS = 10_000;
/**
 * Re-check a healthy backend rarely. The hosted backend scales to zero after a few idle minutes; frequent polling
 * from an open tab would keep it (and its bill) running forever.
 */
const REFRESH_HEALTHY_MS = 5 * 60_000;

interface HealthState {
  health: Health | null;
  error: string | null;
  unreachable: boolean;
  checking: boolean;
}

// One poller shared by every component that needs the backend status.
let state: HealthState = { health: null, error: null, unreachable: false, checking: true };
const listeners = new Set<() => void>();
let timer: ReturnType<typeof setTimeout> | undefined;
let inFlight: Promise<void> | null = null;

function emit(next: Partial<HealthState>) {
  state = { ...state, ...next };
  listeners.forEach((l) => l());
}

function schedule() {
  clearTimeout(timer);
  if (listeners.size === 0 || (typeof document !== "undefined" && document.hidden)) return;
  timer = setTimeout(check, state.unreachable || !state.health ? RETRY_UNREACHABLE_MS : REFRESH_HEALTHY_MS);
}

function check(): Promise<void> {
  if (inFlight) return inFlight;
  emit({ checking: true });
  inFlight = api
    .health()
    .then((health) => emit({ health, error: null, unreachable: false }))
    .catch((err) =>
      emit({ health: null, error: errorMessage(err), unreachable: err instanceof ApiError && err.status === 0 }),
    )
    .finally(() => {
      inFlight = null;
      emit({ checking: false });
      schedule();
    });
  return inFlight;
}

function onVisibility() {
  if (!document.hidden) void check(); // refresh when the tab comes back; nothing is polled while hidden
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  if (listeners.size === 1) {
    document.addEventListener("visibilitychange", onVisibility);
    void check();
  }
  return () => {
    listeners.delete(listener);
    if (listeners.size === 0) {
      clearTimeout(timer);
      document.removeEventListener("visibilitychange", onVisibility);
    }
  };
}

const serverSnapshot: HealthState = { health: null, error: null, unreachable: false, checking: true };

/**
 * Backend + image-generator status, shared across components. "Unreachable" (network failure) is normal while a
 * local backend is not started or a hosted one is waking up, and is re-checked every 10 s until it answers.
 */
export function useHealth() {
  const snapshot = useSyncExternalStore(
    subscribe,
    () => state,
    () => serverSnapshot,
  );
  const refresh = useCallback(() => check(), []);
  return {
    ...snapshot,
    refresh,
    reachable: snapshot.health !== null,
    /** true only before the very first answer */
    loading: snapshot.checking && snapshot.health === null && snapshot.error === null,
  };
}
