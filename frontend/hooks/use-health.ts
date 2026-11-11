"use client";

import { useCallback, useEffect, useState } from "react";

import { ApiError, api, errorMessage } from "@/lib/api";
import type { Health } from "@/types/api";

const REFRESH_MS = 15000;
const RETRY_UNREACHABLE_MS = 5000;

/**
 * Backend + image-generator status. StoryPix is local-first: the backend runs on the user's machine, so
 * "unreachable" (network failure) is a normal first-run state and is re-checked every 5 s until it appears.
 */
export function useHealth() {
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [unreachable, setUnreachable] = useState(false);
  const [checking, setChecking] = useState(true);

  const refresh = useCallback(async () => {
    setChecking(true);
    try {
      const data = await api.health();
      setHealth(data);
      setError(null);
      setUnreachable(false);
    } catch (err) {
      setHealth(null);
      setError(errorMessage(err));
      setUnreachable(err instanceof ApiError && err.status === 0);
    } finally {
      setChecking(false);
    }
  }, []);

  useEffect(() => {
    const first = setTimeout(refresh, 0);
    const timer = setInterval(refresh, unreachable ? RETRY_UNREACHABLE_MS : REFRESH_MS);
    return () => {
      clearTimeout(first);
      clearInterval(timer);
    };
  }, [refresh, unreachable]);

  return {
    health,
    error,
    unreachable,
    checking,
    refresh,
    reachable: health !== null,
    /** true only before the very first answer */
    loading: checking && health === null && error === null,
  };
}
