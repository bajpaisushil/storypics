"use client";

import { useCallback, useEffect, useReducer } from "react";

import { ApiError, api, errorMessage, eventsUrl } from "@/lib/api";
import { isActive } from "@/lib/format";
import type { Job, TaskProgress } from "@/types/api";

export type Connection = "idle" | "connecting" | "live" | "polling";

interface State {
  jobId: string | null;
  job: Job | null;
  error: string | null;
  notFound: boolean;
  connection: Connection;
  /** the job was queued/generating at some point while this view was open */
  everActive: boolean;
}

type Action =
  | { type: "job"; job: Job }
  | { type: "progress"; current: TaskProgress }
  | { type: "error"; message: string; notFound?: boolean }
  | { type: "connection"; value: Connection };

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case "job":
      if (action.job.id !== state.jobId) return state;
      // ignore stale snapshots that arrive out of order
      if (state.job && Date.parse(action.job.updated_at) < Date.parse(state.job.updated_at)) return state;
      return { ...state, job: action.job, error: null, notFound: false, everActive: state.everActive || isActive(action.job) };
    case "progress":
      return state.job ? { ...state, job: { ...state.job, current: action.current } } : state;
    case "error":
      return { ...state, error: action.message, notFound: Boolean(action.notFound) };
    case "connection":
      return state.connection === action.value ? state : { ...state, connection: action.value };
  }
}

const POLL_MS = 2000;
const MAX_SSE_RETRIES = 4;

/**
 * Live job state: Server-Sent Events (`job` snapshots + `progress` steps) with exponential-backoff
 * reconnects, falling back to polling every 2 s when the event stream keeps failing.
 * Mount one instance per job (key the consumer by job id).
 */
export function useJob(jobId: string | null, initialJob: Job | null = null) {
  const [state, dispatch] = useReducer(reducer, {
    jobId,
    job: initialJob && initialJob.id === jobId ? initialJob : null,
    error: null,
    notFound: false,
    connection: jobId ? "connecting" : "idle",
    everActive: initialJob ? isActive(initialJob) : false,
  });

  // Only keep a live connection while work is in progress: an open stream (or polling) keeps a scale-to-zero
  // backend awake. A finished storybook is fetched once; regenerating makes it active again and reconnects.
  const active = state.job ? isActive(state.job) : true;

  useEffect(() => {
    if (!jobId) return;

    let source: EventSource | null = null;
    let pollTimer: ReturnType<typeof setInterval> | undefined;
    let retryTimer: ReturnType<typeof setTimeout> | undefined;
    let retries = 0;
    let disposed = false;

    const load = async () => {
      try {
        const job = await api.getJob(jobId);
        if (!disposed) dispatch({ type: "job", job });
      } catch (err) {
        if (disposed) return;
        const notFound = err instanceof ApiError && err.status === 404;
        dispatch({ type: "error", message: errorMessage(err), notFound });
        if (notFound) stopAll();
      }
    };

    const startPolling = () => {
      if (pollTimer || disposed) return;
      dispatch({ type: "connection", value: "polling" });
      pollTimer = setInterval(load, POLL_MS);
    };

    const stopAll = () => {
      source?.close();
      source = null;
      if (pollTimer) clearInterval(pollTimer);
      if (retryTimer) clearTimeout(retryTimer);
      pollTimer = undefined;
    };

    const connect = () => {
      if (disposed) return;
      source = new EventSource(eventsUrl(jobId));
      source.addEventListener("job", (e) => {
        retries = 0;
        dispatch({ type: "connection", value: "live" });
        try {
          dispatch({ type: "job", job: JSON.parse((e as MessageEvent<string>).data).job as Job });
        } catch {
          /* malformed event: the next snapshot will correct it */
        }
      });
      source.addEventListener("progress", (e) => {
        try {
          dispatch({ type: "progress", current: JSON.parse((e as MessageEvent<string>).data).current as TaskProgress });
        } catch {
          /* ignore */
        }
      });
      source.onerror = () => {
        source?.close();
        source = null;
        if (disposed) return;
        retries += 1;
        if (retries > MAX_SSE_RETRIES) {
          startPolling();
          return;
        }
        dispatch({ type: "connection", value: "connecting" });
        retryTimer = setTimeout(connect, Math.min(8000, 500 * 2 ** retries));
        void load(); // keep the UI fresh while reconnecting
      };
    };

    void load();
    if (active) connect();
    else dispatch({ type: "connection", value: "idle" });
    return () => {
      disposed = true;
      stopAll();
    };
  }, [jobId, active]);

  /** Apply a Job returned by a mutation immediately (the SSE snapshot will follow). */
  const setJob = useCallback((job: Job) => dispatch({ type: "job", job }), []);

  return { ...state, setJob };
}
