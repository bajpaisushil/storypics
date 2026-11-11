import type { UsageReport,
  CreateJobRequest,
  Health,
  Job,
  JobSummary,
  RegenerateSceneRequest,
  StyleInfo,
  UpdateCharacterRequest,
  UploadResponse,
} from "@/types/api";

/** Backend base URL (FastAPI). Media and downloads are served from the same origin. */
export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/+$/, "");

export function apiUrl(path: string): string {
  return `${API_URL}${path.startsWith("/") ? path : `/${path}`}`;
}

/** Backend media paths are root-relative (/media/...); absolute URLs pass through unchanged. */
export function mediaUrl(path: string | null | undefined): string | undefined {
  if (!path) return undefined;
  return /^https?:\/\//.test(path) ? path : apiUrl(path);
}

export class ApiError extends Error {
  /** HTTP status; 0 means the backend could not be reached at all */
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

type ValidationItem = { loc?: (string | number)[]; msg?: string };

/** Turn FastAPI's `{detail}` (string or validation list) into one readable sentence. */
export function formatDetail(detail: unknown, fallback: string): string {
  if (typeof detail === "string" && detail.trim()) return detail;
  if (Array.isArray(detail) && detail.length) {
    return (detail as ValidationItem[])
      .map((d) => {
        const field = (d.loc ?? []).filter((p) => p !== "body").join(".");
        return field ? `${field}: ${d.msg ?? "invalid"}` : (d.msg ?? "invalid value");
      })
      .join("; ");
  }
  return fallback;
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  let res: Response;
  try {
    res = await fetch(apiUrl(path), {
      ...init,
      headers: init.body instanceof FormData ? init.headers : { "Content-Type": "application/json", ...init.headers },
      cache: "no-store",
    });
  } catch {
    throw new ApiError(0, `Can't reach the StoryPix backend at ${API_URL}. Is it running?`);
  }
  if (!res.ok) {
    let detail: unknown;
    try {
      detail = (await res.json())?.detail;
    } catch {
      detail = undefined;
    }
    throw new ApiError(res.status, formatDetail(detail, `Request failed (${res.status} ${res.statusText})`));
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

const json = (body: unknown): RequestInit => ({ method: "POST", body: JSON.stringify(body) });

export const api = {
  health: () => request<Health>("/api/health"),
  usage: () => request<UsageReport>("/api/usage"),
  styles: () => request<StyleInfo[]>("/api/styles"),
  listJobs: () => request<JobSummary[]>("/api/jobs"),
  getJob: (id: string) => request<Job>(`/api/jobs/${id}`),
  createJob: (body: CreateJobRequest) => request<Job>("/api/jobs", json(body)),
  deleteJob: (id: string) => request<void>(`/api/jobs/${id}`, { method: "DELETE" }),
  cancelJob: (id: string) => request<Job>(`/api/jobs/${id}/cancel`, { method: "POST" }),
  resumeJob: (id: string) => request<Job>(`/api/jobs/${id}/resume`, { method: "POST" }),
  regenerateScene: (id: string, index: number, body: RegenerateSceneRequest) =>
    request<Job>(`/api/jobs/${id}/scenes/${index}/regenerate`, json(body)),
  selectVersion: (id: string, index: number, versionId: string) =>
    request<Job>(`/api/jobs/${id}/scenes/${index}/select`, json({ version_id: versionId })),
  updateCharacter: (id: string, characterId: string, body: UpdateCharacterRequest) =>
    request<Job>(`/api/jobs/${id}/characters/${characterId}`, { method: "PATCH", body: JSON.stringify(body) }),
  regenerateCharacter: (id: string, characterId: string) =>
    request<Job>(`/api/jobs/${id}/characters/${characterId}/regenerate`, { method: "POST" }),
  upload: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<UploadResponse>("/api/uploads", { method: "POST", body: form });
  },
};

/**
 * Download links. The backend answers these with `Content-Disposition: attachment` and a readable
 * filename, so a plain <a href> downloads the file (cross-origin `download` attributes are ignored).
 */
export const downloads = {
  sceneImage: (jobId: string, index: number, versionId?: string) =>
    apiUrl(
      `/api/jobs/${jobId}/scenes/${index}/download${versionId ? `?version=${encodeURIComponent(versionId)}` : ""}`,
    ),
  characterImage: (jobId: string, characterId: string) =>
    apiUrl(`/api/jobs/${jobId}/characters/${characterId}/download`),
  zip: (jobId: string) => apiUrl(`/api/jobs/${jobId}/export.zip`),
  pdf: (jobId: string) => apiUrl(`/api/jobs/${jobId}/export.pdf`),
};

export const eventsUrl = (jobId: string) => apiUrl(`/api/jobs/${jobId}/events`);

export function errorMessage(error: unknown): string {
  if (error instanceof Error) return error.message;
  return "Something went wrong.";
}
