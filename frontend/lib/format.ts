import type { Job, SceneState, TaskStatus, VersionMode } from "@/types/api";

export const MAX_STORY_CHARS = 6000;
export const MIN_STORY_CHARS = 20;

export function wordCount(text: string): number {
  const trimmed = text.trim();
  return trimmed ? trimmed.split(/\s+/).length : 0;
}

export function formatNumber(n: number): string {
  return new Intl.NumberFormat("en-US").format(n);
}

/** 0.8234 → "82%"; null → "—" */
export function percent(value: number | null | undefined): string {
  return value === null || value === undefined || Number.isNaN(value) ? "—" : `${Math.round(value * 100)}%`;
}

export const ACTIVE_TASK: TaskStatus[] = ["queued", "generating"];

export function isActive(job: Job): boolean {
  return job.status === "queued" || job.status === "generating";
}

export function selectedVersion(scene: SceneState) {
  return scene.versions.find((v) => v.id === scene.selected_version) ?? null;
}

export function hasImage(scene: SceneState): boolean {
  return selectedVersion(scene) !== null;
}

/** Fine-grained progress: finished tasks plus the fraction of the running task. */
export function jobProgress(job: Job): number {
  const tasks = [
    ...job.characters.filter((c) => c.anchor_status !== "skipped").map((c) => c.anchor_status),
    ...job.scenes.map((s) => s.status),
  ];
  if (!tasks.length) return 0;
  const done = tasks.filter((t) => t === "completed" || t === "failed" || t === "cancelled").length;
  const partial = job.current && job.current.total_steps ? job.current.step / job.current.total_steps : 0;
  return Math.min(1, (done + partial) / tasks.length);
}

export function currentTaskLabel(job: Job): string | null {
  const c = job.current;
  if (!c) return null;
  let label = c.label;
  if (c.kind === "scene" && /scene \d+$/i.test(label)) label += ` of ${job.scenes.length}`;
  if (c.total_steps) label += ` · step ${c.step}/${c.total_steps}`;
  if (c.attempt > 1) label += ` · attempt ${c.attempt}`;
  return label;
}

export const STATUS_LABEL: Record<TaskStatus, string> = {
  pending: "Pending",
  queued: "Queued",
  generating: "Generating",
  completed: "Done",
  failed: "Failed",
  cancelled: "Cancelled",
  skipped: "Not needed",
};

export const JOB_STATUS_LABEL: Record<Job["status"], string> = {
  queued: "Queued",
  generating: "Generating",
  completed: "Complete",
  partial: "Partly complete",
  failed: "Failed",
  cancelled: "Cancelled",
  interrupted: "Interrupted",
};

export const VERSION_MODE_LABEL: Record<VersionMode, string> = {
  initial: "Original",
  keep_composition: "Same composition",
  new_variation: "New variation",
  retry: "Auto-retried",
  upload: "From your reference",
};

export function formatDate(iso: string): string {
  const d = new Date(iso);
  return Number.isNaN(d.getTime())
    ? ""
    : d.toLocaleDateString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

export function capitalize(text: string): string {
  return text ? text[0].toUpperCase() + text.slice(1) : text;
}
