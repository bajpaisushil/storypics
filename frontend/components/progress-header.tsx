"use client";

import { Loader2, RotateCcw, Square } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import type { Connection } from "@/hooks/use-job";
import { api, errorMessage } from "@/lib/api";
import { currentTaskLabel, isActive, JOB_STATUS_LABEL, jobProgress } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { Job } from "@/types/api";

const RESUMABLE = new Set<Job["status"]>(["failed", "partial", "interrupted", "cancelled"]);

export function ProgressHeader({
  job,
  connection,
  onJob,
}: {
  job: Job;
  connection: Connection;
  onJob: (job: Job) => void;
}) {
  const [busy, setBusy] = useState(false);
  const active = isActive(job);
  const value = Math.round(jobProgress(job) * 100);
  const label = currentTaskLabel(job);
  const incomplete =
    job.scenes.some((s) => s.status !== "completed") ||
    job.characters.some((c) => c.is_recurring && ["failed", "cancelled"].includes(c.anchor_status));
  const failedScenes = job.scenes.filter((s) => s.status === "failed").length;

  if (!active && !(RESUMABLE.has(job.status) && incomplete)) return null;

  const run = async (fn: () => Promise<Job>, done: string) => {
    setBusy(true);
    try {
      onJob(await fn());
      toast.success(done);
    } catch (err) {
      toast.error(errorMessage(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-2 rounded-xl border bg-muted/40 p-3">
      <div className="flex items-center gap-3">
        <div className="min-w-0 flex-1" aria-live="polite">
          <p className="flex items-center gap-2 text-sm font-medium">
            {active && <Loader2 className="size-3.5 shrink-0 animate-spin text-primary" />}
            <span className="truncate">
              {active ? (label ?? "Waiting for the image generator…") : JOB_STATUS_LABEL[job.status]}
            </span>
          </p>
          {!active && (
            <p className="text-xs text-muted-foreground">
              {failedScenes ? `${failedScenes} scene${failedScenes > 1 ? "s" : ""} failed. ` : ""}
              {job.error ?? "Resume to finish the remaining scenes."}
            </p>
          )}
        </div>
        {active && (
          <span
            className={cn(
              "hidden items-center gap-1.5 text-xs text-muted-foreground sm:flex",
              connection === "live" ? "" : "text-amber-600 dark:text-amber-400",
            )}
            title={connection === "live" ? "Receiving live updates" : "Live updates interrupted — refreshing periodically"}
          >
            <span className={cn("size-1.5 rounded-full", connection === "live" ? "bg-emerald-500" : "bg-amber-500")} />
            {connection === "live" ? "Live" : connection === "polling" ? "Polling" : "Reconnecting"}
          </span>
        )}
        <span className="text-xs text-muted-foreground tabular-nums">{value}%</span>
        {active ? (
          <Button
            size="sm"
            variant="outline"
            disabled={busy}
            onClick={() => run(() => api.cancelJob(job.id), "Generation cancelled.")}
          >
            <Square className="fill-current" /> Cancel
          </Button>
        ) : (
          <Button size="sm" disabled={busy} onClick={() => run(() => api.resumeJob(job.id), "Resuming generation.")}>
            <RotateCcw /> Resume
          </Button>
        )}
      </div>
      <Progress value={value} aria-label="Overall progress" className="h-1.5" />
    </div>
  );
}
