"use client";

import { Info, Loader2, SearchX } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";

import { CastPanel } from "@/components/cast-panel";
import { DownloadMenu } from "@/components/download-menu";
import { EmptyState } from "@/components/empty-state";
import { ProgressHeader } from "@/components/progress-header";
import { ScenePreview } from "@/components/scene-preview";
import { ThumbnailStrip } from "@/components/thumbnail-strip";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { useJob } from "@/hooks/use-job";
import { api, errorMessage } from "@/lib/api";
import { isActive, JOB_STATUS_LABEL } from "@/lib/format";
import { styleMeta } from "@/lib/styles";
import type { Job } from "@/types/api";

interface Props {
  jobId: string | null;
  initialJob?: Job | null;
  onLoaded?: (job: Job) => void;
  onNew?: () => void;
}

export function StoryViewer({ jobId, initialJob = null, onLoaded, onNew }: Props) {
  if (!jobId) return <EmptyState />;
  // keyed by job id so each story gets a fresh live connection and view state
  return <JobViewer key={jobId} jobId={jobId} initialJob={initialJob} onLoaded={onLoaded} onNew={onNew} />;
}

function JobViewer({ jobId, initialJob, onLoaded, onNew }: Props & { jobId: string }) {
  const { job, error, notFound, connection, everActive, setJob } = useJob(jobId, initialJob ?? null);
  const [selected, setSelected] = useState<number | null>(null); // null = follow generation progress
  const reported = useRef(false);

  useEffect(() => {
    if (job && !reported.current) {
      reported.current = true;
      onLoaded?.(job);
    }
  }, [job, onLoaded]);

  const activeIndex = useMemo(() => {
    if (!job || !job.scenes.length) return 1;
    if (selected !== null && job.scenes.some((s) => s.index === selected)) return selected;
    const generating = job.scenes.find((s) => s.status === "generating");
    if (generating) return generating.index;
    if (everActive) {
      const done = job.scenes.filter((s) => s.status === "completed");
      if (done.length) return done[done.length - 1].index;
    }
    return job.scenes[0].index;
  }, [job, selected, everActive]);

  const sceneCount = job?.scenes.length ?? 0;
  const go = useCallback(
    (delta: number) => setSelected(Math.min(Math.max(1, activeIndex + delta), Math.max(1, sceneCount))),
    [activeIndex, sceneCount],
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey) return;
      const target = e.target as HTMLElement | null;
      if (target?.closest("input, textarea, select, [contenteditable=true], [role=dialog], [role=menu]")) return;
      if (document.querySelector("[role=dialog]")) return;
      if (e.key === "ArrowLeft") go(-1);
      if (e.key === "ArrowRight") go(1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [go]);

  const retryScene = async (index: number) => {
    if (!job) return;
    try {
      setJob(await api.regenerateScene(job.id, index, { mode: "new_variation" }));
      setSelected(index);
      toast.success(`Retrying scene ${index}…`);
    } catch (err) {
      toast.error(errorMessage(err));
    }
  };

  if (!job) {
    if (error) {
      return (
        <Card className="rounded-2xl">
          <CardContent className="flex flex-col items-center gap-3 py-16 text-center">
            {notFound ? <SearchX className="size-6 text-muted-foreground" /> : <Loader2 className="size-6 animate-spin text-muted-foreground" />}
            <p className="font-medium">{notFound ? "This story could not be found" : "Waiting for the backend…"}</p>
            <p className="max-w-md text-sm text-muted-foreground">
              {notFound ? "It may have been deleted, or it was created by a different StoryPix backend." : error}
            </p>
            {onNew && (
              <Button variant="outline" onClick={onNew}>
                Start a new story
              </Button>
            )}
          </CardContent>
        </Card>
      );
    }
    return (
      <Card className="rounded-2xl">
        <CardContent className="space-y-4">
          <Skeleton className="h-6 w-1/2" />
          <Skeleton className="aspect-[4/3] w-full rounded-2xl" />
          <div className="flex gap-2">
            {[0, 1, 2, 3].map((i) => (
              <Skeleton key={i} className="h-20 w-28 rounded-xl" />
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  const scene = job.scenes.find((s) => s.index === activeIndex) ?? null;
  const style = styleMeta(job.settings.style);

  return (
    <div className="space-y-6">
      <Card className="rounded-2xl border-border/70 shadow-sm shadow-violet-500/5">
        <CardContent className="space-y-5">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <h2 className="truncate text-xl font-semibold tracking-tight">{job.title}</h2>
              <p className="text-xs text-muted-foreground">
                {style.name} · {job.settings.aspect_ratio} · {job.scenes.length} scenes · {JOB_STATUS_LABEL[job.status]}
              </p>
            </div>
            <DownloadMenu job={job} scene={scene} />
          </div>

          <ProgressHeader job={job} connection={connection} onJob={setJob} />

          {job.warnings.length > 0 && (
            <ul className="space-y-1 text-xs text-muted-foreground">
              {job.warnings.map((w) => (
                <li key={w} className="flex gap-1.5">
                  <Info className="mt-0.5 size-3.5 shrink-0" /> {w}
                </li>
              ))}
            </ul>
          )}

          <ThumbnailStrip job={job} activeIndex={activeIndex} onSelect={setSelected} onRetry={retryScene} />

          {scene && (
            <ScenePreview key={scene.index} job={job} scene={scene} onPrev={() => go(-1)} onNext={() => go(1)} onJob={setJob} />
          )}

          {isActive(job) && (
            <p className="text-center text-xs text-muted-foreground">
              Tip: use ← → to move between scenes. You can redraw any finished scene while the rest are generating.
            </p>
          )}
        </CardContent>
      </Card>
      <CastPanel job={job} onJob={setJob} />
    </div>
  );
}
