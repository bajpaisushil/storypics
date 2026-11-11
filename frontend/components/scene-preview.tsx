"use client";

import {
  ChevronLeft,
  ChevronRight,
  Clock,
  CloudSun,
  Columns2,
  Download,
  Loader2,
  MapPin,
  RefreshCw,
  RotateCcw,
  Smile,
  TriangleAlert,
  X,
} from "lucide-react";
import { useState } from "react";

import { PromptDetails } from "@/components/prompt-details";
import { RegenerateDialog } from "@/components/regenerate-dialog";
import { VersionCompare } from "@/components/version-compare";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { downloads, mediaUrl } from "@/lib/api";
import { capitalize, selectedVersion } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { Job, SceneState } from "@/types/api";

interface Props {
  job: Job;
  scene: SceneState;
  onPrev: () => void;
  onNext: () => void;
  onJob: (job: Job) => void;
}

export function ScenePreview({ job, scene, onPrev, onNext, onJob }: Props) {
  const [regenOpen, setRegenOpen] = useState(false);
  const [compare, setCompare] = useState(false);
  const version = selectedVersion(scene);
  const total = job.scenes.length;
  const { width, height } = job.settings;
  const isFirst = scene.index === 1;
  const isLast = scene.index === total;
  const working = job.current?.kind === "scene" && job.current.target === String(scene.index) ? job.current : null;
  const anchorsPending = job.characters.some((c) => c.anchor_status === "generating" || c.anchor_status === "queued");
  const characters = job.characters.filter((c) => scene.character_ids.includes(c.id));

  return (
    <article aria-label={`Scene ${scene.index} of ${total}`} className="space-y-4">
      <div className="relative">
        <div
          className="relative mx-auto w-full overflow-hidden rounded-2xl bg-muted ring-1 ring-border/60"
          style={{ aspectRatio: `${width} / ${height}`, maxWidth: `calc(min(68vh, 760px) * ${width / height})` }}
        >
          {version ? (
            // eslint-disable-next-line @next/next/no-img-element -- backend-served image
            <img
              key={version.id}
              src={mediaUrl(version.url)}
              alt={`Scene ${scene.index}: ${scene.summary}`}
              className="size-full object-cover"
            />
          ) : (
            <div className="absolute inset-0 grid place-items-center p-6 text-center">
              {scene.status === "generating" && (
                <>
                  <Skeleton className="absolute inset-0 rounded-none" />
                  <div className="relative flex flex-col items-center gap-2 text-sm">
                    <Loader2 className="size-6 animate-spin text-primary" />
                    <span className="font-medium">Illustrating scene {scene.index}…</span>
                    {working && working.total_steps > 0 && (
                      <span className="text-xs text-muted-foreground tabular-nums">
                        step {working.step} of {working.total_steps}
                        {working.attempt > 1 ? ` · attempt ${working.attempt}` : ""}
                      </span>
                    )}
                  </div>
                </>
              )}
              {(scene.status === "queued" || scene.status === "pending") && (
                <div className="flex flex-col items-center gap-2 text-sm text-muted-foreground">
                  <Clock className="size-5" />
                  {anchorsPending ? "Designing the characters first…" : "Waiting its turn…"}
                </div>
              )}
              {scene.status === "failed" && (
                <div className="flex max-w-sm flex-col items-center gap-3 text-sm">
                  <TriangleAlert className="size-6 text-destructive" />
                  <p className="font-medium">This scene couldn&apos;t be illustrated</p>
                  {scene.error && <p className="text-xs text-muted-foreground">{scene.error}</p>}
                  <Button size="sm" onClick={() => setRegenOpen(true)}>
                    <RotateCcw /> Retry
                  </Button>
                </div>
              )}
              {scene.status === "cancelled" && (
                <div className="flex flex-col items-center gap-3 text-sm text-muted-foreground">
                  <X className="size-5" /> Cancelled
                  <Button size="sm" variant="outline" onClick={() => setRegenOpen(true)}>
                    <RotateCcw /> Generate this scene
                  </Button>
                </div>
              )}
            </div>
          )}
          {version && scene.status === "generating" && (
            <div className="absolute inset-x-3 bottom-3 flex items-center gap-2 rounded-lg bg-black/60 px-3 py-2 text-xs text-white backdrop-blur">
              <Loader2 className="size-3.5 animate-spin" /> Redrawing
              {working?.total_steps ? ` · step ${working.step}/${working.total_steps}` : "…"}
            </div>
          )}
          <Badge className="absolute top-3 left-3 bg-black/55 text-white tabular-nums">
            Scene {scene.index} / {total}
            {scene.versions.length > 1 && version ? ` · ${version.id}` : ""}
          </Badge>
        </div>
        <Button
          variant="secondary"
          size="icon"
          aria-label="Previous scene"
          disabled={isFirst}
          onClick={onPrev}
          className="absolute top-1/2 left-2 -translate-y-1/2 rounded-full shadow-md sm:left-3"
        >
          <ChevronLeft />
        </Button>
        <Button
          variant="secondary"
          size="icon"
          aria-label="Next scene"
          disabled={isLast}
          onClick={onNext}
          className="absolute top-1/2 right-2 -translate-y-1/2 rounded-full shadow-md sm:right-3"
        >
          <ChevronRight />
        </Button>
      </div>

      <div className="flex flex-col gap-3 sm:flex-row sm:items-start">
        <div className="min-w-0 flex-1 space-y-2">
          <h3 className="text-lg font-semibold tracking-tight">
            <span className="text-muted-foreground">Scene {scene.index} · </span>
            {scene.title}
          </h3>
          <p className="text-sm leading-relaxed">{scene.summary}</p>
          {scene.text && scene.text !== scene.summary && (
            <blockquote className="border-l-2 border-primary/40 pl-3 text-sm text-muted-foreground italic">
              {scene.text}
            </blockquote>
          )}
          <div className="flex flex-wrap gap-1.5 pt-1">
            {scene.location && (
              <Badge variant="outline" className="gap-1 font-normal">
                <MapPin className="size-3" /> {capitalize(scene.location.name === "setting" ? "story setting" : scene.location.name)}
              </Badge>
            )}
            {scene.time_of_day && (
              <Badge variant="outline" className="gap-1 font-normal">
                <CloudSun className="size-3" /> {capitalize(scene.time_of_day)}
              </Badge>
            )}
            {scene.emotion !== "neutral" && (
              <Badge variant="outline" className="gap-1 font-normal">
                <Smile className="size-3" /> {capitalize(scene.emotion)}
              </Badge>
            )}
            {characters.map((c) => (
              <Badge key={c.id} variant="secondary" className="font-normal">
                {c.name}
              </Badge>
            ))}
          </div>
        </div>
        <div className="flex shrink-0 flex-wrap gap-2 sm:flex-col sm:items-stretch">
          {version ? (
            <Button asChild>
              <a href={downloads.sceneImage(job.id, scene.index)}>
                <Download /> Download image
              </a>
            </Button>
          ) : (
            <Button disabled>
              <Download /> Download image
            </Button>
          )}
          <Button
            variant="outline"
            onClick={() => setRegenOpen(true)}
            disabled={scene.status === "generating" || scene.status === "queued"}
          >
            <RefreshCw /> Regenerate
          </Button>
          {scene.versions.length > 1 && (
            <Button variant={compare ? "secondary" : "ghost"} onClick={() => setCompare((v) => !v)} aria-pressed={compare}>
              <Columns2 /> {compare ? "Hide versions" : `Versions (${scene.versions.length})`}
            </Button>
          )}
        </div>
      </div>

      {compare && scene.versions.length > 1 && <VersionCompare job={job} scene={scene} onJob={onJob} />}
      {version && <PromptDetails version={version} characters={job.characters} />}

      <RegenerateDialog
        key={`${scene.index}-${scene.versions.length}`}
        job={job}
        scene={scene}
        open={regenOpen}
        onOpenChange={setRegenOpen}
        onJob={onJob}
      />
      <span className={cn("sr-only")} aria-live="polite">
        {scene.status === "completed" ? `Scene ${scene.index} ready` : ""}
      </span>
    </article>
  );
}
