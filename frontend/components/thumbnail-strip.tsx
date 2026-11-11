"use client";

import { Clock, Download, Loader2, RotateCcw, TriangleAlert, X } from "lucide-react";

import { downloads, mediaUrl } from "@/lib/api";
import { selectedVersion, STATUS_LABEL } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { Job } from "@/types/api";

interface Props {
  job: Job;
  activeIndex: number;
  onSelect: (index: number) => void;
  onRetry: (index: number) => void;
}

export function ThumbnailStrip({ job, activeIndex, onSelect, onRetry }: Props) {
  const ratio = `${job.settings.width} / ${job.settings.height}`;
  return (
    <nav aria-label="Scenes" className="scrollbar-thin -mx-1 overflow-x-auto px-1 pt-1 pb-2">
      <ol className="flex gap-2.5">
        {job.scenes.map((scene) => {
          const version = selectedVersion(scene);
          const active = scene.index === activeIndex;
          return (
            <li key={scene.index} className="group relative w-28 shrink-0 sm:w-32">
              <button
                type="button"
                onClick={() => onSelect(scene.index)}
                aria-current={active ? "step" : undefined}
                aria-label={`Scene ${scene.index}: ${scene.title} (${STATUS_LABEL[scene.status]})`}
                className={cn(
                  "relative block w-full overflow-hidden rounded-xl bg-muted ring-1 ring-border outline-none transition",
                  "focus-visible:ring-3 focus-visible:ring-ring/60",
                  active ? "ring-2 ring-primary" : "opacity-85 hover:opacity-100",
                )}
                style={{ aspectRatio: ratio }}
              >
                {version ? (
                  // eslint-disable-next-line @next/next/no-img-element -- backend-served thumbnail
                  <img src={mediaUrl(version.thumb_url)} alt="" className="size-full object-cover" loading="lazy" />
                ) : (
                  <span className="absolute inset-0 grid place-items-center text-muted-foreground">
                    {scene.status === "generating" && <Loader2 className="size-5 animate-spin text-primary" />}
                    {(scene.status === "queued" || scene.status === "pending") && <Clock className="size-4" />}
                    {scene.status === "failed" && <TriangleAlert className="size-5 text-destructive" />}
                    {scene.status === "cancelled" && <X className="size-4" />}
                  </span>
                )}
                {version && scene.status === "generating" && (
                  <span className="absolute inset-0 grid place-items-center bg-background/50">
                    <Loader2 className="size-5 animate-spin text-primary" />
                  </span>
                )}
                <span className="absolute bottom-1 left-1 rounded-md bg-black/55 px-1.5 py-0.5 text-[10px] font-medium text-white tabular-nums">
                  {scene.index}
                </span>
              </button>
              {version && (
                <a
                  href={downloads.sceneImage(job.id, scene.index)}
                  aria-label={`Download scene ${scene.index} image`}
                  title="Download image"
                  className={cn(
                    "absolute top-1 right-1 grid size-7 place-items-center rounded-lg bg-black/55 text-white opacity-0 transition-opacity",
                    "group-hover:opacity-100 group-focus-within:opacity-100 focus-visible:opacity-100 focus-visible:outline-2 focus-visible:outline-white",
                  )}
                >
                  <Download className="size-3.5" />
                </a>
              )}
              <p className={cn("mt-1 truncate text-[11px]", active ? "font-medium" : "text-muted-foreground")}>
                {scene.title}
              </p>
              {scene.status === "failed" && (
                <button
                  type="button"
                  onClick={() => onRetry(scene.index)}
                  className="mt-0.5 flex items-center gap-1 text-[11px] font-medium text-destructive hover:underline"
                >
                  <RotateCcw className="size-3" /> Retry
                </button>
              )}
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
