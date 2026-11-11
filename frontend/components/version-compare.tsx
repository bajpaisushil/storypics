"use client";

import { Check, Download } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { api, downloads, errorMessage, mediaUrl } from "@/lib/api";
import { percent, VERSION_MODE_LABEL } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { Job, SceneState } from "@/types/api";

/** Before/after slider between two versions of a scene, plus the full version history. */
export function VersionCompare({ job, scene, onJob }: { job: Job; scene: SceneState; onJob: (job: Job) => void }) {
  const versions = scene.versions;
  const [beforeId, setBeforeId] = useState(versions.at(-2)?.id ?? versions[0]?.id);
  const [afterId, setAfterId] = useState(versions.at(-1)?.id ?? versions[0]?.id);
  const [split, setSplit] = useState(50);
  const [busy, setBusy] = useState<string | null>(null);
  const before = versions.find((v) => v.id === beforeId) ?? versions[0];
  const after = versions.find((v) => v.id === afterId) ?? versions.at(-1);
  if (!before || !after) return null;

  const select = async (versionId: string) => {
    setBusy(versionId);
    try {
      onJob(await api.selectVersion(job.id, scene.index, versionId));
      toast.success(`Scene ${scene.index} now uses ${versionId} in the storybook.`);
    } catch (err) {
      toast.error(errorMessage(err));
    } finally {
      setBusy(null);
    }
  };

  return (
    <section aria-label="Compare versions" className="space-y-3 rounded-xl border p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-sm font-medium">Compare versions</p>
        <p className="text-xs text-muted-foreground">Drag the slider · click a version to set before / after</p>
      </div>

      <div
        className="relative mx-auto w-full overflow-hidden rounded-xl bg-muted select-none"
        style={{ aspectRatio: `${job.settings.width} / ${job.settings.height}`, maxWidth: 720 }}
      >
        {/* eslint-disable-next-line @next/next/no-img-element -- backend image */}
        <img src={mediaUrl(after.url)} alt={`After: ${after.id}`} className="absolute inset-0 size-full object-cover" />
        {/* eslint-disable-next-line @next/next/no-img-element -- backend image */}
        <img
          src={mediaUrl(before.url)}
          alt={`Before: ${before.id}`}
          className="absolute inset-0 size-full object-cover"
          style={{ clipPath: `inset(0 ${100 - split}% 0 0)` }}
        />
        <div aria-hidden className="absolute inset-y-0 w-0.5 bg-white shadow-[0_0_0_1px_rgba(0,0,0,0.15)]" style={{ left: `${split}%` }} />
        <Badge className="absolute top-2 left-2 bg-black/60 text-white">Before · {before.id}</Badge>
        <Badge className="absolute top-2 right-2 bg-black/60 text-white">After · {after.id}</Badge>
        <input
          type="range"
          min={0}
          max={100}
          value={split}
          onChange={(e) => setSplit(Number(e.target.value))}
          aria-label="Before/after split position"
          className="absolute inset-0 size-full cursor-ew-resize opacity-0"
        />
      </div>

      <ul className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4">
        {versions.map((v) => {
          const isSelected = v.id === scene.selected_version;
          const role = v.id === before.id ? "before" : v.id === after.id ? "after" : null;
          return (
            <li key={v.id} className={cn("rounded-xl border p-2 text-xs", isSelected && "border-primary/60 bg-accent/40")}>
              <button
                type="button"
                className="block w-full overflow-hidden rounded-lg bg-muted outline-none focus-visible:ring-3 focus-visible:ring-ring/50"
                style={{ aspectRatio: `${job.settings.width} / ${job.settings.height}` }}
                onClick={() => {
                  if (role === "after") return;
                  setBeforeId(after.id);
                  setAfterId(v.id);
                }}
                aria-label={`Compare ${v.id} as the after image`}
              >
                {/* eslint-disable-next-line @next/next/no-img-element -- backend thumbnail */}
                <img src={mediaUrl(v.thumb_url)} alt="" className="size-full object-cover" loading="lazy" />
              </button>
              <div className="mt-1.5 flex items-center gap-1">
                <span className="font-medium">{v.id}</span>
                {role && <span className="text-muted-foreground">· {role}</span>}
                <a
                  href={downloads.sceneImage(job.id, scene.index, v.id)}
                  aria-label={`Download version ${v.id}`}
                  title={`Download ${v.id}`}
                  className="ml-auto grid size-6 place-items-center rounded-md text-muted-foreground hover:bg-muted hover:text-foreground focus-visible:outline-2 focus-visible:outline-ring"
                >
                  <Download className="size-3.5" />
                </a>
              </div>
              <p className="text-muted-foreground">{VERSION_MODE_LABEL[v.mode]}</p>
              {v.instructions && <p className="line-clamp-2 text-muted-foreground italic">“{v.instructions}”</p>}
              {v.scores?.overall !== null && v.scores?.overall !== undefined && (
                <p className="text-muted-foreground">Score {percent(v.scores.overall)}</p>
              )}
              <Button
                size="xs"
                variant={isSelected ? "secondary" : "outline"}
                className="mt-1.5 w-full"
                disabled={isSelected || busy !== null}
                onClick={() => select(v.id)}
              >
                {isSelected ? (
                  <>
                    <Check /> In storybook
                  </>
                ) : (
                  "Use this version"
                )}
              </Button>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
