import { ChevronRight } from "lucide-react";

import { percent } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { CharacterState, ImageVersion } from "@/types/api";

function Meter({ label, value, hint }: { label: string; value: number | null | undefined; hint?: string }) {
  const pct = value === null || value === undefined ? null : Math.max(0, Math.min(1, value));
  return (
    <div className="space-y-1" title={hint}>
      <div className="flex justify-between text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-medium tabular-nums">{percent(pct)}</span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-muted">
        {pct !== null && (
          <div
            className={cn("h-full rounded-full", pct >= 0.66 ? "bg-emerald-500" : pct >= 0.45 ? "bg-amber-500" : "bg-rose-500")}
            style={{ width: `${pct * 100}%` }}
          />
        )}
      </div>
    </div>
  );
}

/** Full transparency for the AI pipeline: the exact prompt, seed, sampler settings and evaluator scores. */
export function PromptDetails({ version, characters }: { version: ImageVersion; characters: CharacterState[] }) {
  const g = version.generation;
  const s = version.scores;
  const names = Object.fromEntries(characters.map((c) => [c.id, c.name]));
  const rows: [string, string | number | null | undefined][] = [
    ["Seed", version.seed],
    ["Steps", g.steps],
    ["Guidance", g.guidance_scale],
    ["Size", g.width && g.height ? `${g.width} × ${g.height}` : null],
    ["Profile", g.profile],
    ["Device", g.device],
    ["References", g.reference_mode],
    ["Time", g.duration_s ? `${g.duration_s.toFixed(1)} s` : null],
    ["Attempts", version.attempts],
    ["Peak memory", g.peak_memory_gb ? `${g.peak_memory_gb} GB` : null],
  ];
  return (
    <details className="group rounded-xl border bg-muted/30 open:bg-muted/40">
      <summary className="flex cursor-pointer list-none items-center gap-2 rounded-xl px-3 py-2.5 text-sm font-medium outline-none select-none focus-visible:ring-3 focus-visible:ring-ring/50 [&::-webkit-details-marker]:hidden">
        <ChevronRight className="size-4 transition-transform group-open:rotate-90" />
        Prompt &amp; scores
        <span className="ml-auto text-xs font-normal text-muted-foreground">
          {s?.overall !== null && s?.overall !== undefined ? `overall ${percent(s.overall)}` : `seed ${version.seed}`}
        </span>
      </summary>
      <div className="grid gap-4 px-3 pb-3 md:grid-cols-[1fr_220px]">
        <div className="space-y-3 text-xs">
          <div>
            <p className="mb-1 font-medium">Prompt</p>
            <p className="rounded-lg bg-background/80 p-2.5 font-mono leading-relaxed break-words text-muted-foreground">
              {version.prompt}
            </p>
          </div>
          {version.instructions && (
            <p>
              <span className="font-medium">Your instructions:</span>{" "}
              <span className="text-muted-foreground">{version.instructions}</span>
            </p>
          )}
          {version.negative_prompt && g.guidance_scale !== undefined && g.guidance_scale > 1 && (
            <div>
              <p className="mb-1 font-medium">Negative prompt</p>
              <p className="font-mono leading-relaxed break-words text-muted-foreground">{version.negative_prompt}</p>
            </div>
          )}
          <dl className="grid grid-cols-2 gap-x-4 gap-y-1 sm:grid-cols-3">
            {rows
              .filter(([, v]) => v !== null && v !== undefined && v !== "")
              .map(([k, v]) => (
                <div key={k} className="flex justify-between gap-2 border-b border-dashed border-border/70 py-1">
                  <dt className="text-muted-foreground">{k}</dt>
                  <dd className="truncate font-medium tabular-nums">{String(v)}</dd>
                </div>
              ))}
          </dl>
        </div>
        <div className="space-y-2.5">
          <p className="text-xs font-medium">Evaluation</p>
          {s ? (
            <>
              <Meter label="Matches the scene" value={s.alignment} hint="Model C: text–image alignment" />
              <Meter label="Image quality" value={s.quality} hint="Model D: predicted fidelity" />
              {Object.entries(s.consistency).map(([cid, v]) => (
                <Meter
                  key={cid}
                  label={`${names[cid] ?? cid} looks consistent`}
                  value={v}
                  hint="DINOv2 similarity to the character's reference portrait"
                />
              ))}
              <Meter label="Overall" value={s.overall} />
            </>
          ) : (
            <p className="text-xs text-muted-foreground">No scores — the evaluator was not available for this image.</p>
          )}
        </div>
      </div>
    </details>
  );
}
