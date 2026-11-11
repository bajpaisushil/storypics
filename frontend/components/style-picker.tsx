"use client";

import { Check } from "lucide-react";
import { useState } from "react";

import { useStyles } from "@/hooks/use-styles";
import type { StyleMeta } from "@/lib/styles";
import { cn } from "@/lib/utils";
import type { StyleId } from "@/types/api";

function StylePreview({ style }: { style: StyleMeta }) {
  const [missing, setMissing] = useState(false);
  const Icon = style.icon;
  if (missing) {
    // No sample image yet (they are produced by running the real pipeline) → neutral swatch, not a fake output.
    return (
      <div className={cn("grid size-full place-items-center bg-gradient-to-br", style.swatch)}>
        <Icon className="size-6 text-white/90 drop-shadow" aria-hidden />
      </div>
    );
  }
  return (
    // eslint-disable-next-line @next/next/no-img-element -- static sample from /public, optional
    <img
      src={style.preview_url}
      alt=""
      loading="lazy"
      className="size-full object-cover transition-transform duration-300 group-hover:scale-[1.04]"
      onError={() => setMissing(true)}
      // The image can fail before React hydrates and attaches onError (static prerender) — catch that too.
      ref={(img) => {
        if (img && img.complete && img.naturalWidth === 0) setMissing(true);
      }}
    />
  );
}

export function StylePicker({ value, onChange }: { value: StyleId; onChange: (id: StyleId) => void }) {
  const styles = useStyles();
  const selected = styles.find((s) => s.id === value);
  return (
    <fieldset className="space-y-2.5">
      <legend className="text-sm font-medium">Illustration style</legend>
      <div role="radiogroup" aria-label="Illustration style" className="grid grid-cols-4 gap-2">
        {styles.map((style) => {
          const active = style.id === value;
          return (
            <button
              key={style.id}
              type="button"
              role="radio"
              aria-checked={active}
              title={`${style.name} — ${style.tagline}`}
              onClick={() => onChange(style.id)}
              className={cn(
                "group relative flex flex-col gap-1.5 rounded-xl p-1 text-left outline-none transition-colors",
                "focus-visible:ring-3 focus-visible:ring-ring/50",
                active ? "bg-accent" : "hover:bg-muted",
              )}
            >
              <span
                className={cn(
                  "relative block aspect-square overflow-hidden rounded-lg ring-1 ring-border transition-shadow",
                  active && "ring-2 ring-primary shadow-md shadow-violet-500/20",
                )}
              >
                <StylePreview style={style} />
                {active && (
                  <span className="absolute top-1 right-1 grid size-4.5 place-items-center rounded-full bg-primary text-primary-foreground">
                    <Check className="size-3" strokeWidth={3} />
                  </span>
                )}
              </span>
              <span className={cn("line-clamp-2 px-0.5 text-[11px] leading-tight", active ? "font-medium" : "text-muted-foreground")}>
                {style.name}
              </span>
            </button>
          );
        })}
      </div>
      {selected && (
        <p className="text-xs text-muted-foreground">
          <span className="font-medium text-foreground">{selected.name}</span> — {selected.tagline}
        </p>
      )}
    </fieldset>
  );
}
