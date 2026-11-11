"use client";

import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { cn } from "@/lib/utils";
import type { AspectRatio, SceneCountChoice } from "@/types/api";

const SCENE_OPTIONS: SceneCountChoice[] = ["auto", 3, 5, 8, 10];
const ASPECTS: { value: AspectRatio; w: number; h: number }[] = [
  { value: "1:1", w: 12, h: 12 },
  { value: "4:3", w: 14, h: 10.5 },
  { value: "16:9", w: 16, h: 9 },
  { value: "9:16", w: 9, h: 16 },
];

export function SceneCountSelector({
  value,
  onChange,
}: {
  value: SceneCountChoice;
  onChange: (v: SceneCountChoice) => void;
}) {
  return (
    <div className="space-y-2">
      <p id="scene-count-label" className="text-sm font-medium">
        Scenes
      </p>
      <ToggleGroup
        type="single"
        variant="outline"
        spacing={0}
        aria-labelledby="scene-count-label"
        value={String(value)}
        onValueChange={(v) => v && onChange(v === "auto" ? "auto" : Number(v))}
        className="w-full"
      >
        {SCENE_OPTIONS.map((opt) => (
          <ToggleGroupItem
            key={String(opt)}
            value={String(opt)}
            aria-label={opt === "auto" ? "Let StoryPix decide" : `${opt} scenes`}
            className="flex-1 data-[state=on]:bg-accent data-[state=on]:text-accent-foreground"
          >
            {opt === "auto" ? "Auto" : opt}
          </ToggleGroupItem>
        ))}
      </ToggleGroup>
    </div>
  );
}

export function AspectRatioSelector({ value, onChange }: { value: AspectRatio; onChange: (v: AspectRatio) => void }) {
  return (
    <div className="space-y-2">
      <p id="aspect-label" className="text-sm font-medium">
        Aspect ratio
      </p>
      <ToggleGroup
        type="single"
        variant="outline"
        spacing={0}
        aria-labelledby="aspect-label"
        value={value}
        onValueChange={(v) => v && onChange(v as AspectRatio)}
        className="w-full"
      >
        {ASPECTS.map((a) => (
          <ToggleGroupItem
            key={a.value}
            value={a.value}
            aria-label={`Aspect ratio ${a.value}`}
            className="flex-1 gap-1.5 data-[state=on]:bg-accent data-[state=on]:text-accent-foreground"
          >
            <span
              aria-hidden
              className={cn("inline-block rounded-[3px] border-[1.5px] border-current opacity-70")}
              style={{ width: a.w, height: a.h }}
            />
            <span className="text-xs">{a.value}</span>
          </ToggleGroupItem>
        ))}
      </ToggleGroup>
    </div>
  );
}
