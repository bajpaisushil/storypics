"use client";

import { Loader2, Sparkles, Wand2 } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";

import { AspectRatioSelector, SceneCountSelector } from "@/components/option-selectors";
import { ReferenceUploader, type ReferenceItem } from "@/components/reference-uploader";
import { StylePicker } from "@/components/style-picker";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Textarea } from "@/components/ui/textarea";
import { api, errorMessage } from "@/lib/api";
import { formatNumber, MAX_STORY_CHARS, MIN_STORY_CHARS, wordCount } from "@/lib/format";
import { EXAMPLE_STORY } from "@/lib/styles";
import { cn } from "@/lib/utils";
import type { AspectRatio, Job, SceneCountChoice, StyleId } from "@/types/api";

export interface StoryFormState {
  story: string;
  style: StyleId;
  sceneCount: SceneCountChoice;
  aspectRatio: AspectRatio;
  references: ReferenceItem[];
}

export const DEFAULT_FORM: StoryFormState = {
  story: "",
  style: "storybook",
  sceneCount: "auto",
  aspectRatio: "4:3",
  references: [],
};

interface Props {
  value: StoryFormState;
  onChange: (update: (prev: StoryFormState) => StoryFormState) => void;
  onCreated: (job: Job) => void;
  backendReachable: boolean;
  generatorAvailable: boolean;
  checkingBackend?: boolean;
}

export function StoryForm({ value, onChange, onCreated, backendReachable, generatorAvailable, checkingBackend }: Props) {
  const [submitting, setSubmitting] = useState(false);
  const chars = value.story.length;
  const words = wordCount(value.story);
  const uploading = value.references.some((r) => r.status === "uploading");
  const unnamed = value.references.some((r) => r.status === "ready" && !r.name.trim());

  const blocker = (() => {
    if (!value.story.trim()) return "Write or paste a story to begin.";
    if (value.story.trim().length < MIN_STORY_CHARS) return `Stories need at least ${MIN_STORY_CHARS} characters.`;
    if (chars > MAX_STORY_CHARS) return `Stories can be up to ${formatNumber(MAX_STORY_CHARS)} characters.`;
    if (!backendReachable) {
      return checkingBackend ? "Connecting to the StoryPix backend…" : "Start the local StoryPix backend to generate (see above).";
    }
    if (!generatorAvailable) return "Image generation is unavailable — see the notice above.";
    if (uploading) return "Waiting for reference uploads to finish…";
    if (unnamed) return "Name each reference image so it can be matched to a character.";
    return null;
  })();

  const set = <K extends keyof StoryFormState>(key: K, v: StoryFormState[K]) =>
    onChange((prev) => ({ ...prev, [key]: v }));

  const submit = async () => {
    if (blocker || submitting) return;
    setSubmitting(true);
    try {
      const job = await api.createJob({
        story: value.story.trim(),
        style: value.style,
        scene_count: value.sceneCount,
        aspect_ratio: value.aspectRatio,
        references: value.references
          .filter((r) => r.status === "ready" && r.uploadId && r.name.trim())
          .map((r) => ({ upload_id: r.uploadId as string, character_name: r.name.trim() })),
      });
      toast.success(`Found ${job.scenes.length} scenes and ${job.characters.length} characters — illustrating now.`);
      onCreated(job);
    } catch (err) {
      toast.error(errorMessage(err));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Card className="rounded-2xl border-border/70 shadow-sm shadow-violet-500/5">
      <CardContent className="space-y-6">
        <form
          className="space-y-6"
          onSubmit={(e) => {
            e.preventDefault();
            void submit();
          }}
        >
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label htmlFor="story" className="text-sm font-medium">
                Your story
              </label>
              <Button
                type="button"
                variant="ghost"
                size="xs"
                className="text-primary"
                onClick={() => set("story", EXAMPLE_STORY)}
              >
                <Sparkles /> Try an example
              </Button>
            </div>
            <Textarea
              id="story"
              value={value.story}
              onChange={(e) => set("story", e.target.value)}
              placeholder="Once upon a time…"
              rows={10}
              aria-describedby="story-count"
              aria-invalid={chars > MAX_STORY_CHARS}
              className="min-h-56 resize-y rounded-xl bg-background/70 text-[15px] leading-relaxed"
            />
            <p id="story-count" className="flex justify-between text-xs text-muted-foreground tabular-nums">
              <span>
                {formatNumber(words)} {words === 1 ? "word" : "words"}
              </span>
              <span className={cn(chars > MAX_STORY_CHARS && "font-medium text-destructive")}>
                {formatNumber(chars)} / {formatNumber(MAX_STORY_CHARS)}
              </span>
            </p>
          </div>

          <StylePicker value={value.style} onChange={(v) => set("style", v)} />

          <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-1 xl:grid-cols-2">
            <SceneCountSelector value={value.sceneCount} onChange={(v) => set("sceneCount", v)} />
            <AspectRatioSelector value={value.aspectRatio} onChange={(v) => set("aspectRatio", v)} />
          </div>

          <Separator />

          <ReferenceUploader
            items={value.references}
            onChange={(update) => onChange((prev) => ({ ...prev, references: update(prev.references) }))}
          />

          <div className="space-y-2">
            <Button
              type="submit"
              size="lg"
              disabled={Boolean(blocker) || submitting}
              className="h-11 w-full rounded-xl bg-gradient-to-r from-violet-600 to-indigo-600 text-[15px] text-white shadow-md shadow-violet-500/20 hover:from-violet-600/90 hover:to-indigo-600/90 dark:from-violet-500 dark:to-indigo-500"
            >
              {submitting ? <Loader2 className="animate-spin" /> : <Wand2 />}
              {submitting ? "Reading your story…" : "Generate Story"}
            </Button>
            {blocker && value.story.trim() && (
              <p role="status" className="text-center text-xs text-muted-foreground">
                {blocker}
              </p>
            )}
          </div>
        </form>
      </CardContent>
    </Card>
  );
}
