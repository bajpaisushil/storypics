"use client";

import { Info, PlugZap, ServerCrash } from "lucide-react";

import { BackendSetupPanel } from "@/components/backend-setup-panel";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import type { Health } from "@/types/api";

interface Props {
  health: Health | null;
  error: string | null;
  unreachable: boolean;
  checking: boolean;
  onRetry: () => void;
}

/** Tells the user exactly why generation is blocked — the app never fakes images. */
export function GeneratorStatusBanner({ health, error, unreachable, checking, onRetry }: Props) {
  if (unreachable) return <BackendSetupPanel onRetry={onRetry} checking={checking} />;
  if (error) {
    return (
      <Alert variant="destructive" className="rounded-2xl">
        <ServerCrash />
        <AlertTitle>The StoryPix backend returned an error</AlertTitle>
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    );
  }
  if (health && !health.generator.available) {
    return (
      <Alert className="rounded-2xl border-amber-300/70 bg-amber-50 text-amber-950 dark:border-amber-500/30 dark:bg-amber-950/30 dark:text-amber-100">
        <PlugZap />
        <AlertTitle>Image generation is unavailable</AlertTitle>
        <AlertDescription className="text-amber-900/90 dark:text-amber-100/80">
          <p>{health.generator.reason ?? "The image generator is not ready."}</p>
          <p className="mt-1 text-xs">Story analysis still works; images can be generated once this is fixed.</p>
        </AlertDescription>
      </Alert>
    );
  }
  if (health?.generator.notice) {
    return (
      <Alert className="rounded-2xl border-sky-200 bg-sky-50/80 text-sky-950 dark:border-sky-500/30 dark:bg-sky-950/30 dark:text-sky-100">
        <Info />
        <AlertTitle>Generation works with a limitation</AlertTitle>
        <AlertDescription className="text-sky-900/90 dark:text-sky-100/80">{health.generator.notice}</AlertDescription>
      </Alert>
    );
  }
  return null;
}
