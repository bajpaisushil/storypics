"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { useCallback, useRef, useState } from "react";

import { ErrorBoundary } from "@/components/error-boundary";
import { GeneratorStatusBanner } from "@/components/generator-status-banner";
import { DEFAULT_FORM, StoryForm, type StoryFormState } from "@/components/story-form";
import { StoryViewer } from "@/components/story-viewer";
import { useHealth } from "@/hooks/use-health";
import type { Job } from "@/types/api";

/** Main page: story creation on the left, the live illustrated story on the right. */
export function Studio() {
  const params = useSearchParams();
  const jobId = params.get("job");
  const router = useRouter();
  const { health, error, unreachable, checking, refresh, reachable, loading } = useHealth();
  const [form, setForm] = useState<StoryFormState>(DEFAULT_FORM);
  const [createdJob, setCreatedJob] = useState<Job | null>(null);
  const viewerRef = useRef<HTMLElement>(null);

  const handleCreated = (job: Job) => {
    setCreatedJob(job);
    router.push(`/?job=${job.id}`, { scroll: false });
    if (window.matchMedia("(max-width: 1023px)").matches) {
      viewerRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  };

  // Opening a saved story pre-fills an empty form so it can be tweaked and generated again.
  const handleLoaded = useCallback((job: Job) => {
    setForm((prev) =>
      prev.story.trim()
        ? prev
        : {
            ...prev,
            story: job.story,
            style: job.settings.style,
            aspectRatio: job.settings.aspect_ratio,
            sceneCount: job.settings.scene_count,
          },
    );
  }, []);

  return (
    <div className="space-y-6">
      <GeneratorStatusBanner health={health} error={error} unreachable={unreachable} checking={checking} onRetry={refresh} />
      <div className="grid gap-6 lg:grid-cols-[420px_minmax(0,1fr)] lg:gap-8">
        <aside aria-label="Create a story" className="min-w-0">
          <StoryForm
            value={form}
            onChange={setForm}
            onCreated={handleCreated}
            backendReachable={reachable}
            generatorAvailable={Boolean(health?.generator.available)}
            checkingBackend={loading}
          />
        </aside>
        <section
          ref={viewerRef}
          aria-label="Illustrated story"
          className="scrollbar-thin min-w-0 scroll-mt-20 lg:sticky lg:top-20 lg:max-h-[calc(100dvh-6rem)] lg:self-start lg:overflow-y-auto lg:pr-1"
        >
          <ErrorBoundary label="The story viewer ran into a problem.">
            <StoryViewer jobId={jobId} initialJob={createdJob} onLoaded={handleLoaded} onNew={() => router.push("/")} />
          </ErrorBoundary>
        </section>
      </div>
    </div>
  );
}
