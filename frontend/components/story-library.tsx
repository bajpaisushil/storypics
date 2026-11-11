"use client";

import { BookOpen, ImageIcon, Plus, Trash2 } from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";

import { BackendSetupPanel } from "@/components/backend-setup-panel";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { ApiError, api, errorMessage, mediaUrl } from "@/lib/api";
import { formatDate, JOB_STATUS_LABEL } from "@/lib/format";
import { styleMeta } from "@/lib/styles";
import type { JobSummary } from "@/types/api";

type LoadState =
  | { kind: "loading" }
  | { kind: "ready"; jobs: JobSummary[] }
  | { kind: "unreachable" }
  | { kind: "error"; message: string };

export function StoryLibrary() {
  const [state, setState] = useState<LoadState>({ kind: "loading" });
  const [checking, setChecking] = useState(false);

  const load = useCallback(async () => {
    setChecking(true);
    try {
      setState({ kind: "ready", jobs: await api.listJobs() });
    } catch (err) {
      setState(err instanceof ApiError && err.status === 0 ? { kind: "unreachable" } : { kind: "error", message: errorMessage(err) });
    } finally {
      setChecking(false);
    }
  }, []);

  useEffect(() => {
    const first = setTimeout(load, 0);
    return () => clearTimeout(first);
  }, [load]);

  // keep retrying every 5 s while the local backend is down
  useEffect(() => {
    if (state.kind !== "unreachable") return;
    const timer = setInterval(load, 5000);
    return () => clearInterval(timer);
  }, [state.kind, load]);

  const remove = async (job: JobSummary) => {
    try {
      await api.deleteJob(job.id);
      setState((s) => (s.kind === "ready" ? { kind: "ready", jobs: s.jobs.filter((j) => j.id !== job.id) } : s));
      toast.success(`Deleted “${job.title}”.`);
    } catch (err) {
      toast.error(errorMessage(err));
    }
  };

  if (state.kind === "loading") {
    return (
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {[0, 1, 2, 3].map((i) => (
          <Skeleton key={i} className="h-64 rounded-2xl" />
        ))}
      </div>
    );
  }
  if (state.kind === "unreachable") return <BackendSetupPanel onRetry={load} checking={checking} />;
  if (state.kind === "error") {
    return <p className="rounded-2xl border bg-card p-6 text-sm text-destructive">{state.message}</p>;
  }
  if (!state.jobs.length) {
    return (
      <div className="flex flex-col items-center gap-3 rounded-2xl border border-dashed bg-background/50 px-6 py-16 text-center">
        <BookOpen className="size-6 text-muted-foreground" />
        <p className="font-medium">No stories yet</p>
        <p className="text-sm text-muted-foreground">Write a story and StoryPix will illustrate it scene by scene.</p>
        <Button asChild>
          <Link href="/">
            <Plus /> Create a story
          </Link>
        </Button>
      </div>
    );
  }
  return (
    <ul className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
      {state.jobs.map((job) => (
        <li key={job.id} className="group overflow-hidden rounded-2xl border bg-card shadow-sm transition-shadow hover:shadow-md">
          <Link href={`/?job=${job.id}`} className="block outline-none focus-visible:ring-3 focus-visible:ring-ring/50">
            <div className="aspect-[4/3] bg-muted">
              {job.cover_url ? (
                // eslint-disable-next-line @next/next/no-img-element -- backend-served thumbnail
                <img src={mediaUrl(job.cover_url)} alt="" className="size-full object-cover" loading="lazy" />
              ) : (
                <div className="grid size-full place-items-center text-muted-foreground">
                  <ImageIcon className="size-6" />
                </div>
              )}
            </div>
            <div className="space-y-1 p-3">
              <p className="truncate font-medium">{job.title}</p>
              <p className="text-xs text-muted-foreground">
                {styleMeta(job.style).name} · {job.scene_count} scenes · {formatDate(job.created_at)}
              </p>
            </div>
          </Link>
          <div className="flex items-center justify-between px-3 pb-3">
            <Badge variant={job.status === "completed" ? "secondary" : "outline"}>{JOB_STATUS_LABEL[job.status]}</Badge>
            <AlertDialog>
              <AlertDialogTrigger asChild>
                <Button variant="ghost" size="icon-sm" aria-label={`Delete ${job.title}`}>
                  <Trash2 />
                </Button>
              </AlertDialogTrigger>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>Delete “{job.title}”?</AlertDialogTitle>
                  <AlertDialogDescription>
                    Its images and storybook files are removed from your computer. This can&apos;t be undone.
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel>Cancel</AlertDialogCancel>
                  <AlertDialogAction onClick={() => remove(job)}>Delete</AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          </div>
        </li>
      ))}
    </ul>
  );
}
