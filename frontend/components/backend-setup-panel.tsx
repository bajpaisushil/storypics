"use client";

import { Check, Cloud, Copy, ExternalLink, Laptop, Loader2, RefreshCw } from "lucide-react";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { API_URL } from "@/lib/api";

const README_URL = "https://github.com/bajpaisushil/storypics#readme";
/** A backend on this machine needs starting; a hosted one (e.g. a free Hugging Face Space) only needs waking. */
const IS_LOCAL_BACKEND = /^https?:\/\/(localhost|127\.0\.0\.1|\[::1\])(:|\/|$)/.test(API_URL);
const COMMANDS = ["cd backend && source .venv/bin/activate", "uvicorn app.main:app --port 8000"];

function CommandLine({ command }: { command: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <div className="flex items-center gap-2 rounded-lg border bg-background/80 py-1 pr-1 pl-3 font-mono text-[13px]">
      <span aria-hidden className="text-muted-foreground select-none">
        $
      </span>
      <code className="min-w-0 flex-1 overflow-x-auto whitespace-nowrap">{command}</code>
      <Button
        variant="ghost"
        size="icon-sm"
        aria-label={`Copy command: ${command}`}
        onClick={() => {
          void navigator.clipboard?.writeText(command).then(() => {
            setCopied(true);
            setTimeout(() => setCopied(false), 1500);
          });
        }}
      >
        {copied ? <Check className="text-emerald-600" /> : <Copy />}
      </Button>
    </div>
  );
}

function HostedServerPanel({ onRetry, checking }: { onRetry: () => void; checking: boolean }) {
  return (
    <section
      aria-labelledby="setup-title"
      className="rounded-2xl border border-violet-200/80 bg-gradient-to-br from-white to-violet-50/70 p-5 shadow-sm shadow-violet-500/5 sm:p-6 dark:border-violet-500/20 dark:from-card dark:to-violet-950/20"
    >
      <div className="flex flex-col gap-5 md:flex-row md:items-start">
        <div className="grid size-11 shrink-0 place-items-center rounded-xl bg-violet-600 text-white shadow-sm">
          <Cloud className="size-5" />
        </div>
        <div className="min-w-0 flex-1 space-y-3">
          <h2 id="setup-title" className="text-base font-semibold tracking-tight">
            Waking up the StoryPix server…
          </h2>
          <p className="text-sm text-muted-foreground">
            StoryPix runs its AI models on a free CPU server that sleeps when nobody is using it. Waking it usually
            takes 1–2 minutes; the very first start also loads the image model, which can take a few minutes more.
            This page reconnects on its own.
          </p>
          <div className="flex flex-wrap items-center gap-2">
            <Button onClick={onRetry} disabled={checking}>
              {checking ? <Loader2 className="animate-spin" /> : <RefreshCw />}
              Retry now
            </Button>
            <Button variant="ghost" asChild>
              <a href={README_URL} target="_blank" rel="noreferrer">
                How StoryPix works <ExternalLink className="opacity-60" />
              </a>
            </Button>
            <span className="text-xs text-muted-foreground" aria-live="polite">
              {checking ? "Checking…" : "Retrying automatically every 5 seconds."}
            </span>
          </div>
        </div>
      </div>
    </section>
  );
}

/** Shown when the backend can't be reached (network failure on /api/health). */
export function BackendSetupPanel({ onRetry, checking }: { onRetry: () => void; checking: boolean }) {
  if (!IS_LOCAL_BACKEND) return <HostedServerPanel onRetry={onRetry} checking={checking} />;
  return (
    <section
      aria-labelledby="setup-title"
      className="rounded-2xl border border-violet-200/80 bg-gradient-to-br from-white to-violet-50/70 p-5 shadow-sm shadow-violet-500/5 sm:p-6 dark:border-violet-500/20 dark:from-card dark:to-violet-950/20"
    >
      <div className="flex flex-col gap-5 md:flex-row md:items-start">
        <div className="grid size-11 shrink-0 place-items-center rounded-xl bg-violet-600 text-white shadow-sm">
          <Laptop className="size-5" />
        </div>
        <div className="min-w-0 flex-1 space-y-4">
          <div>
            <h2 id="setup-title" className="text-base font-semibold tracking-tight">
              StoryPix runs on your own computer — start the local backend
            </h2>
            <p className="mt-1 text-sm text-muted-foreground">
              The AI models run locally, so this page talks to the StoryPix backend at{" "}
              <code className="font-mono text-xs">{API_URL}</code>. Start it from the project folder:
            </p>
          </div>
          <div className="max-w-xl space-y-2">
            {COMMANDS.map((c) => (
              <CommandLine key={c} command={c} />
            ))}
          </div>
          <p className="text-xs text-muted-foreground">
            If your browser asks whether this site may access devices on your local network, click{" "}
            <span className="font-medium text-foreground">Allow</span> — that is how it reaches the backend on your
            machine.
          </p>
          <div className="flex flex-wrap items-center gap-2">
            <Button onClick={onRetry} disabled={checking}>
              {checking ? <Loader2 className="animate-spin" /> : <RefreshCw />}
              Retry connection
            </Button>
            <Button variant="ghost" asChild>
              <a href={README_URL} target="_blank" rel="noreferrer">
                Setup guide <ExternalLink className="opacity-60" />
              </a>
            </Button>
            <span className="text-xs text-muted-foreground" aria-live="polite">
              {checking ? "Checking…" : "Retrying automatically every 5 seconds."}
            </span>
          </div>
        </div>
      </div>
    </section>
  );
}
