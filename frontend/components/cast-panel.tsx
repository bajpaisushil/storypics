"use client";

import { Download, ImageIcon, Loader2, Pencil, RefreshCw, TriangleAlert, Users } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";

import { EditCharacterDialog } from "@/components/edit-character-dialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { api, downloads, errorMessage, mediaUrl } from "@/lib/api";
import { capitalize } from "@/lib/format";
import type { CharacterState, Job } from "@/types/api";

function CharacterCard({ job, character, onJob }: { job: Job; character: CharacterState; onJob: (job: Job) => void }) {
  const [editing, setEditing] = useState(false);
  const [busy, setBusy] = useState(false);
  const anchor = character.anchor_versions.find((v) => v.id === character.selected_anchor) ?? null;
  const working = character.anchor_status === "generating" || character.anchor_status === "queued";
  const attributes = Object.entries(character.attributes).slice(0, 5);

  const redraw = async () => {
    setBusy(true);
    try {
      onJob(await api.regenerateCharacter(job.id, character.id));
      toast.success(`Redrawing ${character.name}…`);
    } catch (err) {
      toast.error(errorMessage(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <li className="flex gap-3 rounded-xl border bg-card p-3">
      <div className="group relative size-24 shrink-0 overflow-hidden rounded-lg bg-muted sm:size-28">
        {anchor ? (
          // eslint-disable-next-line @next/next/no-img-element -- backend-served portrait
          <img src={mediaUrl(anchor.thumb_url)} alt={`${character.name} reference portrait`} className="size-full object-cover" />
        ) : working ? (
          <Skeleton className="size-full rounded-none" />
        ) : (
          <div className="grid size-full place-items-center text-muted-foreground">
            {character.anchor_status === "failed" ? (
              <TriangleAlert className="size-5 text-destructive" />
            ) : (
              <ImageIcon className="size-5" />
            )}
          </div>
        )}
        {working && (
          <span className="absolute inset-0 grid place-items-center bg-background/40">
            <Loader2 className="size-5 animate-spin text-primary" aria-label="Drawing portrait" />
          </span>
        )}
        {anchor && (
          <a
            href={downloads.characterImage(job.id, character.id)}
            aria-label={`Download ${character.name} portrait`}
            title="Download portrait"
            className="absolute top-1 right-1 grid size-7 place-items-center rounded-lg bg-black/55 text-white opacity-0 transition-opacity group-hover:opacity-100 focus-visible:opacity-100 focus-visible:outline-2 focus-visible:outline-white"
          >
            <Download className="size-3.5" />
          </a>
        )}
      </div>
      <div className="min-w-0 flex-1 space-y-1.5">
        <div className="flex flex-wrap items-center gap-1.5">
          <p className="font-medium">{character.name}</p>
          <Badge variant="outline" className="font-normal">
            {capitalize(character.species)}
          </Badge>
          {character.reference_upload_id && <Badge variant="secondary">Your reference</Badge>}
        </div>
        {attributes.length > 0 && (
          <dl className="grid grid-cols-[auto_1fr] gap-x-2 text-xs">
            {attributes.map(([k, v]) => (
              <div key={k} className="contents">
                <dt className="text-muted-foreground">{k}</dt>
                <dd className="truncate" title={v}>
                  {v}
                </dd>
              </div>
            ))}
          </dl>
        )}
        {character.anchor_status === "failed" && character.anchor_error && (
          <p className="text-xs text-destructive">{character.anchor_error}</p>
        )}
        <div className="flex flex-wrap gap-1 pt-0.5">
          <Button size="xs" variant="outline" onClick={() => setEditing(true)}>
            <Pencil /> Edit
          </Button>
          <Button size="xs" variant="ghost" onClick={redraw} disabled={busy || working}>
            <RefreshCw /> Redraw
          </Button>
          {anchor && (
            <Button size="xs" variant="ghost" asChild>
              <a href={downloads.characterImage(job.id, character.id)}>
                <Download /> Download
              </a>
            </Button>
          )}
        </div>
      </div>
      <EditCharacterDialog
        key={`${character.id}-${character.descriptor}`}
        job={job}
        character={character}
        open={editing}
        onOpenChange={setEditing}
        onJob={onJob}
      />
    </li>
  );
}

export function CastPanel({ job, onJob }: { job: Job; onJob: (job: Job) => void }) {
  const main = job.characters.filter((c) => c.is_recurring);
  const minor = job.characters.filter((c) => !c.is_recurring);
  if (!job.characters.length) return null;
  return (
    <Card className="rounded-2xl border-border/70 shadow-sm">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <Users className="size-4 text-primary" /> Characters
        </CardTitle>
        <CardDescription>
          Recurring characters get a reference portrait first; every scene is conditioned on it so they keep the
          same look.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {main.length > 0 && (
          <ul className="grid gap-3 xl:grid-cols-2">
            {main.map((c) => (
              <CharacterCard key={c.id} job={job} character={c} onJob={onJob} />
            ))}
          </ul>
        )}
        {minor.length > 0 && (
          <p className="text-xs text-muted-foreground">
            Also in the story:{" "}
            {minor.map((c, i) => (
              <span key={c.id} title={c.descriptor}>
                <span className="font-medium text-foreground">{c.name}</span>
                {c.scene_indices.length ? ` (scene ${c.scene_indices.join(", ")})` : ""}
                {i < minor.length - 1 ? ", " : ""}
              </span>
            ))}{" "}
            — described in text only.
          </p>
        )}
      </CardContent>
    </Card>
  );
}
