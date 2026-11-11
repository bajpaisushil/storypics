"use client";

import { ChevronDown, Download, FileArchive, FileImage, FileText } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { downloads } from "@/lib/api";
import { hasImage } from "@/lib/format";
import type { Job, SceneState } from "@/types/api";

export function DownloadMenu({ job, scene }: { job: Job; scene: SceneState | null }) {
  const anyImage = job.scenes.some(hasImage);
  const sceneReady = scene !== null && hasImage(scene);
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" disabled={!anyImage} aria-label="Download options">
          <Download /> Download <ChevronDown className="opacity-60" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-60">
        <DropdownMenuLabel className="text-xs text-muted-foreground">Download</DropdownMenuLabel>
        <DropdownMenuItem asChild disabled={!sceneReady}>
          <a href={sceneReady && scene ? downloads.sceneImage(job.id, scene.index) : undefined}>
            <FileImage />
            <span className="flex-1">Current image (PNG)</span>
            {scene && <span className="text-xs text-muted-foreground">Scene {scene.index}</span>}
          </a>
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem asChild>
          <a href={downloads.zip(job.id)}>
            <FileArchive />
            <span className="flex-1">All images (ZIP)</span>
          </a>
        </DropdownMenuItem>
        <DropdownMenuItem asChild>
          <a href={downloads.pdf(job.id)}>
            <FileText />
            <span className="flex-1">Storybook (PDF)</span>
          </a>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
