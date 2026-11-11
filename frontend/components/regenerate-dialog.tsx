"use client";

import { Loader2, RefreshCw } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { api, errorMessage } from "@/lib/api";
import { hasImage } from "@/lib/format";
import type { Job, SceneState } from "@/types/api";

type Mode = "keep_composition" | "new_variation";

interface Props {
  job: Job;
  scene: SceneState;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onJob: (job: Job) => void;
}

export function RegenerateDialog({ job, scene, open, onOpenChange, onJob }: Props) {
  const canKeep = hasImage(scene);
  const [instructions, setInstructions] = useState("");
  const [mode, setMode] = useState<Mode>(canKeep ? "keep_composition" : "new_variation");
  const [busy, setBusy] = useState(false);
  const firstCharacter = job.characters.find((c) => scene.character_ids.includes(c.id));
  const suggestions = [
    firstCharacter ? `Make the ${firstCharacter.name.toLowerCase()} happier` : "Make everyone smile",
    "Show it at sunset",
    "Closer view of the characters",
  ];

  const submit = async () => {
    setBusy(true);
    try {
      const updated = await api.regenerateScene(job.id, scene.index, {
        instructions: instructions.trim() || null,
        mode: canKeep ? mode : "new_variation",
      });
      onJob(updated);
      toast.success(`Redrawing scene ${scene.index}…`);
      onOpenChange(false);
      setInstructions("");
    } catch (err) {
      toast.error(errorMessage(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>Regenerate scene {scene.index}</DialogTitle>
          <DialogDescription>
            “{scene.title}”. The current image stays in the version history, so you can compare and switch back.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="regen-instructions">Instructions (optional)</Label>
            <Textarea
              id="regen-instructions"
              value={instructions}
              maxLength={300}
              rows={3}
              placeholder="e.g. Make the hen happier, show the cow standing near the tree"
              onChange={(e) => setInstructions(e.target.value)}
            />
            <div className="flex flex-wrap gap-1.5">
              {suggestions.map((s) => (
                <button
                  key={s}
                  type="button"
                  onClick={() => setInstructions(s)}
                  className="rounded-full border px-2.5 py-1 text-xs text-muted-foreground transition-colors hover:border-primary/50 hover:bg-accent hover:text-accent-foreground"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
          <div className="space-y-2">
            <p id="regen-mode" className="text-sm font-medium">
              Composition
            </p>
            <ToggleGroup
              type="single"
              variant="outline"
              spacing={0}
              aria-labelledby="regen-mode"
              value={canKeep ? mode : "new_variation"}
              onValueChange={(v) => v && setMode(v as Mode)}
              className="w-full"
            >
              <ToggleGroupItem
                value="keep_composition"
                disabled={!canKeep}
                className="flex-1 data-[state=on]:bg-accent data-[state=on]:text-accent-foreground"
              >
                Keep composition
              </ToggleGroupItem>
              <ToggleGroupItem
                value="new_variation"
                className="flex-1 data-[state=on]:bg-accent data-[state=on]:text-accent-foreground"
              >
                New variation
              </ToggleGroupItem>
            </ToggleGroup>
            <p className="text-xs text-muted-foreground">
              {canKeep && mode === "keep_composition"
                ? "Reuses the same seed, so the layout stays similar while your instructions change the details."
                : "Draws the scene again from a fresh random seed."}
            </p>
          </div>
        </div>
        <DialogFooter>
          <Button variant="ghost" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={submit} disabled={busy}>
            {busy ? <Loader2 className="animate-spin" /> : <RefreshCw />} Regenerate
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
