"use client";

import { Loader2, Plus, Trash2 } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { api, errorMessage } from "@/lib/api";
import type { CharacterState, Job } from "@/types/api";

interface Props {
  job: Job;
  character: CharacterState;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onJob: (job: Job) => void;
}

export function EditCharacterDialog({ job, character, open, onOpenChange, onJob }: Props) {
  const [name, setName] = useState(character.name);
  const [descriptor, setDescriptor] = useState(character.descriptor);
  const [rows, setRows] = useState<[string, string][]>(Object.entries(character.attributes));
  const [regenerate, setRegenerate] = useState(true);
  const [busy, setBusy] = useState(false);
  const affected = job.scenes.filter((s) => s.character_ids.includes(character.id)).length;

  const save = async () => {
    setBusy(true);
    try {
      const attributes = Object.fromEntries(rows.filter(([k, v]) => k.trim() && v.trim()).map(([k, v]) => [k.trim(), v.trim()]));
      const descriptorChanged = descriptor.trim() !== character.descriptor;
      const updated = await api.updateCharacter(job.id, character.id, {
        name: name.trim() || null,
        attributes,
        // when only attributes changed, let the backend rebuild the descriptor from them
        descriptor: descriptorChanged ? descriptor.trim() : null,
        regenerate,
      });
      onJob(updated);
      toast.success(regenerate ? `Redrawing ${name || character.name} and ${affected} scene(s)…` : "Character saved.");
      onOpenChange(false);
    } catch (err) {
      toast.error(errorMessage(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[90vh] overflow-y-auto sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>Edit {character.name}</DialogTitle>
          <DialogDescription>
            This description is reused in every scene so {character.name} looks the same throughout the story.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="char-name">Name</Label>
            <Input id="char-name" value={name} maxLength={60} onChange={(e) => setName(e.target.value)} />
          </div>
          <div className="space-y-2">
            <Label>Appearance</Label>
            <ul className="space-y-2">
              {rows.map(([key, value], i) => (
                <li key={i} className="flex gap-2">
                  <Input
                    aria-label="Attribute"
                    value={key}
                    className="w-32 shrink-0"
                    onChange={(e) => setRows((r) => r.map((row, j) => (j === i ? [e.target.value, row[1]] : row)))}
                  />
                  <Input
                    aria-label={`${key || "Attribute"} value`}
                    value={value}
                    onChange={(e) => setRows((r) => r.map((row, j) => (j === i ? [row[0], e.target.value] : row)))}
                  />
                  <Button
                    variant="ghost"
                    size="icon"
                    aria-label={`Remove ${key || "attribute"}`}
                    onClick={() => setRows((r) => r.filter((_, j) => j !== i))}
                  >
                    <Trash2 />
                  </Button>
                </li>
              ))}
            </ul>
            <Button variant="ghost" size="sm" onClick={() => setRows((r) => [...r, ["", ""]])}>
              <Plus /> Add attribute
            </Button>
          </div>
          <div className="space-y-2">
            <Label htmlFor="char-descriptor">Prompt description</Label>
            <Textarea
              id="char-descriptor"
              rows={3}
              value={descriptor}
              maxLength={300}
              onChange={(e) => setDescriptor(e.target.value)}
            />
            <p className="text-xs text-muted-foreground">
              Leave as is to rebuild it from the attributes above, or write your own.
            </p>
          </div>
          <label className="flex items-start gap-2.5 text-sm">
            <Checkbox checked={regenerate} onCheckedChange={(v) => setRegenerate(v === true)} className="mt-0.5" />
            <span>
              Redraw the portrait and update the {affected} scene{affected === 1 ? "" : "s"} with {character.name}
              <span className="block text-xs text-muted-foreground">Scenes keep their composition (same seed).</span>
            </span>
          </label>
        </div>
        <DialogFooter>
          <Button variant="ghost" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={save} disabled={busy || !name.trim()}>
            {busy && <Loader2 className="animate-spin" />} Save
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
