"use client";

import { ImagePlus, Loader2, TriangleAlert, X } from "lucide-react";
import { useRef, useState } from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { api, errorMessage } from "@/lib/api";
import { cn } from "@/lib/utils";

export interface ReferenceItem {
  localId: string;
  previewUrl: string;
  fileName: string;
  name: string;
  uploadId: string | null;
  status: "uploading" | "ready" | "error";
  error?: string;
}

const MAX_REFERENCES = 6;
const MAX_MB = 10;
const ACCEPT = ["image/png", "image/jpeg", "image/webp"];

export function ReferenceUploader({
  items,
  onChange,
}: {
  items: ReferenceItem[];
  onChange: (update: (items: ReferenceItem[]) => ReferenceItem[]) => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const addFiles = (files: FileList | File[]) => {
    const room = MAX_REFERENCES - items.length;
    const accepted = Array.from(files).filter((f) => {
      if (!ACCEPT.includes(f.type)) {
        toast.error(`${f.name}: use a PNG, JPEG or WebP image.`);
        return false;
      }
      if (f.size > MAX_MB * 1024 * 1024) {
        toast.error(`${f.name} is larger than ${MAX_MB} MB.`);
        return false;
      }
      return true;
    });
    if (accepted.length > room) toast.info(`You can add up to ${MAX_REFERENCES} reference images.`);
    for (const file of accepted.slice(0, Math.max(0, room))) {
      const item: ReferenceItem = {
        localId: newLocalId(),
        previewUrl: URL.createObjectURL(file),
        fileName: file.name,
        name: guessName(file.name),
        uploadId: null,
        status: "uploading",
      };
      onChange((prev) => [...prev, item]);
      api
        .upload(file)
        .then((res) =>
          onChange((prev) =>
            prev.map((it) => (it.localId === item.localId ? { ...it, uploadId: res.upload_id, status: "ready" } : it)),
          ),
        )
        .catch((err) => {
          const message = errorMessage(err);
          toast.error(`Couldn't upload ${file.name}: ${message}`);
          onChange((prev) =>
            prev.map((it) => (it.localId === item.localId ? { ...it, status: "error", error: message } : it)),
          );
        });
    }
  };

  const remove = (item: ReferenceItem) => {
    URL.revokeObjectURL(item.previewUrl);
    onChange((prev) => prev.filter((it) => it.localId !== item.localId));
  };

  return (
    <div className="space-y-2.5">
      <div className="flex items-baseline justify-between">
        <p className="text-sm font-medium">
          Character references <span className="font-normal text-muted-foreground">(optional)</span>
        </p>
        <span className="text-xs text-muted-foreground">
          {items.length}/{MAX_REFERENCES}
        </span>
      </div>

      {items.length > 0 && (
        <ul className="space-y-2">
          {items.map((item) => (
            <li key={item.localId} className="flex items-center gap-3 rounded-xl border bg-background/60 p-2">
              <div className="relative size-12 shrink-0 overflow-hidden rounded-lg bg-muted">
                {/* eslint-disable-next-line @next/next/no-img-element -- local object URL preview */}
                <img src={item.previewUrl} alt={`Reference ${item.fileName}`} className="size-full object-cover" />
                {item.status === "uploading" && (
                  <span className="absolute inset-0 grid place-items-center bg-background/60">
                    <Loader2 className="size-4 animate-spin" aria-label="Uploading" />
                  </span>
                )}
                {item.status === "error" && (
                  <span className="absolute inset-0 grid place-items-center bg-destructive/20" title={item.error}>
                    <TriangleAlert className="size-4 text-destructive" aria-label="Upload failed" />
                  </span>
                )}
              </div>
              <div className="min-w-0 flex-1">
                <label htmlFor={`ref-${item.localId}`} className="sr-only">
                  Who is in {item.fileName}?
                </label>
                <Input
                  id={`ref-${item.localId}`}
                  value={item.name}
                  maxLength={60}
                  placeholder="Who is this? e.g. Hen"
                  aria-invalid={item.status === "ready" && !item.name.trim()}
                  onChange={(e) =>
                    onChange((prev) =>
                      prev.map((it) => (it.localId === item.localId ? { ...it, name: e.target.value } : it)),
                    )
                  }
                />
                <p className="mt-1 truncate text-[11px] text-muted-foreground">
                  {item.status === "error" ? item.error : item.fileName}
                </p>
              </div>
              <Button variant="ghost" size="icon-sm" aria-label={`Remove ${item.fileName}`} onClick={() => remove(item)}>
                <X />
              </Button>
            </li>
          ))}
        </ul>
      )}

      {items.length < MAX_REFERENCES && (
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            addFiles(e.dataTransfer.files);
          }}
          className={cn(
            "flex w-full items-center gap-3 rounded-xl border border-dashed px-3 py-3 text-left text-sm text-muted-foreground transition-colors outline-none",
            "hover:border-primary/50 hover:bg-accent/50 focus-visible:ring-3 focus-visible:ring-ring/50",
            dragging && "border-primary bg-accent",
          )}
        >
          <span className="grid size-9 shrink-0 place-items-center rounded-lg bg-muted">
            <ImagePlus className="size-4" />
          </span>
          <span>
            <span className="font-medium text-foreground">Add a photo or drawing</span> of a character to keep their
            look. PNG, JPEG or WebP, up to {MAX_MB} MB.
          </span>
        </button>
      )}
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT.join(",")}
        multiple
        hidden
        onChange={(e) => {
          if (e.target.files) addFiles(e.target.files);
          e.target.value = "";
        }}
      />
    </div>
  );
}

function newLocalId(): string {
  // crypto.randomUUID only exists in secure contexts (https / localhost)
  return typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

/** "my-hen_photo.jpg" → "my hen photo" (a starting point the user can edit). */
function guessName(fileName: string): string {
  const base = fileName.replace(/\.[a-z0-9]+$/i, "").replace(/[_-]+/g, " ").trim();
  return /^(img|image|photo|dsc|screenshot|pxl)\b/i.test(base) || /^\d+$/.test(base) ? "" : base.slice(0, 60);
}
