import { Box, BookOpen, Camera, Droplets, Palette, Smile, Sparkles, Zap, type LucideIcon } from "lucide-react";

import type { StyleId, StyleInfo } from "@/types/api";

export interface StyleMeta extends StyleInfo {
  icon: LucideIcon;
  /** Tailwind gradient shown until a real preview image exists in /public/styles */
  swatch: string;
}

/** Mirrors backend/app/services/styles.py so the picker renders even when the backend is offline. */
export const STYLE_CATALOG: StyleMeta[] = [
  { id: "storybook", name: "Children's Storybook", tagline: "Warm, whimsical picture-book art", icon: BookOpen,
    swatch: "from-amber-200 via-rose-200 to-orange-300", preview_url: "/styles/storybook.webp" },
  { id: "watercolor", name: "Watercolor", tagline: "Soft washes on textured paper", icon: Droplets,
    swatch: "from-teal-200 via-sky-200 to-rose-200", preview_url: "/styles/watercolor.webp" },
  { id: "cartoon", name: "Cartoon", tagline: "Bold outlines, bright flat colours", icon: Smile,
    swatch: "from-amber-300 via-orange-400 to-rose-400", preview_url: "/styles/cartoon.webp" },
  { id: "3d", name: "3D Animation", tagline: "Feature-film CGI look", icon: Box,
    swatch: "from-violet-400 via-purple-500 to-indigo-600", preview_url: "/styles/3d.webp" },
  { id: "anime", name: "Anime", tagline: "Clean line art and cel shading", icon: Sparkles,
    swatch: "from-sky-300 via-indigo-400 to-fuchsia-400", preview_url: "/styles/anime.webp" },
  { id: "realistic", name: "Realistic", tagline: "Lifelike, photographic detail", icon: Camera,
    swatch: "from-stone-300 via-stone-500 to-stone-700", preview_url: "/styles/realistic.webp" },
  { id: "digital_painting", name: "Digital Painting", tagline: "Painterly, atmospheric concept art",
    icon: Palette, swatch: "from-emerald-400 via-teal-600 to-slate-700",
    preview_url: "/styles/digital_painting.webp" },
  { id: "comic", name: "Comic Book", tagline: "Inked lines, halftone, dynamic panels", icon: Zap,
    swatch: "from-yellow-300 via-red-400 to-blue-500", preview_url: "/styles/comic.webp" },
];

export function styleMeta(id: StyleId): StyleMeta {
  return STYLE_CATALOG.find((s) => s.id === id) ?? STYLE_CATALOG[0];
}

export const EXAMPLE_STORY =
  "Once upon a time, there was a hen who lived on a farm. One day, she met a cow. They became good friends " +
  "and explored the farm together. One evening, they helped a lost little rabbit find its home.";
