import type { Metadata } from "next";

import { StoryLibrary } from "@/components/story-library";

export const metadata: Metadata = { title: "My stories" };

export default function StoriesPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">My stories</h1>
        <p className="text-sm text-muted-foreground">Every story you illustrate is saved by your local StoryPix backend.</p>
      </div>
      <StoryLibrary />
    </div>
  );
}
