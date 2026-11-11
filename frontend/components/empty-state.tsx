import { Images, Palette, PenLine } from "lucide-react";

const STEPS = [
  { icon: PenLine, title: "Write your story", text: "Paste a story or try the example. Any length up to 6,000 characters." },
  { icon: Palette, title: "Choose a style", text: "Pick an illustration style, a scene count and an aspect ratio." },
  {
    icon: Images,
    title: "Watch it come to life",
    text: "StoryPix finds the scenes and characters, draws each character once, then illustrates every scene in order.",
  },
];

export function EmptyState() {
  return (
    <div className="flex min-h-[520px] flex-col items-center justify-center rounded-2xl border border-dashed border-border bg-background/50 px-6 py-12 text-center">
      <div className="mb-6 grid size-14 place-items-center rounded-2xl bg-accent text-accent-foreground">
        <Images className="size-6" />
      </div>
      <h2 className="text-lg font-semibold tracking-tight">Your illustrated story will appear here</h2>
      <p className="mt-1 max-w-md text-sm text-muted-foreground">
        Scenes show up one by one as they are drawn, and you can redraw any of them.
      </p>
      <ol className="mt-8 grid w-full max-w-2xl gap-3 text-left sm:grid-cols-3">
        {STEPS.map(({ icon: Icon, title, text }, i) => (
          <li key={title} className="rounded-xl border bg-card p-4">
            <div className="mb-2 flex items-center gap-2 text-sm font-medium">
              <span className="grid size-6 place-items-center rounded-md bg-muted text-xs tabular-nums">{i + 1}</span>
              <Icon className="size-4 text-primary" />
              {title}
            </div>
            <p className="text-xs leading-relaxed text-muted-foreground">{text}</p>
          </li>
        ))}
      </ol>
    </div>
  );
}
