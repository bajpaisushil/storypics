import Link from "next/link";

import { Button } from "@/components/ui/button";

export default function NotFound() {
  return (
    <div className="mx-auto flex max-w-md flex-col items-center gap-3 py-24 text-center">
      <p className="text-sm font-medium text-primary">404</p>
      <h1 className="text-xl font-semibold">This page doesn&apos;t exist</h1>
      <p className="text-sm text-muted-foreground">Head back and start a new story.</p>
      <Button asChild>
        <Link href="/">Create a story</Link>
      </Button>
    </div>
  );
}
