import { Suspense } from "react";

import { Studio } from "@/components/studio";
import { Skeleton } from "@/components/ui/skeleton";

function StudioFallback() {
  return (
    <div className="grid gap-6 lg:grid-cols-[420px_minmax(0,1fr)] lg:gap-8">
      <Skeleton className="h-[640px] rounded-2xl" />
      <Skeleton className="h-[520px] rounded-2xl" />
    </div>
  );
}

export default function HomePage() {
  // Studio reads ?job= from the URL, so it renders on the client inside a Suspense boundary.
  return (
    <Suspense fallback={<StudioFallback />}>
      <Studio />
    </Suspense>
  );
}
