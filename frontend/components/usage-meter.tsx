"use client";

import { useEffect, useState } from "react";

import { api } from "@/lib/api";
import { cn } from "@/lib/utils";
import type { UsageReport } from "@/types/api";

/** "Free compute this month" meter for the hosted (Modal) backend; renders nothing elsewhere. */
export function UsageMeter() {
  const [usage, setUsage] = useState<UsageReport | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = () =>
      api
        .usage()
        .then((u) => !cancelled && setUsage(u))
        .catch(() => {});
    load(); // once per page load: polling would keep the scale-to-zero backend awake
    return () => {
      cancelled = true;
    };
  }, []);

  if (!usage?.available || usage.used_usd == null || !usage.free_credit_usd) return null;
  const share = Math.min(1, usage.used_usd / usage.free_credit_usd);
  const resets = usage.resets_on
    ? new Date(usage.resets_on + "T00:00:00Z").toLocaleDateString(undefined, { month: "short", day: "numeric" })
    : "next month";
  return (
    <div
      className="flex items-center gap-2"
      title={`Hosted on Modal's free plan: $${usage.free_credit_usd} of compute per month. Updated ${usage.updated_at ?? ""}`}
    >
      <span>
        {usage.exhausted
          ? `This month's free compute is used up — image generation resumes ${resets}.`
          : `Free compute this month: $${usage.used_usd.toFixed(2)} of $${usage.free_credit_usd.toFixed(0)} used`}
      </span>
      <span
        aria-hidden
        className="relative inline-block h-1.5 w-20 overflow-hidden rounded-full bg-muted"
      >
        <span
          className={cn(
            "absolute inset-y-0 left-0 rounded-full",
            share > 0.85 ? "bg-red-500" : share > 0.6 ? "bg-amber-500" : "bg-violet-500",
          )}
          style={{ width: `${Math.max(3, share * 100)}%` }}
        />
      </span>
    </div>
  );
}
