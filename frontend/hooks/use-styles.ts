"use client";

import { useEffect, useState } from "react";

import { api } from "@/lib/api";
import { STYLE_CATALOG, type StyleMeta } from "@/lib/styles";

/** Style catalogue: the bundled list (works offline), refreshed with names/taglines from the backend. */
export function useStyles(): StyleMeta[] {
  const [styles, setStyles] = useState<StyleMeta[]>(STYLE_CATALOG);

  useEffect(() => {
    let cancelled = false;
    api
      .styles()
      .then((remote) => {
        if (cancelled) return;
        setStyles(
          STYLE_CATALOG.map((local) => {
            const match = remote.find((r) => r.id === local.id);
            return match ? { ...local, name: match.name, tagline: match.tagline } : local;
          }),
        );
      })
      .catch(() => {
        /* offline: keep the bundled catalogue */
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return styles;
}
