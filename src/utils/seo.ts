import { Meta } from "@once-ui-system/core";
import type { Metadata } from "next";

type MetaArgs = Parameters<typeof Meta.generate>[0];

/**
 * Meta.generate accepts a `canonical` prop but only writes it out when an
 * `alternates` array is passed as well, so a canonical on its own is dropped.
 * This wraps it and always emits one, falling back to baseURL + path.
 */
export function generateMeta(args: MetaArgs): Metadata {
  const meta = Meta.generate(args);

  const base = args.baseURL.replace(/\/+$/, "");
  const path = args.path ? (args.path.startsWith("/") ? args.path : `/${args.path}`) : "";
  const fallback = `${base}${path}`.replace(/(?<!:)\/+$/, "");

  return {
    ...meta,
    alternates: {
      ...(meta.alternates ?? {}),
      canonical: args.canonical ?? fallback,
    },
  };
}
