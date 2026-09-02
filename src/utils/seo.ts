import { person } from "@/resources";
import { Meta } from "@once-ui-system/core";
import type { Metadata } from "next";

type MetaArgs = Parameters<typeof Meta.generate>[0];

/**
 * Site-wide crawler directives. `index, follow` is already the default when no
 * robots tag is present, so the value here is the `max-*` half: it opts into
 * large image thumbnails (Discover, image search) and removes Google's snippet
 * length cap. Without it Google falls back to a short snippet and a thumbnail.
 */
const DEFAULT_ROBOTS = "index, follow, max-image-preview:large, max-snippet:-1, max-video-preview:-1";

const SITE_NAME = `${person.name}, ${person.role}`;

/** BCP-47 `en` as an OpenGraph locale, which wants the underscored form. */
const OG_LOCALE = "en_US";

/**
 * Meta.generate accepts a `canonical` prop but only writes it out when an
 * `alternates` array is passed as well, so a canonical on its own is dropped.
 * This wraps it and always emits one, falling back to baseURL + path.
 *
 * It also fills in what Meta.generate has no props for: the site-wide robots
 * directives, and og:site_name / og:locale, so every page carries them without
 * each page having to repeat them.
 */
export function generateMeta(args: MetaArgs): Metadata {
  // Only default the robots directives when the caller has expressed no
  // opinion. Passing them alongside `noindex` would override the noindex.
  const robots =
    args.robots ?? (args.noindex || args.nofollow ? undefined : DEFAULT_ROBOTS);

  const meta = Meta.generate({ ...args, robots });

  const base = args.baseURL.replace(/\/+$/, "");
  const path = args.path ? (args.path.startsWith("/") ? args.path : `/${args.path}`) : "";
  const fallback = `${base}${path}`.replace(/(?<!:)\/+$/, "");

  return {
    ...meta,
    openGraph: {
      ...(meta.openGraph ?? {}),
      siteName: SITE_NAME,
      locale: OG_LOCALE,
    },
    alternates: {
      ...(meta.alternates ?? {}),
      canonical: args.canonical ?? fallback,
    },
  };
}
