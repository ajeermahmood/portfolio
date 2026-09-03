import { baseURL } from "@/resources";
import type { MetadataRoute } from "next";

/**
 * Nothing is disallowed. Routes turned off in the config 404 (see
 * requireRouteEnabled), and a Disallow on a 404 does nothing except stop Google
 * from seeing the 404 that would drop the URL. /api/og/generate and /api/rss
 * stay crawlable on purpose: the OG route is the og:image for four pages, and
 * Twitter's card crawler honours robots.txt.
 */
export default function robots(): MetadataRoute.Robots {
  return {
    rules: [
      {
        userAgent: "*",
        allow: "/",
      },
    ],
    sitemap: `${baseURL}/sitemap.xml`,
    host: baseURL,
  };
}
