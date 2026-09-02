import { baseURL, protectedRoutes } from "@/resources";
import type { MetadataRoute } from "next";

/**
 * Password-protected routes answer 200 with a password prompt, so without a
 * Disallow a crawler would index that prompt as the page's content.
 *
 * Routes merely turned off in the config are deliberately not listed: they 404
 * now (see requireRouteEnabled), and a Disallow on a 404 does nothing except
 * stop Google from seeing the 404 that would drop the URL.
 */
function protectedPaths(): string[] {
  return Object.entries(protectedRoutes)
    .filter(([, isProtected]) => isProtected)
    .map(([path]) => path)
    .sort();
}

export default function robots(): MetadataRoute.Robots {
  return {
    rules: [
      {
        userAgent: "*",
        allow: "/",
        disallow: [
          // Named individually rather than blanket-disallowing /api/, which
          // would also block /api/og/generate and /api/rss. The OG route is the
          // og:image for four pages, and Twitter's card crawler honours
          // robots.txt, so blocking it would drop those preview images.
          "/api/authenticate",
          "/api/check-auth",
          ...protectedPaths(),
        ],
      },
    ],
    sitemap: `${baseURL}/sitemap.xml`,
    host: baseURL,
  };
}
