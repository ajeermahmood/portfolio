import { home, person } from "@/resources";
import type { MetadataRoute } from "next";

/**
 * Emitted at /manifest.webmanifest and linked from <head> automatically.
 * Mostly a mobile/installability signal rather than a ranking one, but it is
 * also where the browser and Google's mobile crawler pick up the site name and
 * theme colour, so it is worth keeping in step with the content config.
 */
export default function manifest(): MetadataRoute.Manifest {
  return {
    name: `${person.name}, ${person.role}`,
    short_name: person.name,
    description: home.description,
    start_url: "/",
    display: "standalone",
    // The resolved dark page-background token. A manifest takes a single
    // colour, so it uses the dark one; <meta name="theme-color"> in the layout
    // is what actually follows the light/dark switch.
    background_color: "#0A0A0A",
    theme_color: "#0A0A0A",
    icons: [
      {
        src: "/favicon.png",
        sizes: "192x192",
        type: "image/png",
        purpose: "any",
      },
    ],
  };
}
