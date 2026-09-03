import { baseURL, routes as routesConfig } from "@/resources";
import { getPosts } from "@/utils/utils";
import type { MetadataRoute } from "next";

/** Newest publishedAt in a set of posts, or undefined when there are none. */
function newestDate(posts: { metadata: { publishedAt: string } }[]): string | undefined {
  const dates = posts
    .map((post) => post.metadata.publishedAt)
    .filter(Boolean)
    .sort();
  return dates.at(-1);
}

export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  const posts = getPosts(["src", "app", "blog", "posts"]);
  const projects = getPosts(["src", "app", "work", "projects"]);

  const blogs = posts.map((post) => ({
    url: `${baseURL}/blog/${post.slug}`,
    lastModified: post.metadata.publishedAt,
    changeFrequency: "yearly" as const,
    priority: 0.7,
  }));

  const works = projects.map((post) => ({
    url: `${baseURL}/work/${post.slug}`,
    lastModified: post.metadata.publishedAt,
    changeFrequency: "yearly" as const,
    priority: 0.7,
  }));

  // Index pages are only as fresh as the newest thing they list. Stamping them
  // with the build date instead would tell Google the whole site changed on
  // every deploy, which trains it to ignore lastmod. Routes with no content
  // behind them get no lastmod at all, which is valid and honest.
  const listingDates: Record<string, string | undefined> = {
    "/": newestDate([...posts, ...projects]),
    "/blog": newestDate(posts),
    "/work": newestDate(projects),
  };

  const priorities: Record<string, number> = {
    "/": 1,
    "/work": 0.9,
    "/about": 0.8,
    "/blog": 0.8,
    "/contact": 0.6,
  };

  const activeRoutes = Object.keys(routesConfig).filter(
    (route) => routesConfig[route as keyof typeof routesConfig],
  );

  const staticRoutes = activeRoutes.map((route) => {
    const lastModified = listingDates[route];
    return {
      url: `${baseURL}${route !== "/" ? route : ""}`,
      ...(lastModified ? { lastModified } : {}),
      changeFrequency: "monthly" as const,
      priority: priorities[route] ?? 0.5,
    };
  });

  return [...staticRoutes, ...blogs, ...works];
}
