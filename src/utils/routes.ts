import { routes } from "@/resources";
import { notFound } from "next/navigation";

/**
 * Whether a route is turned on in the static routes config.
 */
export function isRouteEnabled(pathname: string | null): boolean {
  if (!pathname) return false;

  if (pathname in routes) {
    return routes[pathname as keyof typeof routes];
  }

  const dynamicRoutes = ["/blog", "/work"] as const;
  for (const route of dynamicRoutes) {
    if (pathname.startsWith(route) && routes[route]) {
      return true;
    }
  }

  return false;
}

/**
 * Called at the top of every page whose route the config can switch off, so a
 * disabled route answers 404 instead of 200.
 *
 * It has to happen here rather than in RouteGuard: RouteGuard is a client
 * component that discards `children` when a route is off, which swallows the
 * notFound() thrown inside them. The page still rendered a "not found" body,
 * but the response stayed 200 — a soft 404, which Google will happily index
 * and then report as a crawl error.
 */
export function requireRouteEnabled(pathname: string): void {
  if (!isRouteEnabled(pathname)) {
    notFound();
  }
}
