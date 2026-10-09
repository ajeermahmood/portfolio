import fs from "node:fs";
import path from "node:path";
import matter from "gray-matter";

type Team = {
  name: string;
  role: string;
  avatar: string;
  linkedIn: string;
};

type Metadata = {
  title: string;
  subtitle?: string;
  publishedAt: string;
  /** Set when the page is materially revised. Feeds dateModified and the sitemap. */
  updatedAt?: string;
  /** Shown on the listing cards and at the top of the page. */
  summary: string;
  /**
   * Meta description and JSON-LD description, when the summary runs longer than
   * the ~155 characters Google will show. Falls back to summary when unset, so
   * only the long ones need it. Kept separate so tightening the snippet does
   * not shorten the copy on the card.
   */
  description?: string;
  image?: string;
  images: string[];
  tag?: string;
  team: Team[];
  link?: string;
  github?: string;
  stack?: string[];
  /**
   * Position on the work page, lowest first. Ranking lives here rather than in
   * publishedAt because publishedAt feeds the sitemap and JSON-LD datePublished,
   * and reordering the portfolio is not the same as the work changing date.
   * Unset sorts after every ranked entry, newest first.
   */
  order?: number;
};

import { notFound } from "next/navigation";
import { cache } from "react";

function getMDXFiles(dir: string) {
  if (!fs.existsSync(dir)) {
    notFound();
  }

  return fs.readdirSync(dir).filter((file) => path.extname(file) === ".mdx");
}

function readMDXFile(filePath: string) {
  if (!fs.existsSync(filePath)) {
    notFound();
  }

  const rawContent = fs.readFileSync(filePath, "utf-8");
  const { data, content } = matter(rawContent);

  const metadata: Metadata = {
    title: data.title || "",
    subtitle: data.subtitle || "",
    publishedAt: data.publishedAt,
    updatedAt: data.updatedAt || undefined,
    summary: data.summary || "",
    description: data.description || "",
    image: data.image || "",
    images: data.images || [],
    tag: data.tag || "",
    team: data.team || [],
    link: data.link || "",
    github: data.github || "",
    stack: data.stack || [],
    order: typeof data.order === "number" ? data.order : undefined,
  };

  return { metadata, content };
}

function getMDXData(dir: string) {
  const mdxFiles = getMDXFiles(dir);
  return mdxFiles.map((file) => {
    const { metadata, content } = readMDXFile(path.join(dir, file));
    const slug = path.basename(file, path.extname(file));

    return {
      metadata,
      slug,
      content,
    };
  });
}

/**
 * Content lives in exactly two directories, and both are spelled out as string
 * literals on purpose.
 *
 * Building the path by spreading a caller-supplied array made it impossible for
 * the bundler to know what would be read, so Turbopack traced the entire project
 * into the server output, public/ included. Literals let it scope the trace to
 * these two folders.
 */
const CONTENT_DIRS = {
  "src/app/blog/posts": () => path.join(process.cwd(), "src", "app", "blog", "posts"),
  "src/app/work/projects": () => path.join(process.cwd(), "src", "app", "work", "projects"),
} as const;

type ContentKey = keyof typeof CONTENT_DIRS;

/** Work-page order: ranked entries first by `order`, the rest newest first. */
export function byRankThenDate(
  a: { metadata: { order?: number; publishedAt: string } },
  b: { metadata: { order?: number; publishedAt: string } },
): number {
  const ra = a.metadata.order ?? Number.POSITIVE_INFINITY;
  const rb = b.metadata.order ?? Number.POSITIVE_INFINITY;
  if (ra !== rb) return ra - rb;
  return new Date(b.metadata.publishedAt).getTime() - new Date(a.metadata.publishedAt).getTime();
}

/**
 * Memoised per request: the home page and every detail page read the same
 * directory several times, and parsing the MDX once per render is enough.
 */
export const getPosts = cache((segments: readonly string[]) => {
  const key = segments.join("/") as ContentKey;
  const resolve = CONTENT_DIRS[key];

  if (!resolve) {
    throw new Error(
      `getPosts: unknown content directory "${key}". Add it to CONTENT_DIRS as a literal path.`,
    );
  }

  return getMDXData(resolve());
});
