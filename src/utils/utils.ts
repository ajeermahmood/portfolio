import fs from "fs";
import path from "path";
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
  summary: string;
  image?: string;
  images: string[];
  tag?: string;
  team: Team[];
  link?: string;
  github?: string;
  stack?: string[];
};

import { notFound } from "next/navigation";

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
    summary: data.summary || "",
    image: data.image || "",
    images: data.images || [],
    tag: data.tag || [],
    team: data.team || [],
    link: data.link || "",
    github: data.github || "",
    stack: data.stack || [],
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

export function getPosts(segments: readonly string[]) {
  const key = segments.join("/") as ContentKey;
  const resolve = CONTENT_DIRS[key];

  if (!resolve) {
    throw new Error(
      `getPosts: unknown content directory "${key}". Add it to CONTENT_DIRS as a literal path.`,
    );
  }

  return getMDXData(resolve());
}
