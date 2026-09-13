import fs from "node:fs";
import path from "node:path";
import { baseURL, blog, person } from "@/resources";
import { getPosts } from "@/utils/utils";
import { NextResponse } from "next/server";

/** RSS is XML: a title with an ampersand or angle bracket must be escaped or the feed stops parsing. */
function escapeXml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

const MIME: Record<string, string> = {
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".webp": "image/webp",
  ".gif": "image/gif",
};

/** RSS 2.0 requires type and length on an enclosure; both come from the file in public/. */
function enclosure(image: string): string {
  const file = path.join(process.cwd(), "public", image);
  if (!fs.existsSync(file)) return "";
  const type = MIME[path.extname(image).toLowerCase()] ?? "application/octet-stream";
  const length = fs.statSync(file).size;
  return `<enclosure url="${baseURL}${image}" type="${type}" length="${length}" />`;
}

export async function GET() {
  const posts = getPosts(["src", "app", "blog", "posts"]);

  const sortedPosts = [...posts].sort((a, b) => {
    return new Date(b.metadata.publishedAt).getTime() - new Date(a.metadata.publishedAt).getTime();
  });

  const newest = sortedPosts[0]?.metadata;
  const lastBuild = new Date(newest?.updatedAt || newest?.publishedAt || Date.now());
  const editor = `${person.email} (${person.name})`;

  const rssXml = `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">
  <channel>
    <title>${escapeXml(blog.title)}</title>
    <link>${baseURL}/blog</link>
    <description>${escapeXml(blog.description)}</description>
    <language>${person.locale ?? "en"}</language>
    <lastBuildDate>${lastBuild.toUTCString()}</lastBuildDate>
    <atom:link href="${baseURL}/api/rss" rel="self" type="application/rss+xml" />
    <managingEditor>${escapeXml(editor)}</managingEditor>
    <webMaster>${escapeXml(editor)}</webMaster>
    <image>
      <url>${baseURL}${person.avatar}</url>
      <title>${escapeXml(blog.title)}</title>
      <link>${baseURL}/blog</link>
    </image>
    ${sortedPosts
      .map(
        (post) => `
    <item>
      <title>${escapeXml(post.metadata.title)}</title>
      <link>${baseURL}/blog/${post.slug}</link>
      <guid>${baseURL}/blog/${post.slug}</guid>
      <pubDate>${new Date(post.metadata.publishedAt).toUTCString()}</pubDate>
      <description><![CDATA[${post.metadata.summary}]]></description>
      ${post.metadata.image ? enclosure(post.metadata.image) : ""}
      ${post.metadata.tag ? `<category>${escapeXml(post.metadata.tag)}</category>` : ""}
      <author>${escapeXml(editor)}</author>
    </item>`,
      )
      .join("")}
  </channel>
</rss>`;

  return new NextResponse(rssXml, {
    headers: {
      "Content-Type": "application/rss+xml; charset=utf-8",
      "Cache-Control": "public, max-age=3600, s-maxage=3600, stale-while-revalidate=86400",
    },
  });
}
