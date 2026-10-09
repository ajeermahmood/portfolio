import fs from "node:fs";
import path from "node:path";
import { CustomMDX, ScrollToHash } from "@/components";
import { ContactCTA } from "@/components/ContactCTA";
import { JsonLd } from "@/components/JsonLd";
import { about, baseURL, howIWork, person, social } from "@/resources";
import { formatDate } from "@/utils/formatDate";
import { requireRouteEnabled } from "@/utils/routes";
import { generateMeta } from "@/utils/seo";
import { Column, Heading, HeadingNav, Row, Text } from "@once-ui-system/core";
import matter from "gray-matter";

/**
 * Read once at build time. The path is a literal on purpose, for the same reason
 * getPosts spells its directories out: a computed path makes Turbopack trace the
 * whole project into the server output.
 */
function readPage() {
  const file = path.join(process.cwd(), "src", "app", "how-i-work", "how-i-work.mdx");
  const { data, content } = matter(fs.readFileSync(file, "utf-8"));
  return {
    content,
    subtitle: (data.subtitle as string) || "",
    publishedAt: (data.publishedAt as string) || "",
    updatedAt: (data.updatedAt as string) || "",
  };
}

export async function generateMetadata() {
  const page = readPage();
  return generateMeta({
    title: howIWork.title,
    description: howIWork.description,
    baseURL: baseURL,
    type: "article",
    publishedTime: page.publishedAt,
    modifiedTime: page.updatedAt || page.publishedAt,
    author: { name: person.name, url: `${baseURL}${about.path}` },
    image: `/api/og/generate?title=${encodeURIComponent(howIWork.title)}`,
    path: howIWork.path,
    canonical: `${baseURL}${howIWork.path}`,
  });
}

export default function HowIWorkPage() {
  requireRouteEnabled(howIWork.path);
  const page = readPage();
  const url = `${baseURL}${howIWork.path}`;

  return (
    <>
      <JsonLd
        data={{
          "@context": "https://schema.org",
          "@type": "BreadcrumbList",
          itemListElement: [
            { "@type": "ListItem", position: 1, name: "Home", item: baseURL },
            { "@type": "ListItem", position: 2, name: howIWork.label, item: url },
          ],
        }}
      />
      <JsonLd
        data={{
          "@context": "https://schema.org",
          "@type": "Article",
          mainEntityOfPage: { "@type": "WebPage", "@id": url },
          url,
          headline: howIWork.title,
          description: howIWork.description,
          image: [`${baseURL}/api/og/generate?title=${encodeURIComponent(howIWork.title)}`],
          datePublished: page.publishedAt,
          dateModified: page.updatedAt || page.publishedAt,
          inLanguage: "en",
          keywords: "AI coding agents, Claude Code, CI guardrails, MCP, LLM evals",
          author: {
            "@type": "Person",
            name: person.name,
            url: `${baseURL}${about.path}`,
            jobTitle: person.role,
            sameAs: social.filter((s) => s.link.startsWith("http")).map((s) => s.link),
          },
          isPartOf: { "@id": `${baseURL}/#website` },
        }}
      />
      <Row fillWidth>
        <Row maxWidth={12} m={{ hide: true }} />
        <Row fillWidth horizontal="center">
          <Column as="section" maxWidth="m" horizontal="center" gap="l" paddingTop="24">
            <Column maxWidth="s" gap="16" horizontal="center" align="center">
              <Text variant="label-strong-m">{howIWork.label}</Text>
              {page.publishedAt && (
                <Text variant="body-default-xs" onBackground="neutral-weak" marginBottom="12">
                  {formatDate(page.publishedAt)}
                  {page.updatedAt && ` · Updated ${formatDate(page.updatedAt)}`}
                </Text>
              )}
              <Heading variant="display-strong-m">{howIWork.title}</Heading>
              {page.subtitle && (
                <Text
                  variant="body-default-l"
                  onBackground="neutral-weak"
                  align="center"
                  style={{ fontStyle: "italic" }}
                >
                  {page.subtitle}
                </Text>
              )}
            </Column>
            <Column as="article" maxWidth="s" marginTop="24">
              <CustomMDX source={page.content} />
            </Column>
            <ContactCTA />
            <ScrollToHash />
          </Column>
        </Row>
        <Column
          maxWidth={12}
          paddingLeft="40"
          fitHeight
          position="sticky"
          top="80"
          gap="16"
          m={{ hide: true }}
        >
          <HeadingNav fitHeight />
        </Column>
      </Row>
    </>
  );
}
