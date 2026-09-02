import { CustomMDX, ScrollToHash } from "@/components";
import { JsonLd } from "@/components/JsonLd";
import { Projects } from "@/components/work/Projects";
import { about, baseURL, person, social, work } from "@/resources";
import { formatDate } from "@/utils/formatDate";
import { requireRouteEnabled } from "@/utils/routes";
import { generateMeta } from "@/utils/seo";
import { getPosts } from "@/utils/utils";
import {
  Avatar,
  AvatarGroup,
  Button,
  Column,
  Flex,
  Heading,
  Line,
  Media,
  Meta,
  Row,
  SmartLink,
  Text,
} from "@once-ui-system/core";
import type { Metadata } from "next";
import { notFound } from "next/navigation";

export async function generateStaticParams(): Promise<{ slug: string }[]> {
  const posts = getPosts(["src", "app", "work", "projects"]);
  return posts.map((post) => ({
    slug: post.slug,
  }));
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ slug: string | string[] }>;
}): Promise<Metadata> {
  const routeParams = await params;
  const slugPath = Array.isArray(routeParams.slug)
    ? routeParams.slug.join("/")
    : routeParams.slug || "";

  const posts = getPosts(["src", "app", "work", "projects"]);
  const post = posts.find((post) => post.slug === slugPath);

  if (!post) return {};

  return generateMeta({
    title: post.metadata.title,
    description: post.metadata.description || post.metadata.summary,
    baseURL: baseURL,
    type: "article",
    publishedTime: post.metadata.publishedAt,
    author: {
      name: person.name,
      url: `${baseURL}${about.path}`,
    },
    image:
      post.metadata.image ||
      post.metadata.images?.[0] ||
      `/api/og/generate?title=${encodeURIComponent(post.metadata.title)}`,
    path: `${work.path}/${post.slug}`,
    canonical: `${baseURL}${work.path}/${post.slug}`,
  });
}

export default async function Project({
  params,
}: {
  params: Promise<{ slug: string | string[] }>;
}) {
  requireRouteEnabled(work.path);

  const routeParams = await params;
  const slugPath = Array.isArray(routeParams.slug)
    ? routeParams.slug.join("/")
    : routeParams.slug || "";

  const post = getPosts(["src", "app", "work", "projects"]).find((post) => post.slug === slugPath);

  if (!post) {
    notFound();
  }

  const avatars =
    post.metadata.team?.map((person) => ({
      src: person.avatar,
    })) || [];

  return (
    <>
      <JsonLd
        data={{
          "@context": "https://schema.org",
          "@type": "Article",
          mainEntityOfPage: {
            "@type": "WebPage",
            "@id": `${baseURL}${work.path}/${post.slug}`,
          },
          url: `${baseURL}${work.path}/${post.slug}`,
          headline: post.metadata.title,
          description: post.metadata.description || post.metadata.summary,
          image: [
            post.metadata.images?.[0]
              ? `${baseURL}${post.metadata.images[0]}`
              : `${baseURL}/api/og/generate?title=${encodeURIComponent(post.metadata.title)}`,
          ],
          datePublished: post.metadata.publishedAt,
          dateModified: post.metadata.publishedAt,
          inLanguage: "en",
          ...(post.metadata.stack?.length ? { keywords: post.metadata.stack.join(", ") } : {}),
          author: {
            "@type": "Person",
            name: person.name,
            url: `${baseURL}${about.path}`,
            image: `${baseURL}${person.avatar}`,
            jobTitle: person.role,
            sameAs: social.filter((s) => s.link.startsWith("http")).map((s) => s.link),
          },
          publisher: { "@type": "Person", name: person.name, url: baseURL },
        }}
      />
      <JsonLd
        data={{
          "@context": "https://schema.org",
          "@type": "BreadcrumbList",
          itemListElement: [
            { "@type": "ListItem", position: 1, name: "Home", item: baseURL },
            {
              "@type": "ListItem",
              position: 2,
              name: work.label,
              item: `${baseURL}${work.path}`,
            },
            {
              "@type": "ListItem",
              position: 3,
              name: post.metadata.title,
              item: `${baseURL}${work.path}/${post.slug}`,
            },
          ],
        }}
      />
      <Column as="section" maxWidth="m" horizontal="center" gap="l">
        <Column maxWidth="s" gap="16" horizontal="center" align="center">
          <SmartLink href="/work">
            <Text variant="label-strong-m">Projects</Text>
          </SmartLink>
          <Text variant="body-default-xs" onBackground="neutral-weak" marginBottom="12">
            {post.metadata.publishedAt && formatDate(post.metadata.publishedAt)}
          </Text>
          <Heading variant="display-strong-m">{post.metadata.title}</Heading>
        </Column>
        {post.metadata.team?.length > 0 && (
          <Row marginBottom="32" horizontal="center">
            <Row gap="16" vertical="center">
              <AvatarGroup reverse avatars={avatars} size="s" />
              <Text variant="label-default-m" onBackground="brand-weak">
                {post.metadata.team?.map((member, idx) => (
                  <span key={idx}>
                    {idx > 0 && (
                      <Text as="span" onBackground="neutral-weak">
                        ,{" "}
                      </Text>
                    )}
                    <SmartLink href={member.linkedIn}>{member.name}</SmartLink>
                  </span>
                ))}
              </Text>
            </Row>
          </Row>
        )}

        {(post.metadata.link || post.metadata.github) && (
          <Row gap="16" horizontal="center" marginBottom="24" wrap>
            {post.metadata.link && (
              <Button
                href={post.metadata.link}
                variant="secondary"
                size="s"
                suffixIcon="arrowUpRightFromSquare"
                label="View live"
              />
            )}
            {post.metadata.github && (
              <Button
                href={post.metadata.github}
                variant="tertiary"
                size="s"
                prefixIcon="github"
                label="Source"
              />
            )}
          </Row>
        )}
        {post.metadata.images.length > 0 && (
          <Media
            priority
            aspectRatio="16 / 9"
            radius="m"
            alt="image"
            src={post.metadata.images[0]}
          />
        )}
        <Column style={{ margin: "auto" }} as="article" maxWidth="xs">
          <CustomMDX source={post.content} />
        </Column>

        {post.metadata.images.length > 1 && (
          <Column fillWidth gap="24" marginTop="40">
            <Heading as="h2" variant="heading-strong-l" align="center">
              Screens
            </Heading>
            <Column fillWidth gap="16">
              {post.metadata.images.slice(1).map((image, idx) => (
                <Media
                  key={image}
                  enlarge
                  aspectRatio="16 / 9"
                  radius="m"
                  sizes="(max-width: 960px) 100vw, 960px"
                  alt={`${post.metadata.title} screen ${idx + 2}`}
                  src={image}
                />
              ))}
            </Column>
          </Column>
        )}
        <Column fillWidth gap="40" horizontal="center" marginTop="40">
          <Line maxWidth="40" />
          <Heading as="h2" variant="heading-strong-xl" marginBottom="24">
            Related projects
          </Heading>
          <Projects exclude={[post.slug]} range={[1, 2]} />
        </Column>
        <ScrollToHash />
      </Column>
    </>
  );
}
