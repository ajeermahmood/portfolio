import { Mailchimp } from "@/components";
import { Posts } from "@/components/blog/Posts";
import { JsonLd } from "@/components/JsonLd";
import { baseURL, blog, newsletter } from "@/resources";
import { requireRouteEnabled } from "@/utils/routes";
import { generateMeta } from "@/utils/seo";
import { getPosts } from "@/utils/utils";
import { Column, Heading } from "@once-ui-system/core";

export async function generateMetadata() {
  return generateMeta({
    title: blog.title,
    description: blog.description,
    baseURL: baseURL,
    image: `/api/og/generate?title=${encodeURIComponent(blog.title)}`,
    path: blog.path,
    canonical: `${baseURL}${blog.path}`,
  });
}

export default function Blog() {
  requireRouteEnabled(blog.path);

  const posts = getPosts(["src", "app", "blog", "posts"]).sort(
    (a, b) =>
      new Date(b.metadata.publishedAt).getTime() - new Date(a.metadata.publishedAt).getTime(),
  );
  const postCount = posts.length;

  return (
    <Column maxWidth="m" paddingTop="24">
      {/*
        A listing page is a Blog, not a BlogPosting. Server-rendered rather than
        through Once UI's <Schema>, which injects client-side and so is invisible
        to the crawler.
      */}
      <JsonLd
        data={{
          "@context": "https://schema.org",
          "@type": "Blog",
          "@id": `${baseURL}${blog.path}`,
          url: `${baseURL}${blog.path}`,
          name: blog.title,
          description: blog.description,
          inLanguage: "en",
          isPartOf: { "@id": `${baseURL}/#website` },
          author: { "@id": `${baseURL}/#person` },
          publisher: { "@id": `${baseURL}/#person` },
          blogPost: posts.map((post) => ({
            "@type": "BlogPosting",
            "@id": `${baseURL}${blog.path}/${post.slug}`,
            url: `${baseURL}${blog.path}/${post.slug}`,
            headline: post.metadata.title,
            description: post.metadata.summary,
            datePublished: post.metadata.publishedAt,
            author: { "@id": `${baseURL}/#person` },
          })),
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
              name: blog.label,
              item: `${baseURL}${blog.path}`,
            },
          ],
        }}
      />
      <Heading marginBottom="l" variant="heading-strong-xl" marginLeft="24">
        {blog.title}
      </Heading>
      <Column fillWidth flex={1} gap="40">
        <Posts range={[1, 1]} thumbnail />
        {postCount > 1 && <Posts range={[2, 3]} columns="2" thumbnail direction="column" />}
        {newsletter.display && <Mailchimp marginBottom="l" />}
        {postCount > 3 && (
          <>
            <Heading as="h2" variant="heading-strong-xl" marginLeft="l">
              Earlier posts
            </Heading>
            <Posts range={[4]} columns="2" />
          </>
        )}
      </Column>
    </Column>
  );
}
