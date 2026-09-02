import { JsonLd } from "@/components/JsonLd";
import { Projects } from "@/components/work/Projects";
import { baseURL, work } from "@/resources";
import { requireRouteEnabled } from "@/utils/routes";
import { generateMeta } from "@/utils/seo";
import { getPosts } from "@/utils/utils";
import { Column, Heading } from "@once-ui-system/core";

export async function generateMetadata() {
  return generateMeta({
    title: work.title,
    description: work.description,
    baseURL: baseURL,
    image: `/api/og/generate?title=${encodeURIComponent(work.title)}`,
    path: work.path,
    canonical: `${baseURL}${work.path}`,
  });
}

export default function Work() {
  requireRouteEnabled(work.path);

  const projects = getPosts(["src", "app", "work", "projects"]).sort(
    (a, b) =>
      new Date(b.metadata.publishedAt).getTime() - new Date(a.metadata.publishedAt).getTime(),
  );

  return (
    <Column maxWidth="m" paddingTop="24">
      {/*
        Once UI's <Schema> renders through next/script, so its JSON-LD is absent
        from the HTML a crawler reads. These are server-rendered, and describe
        the page for what it is: a list, with each project as a named entry, so
        the individual project pages are discoverable from the listing.
      */}
      <JsonLd
        data={{
          "@context": "https://schema.org",
          "@type": "CollectionPage",
          "@id": `${baseURL}${work.path}`,
          url: `${baseURL}${work.path}`,
          name: work.title,
          description: work.description,
          inLanguage: "en",
          isPartOf: { "@id": `${baseURL}/#website` },
          about: { "@id": `${baseURL}/#person` },
          author: { "@id": `${baseURL}/#person` },
          mainEntity: {
            "@type": "ItemList",
            numberOfItems: projects.length,
            itemListElement: projects.map((project, index) => ({
              "@type": "ListItem",
              position: index + 1,
              name: project.metadata.title,
              url: `${baseURL}${work.path}/${project.slug}`,
            })),
          },
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
          ],
        }}
      />
      <Heading marginBottom="l" variant="heading-strong-xl" align="center">
        {work.title}
      </Heading>
      <Projects />
    </Column>
  );
}
