import { JsonLd } from "@/components/JsonLd";
import { baseURL, contact, person, social } from "@/resources";
import { requireRouteEnabled } from "@/utils/routes";
import { generateMeta } from "@/utils/seo";
import { Button, Column, Heading, Row, Text } from "@once-ui-system/core";

export async function generateMetadata() {
  return generateMeta({
    title: contact.title,
    description: contact.description,
    baseURL: baseURL,
    path: contact.path,
    canonical: `${baseURL}${contact.path}`,
    image: `/api/og/generate?title=${encodeURIComponent(contact.title)}`,
  });
}

export default function Contact() {
  requireRouteEnabled(contact.path);

  const subject = encodeURIComponent(`Hello ${person.firstName}`);

  return (
    <>
      <JsonLd
        data={{
          "@context": "https://schema.org",
          "@type": "ContactPage",
          "@id": `${baseURL}${contact.path}`,
          url: `${baseURL}${contact.path}`,
          name: contact.title,
          description: contact.description,
          inLanguage: "en",
          isPartOf: { "@id": `${baseURL}/#website` },
          mainEntity: { "@id": `${baseURL}/#person` },
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
              name: contact.label,
              item: `${baseURL}${contact.path}`,
            },
          ],
        }}
      />
      <Column maxWidth="s" horizontal="center" align="center" gap="16" paddingY="80" fillWidth>
        <Heading variant="display-strong-l" wrap="balance">
          Get in touch
        </Heading>
        <Text variant="body-default-l" onBackground="neutral-weak" wrap="balance">
          I am open to senior full-stack roles, relocation, and remote contract work. Email is the
          fastest way to reach me and I reply to everything.
        </Text>

        <Row gap="12" paddingTop="20" wrap horizontal="center">
          <Button
            href={`mailto:${person.email}?subject=${subject}`}
            prefixIcon="email"
            variant="primary"
            size="l"
            data-border="rounded"
            label={person.email}
          />
        </Row>

        <Row gap="12" paddingTop="12" wrap horizontal="center" data-border="rounded">
          {social
            .filter((item) => item.name !== "Email" && item.link)
            .map((item) => (
              <Button
                key={item.name}
                href={item.link}
                prefixIcon={item.icon}
                label={item.name}
                size="s"
                weight="default"
                variant="secondary"
              />
            ))}
        </Row>

        <Text variant="body-default-s" onBackground="neutral-weak" paddingTop="24">
          Based in India, open to relocation and remote roles.
        </Text>
      </Column>
    </>
  );
}
