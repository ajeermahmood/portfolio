import { Button, Column, Heading, Row, Text } from "@once-ui-system/core";
import { person } from "@/resources";

/**
 * Email is the only channel worth putting a button on. A contact form needs a
 * backend and silently loses messages when it breaks, and a mailto link works
 * from every device without one.
 */
export function ContactCTA() {
  const subject = encodeURIComponent(`Hello ${person.firstName}`);

  return (
    <Column
      fillWidth
      horizontal="center"
      align="center"
      gap="16"
      paddingY="64"
      paddingX="l"
      marginTop="40"
    >
      <Heading as="h2" variant="display-strong-xs" wrap="balance">
        Get in touch
      </Heading>
      <Text
        variant="body-default-l"
        onBackground="neutral-weak"
        wrap="balance"
        style={{ maxWidth: "34rem" }}
      >
        I am open to senior full-stack roles, relocation, and remote contract work. Email is the
        fastest way to reach me and I reply to everything.
      </Text>
      <Row gap="12" paddingTop="12" wrap horizontal="center">
        <Button
          href={`mailto:${person.email}?subject=${subject}`}
          prefixIcon="email"
          variant="primary"
          size="m"
          data-border="rounded"
          label={person.email}
        />
        <Button
          href="/document/RESUME_AJEER.pdf"
          prefixIcon="document"
          variant="secondary"
          size="m"
          data-border="rounded"
          label="Resume"
        />
      </Row>
    </Column>
  );
}
