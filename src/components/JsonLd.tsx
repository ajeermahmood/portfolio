/**
 * Server-rendered JSON-LD.
 *
 * Once UI's <Schema> uses next/script, which injects the tag client-side, so the
 * structured data is absent from the static HTML a crawler reads first. This
 * renders a plain script tag in the server output instead.
 */
export function JsonLd({ data }: { data: Record<string, unknown> }) {
  return (
    <script
      type="application/ld+json"
      // JSON.stringify output is escaped for the one sequence that can break out
      dangerouslySetInnerHTML={{
        __html: JSON.stringify(data).replace(/</g, "\\u003c"),
      }}
    />
  );
}
