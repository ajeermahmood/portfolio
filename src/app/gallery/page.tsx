import GalleryView from "@/components/gallery/GalleryView";
import { JsonLd } from "@/components/JsonLd";
import { baseURL, gallery } from "@/resources";
import { isRouteEnabled, requireRouteEnabled } from "@/utils/routes";
import { generateMeta } from "@/utils/seo";
import { Flex } from "@once-ui-system/core";

const enabled = isRouteEnabled(gallery.path);

export async function generateMetadata() {
  // A route that is off still gets its metadata generated during the build, so
  // it has to opt out of indexing here as well as 404 below.
  if (!enabled) {
    return { title: "Not found", robots: { index: false, follow: false } };
  }

  return generateMeta({
    title: gallery.title,
    description: gallery.description,
    baseURL: baseURL,
    image: `/api/og/generate?title=${encodeURIComponent(gallery.title)}`,
    path: gallery.path,
    canonical: `${baseURL}${gallery.path}`,
  });
}

export default function Gallery() {
  requireRouteEnabled(gallery.path);

  return (
    <Flex maxWidth="l">
      <JsonLd
        data={{
          "@context": "https://schema.org",
          "@type": "CollectionPage",
          "@id": `${baseURL}${gallery.path}`,
          url: `${baseURL}${gallery.path}`,
          name: gallery.title,
          description: gallery.description,
          inLanguage: "en",
          isPartOf: { "@id": `${baseURL}/#website` },
          about: { "@id": `${baseURL}/#person` },
          author: { "@id": `${baseURL}/#person` },
        }}
      />
      <GalleryView />
    </Flex>
  );
}
