import { ProjectCard } from "@/components";
import { getPosts } from "@/utils/utils";
import { Column } from "@once-ui-system/core";

interface ProjectsProps {
  range?: [number, number?];
  exclude?: string[];
  /** Rank by how many stack entries a project shares with these, newest first as the tiebreak. */
  similarTo?: string[];
}

export function Projects({ range, exclude, similarTo }: ProjectsProps) {
  let allProjects = getPosts(["src", "app", "work", "projects"]);

  // Exclude by slug (exact match)
  if (exclude && exclude.length > 0) {
    allProjects = allProjects.filter((post) => !exclude.includes(post.slug));
  }

  const byDate = (a: (typeof allProjects)[number], b: (typeof allProjects)[number]) =>
    new Date(b.metadata.publishedAt).getTime() - new Date(a.metadata.publishedAt).getTime();

  const overlap = (stack: string[] = []) =>
    similarTo ? stack.filter((item) => similarTo.includes(item)).length : 0;

  const sortedProjects = [...allProjects].sort(
    (a, b) => overlap(b.metadata.stack) - overlap(a.metadata.stack) || byDate(a, b),
  );

  const displayedProjects = range
    ? sortedProjects.slice(range[0] - 1, range[1] ?? sortedProjects.length)
    : sortedProjects;

  return (
    <Column fillWidth gap="xl" marginBottom="40" paddingX="l">
      {displayedProjects.map((post, index) => (
        <ProjectCard
          priority={index < 2}
          key={post.slug}
          href={`/work/${post.slug}`}
          images={post.metadata.images}
          title={post.metadata.title}
          description={post.metadata.summary}
          content={post.content}
          avatars={post.metadata.team?.map((member) => ({ src: member.avatar })) || []}
          link={post.metadata.link || ""}
          github={post.metadata.github || ""}
        />
      ))}
    </Column>
  );
}
