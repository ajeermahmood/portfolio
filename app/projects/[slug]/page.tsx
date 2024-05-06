import { notFound } from "next/navigation";
import allProjects from "../../../data/projects/index";
import { Mdx } from "@/app/components/mdx";
import { Header } from "./header";
import "./mdx.css";
import { ReportView } from "./view";
import Image from "next/image";

export const revalidate = 60;

type Props = {
  params: {
    slug: string;
  };
};

export async function generateStaticParams(): Promise<Props["params"][]> {
  return allProjects
    .filter((p) => p.published)
    .map((p) => ({
      slug: p.slug,
    }));
}

export default async function PostPage({ params }: Props) {
  const slug = params?.slug;
  const project = allProjects.find((project) => project.slug === slug);

  if (!project) {
    notFound();
  }

  const views = allProjects;

  return (
    <div className="bg-zinc-50 min-h-screen">
      <Header project={project} views={views} />
      {/* <ReportView slug={project.slug} /> */}

      <article className="px-4 py-12 mx-auto prose prose-zinc prose-quoteless">
        {/* <Mdx code={project.description} /> */}
        {project.images.length != 0 ? (
          <h6 className="mt-0 scroll-m-20 text-base font-semibold tracking-tight">
            Screenshots:
          </h6>
        ) : (
          <></>
        )}
        {project.images.length != 0 ? (
          project.images.map((img, i) => (
            <Image
              className="rounded-md border border-zinc-200 select-none"
              key={i}
              alt=""
              src={`/images/${img}`}
              width={1115}
              height={800}
            />
          ))
        ) : (
          <></>
        )}
      </article>
    </div>
  );
}
