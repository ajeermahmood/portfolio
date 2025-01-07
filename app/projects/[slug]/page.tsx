import CodeComponent from "@/app/components/codeComponent";
import Image from "next/image";
import { notFound } from "next/navigation";
import allProjects from "../../../data/projects/index";
import { Header } from "./header";
import "./mdx.css";

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
const pythonCode = `
const GroceryItem: React.FC<GroceryItemProps> = ({ item }) => {
  return (
    <div>
      <h2>{item.name}</h2>
      <p>Price: {item.price}</p>
      <p>Quantity: {item.quantity}</p>
    </div>
  );
}
  `;

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

        {project.description != "" ? (
          <div>
            <h6 className="mt-0 scroll-m-20 text-base font-semibold tracking-tight">
              {project.description}
            </h6>
            <br />
          </div>
        ) : (
          <></>
        )}

        {project.slug === "nextjs-ai-chatbot" ? (
          <>
            <h6 className="italic mt-0 scroll-m-20 text-base font-semibold tracking-tight">
              Next.js code available in{" "}
              <a
                href="https://github.com/ajeermahmood/nextjs-ai-chatbot"
                target="_blank"
                className="text-sky-800"
              >
                Github.
              </a>
            </h6>
          </>
        ) : (
          <></>
        )}

        {project.features.length != 0 ? (
          <>
            <h2 className="mt-10 scroll-m-20 border-b border-b-zinc-800 pb-1 text-3xl font-semibold tracking-tight first:mt-0">
              Features
            </h2>
            <ul>
              {project.features.map((f) => (
                <li>
                  <b>{f.title}:</b> {f.desc}
                </li>
              ))}
            </ul>
          </>
        ) : (
          <></>
        )}

        {project.images.length != 0 ? (
          <h6 className="mt-0 scroll-m-20 text-base font-semibold tracking-tight">
            Screenshots:
          </h6>
        ) : (
          <></>
        )}

        {project.mobile ? (
          <div className="flex flex-wrap justify-between">
            {project.images.length != 0 ? (
              project.images.map((img, i) => (
                <Image
                  className="rounded-md border border-zinc-200 select-none max-w-72"
                  key={i}
                  alt=""
                  src={`/images/${img}`}
                  width={591}
                  height={1280}
                />
              ))
            ) : (
              <></>
            )}
          </div>
        ) : (
          <div>
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
          </div>
        )}

        {project.code?.length != 0 ? (
          <div>
            <h2 className="mt-10 scroll-m-20 border-b border-b-zinc-800 pb-1 text-3xl font-semibold tracking-tight first:mt-0">
              {project.slug === "nextjs-ai-chatbot"
                ? "Audio Generation Example"
                : "Usage"}
            </h2>
            {project.code?.map((c, i) => (
              <div key={i} className={`${i > 0 ? "mt-10" : ""}`}>
                <h5 className="mt-0 scroll-m-20 text-lg font-semibold tracking-tight">
                  {i + 1}. {c.title}
                </h5>
                {c.features.length != 0 ? (
                  <ul>
                    {c.features.map((f) => (
                      <li>
                        <b>{f.title}:</b> {f.desc}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <></>
                )}
                <CodeComponent code={c.code} lang={c.lang} />
                <h6 className="mt-8 scroll-m-20 text-base font-semibold tracking-tight">
                  {c.description}
                </h6>
              </div>
            ))}
          </div>
        ) : (
          <></>
        )}
      </article>
    </div>
  );
}
