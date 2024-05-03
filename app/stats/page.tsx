"use client";
import { Github, Mail, Linkedin } from "lucide-react";
import { Navigation } from "../components/nav";
import { Card } from "../components/card";
import Link from "next/link";

const socials = [
  {
    icon: <Mail size={20} />,
    href: "mailto:ajeermahmood@outlook.com",
    label: "Email",
    handle: "ajeermahmood@outlook.com",
  },
  {
    icon: <Linkedin size={20} />,
    href: "https://linkedin.com/in/ajeermahmood/",
    label: "Linkedin",
    handle: "ajeermahmood",
  },
  {
    icon: <Github size={20} />,
    href: "https://github.com/ajeermahmood",
    label: "Github",
    handle: "ajeermahmood",
  },
];

export default function Example() {
  return (
    <div className=" bg-gradient-to-tl from-zinc-900/0 via-zinc-900 to-zinc-900/0">
      <Navigation />
      <div className="container block text-center min-h-screen pt-20 px-4 mx-auto max-w-screen-md">
        <h1 className="mt-4 mb-1 text-white">Front-End Focus</h1>
        <h2 className="text-sm text-zinc-500 mx-2">
          I excel in creating visually stunning and intuitive interfaces using
          technologies like ReactJS, AngularJS, NextJS, VueJS, Svelte, Astro and
          Flutter. My dedication to user-centric design ensures that every
          application I develop not only looks great but also delivers seamless
          navigation and functionality across devices.
        </h2>
        <h1 className="mt-4 mb-1 text-white">Backend Brilliance</h1>
        <h2 className="text-sm text-zinc-500 mx-2">
          Behind the scenes, I leverage my skills in TypeScript, Node.js, Python
          and PHP to build robust backend systems that power scalable and
          efficient web applications. I have extensive experience working with
          both MySQL and NoSQL databases, utilizing their respective strengths
          to design and optimize database structures for various applications.
          Additionally, I am proficient in Firebase, harnessing its real-time
          database and authentication services to create dynamic and responsive
          applications. My familiarity with Google Cloud Platform (GCP) enables
          me to deploy and manage cloud-based solutions with ease, ensuring
          optimal performance and scalability. From designing cloud-based CRM
          systems to orchestrating cross-platform Flutter apps, I thrive on
          architecting solutions that drive business growth and enhance
          operational efficiency.
        </h2>
        <h1 className="mt-4 mb-1 text-white">Machine Learning Maven</h1>
        <h2 className="text-sm text-zinc-500 mx-2">
          With experience in TensorFlow and Python, I have delved into the
          exciting realm of machine learning, developing innovative solutions
          like chatbots for real estate. My passion for exploring emerging
          technologies and pushing the boundaries of what's possible drives me
          to stay at the forefront of technological advancements.
        </h2>
        <h1 className="mt-4 mb-1 text-white">Collaborative Spirit</h1>
        <h2 className="text-sm text-zinc-500 mx-2">
          I believe in the power of teamwork and collaboration, and I thrive in
          environments where ideas are shared, challenges are tackled together,
          and innovation is encouraged. Whether leading a multidisciplinary team
          or collaborating with stakeholders, I bring a proactive and
          collaborative approach to every project.
        </h2>
        <h1 className="mt-4 mb-1 text-white">Continuous Learner</h1>
        <h2 className="text-sm text-zinc-500 mx-2">
          In the fast-paced world of technology, I recognize the importance of
          continuous learning and staying updated with the latest trends and
          best practices. I am always eager to expand my skill set, explore new
          technologies, and embrace new challenges that come my way.
        </h2>
        <h1 className="mt-4 mb-1 text-white">Let's Connect</h1>
        <h2 className="text-sm text-zinc-500 mx-2 mb-7 sm:mb-10">
          If you're looking for a passionate Full-Stack Developer who can bring
          your digital projects to life with creativity, technical expertise,
          and a collaborative spirit, I'd love to connect. Feel free to explore
          my portfolio and reach out to discuss how we can work together to
          bring your vision to reality.
        </h2>
      </div>
      <div className="container block text-center pb-10 px-4 mx-auto max-w-screen-lg">
        <div className="grid w-full grid-cols-1 gap-7 mx-auto sm:grid-cols-3 lg:gap-8">
          {socials.map((s) => (
            <Card>
              <Link
                href={s.href}
                target="_blank"
                className="p-4 relative flex flex-col items-center gap-4 duration-700 group md:gap-8 md:py-24  lg:pb-48  md:p-16"
              >
                {/* <span
                  className="absolute w-px h-2/3 bg-gradient-to-b from-zinc-500 via-zinc-500/50 to-transparent"
                  aria-hidden="true"
                /> */}
                <span className="relative z-10 flex items-center justify-center w-12 h-12 text-sm duration-1000 border rounded-full text-zinc-200 group-hover:text-white group-hover:bg-zinc-900 border-zinc-500 bg-zinc-900 group-hover:border-zinc-200 drop-shadow-orange">
                  {s.icon}
                </span>{" "}
                <div className="z-10 flex flex-col items-center">
                  <span className="lg:text-xl font-medium duration-150 xl:text-3xl text-zinc-200 group-hover:text-white font-display">
                    {s.handle}
                  </span>
                  <span className="mt-4 text-sm text-center duration-1000 text-zinc-400 group-hover:text-zinc-200">
                    {s.label}
                  </span>
                </div>
              </Link>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
}
