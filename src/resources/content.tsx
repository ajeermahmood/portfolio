import { About, Blog, Gallery, Home, Newsletter, Person, Social, Work } from "@/types";
import { Line, Row, Text } from "@once-ui-system/core";

const person: Person = {
  firstName: "Ajeer",
  lastName: "Mohammed",
  name: `Ajeer Mohammed`,
  role: "Senior Full-Stack Engineer",
  avatar: "",
  email: "ajeermahmood@outlook.com",
  location: "Asia/Calcutta",
  languages: [],
  locale: "en",
};

const newsletter: Newsletter = {
  display: false,
  title: <>Subscribe to {person.firstName}'s Newsletter</>,
  description: <>Occasional notes on building and running production systems</>,
};

const social: Social = [
  {
    name: "GitHub",
    icon: "github",
    link: "https://github.com/ajeermahmood",
    essential: true,
  },
  {
    name: "LinkedIn",
    icon: "linkedin",
    link: "https://linkedin.com/in/ajeermahmood",
    essential: true,
  },
  {
    name: "Email",
    icon: "email",
    link: `mailto:${person.email}`,
    essential: true,
  },
];

const home: Home = {
  path: "/",
  image: "/images/og/home.jpg",
  label: "Home",
  title: `${person.name}, ${person.role}`,
  description: `Web, mobile and AI systems, and the infrastructure behind them. Five years of production work across India, the UAE and the United States.`,
  headline: <>I build production systems and run what they sit on</>,
  featured: {
    display: true,
    title: (
      <Row gap="12" vertical="center">
        <strong className="ml-4">Enrixa Store</strong>{" "}
        <Line background="brand-alpha-strong" vert height="20" />
        <Text marginRight="4" onBackground="brand-medium">
          Featured work
        </Text>
      </Row>
    ),
    href: "/work/enrixa-store",
  },
  subline: (
    <>
      I'm {person.firstName}, a full-stack engineer with five years of experience building web,
      mobile and AI systems. Right now I lead development on a multi-tenant e-commerce platform
      <br /> and an AI shopping assistant, and I handle the deployment and operations behind both.
    </>
  ),
};

const about: About = {
  path: "/about",
  label: "About",
  title: `About - ${person.name}`,
  description: `${person.name}, ${person.role}, based in India and open to relocation`,
  tableOfContent: {
    display: true,
    subItems: false,
  },
  avatar: {
    display: false,
  },
  calendar: {
    display: false,
    link: "",
  },
  intro: {
    display: true,
    title: "Introduction",
    description: (
      <>
        Full-stack engineer with five years of experience building production web, mobile and AI
        systems, and running the infrastructure behind them. Currently the lead developer on a
        multi-tenant e-commerce platform and an AI shopping assistant, where the work covers
        architecture, planning, cost decisions and code review for a team of 4 to 6 developers.
        Past projects have shipped to a US state health department, international pharmaceutical
        brands, and retail businesses in India, the UAE and the United States. Based in India and
        open to relocation and remote roles.
      </>
    ),
  },
  work: {
    display: true,
    title: "Work Experience",
    experiences: [
      {
        company: "SEA Media",
        timeframe: "05/2025 - Present",
        role: "Senior Full-Stack Engineer and Project Partner",
        achievements: [
          <>
            Enrixa Store: built a multi-tenant e-commerce platform, now 56 data models and 259 REST
            endpoints across 48 controllers, where every merchant runs an isolated store on its own
            subdomain. NestJS, Prisma, PostgreSQL and Redis on the backend, with separate Next.js
            apps for the merchant admin and the customer storefront. Set the tenancy model, the
            payments layer covering Razorpay and cash on delivery, and the release plan.
          </>,
          <>
            Closed the platform's biggest security risk by turning tenant isolation into a build
            requirement. Every pull request runs an automated isolation check and a migration
            compatibility diff against main across a schema that has taken 83 migrations, so a query
            that could leak one merchant's data into another store fails CI rather than reaching
            customers.
          </>,
          <>
            Enrixa AI: shipped an AI shopping assistant for Shopify. A Fastify service streams
            tool-calling agent replies over SSE, BullMQ workers keep the product catalog and its
            embeddings in sync, and the storefront chat widget runs inside a shadow DOM so it never
            collides with theme CSS. Wrote the product and technical plan behind it, including the
            cost model and the scaling approach, and kept a decision log for choices that would be
            expensive to reverse.
          </>,
          <>
            Led a two-repository build for an infant safety program run by a US state public health
            agency. A Next.js 16 and React 19 admin CMS on Supabase, plus a Flutter app for parents
            tracking their child's health, milestones and vaccinations, both sharing one PostgreSQL
            database of 15 tables across 11 edge functions. Wrote the schema contract and table
            ownership rules that let two teams work on the same database without breaking each
            other's features, backed by about 100 test files across the two repositories.
          </>,
          <>
            Built a Shopify Hydrogen storefront of 96 routes on React Router 7, Oxygen and Sanity
            CMS with four other engineers, and owned the architecture notes and the Oxygen deploy
            pipeline.
          </>,
          <>
            Built a competitor price monitoring app for Shopify that matches external catalogs
            against a merchant's own products and adjusts prices by rule, with repricing held behind
            a dry run until the merchant approves it.
          </>,
          <>
            Wrote the agent instructions and custom tooling that let AI coding agents work safely in
            8 production repositories, including automated audit, tenant-check and review-watcher
            scripts. Agents pass through the same gates the team does, so their changes still get
            reviewed and verified before they land.
          </>,
          <>
            Handled deployment and day-to-day operations across all of it: Docker images, GitHub
            Actions pipelines, Vercel and Shopify Oxygen releases, PM2 process management, and the
            managed PostgreSQL and Redis instances. Keep roughly 185 test files green across the
            portfolio.
          </>,
        ],
        images: [],
      },
      {
        company: "NTAM Group (Delta Plus Event Management)",
        timeframe: "06/2024 - 04/2025",
        role: "Software Developer",
        achievements: [
          <>
            Built a gamified training platform for Abbott's medical representatives, used by over
            1,000 reps worldwide, with per-region admin panels, a CMS and reporting. Angular, PHP,
            Node.js, SQL and microservices.
          </>,
          <>
            Delivered L'Oréal's event app for Android and iOS, covering agendas, badge scanning,
            surveys, live voting and push notifications. Used at more than 5 live events. Flutter,
            .NET and SQL.
          </>,
          <>
            Built two interactive pieces for Novartis and Nestlé: a talking avatar chatbot that
            answered product questions for healthcare professionals, using Next.js, Python, Google
            TTS and a fine-tuned model, and a live camera segmentation app with full-body tracking
            for product demos, using React, Redux, Electron and MediaPipe.
          </>,
        ],
        images: [],
      },
      {
        company: "Indus Real Estate LLC",
        timeframe: "12/2022 - 06/2024",
        role: "Full-Stack Developer",
        achievements: [
          <>
            Built a lease management CRM as a progressive web app covering more than 200 units,
            tracking lease expiry, payments, cheques, maintenance and reporting. Angular, PHP,
            Node.js and SQL on Linux.
          </>,
          <>
            Rebuilt the public property site with better filtering and agent contact flows, cutting
            page load time by 70 percent by fixing first contentful paint. Next.js, Node.js, PHP and
            SQL.
          </>,
          <>
            Added a CMS with an image pipeline that resizes, crops and converts uploads to WebP
            through Sharp, along with reporting tools for the content team.
          </>,
        ],
        images: [],
      },
      {
        company: "DOSII",
        timeframe: "06/2022 - 12/2022",
        role: "Flutter Developer",
        achievements: [
          <>
            Helped build a university admissions platform with predictive analytics that served over
            1,000 students in its first quarter, and set up the project's unit and integration
            tests. Flutter, Python, Firebase, GCP and GetX.
          </>,
        ],
        images: [],
      },
      {
        company: "Brototype",
        timeframe: "11/2021 - 04/2022",
        role: "Flutter Developer Intern",
        achievements: [
          <>Built playlist management, shuffle playback and favourites for a media application.</>,
        ],
        images: [],
      },
    ],
  },
  studies: {
    display: true,
    title: "Education",
    institutions: [
      {
        name: "Crossroads Software Engineering Bootcamp",
        description: (
          <>
            India, 08/2020 to 11/2021. Mobile and web development with backend services. Delivered
            more than 10 projects in agile teams.
          </>
        ),
      },
    ],
  },
  technical: {
    display: true,
    title: "Technical skills",
    skills: [
      {
        title: "Backend and data",
        description: (
          <>
            NestJS, Fastify and Express on Node.js, with Prisma over PostgreSQL. Multi-tenant schema
            design, migration safety, Redis and BullMQ for queues and background work.
          </>
        ),
        tags: [
          { name: "TypeScript", icon: "typescript" },
          { name: "NestJS", icon: "nestjs" },
          { name: "Fastify", icon: "fastify" },
          { name: "Prisma", icon: "prisma" },
          { name: "PostgreSQL", icon: "postgresql" },
          { name: "Redis", icon: "redis" },
        ],
        images: [],
      },
      {
        title: "Frontend and mobile",
        description: (
          <>
            React 19 and Next.js App Router, React Router 7, Angular, and Flutter for mobile. Server
            components, streaming, and state with Redux, Riverpod and TanStack Query.
          </>
        ),
        tags: [
          { name: "React", icon: "react" },
          { name: "Next.js", icon: "nextjs" },
          { name: "Angular", icon: "angular" },
          { name: "Flutter", icon: "flutter" },
          { name: "Tailwind CSS", icon: "tailwind" },
        ],
        images: [],
      },
      {
        title: "AI and agentic engineering",
        description: (
          <>
            LLM integration across Claude, Gemini and Groq, retrieval with vector search and
            embeddings, tool-calling agents, SSE streaming and provider fallback. Also the rarer
            half: writing agent instructions, safety gates and verification harnesses so coding
            agents can work inside a production repository without breaking it.
          </>
        ),
        tags: [
          { name: "Python", icon: "python" },
          { name: "Node.js", icon: "nodejs" },
        ],
        images: [],
      },
      {
        title: "E-commerce",
        description: (
          <>
            Shopify Hydrogen and Oxygen, Shopify apps on Remix with Polaris and App Bridge, Sanity
            CMS and Razorpay.
          </>
        ),
        tags: [{ name: "Shopify", icon: "shopify" }],
        images: [],
      },
      {
        title: "Infrastructure and delivery",
        description: (
          <>
            Docker, GitHub Actions pipelines, Vercel, AWS, GCP and Firebase, Nginx and PM2. pnpm and
            Turborepo monorepos, with Jest, Cypress and Playwright for tests.
          </>
        ),
        tags: [
          { name: "Docker", icon: "docker" },
          { name: "Vercel", icon: "vercel" },
        ],
        images: [],
      },
    ],
  },
};

const blog: Blog = {
  path: "/blog",
  label: "Blog",
  title: "Notes on building and running systems",
  description: `Writing by ${person.name}`,
};

const work: Work = {
  path: "/work",
  label: "Work",
  title: `Projects - ${person.name}`,
  description: `Production systems built by ${person.name}`,
};

const gallery: Gallery = {
  path: "/gallery",
  label: "Gallery",
  title: `Gallery - ${person.name}`,
  description: `A photo collection by ${person.name}`,
  images: [],
};

export { person, social, newsletter, home, about, blog, work, gallery };
