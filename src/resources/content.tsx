import type {
  About,
  Blog,
  Contact,
  Home,
  Person,
  Social,
  Work,
} from "@/types";
import { Line, Row, SmartLink, Text } from "@once-ui-system/core";

const person: Person = {
  firstName: "Ajeer",
  lastName: "Mohammed",
  name: `Ajeer Mohammed`,
  role: "Senior Full-Stack Engineer",
  avatar: "/images/avatar.webp",
  email: "ajeermahmood@outlook.com",
  location: "Asia/Calcutta",
  locationLabel: "India",
  languages: [],
  locale: "en",
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
  {
    name: "Resume",
    icon: "document",
    link: "/document/RESUME_AJEER.pdf",
    essential: true,
  },
];

const home: Home = {
  path: "/",
  image: "/images/og/home.png",
  label: "Home",
  title: `${person.name}, ${person.role}`,
  description: `Senior full-stack engineer. Multi-tenant e-commerce, AI shopping assistants and mobile apps, shipped to a US state health agency and global brands.`,
  headline: <>I build it, then I keep it running</>,
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
      I'm {person.firstName}, a full-stack engineer. Right now that means a multi-tenant e-commerce
      platform and an AI shopping assistant, plus the pipelines, containers and databases that keep
      them up.
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
    display: true,
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
        Full-stack engineer building production web, mobile and AI systems, and running the
        infrastructure behind them. Currently the lead developer on a multi-tenant e-commerce
        platform and an AI shopping assistant, where the work covers architecture, planning, cost
        decisions and code review for a team of 4 to 6 developers. Past projects have shipped to a
        US state health department, international pharmaceutical brands, and retail businesses in
        India, the UAE and the United States. I also publish open-source developer tooling: two
        command-line tools for catching expensive mistakes in a build rather than in production.
        Based in India and open to relocation and remote roles.
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
            W.I.N.S, Alabama Department of Public Health: led a two-repository build for an infant
            safety program. A Next.js 16 and React 19 admin CMS on Supabase, plus a Flutter app for
            parents tracking their child's health, milestones and vaccinations, both sharing one
            PostgreSQL database of 15 tables across 11 edge functions. Wrote the schema contract and
            table ownership rules that let two teams work on the same database without breaking each
            other's features, backed by about 100 test files across the two repositories. Live on
            the App Store and Google Play since July 2026 as the agency's official app for the
            program.
          </>,
          <>
            Built a Shopify Hydrogen storefront serving more than 10,000 daily users across India,
            96 routes on React Router 7, Oxygen and Sanity CMS with four other engineers, and owned
            the architecture notes and the Oxygen deploy pipeline.
          </>,
          <>
            Built a competitor price monitoring app for Shopify that matches external catalogs
            against a merchant's own products and adjusts prices by rule, with repricing held behind
            a dry run until the merchant approves it.
          </>,
          <>
            Wrote the agent instructions and CI gates that let AI coding agents work in 8 production
            repositories, including automated audit, tenant-check and review-watcher scripts. Agents
            pass through the same gates the team does, so their changes still get reviewed and
            verified before they land. The generalised version of those gates is open source as{" "}
            <SmartLink href="/work/bouncer">bouncer</SmartLink>.
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
            Built the interactive pieces for Nestlé's Ascenda launch at the Ritz-Carlton, Jeddah: a
            full-body-tracking growth wall where guests watched themselves grow on a 4 metre screen,
            then printed or shared the photo, and a Mario-style runner played on an Xbox controller.
            More than 150 guests used them in one evening. Electron, React, Redux and MediaPipe.
          </>,
          <>
            Built a talking avatar chatbot for Novartis that answered product questions for
            healthcare professionals, using Next.js, Python, Google TTS and a fine-tuned model.
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
            Docker, GitHub Actions pipelines, Vercel, Cloudflare Workers, Railway, GCP and Firebase,
            Nginx and PM2. pnpm and Turborepo monorepos, with Jest, Vitest, Cypress and Playwright
            for tests.
          </>
        ),
        tags: [
          { name: "Docker", icon: "docker" },
          { name: "Vercel", icon: "vercel" },
          { name: "GitHub Actions", icon: "githubactions" },
        ],
        images: [],
      },
      {
        title: "Open source",
        description: (
          <>
            <SmartLink href="https://github.com/ajeermahmood/bouncer">bouncer</SmartLink> is a set
            of five CI gates that block expensive mistakes before they merge: hardcoded secrets,
            database access that reaches around the tenant-scoped client, float arithmetic on
            currency, migrations that break the running app mid-deploy, and documentation links
            pointing at deleted files. Each gate is a pure function, so one implementation runs in
            the CLI, in a Cloudflare Worker, as a Node service and in the browser. Zero runtime
            dependencies and 86 tests, published as{" "}
            <SmartLink href="https://www.npmjs.com/package/bouncer-gates">bouncer-gates</SmartLink>.
            Most of the work went into being wrong less often: 45 findings on a real repository, 32
            of them false positives, brought down to 13 that were all genuine.{" "}
            <SmartLink href="https://github.com/ajeermahmood/estate">estate</SmartLink> is a linter
            for repositories rather than the code inside them, one Go binary with no dependencies
            and 102 tests. Also two small npm utilities:{" "}
            <SmartLink href="https://npmjs.com/package/web-element-scraper">
              web-element-scraper
            </SmartLink>{" "}
            and{" "}
            <SmartLink href="https://npmjs.com/package/nodejs-performance-profiler">
              nodejs-performance-profiler
            </SmartLink>
            .
          </>
        ),
        tags: [
          { name: "Go", icon: "go" },
          { name: "Node.js", icon: "nodejs" },
          { name: "Cloudflare", icon: "cloudflare" },
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
  description: `Notes by ${person.name} on building and running production systems: what holds up in a real repository, and what breaks once real users arrive.`,
};

const work: Work = {
  path: "/work",
  label: "Work",
  title: `Projects - ${person.name}`,
  description: `Production work by ${person.name}: a multi-tenant e-commerce platform, an AI shopping assistant, Shopify Hydrogen storefronts and open-source CI tooling.`,
};

const contact: Contact = {
  path: "/contact",
  label: "Contact",
  title: `Contact ${person.name}`,
  description: `Get in touch with ${person.name}, ${person.role}, by email`,
};

export { person, social, home, about, blog, work, contact };
