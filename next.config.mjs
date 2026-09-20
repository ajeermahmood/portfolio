import mdx from "@next/mdx";

const withMDX = mdx({
  extension: /\.mdx?$/,
  options: {},
});

/** @type {import('next').NextConfig} */
const nextConfig = {
  // pin the workspace root, otherwise Turbopack walks up and finds a stray
  // lockfile outside the repository
  turbopack: { root: import.meta.dirname },
  pageExtensions: ["ts", "tsx", "md", "mdx"],
  transpilePackages: ["next-mdx-remote"],
  images: {
    remotePatterns: [
      {
        protocol: "https",
        hostname: "www.google.com",
        pathname: "**",
      },
    ],
  },
  sassOptions: {
    compiler: "modern",
    silenceDeprecations: ["legacy-js-api"],
  },
  // The bouncer case study moved to /work/bouncer-gates when the project took
  // the name it ships under on npm. 308 rather than 307 so the move is cached
  // and search engines follow it, and because the old path is never coming back.
  async redirects() {
    return [{ source: "/work/bouncer", destination: "/work/bouncer-gates", permanent: true }];
  },
};

export default withMDX(nextConfig);
