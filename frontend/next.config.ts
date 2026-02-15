import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactCompiler: true,
  ...(process.env.DOCKER_BUILD === "true" && { output: "standalone" }),
};

export default nextConfig;
