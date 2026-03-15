import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export",
  distDir: "dist",
  // Allow dev server to work normally (export only affects build)
};

export default nextConfig;
