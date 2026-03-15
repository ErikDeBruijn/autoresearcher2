import { defineConfig } from "@playwright/test";

const apiUrl =
  process.env.NEXT_PUBLIC_API_URL || "http://dllm-experiment.local:8000";

export default defineConfig({
  testDir: "./e2e",
  timeout: 60000,
  use: {
    baseURL: "http://localhost:3000",
    headless: true,
  },
  webServer: {
    command: "npm run dev",
    port: 3000,
    reuseExistingServer: true,
    env: {
      NEXT_PUBLIC_API_URL: apiUrl,
    },
  },
});
