import { defineConfig } from "vite";

// Served from https://<user>.github.io/prism-plm/ on GitHub Pages; override with BASE_PATH.
export default defineConfig({
  base: process.env.BASE_PATH ?? "./",
  build: {
    target: "es2020",
    chunkSizeWarningLimit: 1500,
  },
});
