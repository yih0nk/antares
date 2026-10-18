import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// ANTARES_API_URL points the dev proxy at another `antares serve` (e.g. a
// second checkout on :8001); the default is the one the README starts.
const api = process.env.ANTARES_API_URL ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { "/api": api },
  },
});
