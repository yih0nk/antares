import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
// IBM Plex Mono, self-hosted: the only typeface (400 body, 500 emphasis, 600 the hero number).
import "@fontsource/ibm-plex-mono/400.css";
import "@fontsource/ibm-plex-mono/500.css";
import "@fontsource/ibm-plex-mono/600.css";
import "./styles.css";
import { applyTheme } from "./theme";
import App from "./App";

// theme.ts is the one source of colour: write it onto :root as CSS custom
// properties before the first render (three.js and the charts import it directly).
applyTheme();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
