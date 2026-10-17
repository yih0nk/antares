import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Grid, OrbitControls } from "@react-three/drei";
import {
  api,
  type DatasetInfo,
  type EpisodeInfo,
  type Geometry,
  type Replay,
} from "./api";
import { Arm, ghostMaterial, realMaterial, type Playhead } from "./Arm";

// Layout: 3D viewer (real arm + ghost sim arm) on the left, gap panel on
// the right, episode timeline along the bottom. The ghost is the simulator
// replaying the real commands with the default parameters.


const deg = (v: number | null | undefined, digits = 2) => (v == null || !isFinite(v) ? "--" : `${v.toFixed(digits)}°`);

// Advances the shared playhead inside the render loop so playback never
// waits on React; the UI copy of the frame is updated when it changes.
function Driver({
  playhead,
  playing,
  fps,
  frames,
  speed,
  onFrame,
}: {
  playhead: Playhead;
  playing: boolean;
  fps: number;
  frames: number;
  speed: number;
  onFrame: (f: number) => void;
}) {
  const last = useRef(-1);
  useFrame((_, dt) => {
    if (frames < 2) return;
    if (playing) playhead.frame = (playhead.frame + Math.min(dt, 0.1) * fps * speed) % frames;
    const f = Math.floor(playhead.frame);
    if (f !== last.current) {
      last.current = f;
      onFrame(f);
    }
  });
  return null;
}

export default function App() {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [dataset, setDataset] = useState<string>("");
  const [dsState, setDsState] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const [episodes, setEpisodes] = useState<EpisodeInfo[]>([]);
  const [episode, setEpisode] = useState<number | null>(null);
  const [geometry, setGeometry] = useState<Geometry | null>(null);
  const [baseline, setBaseline] = useState<Replay | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [playing, setPlaying] = useState(true);
  const [frame, setFrame] = useState(0);

  const playhead = useMemo<Playhead>(() => ({ frame: 0 }), []);

  // ---- initial load ---------------------------------------------------------
  useEffect(() => {
    api.datasets().then(
      (ds) => {
        setDatasets(ds);
        setDataset((cur) => cur || ds.find((d) => d.loaded)?.id || ds[0]?.id || "");
      },
      (e) => setError(`API unreachable: ${e.message}. Is \`antares serve\` running?`),
    );
    api.model().then(setGeometry, (e) => setError(`Could not load the arm model: ${e.message}`));
  }, []);

  // ---- dataset -> episodes ---------------------------------------------------
  useEffect(() => {
    if (!dataset) return;
    let live = true;
    setDsState("loading");
    setEpisodes([]);
    setEpisode(null);
    setBaseline(null);
    api
      .loadDataset(dataset)
      .then((info) => {
        if (!live) return;
        setDatasets((ds) => ds.map((d) => (d.id === info.id ? info : d)));
        return api.episodes(dataset);
      })
      .then((eps) => {
        if (!live || !eps) return;
        setEpisodes(eps);
        setEpisode(eps[0]?.index ?? null);
        setDsState("ready");
      })
      .catch((e) => {
        if (!live) return;
        setDsState("error");
        setError(`Could not load ${dataset}: ${e.message}`);
      });
    return () => {
      live = false;
    };
  }, [dataset]);

  // ---- baseline replay (real arm + default-params ghost) --------------------
  useEffect(() => {
    if (!dataset || episode == null) return;
    let live = true;
    api.replay(dataset, episode, null).then(
      (r) => live && setBaseline(r),
      (e) => live && setError(`Replay failed: ${e.message}`),
    );
    playhead.frame = 0;
    setFrame(0);
    return () => {
      live = false;
    };
  }, [dataset, episode, playhead]);

  // ---- derived ---------------------------------------------------------------
  const replay = baseline;
  const fps = baseline?.fps ?? 30;
  const frames = baseline?.real_poses.length ?? 0;
  const ghostGap = replay?.gap ?? null;
  const currentDs = datasets.find((d) => d.id === dataset);

  const ghostLabel = "default sim";

  return (
    <div className="shell">
      <header className="topbar">
        <span className="brand">antares</span>
        <span className="tagline">sim, calibrated to this robot</span>
        <div className="picker-wrap">
          {dsState === "loading" && <span className="muted small">loading dataset…</span>}
          <select
            className="picker"
            value={dataset}
            disabled={!datasets.length}
            onChange={(e) => setDataset(e.target.value)}
          >
            {datasets.map((d) => (
              <option key={d.id} value={d.id}>
                {d.id}
              </option>
            ))}
          </select>
        </div>
      </header>

      <main className="viewer">
        <Canvas shadows camera={{ position: [0.62, 0.48, 0.72], fov: 38, near: 0.01, far: 20 }}>
          <color attach="background" args={["#0b0d12"]} />
          <ambientLight intensity={0.55} />
          <directionalLight position={[0.8, 1.6, 0.6]} intensity={1.3} castShadow shadow-mapSize={[1024, 1024]} />
          <directionalLight position={[-1, 0.6, -0.8]} intensity={0.35} />
          <Grid
            infiniteGrid
            cellSize={0.05}
            sectionSize={0.25}
            cellColor="#1c2230"
            sectionColor="#2a3346"
            fadeDistance={3}
          />
          {geometry && (
            <Suspense fallback={null}>
              {/* MuJoCo is z-up; three.js is y-up. */}
              <group rotation={[-Math.PI / 2, 0, 0]}>
                <Arm geometry={geometry} poses={baseline?.real_poses ?? null} playhead={playhead} material={realMaterial} />
                <Arm geometry={geometry} poses={replay?.sim_poses ?? null} playhead={playhead} material={ghostMaterial} />
              </group>
            </Suspense>
          )}
          <Driver playhead={playhead} playing={playing} fps={fps} frames={frames} speed={1} onFrame={setFrame} />
          <OrbitControls makeDefault target={[0, 0.16, 0]} />
        </Canvas>

        <div className="legend">
          <span>
            <i className="swatch real" /> real arm (logged)
          </span>
          <span>
            <i className="swatch sim" /> sim ghost, {ghostLabel}
          </span>
        </div>

        {(dsState === "loading" || (!baseline && !error)) && (
          <div className="overlay">
            {dsState === "loading"
              ? currentDs?.loaded
                ? "Loading episodes…"
                : "Downloading the dataset from the Hugging Face Hub…"
              : "Replaying episode…"}
          </div>
        )}
        {error && (
          <div className="error" role="alert">
            {error}
            <button onClick={() => setError(null)} aria-label="Dismiss">
              ×
            </button>
          </div>
        )}
      </main>

      <aside className="panel">
        <div className="label">Sim-to-real gap</div>
        <div className="score">{deg(ghostGap?.overall_deg)}</div>
        <button className="cta" disabled>
          Calibrate
        </button>
      </aside>

      <footer className="timeline">
        <div className="controls">
          <button className="icon" onClick={() => setPlaying((p) => !p)} disabled={!frames} aria-label={playing ? "Pause" : "Play"}>
            {playing ? "❚❚" : "▶"}
          </button>
          <select
            className="picker"
            value={episode ?? ""}
            onChange={(e) => setEpisode(Number(e.target.value))}
            disabled={!episodes.length}
          >
            {episodes.map((e) => (
              <option key={e.index} value={e.index}>
                episode {e.index} · {e.duration_s.toFixed(1)}s
              </option>
            ))}
          </select>
          <span className="mono small">
            {(frame / fps).toFixed(2)}s / {(frames / fps).toFixed(2)}s
          </span>
        </div>
      </footer>
    </div>
  );
}
