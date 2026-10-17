import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Grid, OrbitControls } from "@react-three/drei";
import {
  api,
  lockNameFor,
  type CalibrationResult,
  type DatasetInfo,
  type EpisodeInfo,
  type Geometry,
  type JobEvent,
  type LockSummary,
  type Params,
  type ParamValue,
  type ProgressEvent,
  type Replay,
} from "./api";
import { Arm, ghostMaterial, realMaterial, type Playhead } from "./Arm";
import { JointTrace, Sparkline } from "./Charts";

// Layout: 3D viewer (real arm + ghost sim arm) on the left, gap panel on
// the right, episode timeline along the bottom. The ghost is the simulator
// replaying the real commands with either the default parameters, a saved
// robot lock, or (while calibrating) the best fit so far.

const DEFAULT = "default";
const LIVE = "live";
const PARAM_KEYS: (keyof Params)[] = ["kp", "kv", "damping", "armature", "frictionloss", "latency_frames"];
const SPEEDS = [0.25, 0.5, 1, 2];

type Calibration = {
  jobId: string;
  name: string;
  status: "running" | "done" | "error" | "cancelled";
  progress: ProgressEvent[];
  best: number[];
  result?: CalibrationResult;
  error?: string;
};

const deg = (v: number | null | undefined, digits = 2) => (v == null || !isFinite(v) ? "--" : `${v.toFixed(digits)}°`);

function fmtScalar(k: keyof Params, v: number) {
  if (k === "latency_frames") return `${Math.round(v)} fr`;
  if (Math.abs(v) >= 100) return v.toFixed(1);
  if (Math.abs(v) >= 1) return v.toFixed(3);
  return v.toFixed(4);
}

// A per-joint value shows as its range across the six joints.
function fmtParam(k: keyof Params, v: ParamValue) {
  if (!Array.isArray(v)) return fmtScalar(k, v);
  const lo = Math.min(...v);
  const hi = Math.max(...v);
  return lo === hi ? fmtScalar(k, lo) : `${fmtScalar(k, lo)}–${fmtScalar(k, hi)}`;
}

function sameParam(a: ParamValue, b: ParamValue) {
  const av = Array.isArray(a) ? a : [a];
  const bv = Array.isArray(b) ? b : [b];
  const n = Math.max(av.length, bv.length);
  for (let i = 0; i < n; i++) {
    if (Math.abs(av[i % av.length] - bv[i % bv.length]) > 1e-9) return false;
  }
  return true;
}

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
    // Cap dt only against long stalls (a backgrounded tab), so slow renders
    // still play back in real time.
    if (playing) playhead.frame = (playhead.frame + Math.min(dt, 0.5) * fps * speed) % frames;
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
  const [loadSeconds, setLoadSeconds] = useState(0);
  const [episodes, setEpisodes] = useState<EpisodeInfo[]>([]);
  const [episode, setEpisode] = useState<number | null>(null);
  const [geometry, setGeometry] = useState<Geometry | null>(null);
  const [locks, setLocks] = useState<LockSummary[]>([]);
  const [ghost, setGhost] = useState<string>(DEFAULT);
  const [baseline, setBaseline] = useState<Replay | null>(null);
  const [replay, setReplay] = useState<Replay | null>(null);
  const [replayBusy, setReplayBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [frame, setFrame] = useState(0);
  const [joint, setJoint] = useState("shoulder_pan");
  const [calib, setCalib] = useState<Calibration | null>(null);

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
    api.locks().then(setLocks, () => setLocks([]));
  }, []);

  // ---- dataset -> episodes ---------------------------------------------------
  useEffect(() => {
    if (!dataset) return;
    let live = true;
    setDsState("loading");
    setEpisodes([]);
    setEpisode(null);
    setReplay(null);
    setBaseline(null);
    setGhost(DEFAULT);
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

  // The first load downloads the dataset's parquet files and the API reports
  // no progress, so at least show that time is passing.
  useEffect(() => {
    if (dsState !== "loading") return;
    const t0 = performance.now();
    setLoadSeconds(0);
    const id = setInterval(() => setLoadSeconds(Math.floor((performance.now() - t0) / 1000)), 1000);
    return () => clearInterval(id);
  }, [dsState]);

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

  // ---- ghost replay for a saved lock ----------------------------------------
  useEffect(() => {
    if (!dataset || episode == null || ghost === LIVE) return;
    if (ghost === DEFAULT) {
      setReplay(baseline);
      return;
    }
    let live = true;
    setReplayBusy(true);
    api
      .replay(dataset, episode, ghost)
      .then(
        (r) => live && setReplay(r),
        (e) => live && setError(`Replay with ${ghost} failed: ${e.message}`),
      )
      .finally(() => live && setReplayBusy(false));
    return () => {
      live = false;
    };
  }, [dataset, episode, ghost, baseline]);

  // ---- live ghost while calibrating: replay the best params so far ----------
  // One request in flight at a time; when it lands, the newest best (if it
  // changed meanwhile) goes next. Each landing eases the ghost to the new fit.
  const liveWanted = useRef<Params | null>(null);
  const liveInFlight = useRef(false);
  const liveKey = useRef("");
  const where = useRef({ dataset, episode });
  where.current = { dataset, episode };
  const pumpLive = useCallback(() => {
    const { dataset: ds, episode: ep } = where.current;
    if (liveInFlight.current || !liveWanted.current || ep == null) return;
    const params = liveWanted.current;
    const key = `${ds}#${ep}#${JSON.stringify(params)}`;
    if (key === liveKey.current) return;
    liveInFlight.current = true;
    liveKey.current = key;
    setReplayBusy(true);
    api
      .replayParams(ds, ep, params)
      .then(
        (r) => {
          if (where.current.dataset === ds && where.current.episode === ep) setReplay(r);
        },
        () => {},
      )
      .finally(() => {
        liveInFlight.current = false;
        setReplayBusy(false);
        pumpLive();
      });
  }, []);

  useEffect(() => {
    if (ghost === LIVE) pumpLive(); // also re-runs the latest best on a new episode
  }, [ghost, dataset, episode, pumpLive]);

  // ---- calibration -----------------------------------------------------------
  const sourceRef = useRef<EventSource | null>(null);
  useEffect(() => () => sourceRef.current?.close(), []);

  const startCalibration = async () => {
    if (!dataset) return;
    setError(null);
    const name = lockNameFor(dataset);
    let jobId: string;
    try {
      ({ job_id: jobId } = await api.calibrate(dataset, name));
    } catch (e) {
      setError(`Could not start calibration: ${(e as Error).message}`);
      return;
    }
    setCalib({ jobId, name, status: "running", progress: [], best: [] });
    liveWanted.current = null;
    liveKey.current = "";
    setGhost(LIVE);
    setPlaying(true);

    sourceRef.current?.close();
    const src = api.events(jobId);
    sourceRef.current = src;
    let lastBest = Infinity;
    src.onmessage = (msg) => {
      const ev = JSON.parse(msg.data) as JobEvent;
      if (ev.type === "progress") {
        setCalib((c) =>
          c && c.jobId === jobId
            ? { ...c, progress: [...c.progress, ev], best: [...c.best, ev.best_score ?? NaN] }
            : c,
        );
        if (ev.best_score != null && ev.best_score < lastBest - 1e-9) {
          lastBest = ev.best_score;
          liveWanted.current = ev.best_params;
          pumpLive();
        }
      } else if (ev.type === "done") {
        src.close();
        setCalib((c) => (c && c.jobId === jobId ? { ...c, status: "done", result: ev.result } : c));
        api.locks().then((ls) => {
          setLocks(ls);
          setGhost(ev.lock.name);
        });
      } else if (ev.type === "error") {
        src.close();
        setCalib((c) => (c && c.jobId === jobId ? { ...c, status: "error", error: ev.message } : c));
      }
    };
    src.onerror = () => {
      // EventSource reconnects on its own; a closed stream after "done" is fine.
      if (src.readyState === EventSource.CLOSED) {
        setCalib((c) =>
          c && c.jobId === jobId && c.status === "running" ? { ...c, status: "error", error: "lost the progress stream" } : c,
        );
      }
    };
  };

  const stopCalibration = async () => {
    if (!calib) return;
    await api.cancel(calib.jobId).catch(() => {});
    setCalib((c) => (c ? { ...c, status: c.status === "running" ? "cancelled" : c.status } : c));
    // The worker still finishes its current generation and saves the best fit;
    // the stream's "done" event then selects that lock.
  };

  // ---- derived ---------------------------------------------------------------
  const dsLocks = locks.filter((l) => l.dataset === dataset);
  const running = calib?.status === "running" || calib?.status === "cancelled";
  const fps = baseline?.fps ?? 30;
  const frames = baseline?.real_poses.length ?? 0;
  const ghostGap = replay?.gap ?? null;
  const baseGap = baseline?.gap ?? null;
  const jointNames = baseline?.joints ?? [];
  const scored = new Set(baseGap?.scored_joints ?? []);
  const jIndex = Math.max(0, jointNames.indexOf(joint));
  const maxJointErr = Math.max(1e-6, ...Object.values(baseGap?.per_joint_deg ?? {}), ...Object.values(ghostGap?.per_joint_deg ?? {}));
  const lastProgress = calib?.progress[calib.progress.length - 1];
  const shownParams = replay?.params ?? null;
  const defaultParams = baseline?.params ?? null;
  const currentDs = datasets.find((d) => d.id === dataset);

  const ghostLabel =
    ghost === DEFAULT ? "default sim" : ghost === LIVE ? "best fit so far" : `lock ${ghost}`;

  const seek = (f: number) => {
    playhead.frame = f;
    setFrame(Math.floor(f));
  };

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
            disabled={!datasets.length || running}
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
          <Driver playhead={playhead} playing={playing} fps={fps} frames={frames} speed={speed} onFrame={setFrame} />
          <OrbitControls makeDefault target={[0, 0.16, 0]} />
        </Canvas>

        <div className="legend">
          <span>
            <i className="swatch real" /> real arm (logged)
          </span>
          <span>
            <i className="swatch sim" /> sim ghost, {ghostLabel}
            {replayBusy && <span className="dot" />}
          </span>
        </div>

        {(dsState === "loading" || (!baseline && !error)) && (
          <div className="overlay">
            {dsState === "loading" ? (
              currentDs?.loaded ? (
                "Loading episodes…"
              ) : (
                <div className="loading">
                  <div>Downloading the dataset from the Hugging Face Hub… {loadSeconds}s</div>
                  <div className="muted small">
                    The first load of a dataset can take a couple of minutes. After that it is cached.
                  </div>
                </div>
              )
            ) : (
              "Replaying episode…"
            )}
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
        <section>
          <div className="label">Sim-to-real gap</div>
          <div className="score">{deg(ghostGap?.overall_deg)}</div>
          <div className="muted small">
            RMS joint error on episode {episode ?? "--"}, ghost = {ghostLabel}
            {ghost !== DEFAULT && baseGap && (
              <>
                <br />
                default sim was {deg(baseGap.overall_deg)}
              </>
            )}
          </div>
        </section>

        <section>
          <div className="label">Ghost parameters</div>
          <div className="segmented">
            <button className={ghost === DEFAULT ? "on" : ""} onClick={() => setGhost(DEFAULT)} disabled={running}>
              Default
            </button>
            {calib && (running || ghost === LIVE) && (
              <button className={ghost === LIVE ? "on" : ""} onClick={() => setGhost(LIVE)}>
                Live fit
              </button>
            )}
            {dsLocks.map((l) => (
              <button
                key={l.name}
                className={ghost === l.name ? "on" : ""}
                onClick={() => setGhost(l.name)}
                disabled={running}
                title={`created ${l.created_at}`}
              >
                {l.name}
              </button>
            ))}
          </div>
        </section>

        <section>
          <div className="label">Per joint</div>
          <div className="bars">
            {jointNames.map((j) => {
              const b = baseGap?.per_joint_deg[j];
              const g = ghostGap?.per_joint_deg[j];
              return (
                <button
                  key={j}
                  className={`bar-row ${j === joint ? "sel" : ""} ${scored.has(j) ? "" : "unscored"}`}
                  onClick={() => setJoint(j)}
                  title={scored.has(j) ? "" : "not scored (gripper contact isn't simulated)"}
                >
                  <span className="bar-name">{j}</span>
                  <span className="bar-track">
                    {ghost !== DEFAULT && b != null && <span className="bar base" style={{ width: `${(b / maxJointErr) * 100}%` }} />}
                    {g != null && <span className="bar ghost" style={{ width: `${(g / maxJointErr) * 100}%` }} />}
                  </span>
                  <span className="bar-val">{deg(g, 1)}</span>
                </button>
              );
            })}
          </div>
        </section>

        <section className="calib">
          {!running ? (
            <button className="cta" disabled={dsState !== "ready" || !baseline} onClick={startCalibration}>
              {calib?.status === "done" ? "Calibrate again" : "Calibrate"}
            </button>
          ) : (
            <button className="cta secondary" onClick={stopCalibration} disabled={calib?.status === "cancelled"}>
              {calib?.status === "cancelled" ? "Stopping…" : "Stop and keep best fit"}
            </button>
          )}
          {calib && (
            <div className="calib-body">
              <div className="row">
                <span className="muted small">
                  {calib.status === "running" && `fitting on train episodes · eval ${lastProgress?.evaluation ?? 0}`}
                  {calib.status === "cancelled" && "stopping after this generation"}
                  {calib.status === "done" && `saved as ${calib.name}`}
                  {calib.status === "error" && <span className="bad">{calib.error}</span>}
                </span>
                <span className="mono small">{deg(lastProgress?.best_score)}</span>
              </div>
              <Sparkline values={calib.best.filter((v) => isFinite(v))} total={150} baseline={calib.best[0]} />
              {calib.result && (
                <div className="result">
                  <div>
                    <span className="muted small">held-out episodes</span>
                    <div className="mono">
                      {deg(calib.result.baseline_eval_score)} → <b>{deg(calib.result.eval_score)}</b>
                    </div>
                  </div>
                  <div>
                    <span className="muted small">train episodes</span>
                    <div className="mono">
                      {deg(calib.result.baseline_train_score)} → {deg(calib.result.train_score)}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </section>

        {shownParams && defaultParams && (
          <section>
            <div className="label">Parameters</div>
            <table className="params">
              <thead>
                <tr>
                  <th />
                  <th>default</th>
                  <th>ghost</th>
                </tr>
              </thead>
              <tbody>
                {PARAM_KEYS.map((k) => (
                  <tr key={k}>
                    <td>{k}</td>
                    <td>{fmtParam(k, defaultParams[k])}</td>
                    <td className={sameParam(shownParams[k], defaultParams[k]) ? "" : "changed"}>
                      {fmtParam(k, shownParams[k])}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        )}
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
          <span className="muted small trace-label">
            {joint}: <i className="swatch real" /> real <i className="swatch sim" /> sim
          </span>
          <div className="speeds">
            {SPEEDS.map((s) => (
              <button key={s} className={s === speed ? "on" : ""} onClick={() => setSpeed(s)}>
                {s}×
              </button>
            ))}
          </div>
        </div>
        {baseline && (
          <JointTrace
            real={baseline.real_q.map((r) => r[jIndex])}
            sim={(replay ?? baseline).sim_q.map((r) => r[jIndex])}
            frame={frame}
            onSeek={seek}
          />
        )}
      </footer>
    </div>
  );
}
