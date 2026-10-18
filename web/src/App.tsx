import { Suspense, useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Grid, OrbitControls } from "@react-three/drei";
import {
  api,
  lockNameFor,
  type CalibrationResult,
  type Compensation,
  type CrossMatrix,
  type DatasetInfo,
  type EpisodeInfo,
  type Geometry,
  type JobEvent,
  type LockSummary,
  type ObjectsFit,
  type ObjectTrack,
  type NumericParamKey,
  type Params,
  type ParamValue,
  type PolicyEvalEvent,
  type PolicyInfo,
  type PolicyMode,
  type PolicyResults,
  type PolicyTrainEvent,
  type PolicyTrainProgress,
  type PolicyViewer,
  type ProgressEvent,
  type Replay,
  type Residual,
  type SceneFit,
} from "./api";
import * as THREE from "three";
import { Arm, compMaterial, ghostMaterial, policyDefaultMaterial, policyLockMaterial, realMaterial, type Playhead } from "./Arm";
import { CameraView } from "./CameraView";
import { SceneGeometry } from "./Scene";
import { ObjectMarker } from "./Objects";
import { CrossHeatmap, JointTrace, ResidualStrip, Sparkline } from "./Charts";
import { AddDataset } from "./AddDataset";
import { Factors } from "./Factors";
import { FleetView } from "./Fleet";
import { ARM_JOINTS, JointArm } from "./JointArm";

// Layout: 3D viewer (real arm + ghost sim arm) on the left, gap panel on
// the right, episode timeline along the bottom. The ghost is the simulator
// replaying the real commands with either the default parameters, a saved
// robot lock, or (while calibrating) the best fit so far.

const DEFAULT = "default";
const LIVE = "live";
// Parameters set by hand in the tuning panel.
const MANUAL = "manual";
// What the tuning panel offers: a log-scale multiplier on the starting value
// (per-joint values keep their shape), or whole frames added to the latency.
const TUNE_KEYS: { key: NumericParamKey; kind: "scale" | "frames"; hint: string }[] = [
  { key: "kp", kind: "scale", hint: "servo stiffness: how hard it pulls toward the target" },
  { key: "kv", kind: "scale", hint: "velocity gain: how much it resists moving fast" },
  { key: "damping", kind: "scale", hint: "joint damping: drag that slows motion" },
  { key: "armature", kind: "scale", hint: "motor inertia felt at the joint" },
  { key: "frictionloss", kind: "scale", hint: "dry friction: torque needed to start moving" },
  { key: "gravity_scale", kind: "scale", hint: "how much of the links' weight gravity pulls on" },
  { key: "latency_frames", kind: "frames", hint: "frames between a command and the servo acting on it" },
];
const TUNE_LOG_RANGE = 2; // the multiplier spans 10^-2 to 10^2
const TUNE_FRAMES = 6;
// What the server accepts as a lock name (its check also refuses "..").
const LOCK_NAME = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;
const validLockName = (s: string) => LOCK_NAME.test(s) && !s.includes("..");
const PARAM_KEYS: NumericParamKey[] = ["kp", "kv", "damping", "armature", "frictionloss", "latency_frames", "forcerange", "gravity_scale", "deadband"];
// Terms a lock omits at their defaults; the row shows only when the lock carries one.
const OPTIONAL_DEFAULTS: Partial<Record<NumericParamKey, number>> = { gravity_scale: 1, deadband: 0 };
// The STS3215 servo model's fields, written only by a --servo fit; shown when either side has them.
const SERVO_KEYS: NumericParamKey[] = ["servo_gain", "p_coefficient", "vin", "kt", "resistance", "max_velocity"];
// With a servo model these hold its linearized PD equivalent, not fitted values.
const LINEARIZED_KEYS: ReadonlySet<NumericParamKey> = new Set<NumericParamKey>(["kp", "kv", "forcerange"]);
const servoOf = (p: Params) => p.servo ?? "position";
const SPEEDS = [0.25, 0.5, 1, 2];
// Which arms the viewer draws: both overlaid, or one at a time.
const SHOW = ["both", "real", "sim"] as const;
type Show = (typeof SHOW)[number];
// CMA-ES budget for a UI run; also sizes the sparkline's x axis.
const MAX_EVALUATIONS = 150;
// Budget of the optional per-joint refinement stage (the API's default).
const PER_JOINT_EVALUATIONS = 600;

type Calibration = {
  jobId: string;
  name: string;
  // Evaluations the run was asked for over all its stages; sizes the sparkline.
  total: number;
  perJoint: boolean;
  status: "running" | "done" | "error" | "cancelled";
  progress: ProgressEvent[];
  best: number[];
  result?: CalibrationResult;
  error?: string;
};

const noop = () => {};

// A policy training or evaluation job streamed from the API.
type PolicyJob = {
  jobId: string;
  kind: "train" | "evaluate";
  mode?: PolicyMode;
  status: "running" | "done" | "error";
  progress: PolicyTrainProgress[];
  // Evaluation progress: episodes done / total.
  i: number;
  n: number;
  error?: string;
};

// Epochs a UI training run asks for (early stopping usually ends it sooner).
const POLICY_EPOCHS = 200;

const short = (id: string) => id.split("/").pop() ?? id;

const deg = (v: number | null | undefined, digits = 2) => (v == null || !isFinite(v) ? "--" : `${v.toFixed(digits)}°`);
const cm = (v: number | null | undefined, digits = 1) => (v == null || !isFinite(v) ? "--" : `${v.toFixed(digits)} cm`);

function fmtScalar(k: NumericParamKey, v: number) {
  if (k === "latency_frames") return `${Math.round(v)} fr`;
  if (Math.abs(v) >= 100) return v.toFixed(1);
  if (Math.abs(v) >= 1) return v.toFixed(3);
  return v.toFixed(4);
}

// A per-joint value shows as its range across the six joints.
function fmtParam(k: NumericParamKey, v: ParamValue) {
  if (!Array.isArray(v)) return fmtScalar(k, v);
  const lo = Math.min(...v);
  const hi = Math.max(...v);
  return lo === hi ? fmtScalar(k, lo) : `${fmtScalar(k, lo)}–${fmtScalar(k, hi)}`;
}

const fmtMult = (m: number) => (m >= 10 ? m.toFixed(0) : m >= 1 ? m.toFixed(1) : m.toFixed(2));

// The tuning panel's sliders applied to the parameters it started from.
function applyTune(base: Params, adj: Partial<Record<NumericParamKey, number>>): Params {
  const out: Params = { ...base };
  for (const { key, kind } of TUNE_KEYS) {
    const a = adj[key] ?? 0;
    // A lock without a gravity term gets one only once its slider moves.
    const v = base[key] ?? (key === "gravity_scale" && a !== 0 ? 1 : undefined);
    if (v == null) continue;
    const f = kind === "frames" ? (x: number) => Math.max(0, Math.round(x + a)) : (x: number) => x * 10 ** a;
    out[key] = Array.isArray(v) ? v.map(f) : f(v);
  }
  return out;
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

const CAMERA_POSITION: [number, number, number] = [0.62, 0.48, 0.72];
const CAMERA_TARGET: [number, number, number] = [0, 0.16, 0];
// Widest view the default camera frames the whole arm in; narrower panes
// (side by side) pull the camera back so the arm still fits.
const FRAMED_ASPECT = 1.6;

function FitCamera() {
  const camera = useThree((s) => s.camera);
  const aspect = useThree((s) => s.size.width / Math.max(1, s.size.height));
  // The pull-back factor last applied; a later aspect change rescales the
  // camera's current offset from the target, so an orbit the user made survives
  // a resize or the side-by-side toggle.
  const lastK = useRef<number | null>(null);
  useEffect(() => {
    const k = Math.max(1, FRAMED_ASPECT / aspect);
    const [tx, ty, tz] = CAMERA_TARGET;
    if (lastK.current == null) {
      const [x, y, z] = CAMERA_POSITION;
      camera.position.set(tx + (x - tx) * k, ty + (y - ty) * k, tz + (z - tz) * k);
    } else if (lastK.current !== k) {
      const r = k / lastK.current;
      const { x, y, z } = camera.position;
      camera.position.set(tx + (x - tx) * r, ty + (y - ty) * r, tz + (z - tz) * r);
    }
    lastK.current = k;
  }, [camera, aspect]);
  return null;
}

// A pane that mirrors another pane's playhead: the second episode is a pure
// function of the first one's clock (scaled by frame rate, wrapped to its
// own length), so the two never drift apart.
function Follow({
  lead,
  leadFps,
  playhead,
  fps,
  frames,
}: {
  lead: Playhead;
  leadFps: number;
  playhead: Playhead;
  fps: number;
  frames: number;
}) {
  useFrame(() => {
    if (frames < 2) return;
    playhead.frame = ((lead.frame / Math.max(1, leadFps)) * fps) % frames;
  });
  return null;
}

// One 3D view: the real arm, the sim ghost and (optionally) the
// lag-compensated sim for one episode, with its own playhead and camera.
function Scene({
  geometry,
  realPoses,
  simPoses,
  compPoses,
  hotSpans,
  clip,
  show,
  playhead,
  playing,
  fps,
  frames,
  speed,
  onFrame,
  ghostMat = ghostMaterial,
  children,
  follow,
}: {
  geometry: Geometry | null;
  realPoses: number[][][] | null;
  simPoses: number[][][] | null;
  compPoses: number[][][] | null;
  hotSpans: number[][] | null;
  clip: string;
  show: Show;
  playhead: Playhead;
  playing: boolean;
  fps: number;
  frames: number;
  speed: number;
  onFrame: (f: number) => void;
  // A second pane gets its own copy: the warm glow on hot spans animates it.
  ghostMat?: THREE.Material;
  // Extra things drawn in the arm's frame (policy ghosts, the fitted scene, the tracked object).
  children?: ReactNode;
  // Follow another pane's playhead (scaled by fps) instead of running its own clock.
  follow?: { lead: Playhead; leadFps: number };
}) {
  return (
    <Canvas shadows camera={{ position: CAMERA_POSITION, fov: 38, near: 0.01, far: 20 }}>
      <FitCamera />
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
          {/* MuJoCo is z-up; three.js is y-up. Keyed by model so a robot switch swaps the meshes cleanly. */}
          <group key={geometry.robot} rotation={[-Math.PI / 2, 0, 0]}>
            <Arm
              geometry={geometry}
              poses={realPoses}
              clip={clip}
              playhead={playhead}
              material={realMaterial}
              hidden={show === "sim"}
            />
            <Arm
              geometry={geometry}
              poses={simPoses}
              clip={clip}
              playhead={playhead}
              material={ghostMat}
              hotSpans={hotSpans}
              hidden={show === "real"}
            />
            <Arm
              geometry={geometry}
              poses={compPoses}
              clip={clip}
              playhead={playhead}
              material={compMaterial}
              hidden={show === "real"}
            />
            {children}
          </group>
        </Suspense>
      )}
      {follow ? (
        <Follow lead={follow.lead} leadFps={follow.leadFps} playhead={playhead} fps={fps} frames={frames} />
      ) : (
        <Driver playhead={playhead} playing={playing} fps={fps} frames={frames} speed={speed} onFrame={onFrame} />
      )}
      <OrbitControls makeDefault target={CAMERA_TARGET} />
    </Canvas>
  );
}

export default function App() {
  // Top-level screen: the single-arm viewer, or the fleet dashboard. The
  // hash deep-links to the fleet (`#fleet`, or `#fleet/<lock>` with a row open).
  const [view, setViewState] = useState<"viewer" | "fleet">(() =>
    window.location.hash.startsWith("#fleet") ? "fleet" : "viewer",
  );
  const setView = (v: "viewer" | "fleet") => {
    setViewState(v);
    window.history.replaceState(null, "", v === "fleet" ? "#fleet" : window.location.pathname + window.location.search);
  };
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
  const [show, setShow] = useState<Show>("both");
  // Per-joint error as rows, or drawn on a sketch of the arm (default or ghost).
  const [jointView, setJointView] = useState<"rows" | "arm">("rows");
  const [armOf, setArmOf] = useState<"ghost" | "default">("ghost");
  // Hand tuning: the parameters it started from, what it started from, and
  // each slider's setting (log10 multiplier, or frames for latency).
  const [tuneBase, setTuneBase] = useState<Params | null>(null);
  const [tuneFrom, setTuneFrom] = useState("");
  // The saved lock the tuning started from (its command map and unit map
  // stay with the tuned parameters), or null for the default/live sim.
  const [tuneLock, setTuneLock] = useState<string | null>(null);
  // Saving the tuned values as a lock: its name, and the request's state.
  const [saveName, setSaveName] = useState("");
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [tuneAdj, setTuneAdj] = useState<Partial<Record<NumericParamKey, number>>>({});
  // A second episode shown side by side with the first; null when off.
  const [compare, setCompare] = useState<number | null>(null);
  const [compareBase, setCompareBase] = useState<Replay | null>(null);
  const [compareReplay, setCompareReplay] = useState<Replay | null>(null);
  const [frame, setFrame] = useState(0);
  const [joint, setJoint] = useState("shoulder_pan");
  const [calib, setCalib] = useState<Calibration | null>(null);
  const [perJoint, setPerJoint] = useState(false);
  // null: the built-in factors; otherwise the shared-stage names to fit.
  const [fitFactors, setFitFactors] = useState<string[] | null>(null);
  const [fitGravity, setFitGravity] = useState(false);
  const [fitDeadband, setFitDeadband] = useState(false);
  const [residual, setResidual] = useState<Residual | null>(null);
  const [crossOpen, setCrossOpen] = useState(false);
  // Cached per (loaded datasets × saved locks); cleared when either changes.
  const [cross, setCross] = useState<{ key: string; data: CrossMatrix } | null>(null);
  const [crossBusy, setCrossBusy] = useState(false);
  const [crossError, setCrossError] = useState<string | null>(null);
  const [showComp, setShowComp] = useState(false);
  const [comp, setComp] = useState<Compensation | null>(null);
  const [compBusy, setCompBusy] = useState(false);
  // The static scene (table, workspace) fitted for the camera lock, if any.
  const [scene, setScene] = useState<SceneFit | null>(null);
  // A policy in the loop: the lock's saved policies, their evaluations, the
  // running job, and the evaluated episode's ghosts in both sims.
  const [policies, setPolicies] = useState<PolicyInfo[]>([]);
  const [polResults, setPolResults] = useState<PolicyResults | null>(null);
  const [polJob, setPolJob] = useState<PolicyJob | null>(null);
  const [polMode, setPolMode] = useState<PolicyMode>("shadow");
  const [showPolicy, setShowPolicy] = useState(false);
  const [polViewer, setPolViewer] = useState<PolicyViewer | null>(null);
  const [polViewerNote, setPolViewerNote] = useState<string | null>(null);
  const [polViewerBusy, setPolViewerBusy] = useState(false);

  // The manipulated object fitted for the camera lock, and this episode's track.
  const [objects, setObjects] = useState<ObjectsFit | null>(null);
  const [track, setTrack] = useState<ObjectTrack | null>(null);

  const playhead = useMemo<Playhead>(() => ({ frame: 0 }), []);
  const comparePlayhead = useMemo<Playhead>(() => ({ frame: 0 }), []);
  const compareGhostMat = useMemo(() => ghostMaterial.clone(), []);

  // ---- initial load ---------------------------------------------------------
  useEffect(() => {
    api.datasets().then(
      (ds) => {
        setDatasets(ds);
        setDataset((cur) => cur || ds.find((d) => d.loaded)?.id || ds[0]?.id || "");
      },
      (e) => setError(`API unreachable: ${e.message}. Is \`antares serve\` running?`),
    );
    api.locks().then(setLocks, () => setLocks([]));
  }, []);

  // ---- arm model for the selected dataset's robot ---------------------------
  // SO-100 and SO-101 datasets may be simulated with different MJCFs; the
  // geometry (and its meshes) follows whatever model the API uses for it.
  const currentDs = datasets.find((d) => d.id === dataset);
  const robot = currentDs?.robot ?? "so101";
  useEffect(() => {
    let live = true;
    api.model(robot).then(
      (g) => live && setGeometry(g),
      (e) => live && setError(`Could not load the arm model: ${e.message}`),
    );
    return () => {
      live = false;
    };
  }, [robot]);

  // A dataset added from the "+ add dataset" popover: list it and select it;
  // the dataset effect below then loads it behind the usual overlay.
  const addDataset = (info: DatasetInfo) => {
    setDatasets((ds) => (ds.some((d) => d.id === info.id) ? ds.map((d) => (d.id === info.id ? info : d)) : [...ds, info]));
    setError(null);
    setDataset(info.id);
  };

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
    setCompare(null);
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
    comparePlayhead.frame = 0;
    setFrame(0);
    return () => {
      live = false;
    };
  }, [dataset, episode, playhead, comparePlayhead]);

  // ---- second episode, side by side -----------------------------------------
  // Its ghost follows the first pane's choice: default, the same lock, or the
  // live fit's current best parameters. Both panes restart together.
  useEffect(() => {
    setCompareBase(null);
    setCompareReplay(null);
    if (!dataset || compare == null) return;
    let live = true;
    api.replay(dataset, compare, null).then(
      (r) => live && setCompareBase(r),
      (e) => live && setError(`Replay of episode ${compare} failed: ${e.message}`),
    );
    return () => {
      live = false;
    };
  }, [dataset, compare]);

  useEffect(() => {
    playhead.frame = 0;
    comparePlayhead.frame = 0;
    setFrame(0);
  }, [compare, playhead, comparePlayhead]);

  // The second pane's ghost with a lock or the live fit is fetched for its
  // own episode; with the default ghost it is simply the base replay, read at
  // render time so no request is made for it. Requests run one at a time:
  // while one is in flight the newest wanted one waits (the live fit lands a
  // new best every few events), and a reply for an older episode or ghost is
  // dropped.
  const tuned = useMemo(() => (tuneBase ? applyTune(tuneBase, tuneAdj) : null), [tuneBase, tuneAdj]);
  const liveParams = ghost === MANUAL ? tuned : ghost === LIVE ? replay?.params ?? null : null;
  const [compareBusy, setCompareBusy] = useState(false);
  const compareKey = useRef("");
  const compareWanted = useRef<{ key: string; episode: number; run: () => Promise<Replay> } | null>(null);
  const compareInFlight = useRef(false);
  const pumpCompare = useCallback(() => {
    if (compareInFlight.current || !compareWanted.current) return;
    const { key, episode: ep, run } = compareWanted.current;
    compareWanted.current = null;
    compareInFlight.current = true;
    setCompareBusy(true);
    run()
      .then(
        (r) => key === compareKey.current && setCompareReplay(r),
        (e) => key === compareKey.current && setError(`Replay of episode ${ep} failed: ${(e as Error).message}`),
      )
      .finally(() => {
        compareInFlight.current = false;
        setCompareBusy(false);
        pumpCompare();
      });
  }, []);
  useEffect(() => {
    compareKey.current = `${dataset}#${compare}#${ghost}`;
    setCompareReplay(null);
  }, [dataset, compare, ghost]);
  useEffect(() => {
    if (!dataset || compare == null || ghost === DEFAULT) return;
    if ((ghost === LIVE || ghost === MANUAL) && !liveParams) return;
    const ep = compare;
    const key = compareKey.current;
    compareWanted.current = {
      key,
      episode: ep,
      run: () =>
        (ghost === LIVE || ghost === MANUAL) && liveParams
          ? api.replayParams(dataset, ep, liveParams, ghost === MANUAL ? tuneLock : null)
          : api.replay(dataset, ep, ghost),
    };
    pumpCompare();
  }, [dataset, compare, ghost, liveParams, tuneLock, pumpCompare]);
  const shownCompare = ghost === DEFAULT ? compareBase : compareReplay;

  // ---- ghost replay for a saved lock ----------------------------------------
  useEffect(() => {
    if (!dataset || episode == null || ghost === LIVE || ghost === MANUAL) return;
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

  // ---- residual (real - sim) for the ghost's parameters ---------------------
  // The strip follows whatever lock the ghost is showing; the live fit has no
  // saved baseline to z-score against, so it keeps the last strip.
  useEffect(() => {
    if (!dataset || episode == null || ghost === LIVE || ghost === MANUAL) return;
    let live = true;
    setResidual(null);
    api.residual(dataset, episode, ghost === DEFAULT ? null : ghost).then(
      (r) => live && setResidual(r),
      () => {}, // the strip is a bonus; replay errors are already reported
    );
    return () => {
      live = false;
    };
  }, [dataset, episode, ghost]);

  // ---- lag-compensated ghost (sim-validated only) ---------------------------
  // ILC runs in the calibrated sim for the ghost's parameters; a few hundred
  // ms per episode, so it is fetched only while the toggle is on.
  useEffect(() => {
    if (!showComp || !dataset || episode == null || ghost === LIVE || ghost === MANUAL) return;
    let live = true;
    setComp(null);
    setCompBusy(true);
    api
      .compensate(dataset, episode, ghost === DEFAULT ? null : ghost)
      .then(
        (c) => live && setComp(c),
        (e) => live && setError(`Lag compensation failed: ${e.message}`),
      )
      .finally(() => live && setCompBusy(false));
    return () => {
      live = false;
    };
  }, [showComp, dataset, episode, ghost]);

  // ---- live ghost while calibrating: replay the best params so far ----------
  // One request in flight at a time; when it lands, the newest best (if it
  // changed meanwhile) goes next. Each landing eases the ghost to the new fit.
  // The same pump serves the hand-tuned ghost: `liveLock` is the lock the
  // tuned params started from (its maps travel with the request) and
  // `liveReport` says whether a failed replay is shown (hand tuning) or
  // stays quiet (a transient failure during a fit).
  const liveWanted = useRef<Params | null>(null);
  const liveInFlight = useRef(false);
  const liveKey = useRef("");
  const liveLock = useRef<string | null>(null);
  const liveReport = useRef(false);
  const where = useRef({ dataset, episode });
  where.current = { dataset, episode };
  const pumpLive = useCallback(() => {
    const { dataset: ds, episode: ep } = where.current;
    if (liveInFlight.current || !liveWanted.current || ep == null) return;
    const params = liveWanted.current;
    const lock = liveLock.current;
    const key = `${ds}#${ep}#${lock ?? ""}#${JSON.stringify(params)}`;
    if (key === liveKey.current) return;
    liveInFlight.current = true;
    liveKey.current = key;
    setReplayBusy(true);
    api
      .replayParams(ds, ep, params, lock)
      .then(
        (r) => {
          if (where.current.dataset === ds && where.current.episode === ep) setReplay(r);
        },
        (e) => liveReport.current && setError(`Replay with your parameters failed: ${(e as Error).message}`),
      )
      .finally(() => {
        // Cleared whatever the ghost is now, so the busy dot never sticks.
        liveInFlight.current = false;
        setReplayBusy(false);
        pumpLive();
      });
  }, []);

  useEffect(() => {
    if (ghost !== MANUAL) {
      liveLock.current = null;
      liveReport.current = false;
    }
    if (ghost === LIVE) pumpLive(); // also re-runs the latest best on a new episode
  }, [ghost, dataset, episode, pumpLive]);

  // ---- hand-tuned ghost: replay whenever a slider settles -------------------
  // Through the same one-in-flight pump as the live fit: a dragged slider
  // never queues more than one request on the server, and the newest value
  // goes next when the current replay lands.
  const tunedKey = tuned ? JSON.stringify(tuned) : "";
  useEffect(() => {
    if (ghost !== MANUAL || !tuned || !dataset || episode == null) return;
    const t = setTimeout(() => {
      liveWanted.current = tuned;
      pumpLive();
    }, 200);
    return () => clearTimeout(t);
    // tunedKey stands for tuned's contents
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ghost, dataset, episode, tunedKey, pumpLive]);

  const startTuning = () => {
    const fromLock = ghost !== DEFAULT && ghost !== LIVE && ghost !== MANUAL ? ghost : null;
    setTuneBase(replay?.params ?? baseline?.params ?? null);
    setTuneFrom(ghostLabel);
    setTuneLock(fromLock);
    setTuneAdj({});
    setSaveName(`${lockNameFor(dataset)}-tuned`);
    setSaveError(null);
    liveLock.current = fromLock;
    liveReport.current = true;
    liveKey.current = ""; // the lock's own replay came from GET /api/replay; replay the base once
    setGhost(MANUAL);
  };

  const saveTuned = async () => {
    if (!tuned || !dataset) return;
    setSaving(true);
    setSaveError(null);
    try {
      const saved = await api.saveLock(dataset, saveName.trim(), tuned, tuneLock);
      setLocks(await api.locks());
      setGhost(saved.name); // the ghost now replays the saved lock
    } catch (e) {
      setSaveError((e as Error).message);
    } finally {
      setSaving(false);
    }
  };

  // ---- calibration -----------------------------------------------------------
  const sourceRef = useRef<EventSource | null>(null);
  useEffect(() => () => sourceRef.current?.close(), []);

  const startCalibration = async () => {
    if (!dataset) return;
    setError(null);
    const name = lockNameFor(dataset);
    let jobId: string;
    const total = MAX_EVALUATIONS + (perJoint ? PER_JOINT_EVALUATIONS : 0);
    try {
      ({ job_id: jobId } = await api.calibrate(
        dataset,
        name,
        MAX_EVALUATIONS,
        perJoint ? { maxEvaluations: PER_JOINT_EVALUATIONS } : undefined,
        fitFactors,
        { fitGravity, fitDeadband },
      ));
    } catch (e) {
      setError(`Could not start calibration: ${(e as Error).message}`);
      return;
    }
    setCalib({ jobId, name, total, perJoint, status: "running", progress: [], best: [] });
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

  // ---- cross matrix: every lock on every loaded dataset ----------------------
  const loadedIds = datasets.filter((d) => d.loaded).map((d) => d.id);
  const lockNames = locks.map((l) => l.name);
  const crossKey = `${loadedIds.join(",")}|${lockNames.join(",")}`;
  const crossData = cross?.key === crossKey ? cross.data : null;
  const runCross = async () => {
    setCrossBusy(true);
    setCrossError(null);
    try {
      const data = await api.cross(loadedIds, lockNames);
      setCross({ key: crossKey, data });
    } catch (e) {
      setCrossError((e as Error).message);
    } finally {
      setCrossBusy(false);
    }
  };
  useEffect(() => {
    if (!crossOpen) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setCrossOpen(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [crossOpen]);

  // ---- derived ---------------------------------------------------------------
  const dsLocks = locks.filter((l) => l.dataset === dataset);
  const lockHome = Object.fromEntries(locks.map((l) => [l.name, l.dataset]));
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
  const stageLabel =
    lastProgress?.stage === "per_joint" ? "per-joint refinement"
    : lastProgress?.stage === "command_map" ? "fitting sensor map"
    : "fitting shared params";
  const shownParams = replay?.params ?? null;
  const defaultParams = baseline?.params ?? null;
  const clip = `${dataset}#${episode ?? ""}`;
  const dsWarnings = currentDs?.warnings ?? [];

  // The trace arrays only change with the replay or the joint, not per frame.
  const traceReal = useMemo(() => baseline?.real_q.map((r) => r[jIndex]) ?? [], [baseline, jIndex]);
  const traceSim = useMemo(() => (replay ?? baseline)?.sim_q.map((r) => r[jIndex]) ?? [], [replay, baseline, jIndex]);
  // Summed |residual| over the scored joints (the gripper is not simulated).
  const residualSum = useMemo(() => {
    if (!residual) return [];
    const cols = residual.scored_joints.map((j) => residual.joints.indexOf(j)).filter((i) => i >= 0);
    return residual.residual.map((row) => cols.reduce((s, i) => s + Math.abs(row[i]), 0));
  }, [residual]);
  const hotSpans = useMemo(() => residual?.events.map((e) => [e.start, e.end]) ?? null, [residual]);
  const inEvent = residual?.events.find((e) => frame >= e.start && frame < e.end);
  const compPoses = showComp && ghost !== LIVE && ghost !== MANUAL && comp ? comp.sim_comp_poses : null;

  // The arm sketch is drawn for the SO-100/101 joint layout only.
  const canSketch = ARM_JOINTS.every((j) => jointNames.includes(j)) && jointNames.length === ARM_JOINTS.length;
  const ghostLabel =
    ghost === DEFAULT ? "default sim"
    : ghost === LIVE ? "best fit so far"
    : ghost === MANUAL ? "your parameters"
    : `lock ${ghost}`;
  // Camera fits live with a lock: the ghost's, or the dataset's first saved one.
  const cameraLock = ghost !== DEFAULT && ghost !== LIVE && ghost !== MANUAL ? ghost : dsLocks[0]?.name ?? null;

  const compareFrames = compareBase?.real_poses.length ?? 0;
  // ---- scene fit saved for the camera lock -----------------------------------
  useEffect(() => {
    if (!cameraLock) {
      setScene(null);
      return;
    }
    let live = true;
    api.scene(cameraLock).then(
      (r) => live && setScene(r.scene),
      () => live && setScene(null),
    );
    return () => {
      live = false;
    };
  }, [cameraLock]);

  // ---- policy: saved policies and evaluations of the camera lock -------------
  const policyLock = cameraLock;
  const policy = policies[0] ?? null;
  const refreshPolicy = useCallback((lock: string) => {
    api.policies(lock).then((r) => setPolicies(r.policies), () => setPolicies([]));
    api.policyResults(lock).then(setPolResults, () => setPolResults(null));
  }, []);
  useEffect(() => {
    setPolicies([]);
    setPolResults(null);
    setPolViewer(null);
    if (policyLock) refreshPolicy(policyLock);
  }, [policyLock, refreshPolicy]);

  // The two policy ghosts for the current episode, recomputed by the API on
  // request (a fraction of a second): only held-out episodes have them.
  const polBusy = polJob?.status === "running";
  useEffect(() => {
    if (!showPolicy || !policyLock || !policy || episode == null || polBusy) return;
    let live = true;
    setPolViewerBusy(true);
    setPolViewerNote(null);
    api
      .policyViewer(dataset, policyLock, episode, polMode)
      .then(
        (v) => live && setPolViewer(v),
        (e) => {
          if (!live) return;
          setPolViewer(null);
          setPolViewerNote(e.status === 400 ? `episode ${episode} is not held out from the policy's training` : e.message);
        },
      )
      .finally(() => live && setPolViewerBusy(false));
    return () => {
      live = false;
    };
  }, [showPolicy, policyLock, policy, dataset, episode, polMode, polBusy]);

  const policySource = useRef<EventSource | null>(null);
  useEffect(() => () => policySource.current?.close(), []);

  const followPolicyJob = (jobId: string, kind: PolicyJob["kind"], mode?: PolicyMode) => {
    setPolJob({ jobId, kind, mode, status: "running", progress: [], i: 0, n: 0 });
    policySource.current?.close();
    const src = api.jobEvents(jobId);
    policySource.current = src;
    src.onmessage = (msg) => {
      const ev = JSON.parse(msg.data) as PolicyTrainEvent | PolicyEvalEvent;
      if (ev.type === "progress") {
        setPolJob((j) =>
          j && j.jobId === jobId
            ? "epoch" in ev
              ? { ...j, progress: [...j.progress, ev] }
              : { ...j, i: ev.i, n: ev.n }
            : j,
        );
      } else if (ev.type === "done") {
        src.close();
        setPolJob((j) => (j && j.jobId === jobId ? { ...j, status: "done" } : j));
        if (policyLock) refreshPolicy(policyLock);
        if (kind === "evaluate") setShowPolicy(true);
      } else if (ev.type === "error") {
        src.close();
        setPolJob((j) => (j && j.jobId === jobId ? { ...j, status: "error", error: ev.message } : j));
      }
    };
    src.onerror = () => {
      if (src.readyState === EventSource.CLOSED) {
        setPolJob((j) => (j && j.jobId === jobId && j.status === "running" ? { ...j, status: "error", error: "lost the progress stream" } : j));
      }
    };
  };

  const startPolicyTraining = async () => {
    if (!policyLock) return;
    setError(null);
    try {
      const { job_id } = await api.policyTrain(dataset, policyLock, POLICY_EPOCHS);
      followPolicyJob(job_id, "train");
    } catch (e) {
      setError(`Could not start training: ${(e as Error).message}`);
    }
  };

  const startPolicyEval = async (mode: PolicyMode) => {
    if (!policyLock) return;
    setError(null);
    setPolMode(mode);
    try {
      const { job_id } = await api.policyEvaluate(dataset, policyLock, mode);
      followPolicyJob(job_id, "evaluate", mode);
    } catch (e) {
      setError(`Could not start the evaluation: ${(e as Error).message}`);
    }
  };

  const polResult = polMode === "shadow" ? polResults?.shadow ?? null : polResults?.closedloop ?? null;
  const polRows = polResult ? Object.entries(polResult.summary.variants) : [];
  // Typed headline rows per mode (the table below handles both shapes per row).
  const shadowLock = polResults?.shadow?.summary.variants["lock"] ?? null;
  const shadowDefault = polResults?.shadow?.summary.variants["default"] ?? null;
  const clLock = polResults?.closedloop?.summary.variants["lock"] ?? null;
  const clDefault = polResults?.closedloop?.summary.variants["default"] ?? null;
  const polHeldOut = policy?.eval_episodes ?? [];
  const polPoses = (name: string) =>
    showPolicy && polViewer && polViewer.episode === episode && !polBusy ? polViewer.sims[name]?.poses ?? null : null;
  const lastEpoch = polJob?.progress[polJob.progress.length - 1];

  // ---- objects fit saved for the camera lock ---------------------------------
  useEffect(() => {
    if (!cameraLock) {
      setObjects(null);
      return;
    }
    let live = true;
    api.objects(cameraLock).then(
      (r) => live && setObjects(r.objects),
      () => live && setObjects(null),
    );
    return () => {
      live = false;
    };
  }, [cameraLock]);

  // ---- the object's per-frame track for this episode -------------------------
  // Fetched once per episode (the saved track, or a few seconds of tracking
  // on the server the first time); the viewer reads it at the playhead.
  useEffect(() => {
    setTrack(null);
    if (!dataset || episode == null || !cameraLock || !objects || objects.lock !== cameraLock) return;
    let live = true;
    api.objectsTrack(dataset, episode, cameraLock).then(
      (t) => live && setTrack(t),
      () => live && setTrack(null),
    );
    return () => {
      live = false;
    };
  }, [dataset, episode, cameraLock, objects]);

  const seek = (f: number) => {
    playhead.frame = f;
    if (compareFrames) comparePlayhead.frame = f % compareFrames;
    setFrame(Math.floor(f));
  };

  return (
    <div className={`shell${view === "fleet" ? " fleet-shell" : ""}`}>
      <header className="topbar">
        <span className="brand">antares</span>
        <span className="tagline">sim, calibrated to this robot</span>
        <nav className="segmented views" aria-label="view">
          <button className={view === "viewer" ? "on" : ""} onClick={() => setView("viewer")}>
            Viewer
          </button>
          <button className={view === "fleet" ? "on" : ""} onClick={() => setView("fleet")}>
            Fleet
          </button>
        </nav>
        {view === "fleet" ? null : (
        <div className="picker-wrap">
          {dsState === "loading" && <span className="muted small">loading dataset…</span>}
          {currentDs && (
            <span
              className={`chip ${currentDs.source} ${dsWarnings.length ? "warn" : ""}`}
              title={[
                `${currentDs.source} dataset${currentDs.robot_family ? ` · ${currentDs.robot_family} arm` : ""} · sim model ${currentDs.robot}`,
                ...dsWarnings,
              ].join("\n")}
            >
              {currentDs.source}
              {currentDs.robot_family && currentDs.robot_family !== currentDs.robot && ` · ${currentDs.robot_family}→${currentDs.robot}`}
              {dsWarnings.length > 0 && ` · ${dsWarnings.length} warning${dsWarnings.length === 1 ? "" : "s"}`}
            </span>
          )}
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
          <AddDataset disabled={running} onAdded={addDataset} />
        </div>
        )}
      </header>

      {view === "fleet" ? (
        <FleetView initialOpen={decodeURIComponent(window.location.hash.replace(/^#fleet\/?/, "")) || null} />
      ) : (
      <>
      <main className="viewer">
        <div className="panes">
          <div className="pane">
            <Scene
              geometry={geometry}
              realPoses={baseline?.real_poses ?? null}
              simPoses={replay?.sim_poses ?? null}
              compPoses={compPoses}
              hotSpans={hotSpans}
              clip={clip}
              show={show}
              playhead={playhead}
              playing={playing}
              fps={fps}
              frames={frames}
              speed={speed}
              onFrame={setFrame}
            >
              {geometry && (
                <>
                  <Arm geometry={geometry} poses={polPoses("default")} clip={`${clip}#policy`} playhead={playhead} material={policyDefaultMaterial} />
                  <Arm geometry={geometry} poses={polPoses("lock")} clip={`${clip}#policy`} playhead={playhead} material={policyLockMaterial} />
                </>
              )}
              <SceneGeometry scene={scene} />
              <ObjectMarker track={track} playhead={playhead} />
            </Scene>
            {compare != null && <div className="pane-label">episode {episode ?? "--"} · {deg(ghostGap?.overall_deg)}</div>}
          </div>
          {compare != null && (
            <div className="pane">
              <Scene
                geometry={geometry}
                realPoses={compareBase?.real_poses ?? null}
                simPoses={shownCompare?.sim_poses ?? null}
                compPoses={null}
                hotSpans={null}
                clip={`${dataset}#${compare}`}
                show={show}
                playhead={comparePlayhead}
                playing={playing}
                fps={compareBase?.fps ?? fps}
                frames={compareFrames}
                speed={speed}
                onFrame={noop}
                follow={{ lead: playhead, leadFps: fps }}
                ghostMat={compareGhostMat}
              />
              <div className="pane-label">
                <select
                  className="picker"
                  value={compare}
                  onChange={(e) => setCompare(Number(e.target.value))}
                  aria-label="Episode shown beside the first"
                >
                  {episodes
                    .filter((e) => e.index !== episode)
                    .map((e) => (
                      <option key={e.index} value={e.index}>
                        episode {e.index} · {e.duration_s.toFixed(1)}s
                      </option>
                    ))}
                </select>
                {deg(shownCompare?.gap?.overall_deg)}
                {compareBusy && <span className="dot" />}
              </div>
            </div>
          )}
        </div>

        <div className="legend">
          {show !== "sim" && (
            <span>
              <i className="swatch real" /> real arm (logged)
            </span>
          )}
          {show !== "real" && (
            <span>
              <i className="swatch sim" /> sim ghost, {ghostLabel}
              {replayBusy && <span className="dot" />}
            </span>
          )}
          {showComp && ghost !== LIVE && ghost !== MANUAL && show !== "real" && (
            <span>
              <i className="swatch comp" /> sim with lag-compensated commands (sim-validated only)
              {compBusy && <span className="dot" />}
            </span>
          )}
          {showPolicy && policy && (
            <>
              <span>
                <i className="swatch pol-default" /> policy in the default sim
                {polViewerBusy && <span className="dot" />}
              </span>
              <span>
                <i className="swatch pol-lock" /> policy in the calibrated sim ({polMode === "shadow" ? "its commands replayed" : "closed loop"})
              </span>
              {polViewerNote && <span className="muted small">{polViewerNote}</span>}
            </>
          )}
          {scene && (
            <span className="muted small">
              table at {(scene.table.z * 1000).toFixed(0)} mm, workspace box from the logs
            </span>
          )}
          {track && (
            <span className="muted small">
              <i className="swatch" style={{ background: `rgb(${track.rgb.join(",")})` }} /> object from the video
              {track.source[Math.min(track.source.length - 1, frame)] === "held" ? " (held: fingertip from FK)" : ""}
            </span>
          )}
          {inEvent && show !== "real" && (
            <span className="warm">
              <i className="swatch warm" /> sim disagrees here: {inEvent.kind} on {inEvent.joints.join(", ")} (peak{" "}
              {inEvent.peak_deg.toFixed(1)}°)
            </span>
          )}
        </div>

        <div className="view-controls">
          <div className="segmented">
            <button
              className={compare != null ? "on" : ""}
              onClick={() => setCompare(compare != null ? null : episodes.find((e) => e.index !== episode)?.index ?? null)}
              disabled={episodes.length < 2}
              title="Play a second episode side by side with the first"
            >
              Side by side
            </button>
          </div>
          <div className="segmented" role="group" aria-label="Arms shown">
            {SHOW.map((v) => (
              <button key={v} className={show === v ? "on" : ""} onClick={() => setShow(v)}>
                {v === "both" ? "Real + sim" : v === "real" ? "Real only" : "Sim only"}
              </button>
            ))}
          </div>
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
            <button
              className={ghost === MANUAL ? "on" : ""}
              onClick={() => (ghost === MANUAL ? undefined : startTuning())}
              disabled={running || !baseline}
              title="Set the parameters yourself and watch the ghost replay with them"
            >
              Tune by hand
            </button>
          </div>
          {ghost !== DEFAULT && ghost !== LIVE && ghost !== MANUAL && (
            <div className="export">
              <a className="cta secondary" href={api.exportZipUrl(ghost)} download={`${ghost}.zip`}>
                Download MJCF
              </a>
              <span className="muted small">standalone MuJoCo model with the fitted physics</span>
            </div>
          )}
        </section>

        {ghost === MANUAL && tuneBase && tuned && (
          <section className="tune">
            <div className="row">
              <div className="label">Tune by hand</div>
              <button className="link" onClick={() => setTuneAdj({})} disabled={!Object.keys(tuneAdj).length}>
                reset
              </button>
            </div>
            <div className="muted small">
              Starting from {tuneFrom}. Drag a slider and the ghost replays this episode with the new value
              {servoOf(tuneBase) !== "position" && "; this lock uses the servo model, so kp and kv are derived and left out"}.
            </div>
            {TUNE_KEYS.map(({ key, kind, hint }) => {
              if (servoOf(tuneBase) !== "position" && LINEARIZED_KEYS.has(key)) return null;
              const a = tuneAdj[key] ?? 0;
              const v = tuned[key] ?? (key === "gravity_scale" ? 1 : null);
              if (v == null) return null;
              return (
                <label key={key} className="tune-row" title={hint}>
                  <span className="tune-name">{key}</span>
                  <input
                    type="range"
                    min={kind === "frames" ? -TUNE_FRAMES : -TUNE_LOG_RANGE}
                    max={kind === "frames" ? TUNE_FRAMES : TUNE_LOG_RANGE}
                    step={kind === "frames" ? 1 : 0.02}
                    value={a}
                    onChange={(e) => setTuneAdj((t) => ({ ...t, [key]: Number(e.target.value) }))}
                    onDoubleClick={() => setTuneAdj((t) => ({ ...t, [key]: 0 }))}
                  />
                  <span className={`tune-val ${a ? "changed" : ""}`}>
                    {fmtParam(key, v)}
                    <span className="muted">
                      {kind === "frames" ? (a ? ` ${a > 0 ? "+" : ""}${a}` : "") : a ? ` ×${fmtMult(10 ** a)}` : ""}
                    </span>
                  </span>
                </label>
              );
            })}
            <button
              className="cta secondary"
              onClick={() => {
                seek(0);
                setPlaying(true);
              }}
            >
              Replay from the start
            </button>
            <div className="save-lock">
              <input
                className="picker"
                value={saveName}
                onChange={(e) => setSaveName(e.target.value)}
                aria-label="Name for the new lock"
                spellCheck={false}
              />
              <button className="cta" onClick={saveTuned} disabled={saving || !validLockName(saveName.trim())}>
                {saving ? "Saving…" : "Save as lock"}
              </button>
            </div>
            {saveError ? (
              <div className="bad small">{saveError}</div>
            ) : (
              <div className="muted small">
                Scored on the same held-out episodes as a calibration, then listed with the other locks
                {tuneLock && `; keeps ${tuneLock}'s sensor map and camera fits`}.
              </div>
            )}
          </section>
        )}

        <section>
          <div className="row">
            <div className="label">Per joint</div>
            {canSketch && (
              <div className="segmented small-seg">
                <button className={jointView === "rows" ? "on" : ""} onClick={() => setJointView("rows")}>
                  rows
                </button>
                <button className={jointView === "arm" ? "on" : ""} onClick={() => setJointView("arm")}>
                  on the arm
                </button>
              </div>
            )}
          </div>
          {canSketch && jointView === "arm" ? (
            <>
              {ghost !== DEFAULT && (
                <div className="segmented small-seg">
                  <button className={armOf === "default" ? "on" : ""} onClick={() => setArmOf("default")}>
                    default sim
                  </button>
                  <button className={armOf === "ghost" ? "on" : ""} onClick={() => setArmOf("ghost")}>
                    {ghostLabel}
                  </button>
                </div>
              )}
              <JointArm
                errors={(ghost !== DEFAULT && armOf === "default" ? baseGap : ghostGap)?.per_joint_deg ?? {}}
                selected={jointNames[jIndex] ?? ""}
                onSelect={setJoint}
              />
            </>
          ) : (
            <div className="bars">
              {jointNames.map((j) => {
                const b = baseGap?.per_joint_deg[j];
                const g = ghostGap?.per_joint_deg[j];
                // Default error (grey dot) to the ghost's error (blue dot); the
                // default ghost has nothing to compare, so it shows one dot.
                const compared = ghost !== DEFAULT && b != null && g != null;
                const pos = (v: number) => `${Math.min(100, (v / maxJointErr) * 100)}%`;
                const pct = compared && b > 0 ? Math.round(((g - b) / b) * 100) : null;
                return (
                  <button
                    key={j}
                    className={`db-row ${j === jointNames[jIndex] ? "sel" : ""} ${scored.has(j) ? "" : "unscored"}`}
                    onClick={() => setJoint(j)}
                    title={scored.has(j) ? "" : "not scored (gripper contact isn't simulated)"}
                  >
                    <span className="bar-name">{j}</span>
                    <span className="db-track">
                      {compared && (
                        <span
                          className="db-span"
                          style={{ left: pos(Math.min(b, g)), width: `calc(${pos(Math.max(b, g))} - ${pos(Math.min(b, g))})` }}
                        />
                      )}
                      {compared && <span className="db-dot base" style={{ left: pos(b) }} title={`default ${deg(b, 2)}`} />}
                      {g != null && <span className="db-dot ghost" style={{ left: pos(g) }} title={`${ghostLabel} ${deg(g, 2)}`} />}
                    </span>
                    <span className="bar-val">{deg(g, 1)}</span>
                    <span className={`db-change ${pct == null || pct === 0 ? "" : pct < 0 ? "good" : "bad"}`}>
                      {pct == null ? "" : pct === 0 ? "0%" : `${pct < 0 ? "−" : "+"}${Math.abs(pct)}%`}
                    </span>
                  </button>
                );
              })}
            </div>
          )}
          {ghost !== DEFAULT && !(canSketch && jointView === "arm") && (
            <div className="muted small">
              <i className="db-dot base inline" /> default <i className="db-dot ghost inline" /> {ghostLabel}
            </div>
          )}
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
          <label className={`toggle ${running ? "locked" : ""}`} title="After the shared fit, refine one value per joint (7 params × 6 joints, +600 evaluations, about a minute)">
            <input type="checkbox" checked={perJoint} disabled={running} onChange={(e) => setPerJoint(e.target.checked)} />
            per-joint refinement
            <span className="muted">+{PER_JOINT_EVALUATIONS} evals</span>
          </label>
          <Factors selected={fitFactors} onChange={setFitFactors} disabled={running} />
          <label className={`toggle ${running ? "locked" : ""}`} title="Also fit gravity_scale, the arm's gravity load: how much it sags per pose, apart from how fast kp makes it track">
            <input type="checkbox" checked={fitGravity} disabled={running} onChange={(e) => setFitGravity(e.target.checked)} />
            fit gravity load
          </label>
          <label className={`toggle ${running ? "locked" : ""}`} title="Also fit the servo deadband (firmware deadzone plus gear backlash); one value per joint with per-joint refinement">
            <input type="checkbox" checked={fitDeadband} disabled={running} onChange={(e) => setFitDeadband(e.target.checked)} />
            fit servo deadband
          </label>
          {calib && (
            <div className="calib-body">
              <div className="row">
                <span className="muted small">
                  {calib.status === "running" &&
                    `${stageLabel} · eval ${lastProgress?.evaluation ?? 0}${calib.perJoint ? ` / ${calib.total}` : ""}`}
                  {calib.status === "cancelled" && "stopping after this generation"}
                  {calib.status === "done" && `saved as ${calib.name}`}
                  {calib.status === "error" && <span className="bad">{calib.error}</span>}
                </span>
                <span className="mono small">{deg(lastProgress?.best_score)}</span>
              </div>
              <Sparkline
                values={calib.best.filter((v) => isFinite(v))}
                total={calib.total}
                baseline={calib.best[0]}
                stageAt={calib.perJoint ? MAX_EVALUATIONS : null}
              />
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

        <CameraView
          dataset={dataset}
          episode={episode}
          frame={frame}
          playing={playing}
          lock={cameraLock}
          busy={running}
          scene={scene}
          onSceneChange={setScene}
          objects={objects}
          onObjectsChange={setObjects}
          track={track}
          onError={setError}
        />

        <section>
          <div className="label">Lag compensation</div>
          <label className={`toggle ${running ? "locked" : ""}`} title="ILC in the calibrated sim: commands that make the model follow the real trajectory despite servo latency">
            <input type="checkbox" checked={showComp} disabled={running} onChange={(e) => setShowComp(e.target.checked)} />
            show lag-compensated commands
          </label>
          {showComp && (
            <div className="rms">
              {(["raw", "shift", "ilc"] as const).map((k) => (
                <div key={k}>
                  <span className="muted small">{k === "raw" ? "raw" : k === "shift" ? "shifted" : "ILC"}</span>
                  <div className={`mono ${k === "ilc" && comp ? "good" : ""}`}>{compBusy ? "…" : deg(comp?.rms[k])}</div>
                </div>
              ))}
              <div className="muted small rms-note">
                RMS tracking error of the sim on the real trajectory: raw commands, commands shifted by the lock's latency,
                and iterative learning control{comp && ` (${comp.ilc_iterations} iterations)`}. Sim-validated only: judged
                in the calibrated MuJoCo model, no hardware driven.
              </div>
            </div>
          )}
        </section>

        <section>
          <div className="label">A policy in the loop</div>
          {!policyLock ? (
            <div className="muted small">Calibrate first: the policy trains on a lock's training episodes and is judged on its held-out ones.</div>
          ) : (
            <>
              <div className="row">
                <button
                  className="cta secondary"
                  disabled={running || polBusy || dsState !== "ready"}
                  onClick={startPolicyTraining}
                  title="State-based chunked policy (ACT without images): the last 2 joint states and the grasp point in, the next 10 commands out. Trained on the lock's training episodes, early-stopped on its held-out ones. Seconds on a CPU; needs the [policy] extra (torch)."
                >
                  {polJob?.kind === "train" && polBusy ? "Training…" : policy ? "Retrain policy" : "Train policy"}
                </button>
                <span className="muted small">
                  {polJob?.kind === "train" && polBusy && lastEpoch
                    ? `epoch ${lastEpoch.epoch} · held-out mse ${lastEpoch.eval_mse.toFixed(3)} (best ${lastEpoch.best_eval_mse.toFixed(3)})`
                    : policy
                      ? `${policy.name} · ${policy.train_episodes.length} train / ${policy.eval_episodes.length} held-out episodes`
                      : `on lock ${policyLock}`}
                </span>
              </div>
              {polJob?.kind === "train" && polBusy && (
                <div className="progress">
                  <span style={{ width: `${Math.min(100, (100 * (lastEpoch?.epoch ?? 0)) / POLICY_EPOCHS)}%` }} />
                </div>
              )}
              {polJob?.status === "error" && <div className="bad small">{polJob.error}</div>}
              {policy && (
                <div className="result">
                  <div>
                    <span className="muted small">held-out chunk error</span>
                    <div className="mono">
                      <b>{deg(policy.metrics.policy.eval.rmse_deg)}</b>
                    </div>
                  </div>
                  <div>
                    <span className="muted small">hold state / repeat command</span>
                    <div className="mono">
                      {deg(policy.metrics.baselines.hold_state.eval.rmse_deg)} / {deg(policy.metrics.baselines.repeat_command.eval.rmse_deg)}
                    </div>
                  </div>
                </div>
              )}
              {policy && (
                <>
                  <div className="segmented">
                    <button className={polMode === "shadow" ? "on" : ""} onClick={() => setPolMode("shadow")} disabled={polBusy}>
                      Shadow
                    </button>
                    <button className={polMode === "closedloop" ? "on" : ""} onClick={() => setPolMode("closedloop")} disabled={polBusy}>
                      Closed loop
                    </button>
                  </div>
                  <div className="row">
                    <button className="cta" disabled={running || polBusy} onClick={() => startPolicyEval(polMode)}>
                      {polJob?.kind === "evaluate" && polBusy
                        ? `Evaluating… ${polJob.i}/${polJob.n || polHeldOut.length}`
                        : `Evaluate (${polMode === "shadow" ? "shadow" : "closed loop"})`}
                    </button>
                    <span className="muted small">
                      {polMode === "shadow"
                        ? "its commands on the real observations, rolled through each sim"
                        : "the policy drives each sim from the real start, same goal"}
                    </span>
                  </div>
                  {polJob?.kind === "evaluate" && polBusy && (
                    <div className="progress">
                      <span style={{ width: `${polJob.n ? (100 * polJob.i) / polJob.n : 5}%` }} />
                    </div>
                  )}
                  {polResult && (
                    <>
                      <div className="subscore">
                        {polMode === "shadow" && shadowLock && shadowDefault ? (
                          <>
                            <span className="mono big">{deg(shadowLock.divergence_deg)}</span>
                            <span className="muted small">
                              {" "}
                              between what the stock and the calibrated sim say this policy does (sim vs real: {deg(shadowDefault.sim_vs_real_deg)} stock, {deg(shadowLock.sim_vs_real_deg)} calibrated)
                            </span>
                          </>
                        ) : polMode === "closedloop" && clLock && clDefault ? (
                          <>
                            <span className="mono big">
                              {clDefault.n_success}/{clDefault.n_judged} → {clLock.n_success}/{clLock.n_judged}
                            </span>
                            <span className="muted small">
                              {" "}
                              held-out episodes where the fingertip closes within 2 cm of the real grasp point: stock sim → calibrated sim
                            </span>
                          </>
                        ) : null}
                      </div>
                      <table className="params">
                        <thead>
                          <tr>
                            <th />
                            {polMode === "shadow" ? (
                              <>
                                <th>sim vs real</th>
                                <th>vs stock sim</th>
                                <th>grasp point</th>
                              </>
                            ) : (
                              <>
                                <th>approach RMS</th>
                                <th>grasp point</th>
                                <th>within 2 cm</th>
                              </>
                            )}
                          </tr>
                        </thead>
                        <tbody>
                          {polRows.map(([name, row]) => (
                            <tr key={name}>
                              <td>{name === "default" ? "stock sim" : name === "lock" ? "calibrated" : name === "perjoint" ? "per-joint lock" : name}</td>
                              {"sim_vs_real_deg" in row ? (
                                <>
                                  <td>{deg(row.sim_vs_real_deg)}</td>
                                  <td>{"divergence_deg" in row ? deg(row.divergence_deg) : "–"}</td>
                                  <td>{cm(row.grasp_error_cm)}</td>
                                </>
                              ) : (
                                <>
                                  <td>{deg(row.approach_rms_deg, 1)}</td>
                                  <td>{cm(row.grasp_error_cm)}</td>
                                  <td className={row.n_success > 0 ? "changed" : ""}>
                                    {row.n_success}/{row.n_judged}
                                  </td>
                                </>
                              )}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                      <div className="muted small">
                        {polResult.summary.n_episodes} held-out episodes, sim-only: nothing was driven on hardware.
                      </div>
                    </>
                  )}
                  <label className={`toggle ${polBusy ? "locked" : ""}`} title="Two more ghosts: the policy's trajectory in the stock sim (grey) and in the calibrated sim (green) for the current held-out episode">
                    <input type="checkbox" checked={showPolicy} disabled={polBusy} onChange={(e) => setShowPolicy(e.target.checked)} />
                    show the policy's ghosts
                  </label>
                  {showPolicy && episode != null && !polHeldOut.includes(episode) && polHeldOut.length > 0 && (
                    <div className="row">
                      <span className="muted small">episode {episode} is in the policy's training set</span>
                      <button className="cta secondary" onClick={() => setEpisode(polHeldOut[0])}>
                        go to held-out episode {polHeldOut[0]}
                      </button>
                    </div>
                  )}
                  {showPolicy && polViewer && polViewer.episode === episode && (
                    <div className="muted small">
                      {polMode === "shadow"
                        ? `this episode: ${deg(polViewer.sims["default"]?.sim_vs_real_deg)} stock vs ${deg(polViewer.sims["lock"]?.sim_vs_real_deg)} calibrated against the real arm; ${deg(polViewer.sims["lock"]?.divergence_deg)} between the two sims`
                        : `this episode: grasp point missed by ${cm(polViewer.sims["default"]?.grasp_error_cm)} in the stock sim, ${cm(polViewer.sims["lock"]?.grasp_error_cm)} in the calibrated one`}
                    </div>
                  )}
                </>
              )}
            </>
          )}
        </section>

        <section>
          <div className="label">Every robot is different</div>
          <button className="cta secondary" onClick={() => setCrossOpen(true)} disabled={!locks.length}>
            Compare locks across robots
          </button>
          <div className="muted small">
            {loadedIds.length} dataset{loadedIds.length === 1 ? "" : "s"} loaded × {locks.length} lock
            {locks.length === 1 ? "" : "s"}
            {loadedIds.length < 2 && " · pick another dataset above to add a row"}
          </div>
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
                {(servoOf(defaultParams) !== "position" || servoOf(shownParams) !== "position") && (
                  <tr>
                    <td>servo</td>
                    <td>{servoOf(defaultParams)}</td>
                    <td className={servoOf(shownParams) === servoOf(defaultParams) ? "" : "changed"}>
                      {servoOf(shownParams)}
                    </td>
                  </tr>
                )}
                {PARAM_KEYS.map((k) => {
                  const optional = OPTIONAL_DEFAULTS[k];
                  if (optional != null && defaultParams[k] == null && shownParams[k] == null) return null;
                  const d = defaultParams[k] ?? optional;
                  const g = shownParams[k] ?? optional;
                  if (d == null || g == null) return null; // older locks have no forcerange
                  // a servo fit's kp/kv/forcerange are its linearized equivalent, not fitted
                  const derived = LINEARIZED_KEYS.has(k) && servoOf(shownParams) !== "position";
                  return (
                    <tr key={k}>
                      <td title={derived ? "PD equivalent of the servo model (derived, not fitted)" : undefined}>
                        {k}
                        {derived && <span className="muted"> (PD equiv.)</span>}
                      </td>
                      <td>{fmtParam(k, d)}</td>
                      <td className={sameParam(g, d) ? "" : "changed"}>{fmtParam(k, g)}</td>
                    </tr>
                  );
                })}
                {SERVO_KEYS.map((k) => {
                  const d = defaultParams[k];
                  const g = shownParams[k];
                  if (d == null && g == null) return null; // position-actuator fits have none
                  return (
                    <tr key={k}>
                      <td>{k}</td>
                      <td>{d == null ? "–" : fmtParam(k, d)}</td>
                      <td className={d != null && g != null && sameParam(g, d) ? "" : "changed"}>
                        {g == null ? "–" : fmtParam(k, g)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </section>
        )}
      </aside>

      {crossOpen && (
        <div className="modal-backdrop" onClick={() => setCrossOpen(false)}>
          <div className="modal" role="dialog" aria-modal="true" onClick={(e) => e.stopPropagation()}>
            <div className="row">
              <div>
                <div className="label">Every robot is different</div>
                <div className="muted small">
                  Each saved lock replayed on each loaded dataset's held-out episodes (20 train / 10 eval, seed 0). RMS joint
                  error in degrees; lower is better. The outlined cell is a lock on the robot it was fitted on.
                </div>
              </div>
              <button className="icon" onClick={() => setCrossOpen(false)} aria-label="Close">
                ×
              </button>
            </div>
            {crossData ? (
              <CrossHeatmap
                rows={crossData.rows}
                cols={crossData.cols}
                scores={crossData.scores}
                perJoint={crossData.per_joint}
                home={lockHome}
              />
            ) : (
              <div className="muted small cross-empty">
                {crossBusy
                  ? `Replaying ${loadedIds.length * (locks.length + 1)} lock × dataset pairs…`
                  : "Nothing computed yet for this set of datasets and locks."}
              </div>
            )}
            {crossError && <div className="bad small">{crossError}</div>}
            <div className="row">
              <span className="muted small">
                {loadedIds.map(short).join(", ")} × {lockNames.join(", ")}, default
              </span>
              <button className="cta" onClick={runCross} disabled={crossBusy || !loadedIds.length}>
                {crossBusy ? (
                  <>
                    Scoring… <span className="dot light" />
                  </>
                ) : crossData ? (
                  "Run again"
                ) : (
                  "Run"
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      <footer className="timeline">
        <div className="controls">
          <button className="icon" onClick={() => setPlaying((p) => !p)} disabled={!frames} aria-label={playing ? "Pause" : "Play"}>
            {playing ? "❚❚" : "▶"}
          </button>
          <div className="speeds">
            {SPEEDS.map((s) => (
              <button key={s} className={s === speed ? "on" : ""} onClick={() => setSpeed(s)}>
                {s}×
              </button>
            ))}
          </div>
          <select
            className="picker"
            value={episode ?? ""}
            onChange={(e) => {
              const ep = Number(e.target.value);
              if (ep === compare) setCompare(null);
              setEpisode(ep);
            }}
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
          <span className="muted small trace-label">
            |real − sim| <i className="swatch contact" /> contact <i className="swatch anomaly" /> anomaly{" "}
            <i className="swatch grasp" /> gripper closed
          </span>
        </div>
        {baseline && (
          <>
            <JointTrace real={traceReal} sim={traceSim} frame={frame} onSeek={seek} />
            <ResidualStrip
              sum={residualSum}
              events={residual?.events ?? []}
              grasps={residual?.grasp_windows ?? []}
              frames={frames}
              frame={frame}
              onSeek={seek}
            />
          </>
        )}
      </footer>
      </>
      )}
    </div>
  );
}
