import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties, type ReactNode } from "react";
import {
  api,
  followJob,
  lockNameFor,
  type CalibrationResult,
  type Compensation,
  type CrossMatrix,
  type DatasetInfo,
  type EpisodeInfo,
  type Geometry,
  type JobEvent,
  type NumericParamKey,
  type ObjectsFit,
  type ObjectTrack,
  type Params,
  type PolicyEvalEvent,
  type PolicyInfo,
  type PolicyMode,
  type PolicyResults,
  type PolicyTrainEvent,
  type PolicyViewer,
  type Replay,
  type Residual,
  type SceneFit,
} from "./api";
import { Arm, ghostMaterials, makeGhostMaterial, policyDefaultMaterial, policyLockMaterial, type Playhead } from "./Arm";
import { CameraView, useCameras } from "./CameraView";
import { useUnderlay, VideoBackdrop } from "./VideoUnderlay";
import { SceneGeometry } from "./Scene";
import { ObjectMarker } from "./Objects";
import { AddDataset } from "./AddDataset";
import { FleetView } from "./Fleet";
import { ArmScene, type Show } from "./Viewport";
import { Timeline, episodeOptionLabel } from "./Timeline";
import {
  CalibrateSection,
  GapStat,
  JointBars,
  LIVE_ID,
  MAX_EVALUATIONS,
  PER_JOINT_EVALUATIONS,
  MANUAL_ID,
  ParamsDisclosure,
  PhysicsPicker,
  STOCK,
  TuneSection,
  applyTune,
  validLockName,
  type CalibOptions,
  type Calibration,
  type PhysicsRow,
} from "./CalibrateTab";
import { PolicyTab, POLICY_EPOCHS, type PolicyJob } from "./PolicyTab";
import { ToolsTab } from "./ToolsTab";
import { CrossModal } from "./CrossModal";
import { deg } from "./format";
import {
  defaultSplit,
  fmtWhen,
  isViewerRun,
  lockApi,
  lockKinds,
  lockLabel,
  partialOf,
  readPartialRuns,
  recordPartialRun,
  repeatOf,
  roleOf,
  splitOf,
  ROLE_HELP,
  type LockDetail,
  type LockRow,
  type PartialRun,
} from "./lockInfo";
import type { SimTone } from "./theme";
import {
  ArmKey,
  Button,
  Icon,
  IconButton,
  InlineNotice,
  LegendItem,
  SegmentedControl,
  Select,
  Spinner,
  TabPanel,
  Tabs,
  ToastHost,
  useToasts,
} from "./ui";

// Layout: the 3D viewport (real arm + sim ghost) on the left, the inspector
// (Calibrate / Camera / Policy / Tools) on the right, the episode timeline
// along the bottom. The ghost is the simulator replaying the real commands
// with the stock model, a saved lock, or (while calibrating) the best fit so far.

const DEFAULT = STOCK;
const LIVE = LIVE_ID;
// Parameters set by hand in the Tune by hand section.
const MANUAL = MANUAL_ID;

const noop = () => {};

type Tab = "calibrate" | "camera" | "policy" | "tools";
const TABS: Tab[] = ["calibrate", "camera", "policy", "tools"];
const TAB_KEY = "antares.inspector-tab";

function readTab(): Tab {
  try {
    const t = window.localStorage.getItem(TAB_KEY) as Tab | null;
    return t && TABS.includes(t) ? t : "calibrate";
  } catch {
    return "calibrate";
  }
}

// How long a deleted lock can be restored before the DELETE is sent.
const UNDO_MS = 6000;

// The selection that survives a reload, carried in the query string so the
// address is also a link to this view: `?dataset=…&episode=…&ghost=…`.
// The live fit and hand-tuned parameters are never restored (they live only in this page).
type Restore = { dataset: string; episode: number | null; ghost: string | null };

function readRestore(): Restore {
  try {
    const q = new URLSearchParams(window.location.search);
    const ep = q.get("episode");
    const n = ep == null ? NaN : Number(ep);
    const ghost = q.get("ghost");
    return {
      dataset: q.get("dataset") ?? "",
      episode: Number.isInteger(n) && n >= 0 ? n : null,
      ghost: ghost && ghost !== LIVE && ghost !== MANUAL ? ghost : null,
    };
  } catch {
    return { dataset: "", episode: null, ghost: null };
  }
}

function writeRestore(dataset: string, episode: number | null, ghost: string) {
  try {
    const q = new URLSearchParams(window.location.search);
    if (dataset) q.set("dataset", dataset);
    else q.delete("dataset");
    if (episode != null) q.set("episode", String(episode));
    else q.delete("episode");
    if (ghost !== DEFAULT && ghost !== LIVE && ghost !== MANUAL) q.set("ghost", ghost);
    else q.delete("ghost");
    const search = q.toString();
    const url = window.location.pathname + (search ? "?" + search : "") + window.location.hash;
    if (url !== window.location.pathname + window.location.search + window.location.hash) {
      window.history.replaceState(null, "", url);
    }
  } catch {
    // per-viewer convenience only
  }
}

export default function App() {
  // Top-level screen: the single-arm viewer, or the fleet dashboard. The
  // hash deep-links to the fleet (`#fleet`, or `#fleet/<lock>` with a row open).
  const [view, setViewState] = useState<"viewer" | "fleet">(() => (window.location.hash.startsWith("#fleet") ? "fleet" : "viewer"));
  const setView = (v: "viewer" | "fleet") => {
    setViewState(v);
    window.history.replaceState(null, "", v === "fleet" ? "#fleet" : window.location.pathname + window.location.search);
  };
  const [tab, setTabState] = useState<Tab>(readTab);
  const inspectorBody = useRef<HTMLDivElement | null>(null);
  const setTab = (t: Tab) => {
    setTabState(t);
    // Each tab opens at its top (the panels share one scroller).
    if (inspectorBody.current) inspectorBody.current.scrollTop = 0;
    try {
      window.localStorage.setItem(TAB_KEY, t);
    } catch {
      // per-viewer convenience only
    }
  };
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  // Episode and ghost to restore once the dataset named in the URL has loaded.
  const restore = useRef<Restore>(readRestore());
  const [dataset, setDataset] = useState<string>(() => restore.current.dataset);
  const [dsState, setDsState] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const [loadSeconds, setLoadSeconds] = useState(0);
  const [episodes, setEpisodes] = useState<EpisodeInfo[]>([]);
  const [episode, setEpisode] = useState<number | null>(null);
  const [geometry, setGeometry] = useState<Geometry | null>(null);
  const [locks, setLocks] = useState<LockRow[]>([]);
  const locksRef = useRef<LockRow[]>([]);
  locksRef.current = locks;
  const [ghost, setGhost] = useState<string>(DEFAULT);
  const [baseline, setBaseline] = useState<Replay | null>(null);
  const [replay, setReplay] = useState<Replay | null>(null);
  const [replayBusy, setReplayBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [show, setShow] = useState<Show>("both");
  // Hand tuning: the parameters it started from, what it started from, and
  // each slider's setting (log10 multiplier, or frames for latency).
  const [tuneBase, setTuneBase] = useState<Params | null>(null);
  const [tuneFrom, setTuneFrom] = useState("");
  // The saved lock the tuning started from (its command map and unit map
  // stay with the tuned parameters), or null for the stock model / live fit.
  const [tuneLock, setTuneLock] = useState<string | null>(null);
  // Saving the tuned values as a lock: its name, and the request's state.
  const [saveName, setSaveName] = useState("");
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [tuneAdj, setTuneAdj] = useState<Partial<Record<NumericParamKey, number>>>({});
  // A second episode shown next to the first; null when off.
  const [compare, setCompare] = useState<number | null>(null);
  const [compareBase, setCompareBase] = useState<Replay | null>(null);
  const [compareReplay, setCompareReplay] = useState<Replay | null>(null);
  const [frame, setFrame] = useState(0);
  const [joint, setJoint] = useState("shoulder_pan");
  const [calib, setCalib] = useState<Calibration | null>(null);
  const [opts, setOpts] = useState<CalibOptions>({ perJoint: false, fitGravity: false, fitDeadband: false, factors: null });
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

  // What the inspector knows about each lock beyond /api/locks: its split,
  // kind and stages (GET /api/locks/{name}), and its camera fits.
  const [details, setDetails] = useState<Record<string, LockDetail>>({});
  const [camFits, setCamFits] = useState<Record<string, string[]>>({});
  // Runs stopped from this browser, for the "partial" badge.
  const [stoppedRuns, setStoppedRuns] = useState<Record<string, PartialRun>>(readPartialRuns);
  const [showAllLocks, setShowAllLocks] = useState(false);
  // Locks deleted in the UI whose DELETE waits out the undo window.
  const [pendingDeletes, setPendingDeletes] = useState<Set<string>>(new Set());
  const { toasts, push: pushToast, remove: removeToast } = useToasts();

  const playhead = useMemo<Playhead>(() => ({ frame: 0 }), []);
  const comparePlayhead = useMemo<Playhead>(() => ({ frame: 0 }), []);
  const compareGhostMats = useMemo(() => ({ stock: makeGhostMaterial("stock"), sim: makeGhostMaterial("sim") }), []);

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

  // A dataset added from the "Add dataset" popover: list it and select it;
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
    // Hand-tuned parameters belong to the dataset they were tuned on.
    setTuneBase(null);
    api
      .loadDataset(dataset)
      .then((info) => {
        if (!live) return;
        setDatasets((ds) => (ds.some((d) => d.id === info.id) ? ds.map((d) => (d.id === info.id ? info : d)) : [...ds, info]));
        return api.episodes(dataset);
      })
      .then((eps) => {
        if (!live || !eps) return;
        setEpisodes(eps);
        // A reload lands on the episode and ghost it left, once.
        const r = restore.current;
        const wanted = r.dataset === dataset && r.episode != null && eps.some((e) => e.index === r.episode) ? r.episode : null;
        setEpisode(wanted ?? eps[0]?.index ?? null);
        if (r.dataset === dataset && r.ghost && (r.ghost === DEFAULT || locksRef.current.length === 0 || locksRef.current.some((l) => l.name === r.ghost))) setGhost(r.ghost);
        restore.current = { dataset: "", episode: null, ghost: null };
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

  // The address follows the selection, so a reload (or a pasted link) returns here.
  useEffect(() => {
    writeRestore(dataset, episode, ghost);
  }, [dataset, episode, ghost]);

  // ---- baseline replay (real arm + stock-model ghost) --------------------
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

  // ---- second episode, compared ---------------------------------------------
  // Its ghost follows the first pane's choice: stock, the same lock, or the
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
  // own episode; with the stock ghost it is simply the base replay, read at
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
      // A request dropped mid-flight (another ghost picked) must not leave the spinner on.
      setReplayBusy(false);
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
      setCompBusy(false);
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
        // Cleared whatever the ghost is now, so the busy spinner never sticks.
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

  // ---- calibration -----------------------------------------------------------
  const sourceRef = useRef<EventSource | null>(null);
  useEffect(() => () => sourceRef.current?.close(), []);
  // The job the user asked to stop; its saved lock is partial.
  const stoppedJob = useRef<string | null>(null);

  // Set while the start request is in flight, so a double click cannot start
  // two jobs under the same (to-the-second) lock name.
  const starting = useRef(false);

  const startCalibration = async () => {
    if (!dataset || starting.current) return;
    starting.current = true;
    setError(null);
    const name = lockNameFor(dataset);
    const ds = dataset;
    let jobId: string;
    const perJoint = opts.perJoint;
    const total = MAX_EVALUATIONS + (perJoint ? PER_JOINT_EVALUATIONS : 0);
    try {
      ({ job_id: jobId } = await api.calibrate(
        dataset,
        name,
        MAX_EVALUATIONS,
        perJoint ? { maxEvaluations: PER_JOINT_EVALUATIONS } : undefined,
        opts.factors,
        { fitGravity: opts.fitGravity, fitDeadband: opts.fitDeadband },
      ));
    } catch (e) {
      setError(`Could not start calibration: ${(e as Error).message}`);
      return;
    } finally {
      starting.current = false;
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
    // The run saved its lock (from the stream's "done", or the job's status
    // after the stream was lost): mark it partial if it was stopped, say so,
    // and select it once the lock list has reloaded.
    const finishWithLock = (lockName: string, result: CalibrationResult) => {
      if (stoppedJob.current === jobId && result.n_evaluations < total) {
        setStoppedRuns(recordPartialRun(lockName, { n: result.n_evaluations, total }));
      }
      pushToast({
        tone: "good",
        message: (
          <>
            Saved <b>{lockLabel(lockName, ds)}</b> · held-out {deg(result.eval_score)}
          </>
        ),
        duration: 5000,
      });
      api.locks().then(
        (ls) => {
          setLocks(ls);
          setGhost(lockName);
        },
        (e) => setError(`Saved ${lockName}, but could not reload the lock list: ${(e as Error).message}`),
      );
    };
    followJob<JobEvent>(src, (ev) => {
      if (ev.type === "progress") {
        setCalib((c) => (c && c.jobId === jobId ? { ...c, progress: [...c.progress, ev], best: [...c.best, ev.best_score ?? NaN] } : c));
        if (ev.best_score != null && ev.best_score < lastBest - 1e-9) {
          lastBest = ev.best_score;
          liveWanted.current = ev.best_params;
          pumpLive();
        }
      } else if (ev.type === "done") {
        src.close();
        setCalib((c) => (c && c.jobId === jobId ? { ...c, status: "done", result: ev.result } : c));
        finishWithLock(ev.lock.name, ev.result);
      } else if (ev.type === "error") {
        src.close();
        setCalib((c) => (c && c.jobId === jobId ? { ...c, status: "error", error: ev.message } : c));
      }
    });
    src.onerror = () => {
      // EventSource reconnects on its own; a closed stream after "done" is fine.
      // A stream lost while the job runs (or after Stop, while the worker
      // still finishes the generation and saves the best fit) is not the
      // job's fate: ask the server where it stands and finish from that.
      if (src.readyState !== EventSource.CLOSED) return;
      api.jobStatus(jobId).then(
        (st) => {
          if (st.status === "done") {
            const result = st.result as CalibrationResult;
            setCalib((c) => (c && c.jobId === jobId ? { ...c, status: "done", result } : c));
            if (st.lock) finishWithLock(st.lock.name, result);
          } else if (st.status === "error") {
            setCalib((c) => (c && c.jobId === jobId ? { ...c, status: "error", error: st.error ?? "the fit failed" } : c));
          } else {
            setCalib((c) =>
              c && c.jobId === jobId && (c.status === "running" || c.status === "cancelled")
                ? { ...c, status: "error", error: "lost the progress stream while the fit is still running; reload to see its lock" }
                : c,
            );
          }
        },
        () =>
          setCalib((c) =>
            c && c.jobId === jobId && (c.status === "running" || c.status === "cancelled")
              ? { ...c, status: "error", error: "lost the progress stream" }
              : c,
          ),
      );
    };
  };

  const stopCalibration = async () => {
    if (!calib) return;
    stoppedJob.current = calib.jobId;
    try {
      await api.cancel(calib.jobId);
    } catch (e) {
      // The job keeps running and Stop stays enabled, so the user can retry.
      stoppedJob.current = null;
      setError(`Could not stop the calibration: ${(e as Error).message}`);
      return;
    }
    setCalib((c) => (c ? { ...c, status: c.status === "running" ? "cancelled" : c.status } : c));
    // The worker still finishes its current generation and saves the best fit;
    // the stream's "done" event then selects that lock.
  };

  // ---- locks the UI shows (minus those waiting out their undo) ---------------
  const visibleLocks = useMemo(() => locks.filter((l) => !pendingDeletes.has(l.name)), [locks, pendingDeletes]);
  const dsLocks = useMemo(() => visibleLocks.filter((l) => l.dataset === dataset), [visibleLocks, dataset]);
  const dsLockKey = dsLocks.map((l) => l.name).join(",");

  // Each lock's file, for its split, kind and stages; fetched once per name.
  const detailRequested = useRef(new Set<string>());
  useEffect(() => {
    for (const l of dsLocks) {
      if (detailRequested.current.has(l.name)) continue;
      detailRequested.current.add(l.name);
      lockApi.get(l.name).then(
        (d) => setDetails((x) => ({ ...x, [l.name]: d })),
        () => detailRequested.current.delete(l.name),
      );
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dsLockKey]);

  // ---- cross matrix: every lock on every loaded dataset ----------------------
  const loadedIds = datasets.filter((d) => d.loaded).map((d) => d.id);
  const lockNames = visibleLocks.map((l) => l.name);
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
  const closeCross = useCallback(() => setCrossOpen(false), []);

  // ---- derived ---------------------------------------------------------------
  const lockHome = Object.fromEntries(visibleLocks.map((l) => [l.name, l.dataset]));
  const running = calib?.status === "running" || calib?.status === "cancelled";
  const runningRef = useRef(running);
  runningRef.current = running;
  const fps = baseline?.fps ?? 30;
  const frames = baseline?.real_poses.length ?? 0;
  const ghostGap = replay?.gap ?? null;
  const baseGap = baseline?.gap ?? null;
  const jointNames = baseline?.joints ?? [];
  const scored = useMemo(() => new Set(baseGap?.scored_joints ?? []), [baseGap]);
  const jIndex = Math.max(0, jointNames.indexOf(joint));
  const lastProgress = calib?.progress[calib.progress.length - 1];
  const shownParams = replay?.params ?? null;
  const defaultParams = baseline?.params ?? null;
  const clip = `${dataset}#${episode ?? ""}`;
  const dsWarnings = currentDs?.warnings ?? [];
  const isLock = ghost !== DEFAULT && ghost !== LIVE && ghost !== MANUAL;
  const tone: SimTone = ghost === DEFAULT ? "stock" : "sim";
  const physicsLabel =
    ghost === DEFAULT ? "stock model" : ghost === LIVE ? "live fit" : ghost === MANUAL ? "your parameters" : lockLabel(ghost, dataset);

  // The selected physics' train / held-out split (a lock's own; the default
  // split for the stock model and the live fit, which uses it too).
  const dfltSplit = useMemo(() => defaultSplit(dsLocks, details), [dsLocks, details]);
  const split = isLock ? splitOf(details[ghost]) : dfltSplit;
  const roleFor = (ep: number | null) => roleOf(split, ep);
  const role = roleFor(episode);
  // The stock model's held-out error, from any lock scored on the default split.
  const stockHeldOut = useMemo(() => {
    if (!dfltSplit) return null;
    const key = JSON.stringify(dfltSplit);
    const l = dsLocks.find((x) => JSON.stringify(splitOf(details[x.name])) === key && x.scores.baseline_eval_score != null);
    return l?.scores.baseline_eval_score ?? null;
  }, [dsLocks, details, dfltSplit]);

  const physicsRows: PhysicsRow[] = useMemo(
    () =>
      dsLocks.map((l) => {
        const rep = repeatOf(l, dsLocks);
        return {
          name: l.name,
          label: lockLabel(l.name, l.dataset),
          tooltip: `${l.name}\ncreated ${new Date(l.created_at).toLocaleString()}`,
          kinds: lockKinds(l, details[l.name]),
          when: fmtWhen(l.created_at, isViewerRun(l.name)),
          heldOut: l.scores.eval_score,
          partial: partialOf(l, details[l.name], stoppedRuns),
          repeatOf: rep ? { name: rep.name, label: lockLabel(rep.name, rep.dataset) } : null,
          exportUrl: api.exportZipUrl(l.name),
        };
      }),
    [dsLocks, details, stoppedRuns],
  );

  // The trace arrays only change with the replay or the joint, not per frame.
  const traceReal = useMemo(() => baseline?.real_q.map((r) => r[jIndex]) ?? [], [baseline, jIndex]);
  const traceSim = useMemo(() => (replay ?? baseline)?.sim_q.map((r) => r[jIndex]) ?? [], [replay, baseline, jIndex]);
  // Summed |residual| over the scored joints (the gripper is not simulated).
  const residualSum = useMemo(() => {
    if (!residual) return [];
    const cols = residual.scored_joints.map((j) => residual.joints.indexOf(j)).filter((i) => i >= 0);
    return residual.residual.map((row) => cols.reduce((s, i) => s + Math.abs(row[i]), 0));
  }, [residual]);
  // [start, end, kind]: the ghost glows critical on contact, warn on an anomaly.
  const hotSpans = useMemo(() => residual?.events.map((e) => [e.start, e.end, e.kind === "anomaly" ? 1 : 0]) ?? null, [residual]);
  const inEvent = residual?.events.find((e) => frame >= e.start && frame < e.end);
  const compPoses = showComp && ghost !== LIVE && ghost !== MANUAL && comp ? comp.sim_comp_poses : null;

  // Camera fits live with a lock: the ghost's, or the dataset's newest one.
  const cameraLock = isLock ? ghost : dsLocks[0]?.name ?? null;
  const cameraLockLabel = cameraLock ? lockLabel(cameraLock, dataset) : "";

  // ---- video underlay ----------------------------------------------------------
  // The dataset's video behind the main pane, pinned to the lock's fitted
  // camera when it has one. The cameras are fetched once and shared with the
  // Camera tab, and a fit saved in either place reaches the other.
  const [videoOn, setVideoOn] = useState(false);
  const [videoError, setVideoError] = useState<string | null>(null);
  const [fitsRev, setFitsRev] = useState(0);
  const cameras = useCameras(dataset);
  const video = useUnderlay(dataset, cameras, episode, cameraLock, videoOn, fitsRev, running);
  const underlay = video.underlay;
  const videoAvail = cameras.length > 0;

  // ---- hand tuning -------------------------------------------------------------
  const startTuning = () => {
    const fromLock = isLock ? ghost : null;
    setTuneBase(replay?.params ?? baseline?.params ?? null);
    setTuneFrom(physicsLabel === "stock model" || physicsLabel === "live fit" ? `the ${physicsLabel}` : physicsLabel);
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
      pushToast({
        tone: "good",
        message: (
          <>
            Saved <b>{lockLabel(saved.name, dataset)}</b> · held-out {deg(saved.scores.eval_score)}
          </>
        ),
        duration: 5000,
      });
    } catch (e) {
      setSaveError((e as Error).message);
    } finally {
      setSaving(false);
    }
  };

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

  // ---- camera fits of the dataset's other locks (for "switch to <lock>") -----
  const camRequested = useRef(new Set<string>());
  useEffect(() => {
    if (tab !== "camera") return;
    for (const l of dsLocks) {
      if (camRequested.current.has(l.name)) continue;
      camRequested.current.add(l.name);
      api.cameraFits(l.name).then(
        (r) => setCamFits((x) => ({ ...x, [l.name]: Object.keys(r.cameras) })),
        () => camRequested.current.delete(l.name),
      );
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tab, dsLockKey]);
  const onFitsChange = useCallback((name: string, cams: string[]) => setCamFits((x) => ({ ...x, [name]: cams })), []);
  const altCamLock = dsLocks.find((l) => l.name !== cameraLock && (camFits[l.name]?.length ?? 0) > 0) ?? null;

  // ---- policy: saved policies and evaluations of the camera lock -------------
  const policyLock = cameraLock;
  const policy = policies[0] ?? null;
  const refreshPolicy = useCallback((lock: string) => {
    api.policies(lock).then(
      (r) => setPolicies(r.policies),
      () => setPolicies([]),
    );
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
          setPolViewerNote(e.status === 400 ? `Episode ${episode} is not held out from the policy's training.` : e.message);
        },
      )
      .finally(() => live && setPolViewerBusy(false));
    return () => {
      live = false;
      setPolViewerBusy(false);
    };
  }, [showPolicy, policyLock, policy, dataset, episode, polMode, polBusy]);

  const policySource = useRef<EventSource | null>(null);
  useEffect(() => () => policySource.current?.close(), []);

  const followPolicyJob = (jobId: string, kind: PolicyJob["kind"], mode?: PolicyMode) => {
    setPolJob({ jobId, kind, mode, status: "running", progress: [], i: 0, n: 0 });
    policySource.current?.close();
    const src = api.jobEvents(jobId);
    policySource.current = src;
    followJob<PolicyTrainEvent | PolicyEvalEvent>(src, (ev) => {
      if (ev.type === "progress") {
        setPolJob((j) => (j && j.jobId === jobId ? ("epoch" in ev ? { ...j, progress: [...j.progress, ev] } : { ...j, i: ev.i, n: ev.n }) : j));
      } else if (ev.type === "done") {
        src.close();
        setPolJob((j) => (j && j.jobId === jobId ? { ...j, status: "done" } : j));
        if (policyLock) refreshPolicy(policyLock);
        if (kind === "evaluate") setShowPolicy(true);
      } else if (ev.type === "error") {
        src.close();
        setPolJob((j) => (j && j.jobId === jobId ? { ...j, status: "error", error: ev.message } : j));
      }
    });
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

  const polPoses = (name: string) => (showPolicy && polViewer && polViewer.episode === episode && !polBusy ? polViewer.sims[name]?.poses ?? null : null);
  // While the policy's ghosts are drawn, the replay ghost hides: two blue
  // arms (the replay and the policy in the calibrated sim) would be ambiguous.
  const policyGhostsShown = !!(polPoses("default") || polPoses("lock"));

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

  // ---- delete with undo ---------------------------------------------------------
  // DELETE /api/locks/{name} unlinks the file at once and has no restore, so
  // the row hides now and the request goes out only when the undo window
  // passes. Closing the tab inside the window keeps the lock.
  const datasetRef = useRef(dataset);
  datasetRef.current = dataset;
  const unpend = (name: string) =>
    setPendingDeletes((p) => {
      const n = new Set(p);
      n.delete(name);
      return n;
    });
  const deleteLock = (name: string) => {
    const wasSelected = ghost === name;
    const ds = dataset;
    const label = lockLabel(name, ds);
    setPendingDeletes((p) => new Set(p).add(name));
    if (wasSelected) setGhost(DEFAULT);
    pushToast({
      duration: UNDO_MS,
      message: (
        <>
          Deleted <b>{label}</b>
        </>
      ),
      action: {
        label: "Undo",
        onClick: () => {
          unpend(name);
          if (wasSelected && datasetRef.current === ds && !runningRef.current) setGhost((g) => (g === DEFAULT ? name : g));
        },
      },
      onExpire: () => {
        lockApi.remove(name).then(
          () =>
            api.locks().then(
              (ls) => {
                setLocks(ls);
                unpend(name);
              },
              () => {
                setLocks((ls) => ls.filter((l) => l.name !== name));
                unpend(name);
              },
            ),
          (e) => {
            unpend(name);
            setError(`Could not delete ${name}: ${(e as Error).message}. It is still saved.`);
          },
        );
      },
    });
  };

  // ---- playback ----------------------------------------------------------------------
  const seek = (f: number) => {
    playhead.frame = f;
    if (compareFrames) comparePlayhead.frame = f % compareFrames;
    setFrame(Math.floor(f));
  };

  const selectEpisode = (ep: number) => {
    if (ep === compare) setCompare(null);
    setEpisode(ep);
  };

  // Keyboard: Space play/pause, ←/→ one frame, Shift+←/→ one second, [ ] episode.
  const keys = useRef({ toggle: noop, step: (_: number) => {}, episodeBy: (_: number) => {} });
  keys.current = {
    toggle: () => frames && setPlaying((p) => !p),
    step: (df: number) => {
      if (!frames) return;
      setPlaying(false);
      seek(Math.min(frames - 1, Math.max(0, Math.round(playhead.frame) + df)));
    },
    episodeBy: (d: number) => {
      const i = episodes.findIndex((e) => e.index === episode);
      const next = episodes[i + d];
      if (i >= 0 && next) selectEpisode(next.index);
    },
  };
  useEffect(() => {
    if (view !== "viewer") return;
    // Whether the focused element got focus from a click. `:focus-visible` can't
    // tell: Chrome turns it on as soon as a key is pressed on the element.
    let pointerDown = false;
    let clickFocused = false;
    const onPointer = () => {
      pointerDown = true;
    };
    const onFocus = () => {
      clickFocused = pointerDown;
      pointerDown = false;
    };
    const onKey = (e: KeyboardEvent) => {
      pointerDown = false; // a click that focused nothing must not mark the next Tab focus as a click
      if (e.defaultPrevented || e.metaKey || e.ctrlKey || e.altKey) return;
      const t = e.target as HTMLElement | null;
      if (t?.closest?.("input, select, textarea, [contenteditable='true'], [contenteditable=''], [role=menu], [role=dialog]")) return;
      // Keyboard users moving through the tabs keep their keys; a tab focused by a click does not.
      if (t?.closest?.("[role=tablist]") && !clickFocused) return;
      if (e.key === " ") {
        // A button focused from the keyboard keeps Space for itself.
        if (t?.closest?.("button, a, summary, [role=button]")) {
          if (!clickFocused) return;
          t.blur();
        }
        e.preventDefault();
        keys.current.toggle();
      } else if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
        e.preventDefault();
        keys.current.step((e.key === "ArrowRight" ? 1 : -1) * (e.shiftKey ? Math.round(fps) : 1));
      } else if (e.key === "[") {
        keys.current.episodeBy(-1);
      } else if (e.key === "]") {
        keys.current.episodeBy(1);
      }
    };
    window.addEventListener("pointerdown", onPointer, true);
    window.addEventListener("focusin", onFocus);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("pointerdown", onPointer, true);
      window.removeEventListener("focusin", onFocus);
      window.removeEventListener("keydown", onKey);
    };
  }, [view, fps]);

  // ---- render ---------------------------------------------------------------------------
  const ghostMat = ghostMaterials[tone];
  const calibBlocker = dsState === "loading" ? "Waiting for the dataset to load." : !baseline ? "Waiting for the episode to load." : null;
  const savedLabel = calib?.status === "done" ? lockLabel(calib.name, dataset) : null;
  const savedPartial = calib?.status === "done" ? stoppedRuns[calib.name] ?? null : null;
  const liveMeta = running ? (
    <span className="muted">
      eval {lastProgress?.evaluation ?? 0} / {calib?.total}
    </span>
  ) : calib?.status === "error" ? (
    <span className="muted">run failed; replaying its best</span>
  ) : (
    <span className="muted">best fit of the last run</span>
  );

  const paneLabel = (ep: number | null, gap: number | null | undefined) => {
    const r = roleFor(ep);
    return (
      <>
        Episode {ep ?? "--"}
        {r && (
          <>
            {" · "}
            <span title={ROLE_HELP[r]}>{r}</span>
          </>
        )}
        {" · "}
        <span className="num">{deg(gap)}</span>
      </>
    );
  };

  // What the video underlay is doing, for the legend.
  const lineUp = video.lineUp;
  const lineUpStage = (l: NonNullable<typeof lineUp>) =>
    l.status === "done"
      ? " (saved)"
      : l.last
        ? ` (${
            l.last.stage === "start"
              ? `try ${l.last.start + 1} of ${l.last.n_starts}`
              : l.last.stage === "sample"
                ? "first guesses"
                : l.last.stage === "refine"
                  ? "refining"
                  : "checking"
          }, overlap ${(l.last.best_iou * 100).toFixed(0)}%)`
        : "…";
  const videoNote: ReactNode = !underlay
    ? null
    : videoError
      ? videoError
      : underlay.fit
        ? `Video from the ${underlay.camera.name} camera, view pinned to its fitted pose${underlay.fit.lock && underlay.fit.lock !== cameraLock ? ` (fitted with ${lockLabel(underlay.fit.lock, dataset)})` : ""}, held-out overlap ${(underlay.fit.heldout_iou * 100).toFixed(0)}%`
        : !underlay.camera.fixed
          ? `Video from the ${underlay.camera.name} camera as a plain backdrop: it moves with the arm, so the 3D view cannot be lined up with it`
          : lineUp?.status === "running" || lineUp?.status === "done"
            ? (
              <>
                <Spinner /> Lining the 3D view up with the {underlay.camera.name} camera
                {lineUp.sibling && lineUp.status !== "done" ? `, second pass with the ${lineUp.sibling} camera's field of view` : ""}
                {lineUpStage(lineUp)}{" "}
                {lineUp.status !== "done" && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={video.stopLineUp}
                    disabled={!lineUp.jobId}
                    title={lineUp.jobId ? "Stop lining the view up; the video stays as a plain backdrop" : "Waiting for the fit to start"}
                  >
                    Stop
                  </Button>
                )}
              </>
            )
            : lineUp?.status === "error"
              ? `Could not line the 3D view up with the ${underlay.camera.name} camera: ${lineUp.error}`
              : !cameraLock
                ? `Video from the ${underlay.camera.name} camera as a plain backdrop: calibrate first, then the view lines up with the video`
                : running
                  ? `Video from the ${underlay.camera.name} camera as a plain backdrop: the view lines up with it once calibration finishes`
                  : `Video from the ${underlay.camera.name} camera as a plain backdrop: no fitted camera yet, fit it in the Camera tab to line the arm up`;

  return (
    <div className={`shell${view === "fleet" ? " fleet-shell" : ""}`}>
      <header className="topbar">
        <span className="brand" title="antares: sim, calibrated to this robot">
          antares
        </span>
        <nav className="view-tabs" aria-label="Views">
          <a
            href="#"
            className={view === "viewer" ? "on" : ""}
            aria-current={view === "viewer" ? "page" : undefined}
            onClick={(e) => {
              e.preventDefault();
              setView("viewer");
            }}
          >
            Viewer
          </a>
          <a
            href="#fleet"
            className={view === "fleet" ? "on" : ""}
            aria-current={view === "fleet" ? "page" : undefined}
            onClick={(e) => {
              e.preventDefault();
              setView("fleet");
            }}
          >
            Fleet
          </a>
        </nav>
        {view === "fleet" ? null : (
          <div className="topbar-right">
            {dsState === "loading" && (
              <span className="topbar-note">
                <Spinner /> Loading dataset…
              </span>
            )}
            {running && <span className="topbar-note">Dataset locked while calibrating</span>}
            {currentDs && (
              <span
                className={`ds-chip ${dsWarnings.length ? "tone-warn" : ""}`}
                title={[
                  `${currentDs.source} dataset${currentDs.robot_family ? ` · ${currentDs.robot_family} arm` : ""} · sim model ${currentDs.robot}`,
                  ...dsWarnings,
                ].join("\n")}
              >
                {dsWarnings.length > 0 && <Icon.warn />}
                {currentDs.source}
                {currentDs.robot_family && currentDs.robot_family !== currentDs.robot && ` · ${currentDs.robot_family}→${currentDs.robot}`}
                {dsWarnings.length > 0 && ` · ${dsWarnings.length} warning${dsWarnings.length === 1 ? "" : "s"}`}
              </span>
            )}
            <Select
              label="Dataset"
              hideLabel
              className="dataset-select"
              value={dataset}
              disabled={!datasets.length || running}
              title={running ? "Stop the calibration to switch datasets" : "Dataset"}
              onChange={(e) => setDataset(e.target.value)}
            >
              {datasets.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.id}
                </option>
              ))}
            </Select>
            <AddDataset disabled={running} disabledReason="Stop the calibration to add a dataset" onAdded={addDataset} />
          </div>
        )}
      </header>

      {view === "fleet" ? (
        <FleetView initialOpen={decodeURIComponent(window.location.hash.replace(/^#fleet\/?/, "")) || null} />
      ) : (
        <>
          <main className={`viewer${compare != null ? " comparing" : ""}`} aria-label="3D viewer">
            <div className="panes">
              <div className="pane pane-main">
                <div
                  className={underlay ? "underlay-frame" : "pane-fill"}
                  style={underlay ? ({ "--aspect": underlay.camera.width / Math.max(1, underlay.camera.height) } as CSSProperties) : undefined}
                >
                  {underlay && <VideoBackdrop underlay={underlay} playhead={playhead} playing={playing} speed={speed} onError={setVideoError} />}
                  <ArmScene
                    geometry={geometry}
                    realPoses={baseline?.real_poses ?? null}
                    simPoses={replay?.sim_poses ?? null}
                    compPoses={compPoses}
                    hotSpans={hotSpans}
                    clip={clip}
                    show={show}
                    hideGhost={policyGhostsShown}
                    playhead={playhead}
                    playing={playing}
                    fps={fps}
                    frames={frames}
                    speed={speed}
                    onFrame={setFrame}
                    ghostMat={ghostMat}
                    underlay={underlay}
                    label={`Episode ${episode ?? ""}: the real arm and the sim on the ${physicsLabel}`}
                  >
                    {geometry && (
                      <>
                        <Arm geometry={geometry} poses={polPoses("default")} clip={`${clip}#policy`} playhead={playhead} material={policyDefaultMaterial} />
                        <Arm geometry={geometry} poses={polPoses("lock")} clip={`${clip}#policy`} playhead={playhead} material={policyLockMaterial} />
                      </>
                    )}
                    {/* The fitted table plane is opaque and would hide the footage; the video shows the real table. */}
                    {!underlay && <SceneGeometry scene={scene} />}
                    <ObjectMarker track={track} playhead={playhead} />
                  </ArmScene>
                </div>
                {compare != null && <div className="pane-label">{paneLabel(episode, ghostGap?.overall_deg)}</div>}
              </div>
              {compare != null && (
                <div className="pane">
                  <ArmScene
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
                    ghostMat={compareGhostMats[tone]}
                    label={`Episode ${compare}, compared: the real arm and the sim on the ${physicsLabel}`}
                  />
                  <div className="pane-label">
                    <Select label="Compared episode" hideLabel value={compare} onChange={(e) => setCompare(Number(e.target.value))}>
                      {episodes
                        .filter((e) => e.index !== episode)
                        .map((e) => (
                          <option key={e.index} value={e.index}>
                            {episodeOptionLabel(e, roleFor(e.index))}
                          </option>
                        ))}
                    </Select>
                    <span className="num">{deg(shownCompare?.gap?.overall_deg)}</span>
                    {compareBusy && <Spinner label="Replaying" />}
                  </div>
                </div>
              )}
            </div>

            {inEvent && show !== "real" && (
              <div
                className={`status-pill tone-${inEvent.kind === "contact" ? "critical" : "warn"}`}
                title={`Sim disagrees with the arm here: ${inEvent.kind} on ${inEvent.joints.join(", ")}, peak ${inEvent.peak_deg.toFixed(1)}°`}
              >
                {inEvent.kind === "contact" ? <Icon.warn /> : <Icon.info />}
                <span>
                  Sim disagrees: {inEvent.kind} on{" "}
                  {inEvent.joints.length > 2 ? `${inEvent.joints.slice(0, 2).join(", ")} +${inEvent.joints.length - 2}` : inEvent.joints.join(", ")} · peak{" "}
                  <span className="num">{inEvent.peak_deg.toFixed(1)}°</span>
                </span>
              </div>
            )}

            <div className="view-toolbar">
              <SegmentedControl<Show>
                label="Show"
                value={show}
                onChange={setShow}
                options={[
                  { value: "both", label: "Both", title: "The real arm and the sim" },
                  { value: "real", label: "Real", title: "Only the real arm (as logged)" },
                  { value: "sim", label: "Sim", title: "Only the sim" },
                ]}
              />
              <Button
                aria-pressed={compare != null}
                className={compare != null ? "on" : ""}
                onClick={() => setCompare(compare != null ? null : episodes.find((e) => e.index !== episode)?.index ?? null)}
                disabled={episodes.length < 2}
                title={episodes.length < 2 ? "Needs a dataset with two or more episodes" : "Play a second episode next to this one, in step"}
                icon={compare != null ? <Icon.check /> : undefined}
              >
                Compare episode
              </Button>
              <Button
                aria-pressed={videoOn && videoAvail}
                className={videoOn && videoAvail ? "on" : ""}
                onClick={() => setVideoOn((v) => !v)}
                disabled={!videoAvail}
                title={videoAvail ? "Draw the dataset's camera video behind the arms" : "This dataset has no camera video"}
                icon={videoOn && videoAvail ? <Icon.check /> : undefined}
              >
                Video
              </Button>
              {videoOn && videoAvail && cameras.length > 1 && (
                <Select label="Video camera" hideLabel value={video.camera} onChange={(e) => video.setCamera(e.target.value)}>
                  {cameras.map((c) => (
                    <option key={c.name} value={c.name}>
                      {c.name}
                    </option>
                  ))}
                </Select>
              )}
              {videoOn && videoAvail && (
                <span className="video-delay" role="group" aria-label="Video delay" title="Frames the video lags the joint log (antares camera-delay measures it)">
                  <IconButton label="Video one frame earlier" onClick={() => video.setDelay((d) => d - 1)}>
                    −
                  </IconButton>
                  <span className="num">delay {video.delay}</span>
                  <IconButton label="Video one frame later" onClick={() => video.setDelay((d) => d + 1)}>
                    +
                  </IconButton>
                </span>
              )}
            </div>

            <div className="viewport-foot">
              {error && (
                <InlineNotice tone="error" onDismiss={() => setError(null)} className="viewport-error">
                  {error}
                </InlineNotice>
              )}
              <div className="legend" aria-label="Legend">
                {show !== "sim" && <LegendItem k={<ArmKey tone="real" />}>real arm (logged)</LegendItem>}
                {show !== "real" && !policyGhostsShown && (
                  <LegendItem k={<ArmKey tone={tone} translucent />}>
                    sim · {physicsLabel}
                    {ghost === LIVE && " (best so far)"} {replayBusy && <Spinner />}
                  </LegendItem>
                )}
                {show !== "real" && policyGhostsShown && <span className="legend-note">sim replay hidden while the policy ghosts show</span>}
                {showComp && ghost !== LIVE && ghost !== MANUAL && show !== "real" && (
                  <LegendItem k={<ArmKey tone="comp" translucent />}>
                    sim, lag-compensated (sim-validated) {compBusy && <Spinner />}
                  </LegendItem>
                )}
                {showPolicy && policy && (
                  <>
                    <LegendItem k={<ArmKey tone="stock" translucent />}>
                      policy in the stock model {polViewerBusy && <Spinner />}
                    </LegendItem>
                    <LegendItem k={<ArmKey tone="sim" translucent />}>policy in the calibrated sim ({polMode === "shadow" ? "shadow" : "closed loop"})</LegendItem>
                  </>
                )}
                {underlay && <span className="legend-note">{videoNote}</span>}
                {scene && !underlay && <span className="legend-note">table at {(scene.table.z * 1000).toFixed(0)} mm, workspace box from the logs</span>}
                {track && (
                  <LegendItem k={<span className="key arm-key" style={{ ["--key" as string]: `rgb(${track.rgb.join(",")})` }} />}>
                    object from the video{track.source[Math.min(track.source.length - 1, frame)] === "held" ? " (held: fingertip from FK)" : ""}
                  </LegendItem>
                )}
              </div>
            </div>

            {(dsState === "loading" || (!baseline && !error)) && (
              <div className="overlay">
                {dsState === "loading" ? (
                  currentDs?.loaded ? (
                    <span>
                      <Spinner /> Loading episodes…
                    </span>
                  ) : (
                    <div className="loading">
                      <div>
                        <Spinner /> Downloading the dataset from the Hugging Face Hub… <span className="num">{loadSeconds}s</span>
                      </div>
                      <div className="section-hint">The first load of a dataset can take a couple of minutes. After that it is cached.</div>
                    </div>
                  )
                ) : (
                  <span>
                    <Spinner /> Replaying episode…
                  </span>
                )}
              </div>
            )}
          </main>

          <aside className="inspector" aria-label="Inspector">
            <Tabs<Tab>
              idBase="insp"
              ariaLabel="Inspector"
              value={tab}
              onChange={setTab}
              tabs={[
                { id: "calibrate", label: "Calibrate", badge: running ? <Spinner /> : undefined },
                { id: "camera", label: "Camera" },
                { id: "policy", label: "Policy", badge: polBusy ? <Spinner /> : undefined },
                { id: "tools", label: "Tools" },
              ]}
            />
            <div className="inspector-body" ref={inspectorBody}>
              <TabPanel idBase="insp" id="calibrate" active={tab === "calibrate"} className="tab-panel">
                <GapStat
                  gap={ghostGap?.overall_deg ?? null}
                  stockGap={baseGap?.overall_deg ?? null}
                  episode={episode}
                  role={role}
                  tone={tone}
                  live={ghost === LIVE && running}
                  busy={replayBusy && ghost !== LIVE}
                />
                <PhysicsPicker
                  rows={physicsRows}
                  stockHeldOut={stockHeldOut}
                  stockValue={deg(stockHeldOut)}
                  live={!!calib && (running || ghost === LIVE)}
                  liveValue={deg(lastProgress?.best_score)}
                  liveMeta={liveMeta}
                  manual={ghost === MANUAL && !!tuneBase}
                  manualValue={deg(ghost === MANUAL ? ghostGap?.overall_deg : null)}
                  manualMeta={<span className="muted">from {tuneFrom}</span>}
                  selected={ghost}
                  onSelect={setGhost}
                  running={running}
                  onDelete={deleteLock}
                  showAll={showAllLocks}
                  onShowAll={setShowAllLocks}
                />
                <JointBars
                  jointNames={jointNames}
                  scored={scored}
                  stock={baseGap?.per_joint_deg}
                  selected={ghostGap?.per_joint_deg}
                  tone={tone}
                  physicsLabel={physicsLabel}
                  joint={jointNames[jIndex] ?? joint}
                  onJoint={setJoint}
                />
                <TuneSection
                  active={ghost === MANUAL}
                  base={tuneBase}
                  tuned={tuned}
                  adj={tuneAdj}
                  onAdj={setTuneAdj}
                  from={tuneFrom}
                  fromLock={tuneLock ? lockLabel(tuneLock, dataset) : null}
                  startLabel={physicsLabel}
                  blocker={running ? "Stop the calibration to tune by hand." : !baseline ? "Waiting for the episode to load." : null}
                  onStart={startTuning}
                  onReplay={() => {
                    seek(0);
                    setPlaying(true);
                  }}
                  saveName={saveName}
                  onSaveName={setSaveName}
                  nameValid={validLockName(saveName.trim())}
                  saving={saving}
                  saveError={saveError}
                  onSave={saveTuned}
                />
                <CalibrateSection
                  calib={calib}
                  running={running}
                  blocker={calibBlocker}
                  onStart={startCalibration}
                  onStop={stopCalibration}
                  opts={opts}
                  onOpts={setOpts}
                  savedLabel={savedLabel}
                  partial={savedPartial}
                  onDismissError={() => setCalib(null)}
                />
                <ParamsDisclosure stock={defaultParams} shown={shownParams} label={physicsLabel} fullName={isLock ? ghost : null} />
              </TabPanel>

              <TabPanel idBase="insp" id="camera" active={tab === "camera"} className="tab-panel">
                <CameraView
                  dataset={dataset}
                  cameras={cameras}
                  episode={episode}
                  frame={frame}
                  playing={playing}
                  active={tab === "camera"}
                  lock={cameraLock}
                  lockLabel={cameraLockLabel}
                  borrowed={!isLock && !!cameraLock}
                  altLock={altCamLock ? { name: altCamLock.name, label: lockLabel(altCamLock.name, dataset) } : null}
                  canSwitch={!running}
                  onSwitchLock={setGhost}
                  onGoCalibrate={() => setTab("calibrate")}
                  busy={running}
                  scene={scene}
                  onSceneChange={setScene}
                  objects={objects}
                  onObjectsChange={setObjects}
                  track={track}
                  onFitsChange={onFitsChange}
                  onError={setError}
                  onFitSaved={() => setFitsRev((r) => r + 1)}
                  fitsRev={video.fitsRev}
                />
              </TabPanel>

              <TabPanel idBase="insp" id="policy" active={tab === "policy"} className="tab-panel">
                <PolicyTab
                  lock={policyLock}
                  lockLabel={cameraLockLabel}
                  borrowed={!isLock && !!policyLock}
                  policy={policy}
                  job={polJob}
                  running={running}
                  ready={dsState === "ready"}
                  mode={polMode}
                  onMode={setPolMode}
                  onTrain={startPolicyTraining}
                  onEvaluate={startPolicyEval}
                  results={polResults}
                  showGhosts={showPolicy}
                  onShowGhosts={setShowPolicy}
                  episode={episode}
                  onEpisode={selectEpisode}
                  viewer={polViewer}
                  viewerNote={polViewerNote}
                  viewerBusy={polViewerBusy}
                  onGoCalibrate={() => setTab("calibrate")}
                  onDismissError={() => setPolJob(null)}
                />
              </TabPanel>

              <TabPanel idBase="insp" id="tools" active={tab === "tools"} className="tab-panel">
                <ToolsTab
                  showComp={showComp}
                  onShowComp={setShowComp}
                  comp={comp}
                  compBusy={compBusy}
                  compBlocked={
                    ghost === LIVE
                      ? "Available once the run has saved its lock: it needs fixed physics, not the live fit."
                      : ghost === MANUAL
                        ? "Available once your parameters are saved as a lock: it needs saved physics."
                        : null
                  }
                  physicsLabel={physicsLabel}
                  onOpenCross={() => setCrossOpen(true)}
                  nLoaded={loadedIds.length}
                  nLocks={visibleLocks.length}
                  exportLock={isLock ? ghost : null}
                  exportLabel={physicsLabel}
                  exportUrl={isLock ? api.exportZipUrl(ghost) : null}
                />
              </TabPanel>
            </div>
          </aside>

          {crossOpen && (
            <CrossModal
              onClose={closeCross}
              data={crossData}
              busy={crossBusy}
              error={crossError}
              onRun={runCross}
              datasets={loadedIds}
              locks={lockNames}
              home={lockHome}
            />
          )}

          <Timeline
            playing={playing}
            onPlayPause={() => setPlaying((p) => !p)}
            speed={speed}
            onSpeed={setSpeed}
            episodes={episodes}
            episode={episode}
            onEpisode={selectEpisode}
            roleOf={roleFor}
            frame={frame}
            frames={frames}
            fps={fps}
            joint={jointNames[jIndex] ?? joint}
            jointNames={jointNames}
            onJoint={setJoint}
            traceReal={traceReal}
            traceSim={traceSim}
            simTone={tone}
            simLabel={physicsLabel}
            residualSum={residualSum}
            events={residual?.events ?? []}
            grasps={residual?.grasp_windows ?? []}
            hasData={!!baseline}
            onSeek={seek}
          />
        </>
      )}
      <ToastHost toasts={toasts} onRemove={removeToast} />
    </div>
  );
}
