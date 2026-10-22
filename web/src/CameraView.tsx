import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import {
  api,
  type CameraFit,
  type CameraInfo,
  type CameraJobEvent,
  type CameraOverlay,
  type CameraProgressEvent,
  type ObjectsFit,
  type ObjectsJobEvent,
  type ObjectTrack,
  pickCamera,
  type SceneFit,
  type SceneView,
  followJob,
} from "./api";
import { Sparkline } from "./Charts";
import { Badge, Button, EmptyState, InlineNotice, LegendItem, LineKey, ProgressBar, SegmentedControl, Spinner } from "./ui";

// The Camera tab. Camera, scene and object fits are saved per lock, as three
// steps that build on each other: fit a fixed camera to the robot's own logs
// (a job streamed over SSE) and see the sim arm's outline over the real frame
// at the playhead; recover the table and workspace (the scene) from the logs,
// which also gives the sim's own view of the same instant; then find, track
// and weigh the manipulated object. Frames are fetched when the playhead
// pauses or a scrub settles, never per frame while playing, and only while
// this tab is open.

type Props = {
  dataset: string;
  // The dataset's cameras (`useCameras`), fetched once by the app and shared
  // with the video underlay.
  cameras: CameraInfo[];
  episode: number | null;
  frame: number;
  playing: boolean;
  // This tab is the one on screen; frames are fetched only then.
  active: boolean;
  // Lock the fits belong to (the selected physics' lock, or the dataset's newest).
  lock: string | null;
  lockLabel: string;
  // The stock model is selected, so `lock` is borrowed (the dataset's newest lock).
  borrowed: boolean;
  // Another lock of this dataset that has a camera fit, to offer switching to.
  altLock: { name: string; label: string } | null;
  // Switching physics is allowed (not while calibrating).
  canSwitch: boolean;
  onSwitchLock: (name: string) => void;
  onGoCalibrate: () => void;
  // Another job (calibration) is running; fits are paused meanwhile.
  busy: boolean;
  // The scene fit saved for `lock`, owned by the app (the 3D viewer draws it too).
  scene: SceneFit | null;
  onSceneChange: (scene: SceneFit | null) => void;
  // The objects fit saved for `lock` and this episode's per-frame track,
  // both owned by the app (the 3D viewer draws the tracked box too).
  objects: ObjectsFit | null;
  onObjectsChange: (fit: ObjectsFit | null) => void;
  track: ObjectTrack | null;
  // Tells the app which cameras `lock` has fitted (for "switch to a lock with a camera fit").
  onFitsChange: (lock: string, cameras: string[]) => void;
  onError: (message: string) => void;
  // A camera fit finished and was saved (the video underlay refetches the fits).
  onFitSaved?: () => void;
  // Changes when a fit was saved elsewhere (the video underlay's automatic fit).
  fitsRev?: number;
};

type ObjectsJob = {
  jobId: string;
  status: "running" | "done" | "error" | "cancelled";
  i: number;
  n: number;
  stage: string;
  error?: string;
};

const grams = (kg: number | null | undefined) => (kg == null || !isFinite(kg) ? "--" : `${(kg * 1000).toFixed(0)} g`);

type ViewMode = "overlay" | "sim" | "both";

type Job = {
  jobId: string;
  camera: string;
  status: "running" | "done" | "error" | "cancelled";
  last?: CameraProgressEvent;
  best: number[];
  error?: string;
};

const pct = (v: number | null | undefined) => (v == null || !isFinite(v) ? "--" : `${(v * 100).toFixed(0)}%`);

const stageLabel = (ev?: CameraProgressEvent) => {
  if (!ev) return "Starting";
  if (ev.stage === "sample") return `Sampled ${ev.stage_evaluation} poses`;
  if (ev.stage === "start") return `Start ${ev.start + 1} / ${ev.n_starts}`;
  if (ev.stage === "refine") return "Refining at full resolution";
  return "Scoring held-out frames";
};

const mm = (v: number | null | undefined) => (v == null || !isFinite(v) ? "--" : `${(v * 1000).toFixed(1)} mm`);
const cm = (v: number) => `${(v * 100).toFixed(0)}`;

const NO_CAMERAS: CameraInfo[] = [];

function Step({ n, title, needs, children }: { n: number; title: string; needs?: string | null; children: ReactNode }) {
  return (
    <section className="section step">
      <div className="section-head">
        <h2 className="section-title">
          <span className="step-n" aria-hidden="true">
            {n}
          </span>
          {title}
        </h2>
        {needs && <Badge tone="muted">Needs: {needs}</Badge>}
      </div>
      {children}
    </section>
  );
}

// The cameras in the dataset's info, fetched once per dataset for the Camera
// panel and the video underlay (empty while they load or when there are none).
export function useCameras(dataset: string): CameraInfo[] {
  // Kept with the dataset they belong to, so right after a dataset switch the
  // new dataset is never paired with the old one's cameras (and a clip or
  // overlay asked for with a camera it does not have).
  const [got, setGot] = useState<{ dataset: string; cameras: CameraInfo[] } | null>(null);
  useEffect(() => {
    if (!dataset) return;
    let live = true;
    api.cameras(dataset).then(
      (cs) => live && setGot({ dataset, cameras: cs }),
      () => live && setGot({ dataset, cameras: NO_CAMERAS }),
    );
    return () => {
      live = false;
    };
  }, [dataset]);
  return got && got.dataset === dataset ? got.cameras : NO_CAMERAS;
}

export function CameraView({
  dataset,
  cameras,
  episode,
  frame,
  playing,
  active,
  lock,
  lockLabel,
  borrowed,
  altLock,
  canSwitch,
  onSwitchLock,
  onGoCalibrate,
  busy,
  scene,
  onSceneChange,
  objects,
  onObjectsChange,
  track,
  onFitsChange,
  onError,
  onFitSaved,
  fitsRev,
}: Props) {
  const [objectsJob, setObjectsJob] = useState<ObjectsJob | null>(null);
  const objectsSourceRef = useRef<EventSource | null>(null);
  useEffect(() => () => objectsSourceRef.current?.close(), []);
  const [camera, setCamera] = useState<string>("");
  // Fits saved for `fitsLock`; null until the first answer for it.
  const [fits, setFits] = useState<Record<string, CameraFit> | null>(null);
  const [fitsLock, setFitsLock] = useState<string | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [overlay, setOverlay] = useState<CameraOverlay | null>(null);
  const [overlayBusy, setOverlayBusy] = useState(false);
  const [overlayError, setOverlayError] = useState<string | null>(null);
  const [mode, setMode] = useState<ViewMode>("overlay");
  // The sim view and the scene fit (created_at) it was composited for.
  const [view, setView] = useState<{ v: SceneView; stamp: string } | null>(null);
  const [viewBusy, setViewBusy] = useState(false);
  const [viewError, setViewError] = useState<string | null>(null);
  const [sceneBusy, setSceneBusy] = useState(false);
  const sourceRef = useRef<EventSource | null>(null);
  useEffect(() => () => sourceRef.current?.close(), []);
  const frameRef = useRef(frame);
  frameRef.current = frame;

  // ---- the camera shown: only a fixed one can be fitted ----------------------
  useEffect(() => {
    setOverlay(null);
    setCamera((cur) => pickCamera(cameras.filter((c) => c.fixed), cur));
  }, [cameras]);

  // ---- fits saved for the lock ------------------------------------------------
  const onFitsChangeRef = useRef(onFitsChange);
  onFitsChangeRef.current = onFitsChange;
  const lockRef = useRef(lock);
  lockRef.current = lock;
  const refreshFits = (name: string | null) => {
    if (!name) {
      setFits({});
      setFitsLock(null);
      return;
    }
    api.cameraFits(name).then(
      (r) => {
        onFitsChangeRef.current(name, Object.keys(r.cameras));
        if (lockRef.current !== name) return; // an answer for a lock no longer shown
        setFits(r.cameras);
        setFitsLock(name);
      },
      () => {
        if (lockRef.current !== name) return;
        setFits({});
        setFitsLock(name);
      },
    );
  };
  useEffect(() => {
    setFits(null);
    refreshFits(lock);
  }, [lock]);
  // A fit saved elsewhere (the video underlay's line-up): refetch, keeping the current fits meanwhile.
  const firstRev = useRef(true);
  useEffect(() => {
    if (firstRev.current) {
      firstRev.current = false;
      return;
    }
    refreshFits(lockRef.current);
  }, [fitsRev]);

  const fitsReady = fits != null && fitsLock === lock;
  const fit = camera && fitsReady ? fits[camera] : undefined;
  const info = cameras.find((c) => c.name === camera);
  const running = job?.status === "running" || job?.status === "cancelled";

  // ---- overlay at the playhead ----------------------------------------------------
  // The overlay must match what the panel says it shows: the camera, the
  // episode, the lock whose fit draws the outline, and that fit itself (a
  // refit of the same lock has a new created_at). Anything else is stale.
  const wantLock = fit ? lock : null;
  const fitStamp = fit?.created_at ?? null;
  const overlayMatches = (o: CameraOverlay | null): boolean =>
    !!o && o.dataset === dataset && o.camera === camera && o.episode === episode && o.lock === wantLock && (fit ? o.fit?.created_at === fitStamp : true);
  const overlayRef = useRef(overlay);
  overlayRef.current = overlay;

  const fetchOverlay = (f: number, isLive: () => boolean) => {
    if (episode == null) return;
    setOverlayBusy(true);
    setOverlayError(null);
    api
      .cameraOverlay(dataset, camera, episode, f, wantLock)
      .then(
        (o) => isLive() && setOverlay(o),
        (e) => isLive() && setOverlayError((e as Error).message),
      )
      .finally(() => isLive() && setOverlayBusy(false));
  };

  // Paused: follow the playhead, once a scrub settles.
  useEffect(() => {
    if (!active || !dataset || episode == null || !camera || !info?.fixed || playing || !fitsReady) return;
    let live = true;
    const id = setTimeout(() => fetchOverlay(frame, () => live), 180);
    return () => {
      live = false;
      clearTimeout(id);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, dataset, episode, camera, frame, playing, wantLock, fitStamp, info?.fixed, fitsReady]);

  // Playing: when the lock, the fit, the camera or the episode changes, fetch
  // once (at the frame last shown, or the playhead) so the picture never
  // belongs to another fit than the numbers next to it.
  useEffect(() => {
    if (!active || !dataset || episode == null || !camera || !info?.fixed || !playing || !fitsReady) return;
    if (overlayMatches(overlayRef.current)) return;
    let live = true;
    const prev = overlayRef.current;
    fetchOverlay(prev && prev.episode === episode && prev.dataset === dataset ? prev.frame : frameRef.current, () => live);
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, dataset, episode, camera, playing, wantLock, fitStamp, info?.fixed, fitsReady]);

  // ---- sim view at the playhead: needs the lock's scene fit with this camera's backdrop ----
  const backdrop = scene && scene.lock === lock && camera ? scene.backdrops[camera] : undefined;
  const wantView = mode !== "overlay" && !!backdrop && !!fit;
  const sceneStamp = scene?.created_at ?? "";
  const viewMatches = (x: { v: SceneView; stamp: string } | null) =>
    !!x && x.v.camera === camera && x.v.dataset === dataset && x.v.lock === lock && x.v.episode === episode && x.stamp === sceneStamp;
  const viewRef = useRef(view);
  viewRef.current = view;
  const fetchView = (f: number, isLive: () => boolean) => {
    if (episode == null || !lock) return;
    const stamp = sceneStamp;
    setViewBusy(true);
    setViewError(null);
    api
      .sceneView(dataset, camera, episode, f, lock)
      .then(
        (v) => isLive() && setView({ v, stamp }),
        (e) => isLive() && setViewError((e as Error).message),
      )
      .finally(() => isLive() && setViewBusy(false));
  };
  useEffect(() => {
    if (!active || !wantView || !dataset || episode == null || !camera || !lock || playing) return;
    let live = true;
    const id = setTimeout(() => fetchView(frame, () => live), 180);
    return () => {
      live = false;
      clearTimeout(id);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, wantView, dataset, episode, camera, frame, playing, lock, sceneStamp]);
  useEffect(() => {
    if (!active || !wantView || !dataset || episode == null || !camera || !lock || !playing) return;
    if (viewMatches(viewRef.current)) return;
    let live = true;
    const prev = viewRef.current;
    fetchView(prev && prev.v.episode === episode ? prev.v.frame : frameRef.current, () => live);
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, wantView, dataset, episode, camera, playing, lock, sceneStamp]);

  // ---- scene fit (synchronous: FK over the grasp instants + the backdrops) ----
  const fitScene = async () => {
    if (!dataset || !lock) return;
    setSceneBusy(true);
    try {
      const s = await api.sceneFit(dataset, lock);
      onSceneChange(s);
      setView(null);
    } catch (e) {
      onError(`Scene fit failed: ${(e as Error).message}`);
    } finally {
      setSceneBusy(false);
    }
  };

  // ---- camera fit job ----------------------------------------------------------------
  const startFit = async () => {
    if (!dataset || !camera || !lock) return;
    let jobId: string;
    try {
      ({ job_id: jobId } = await api.cameraFit(dataset, camera, lock));
    } catch (e) {
      onError(`Could not start the camera fit: ${(e as Error).message}`);
      return;
    }
    setJob({ jobId, camera, status: "running", best: [] });
    sourceRef.current?.close();
    const src = api.jobEvents(jobId);
    sourceRef.current = src;
    followJob<CameraJobEvent>(src, (ev) => {
      if (ev.type === "progress") {
        setJob((j) => (j && j.jobId === jobId ? { ...j, last: ev, best: [...j.best, ev.best_iou] } : j));
      } else if (ev.type === "done") {
        src.close();
        setJob((j) => (j && j.jobId === jobId ? { ...j, status: "done" } : j));
        refreshFits(lock);
        onFitSaved?.();
      } else if (ev.type === "error") {
        src.close();
        setJob((j) => (j && j.jobId === jobId ? { ...j, status: "error", error: ev.message } : j));
      }
    });
    src.onerror = () => {
      if (src.readyState === EventSource.CLOSED) {
        setJob((j) => (j && j.jobId === jobId && j.status === "running" ? { ...j, status: "error", error: "lost the progress stream" } : j));
      }
    };
  };

  const stopFit = async () => {
    if (!job) return;
    try {
      await api.cancelJob(job.jobId);
    } catch (e) {
      onError(`Could not stop the camera fit: ${(e as Error).message}`);
      return;
    }
    setJob((j) => (j ? { ...j, status: j.status === "running" ? "cancelled" : j.status } : j));
  };

  // ---- objects fit job: track the object through every episode -------------
  const startObjectsFit = async () => {
    if (!dataset || !lock) return;
    let jobId: string;
    try {
      ({ job_id: jobId } = await api.objectsFit(dataset, lock, fit ? camera : undefined));
    } catch (e) {
      onError(`Could not start the object fit: ${(e as Error).message}`);
      return;
    }
    setObjectsJob({ jobId, status: "running", i: 0, n: 0, stage: "starting" });
    objectsSourceRef.current?.close();
    const src = api.jobEvents(jobId);
    objectsSourceRef.current = src;
    followJob<ObjectsJobEvent>(src, (ev) => {
      if (ev.type === "progress") {
        setObjectsJob((j) =>
          j && j.jobId === jobId
            ? {
                ...j,
                stage: ev.stage,
                i: "i" in ev ? ev.i : j.i,
                n: "n" in ev ? ev.n : j.n,
              }
            : j,
        );
      } else if (ev.type === "done") {
        src.close();
        setObjectsJob((j) => (j && j.jobId === jobId ? { ...j, status: "done" } : j));
        onObjectsChange(ev.fit);
      } else if (ev.type === "error") {
        src.close();
        setObjectsJob((j) => (j && j.jobId === jobId ? { ...j, status: "error", error: ev.message } : j));
      }
    });
    src.onerror = () => {
      if (src.readyState === EventSource.CLOSED) {
        setObjectsJob((j) => (j && j.jobId === jobId && j.status === "running" ? { ...j, status: "error", error: "lost the progress stream" } : j));
      }
    };
  };

  const stopObjectsFit = async () => {
    if (!objectsJob) return;
    try {
      await api.cancelJob(objectsJob.jobId);
    } catch (e) {
      onError(`Could not stop the object fit: ${(e as Error).message}`);
      return;
    }
    setObjectsJob((j) => (j ? { ...j, status: j.status === "running" ? "cancelled" : j.status } : j));
  };

  const shownOverlay = overlay && overlayMatches(overlay) ? overlay : null;
  const polygons = useMemo(() => (shownOverlay?.sim_polygons ?? []).map((p) => p.map(([x, y]) => `${x},${y}`).join(" ")), [shownOverlay]);
  const realPolys = useMemo(() => (shownOverlay?.real_mask_polygon ?? []).map((p) => p.map(([x, y]) => `${x},${y}`).join(" ")), [shownOverlay]);
  const shownView = wantView && viewMatches(view) ? view!.v : null;
  const w = shownOverlay?.width ?? info?.width ?? 640;
  const h = shownOverlay?.height ?? info?.height ?? 480;
  const anyFit = fitsReady && Object.keys(fits).length > 0;
  const sceneForLock = scene && scene.lock === lock ? scene : null;
  const objectsForLock = objects && objects.lock === lock ? objects : null;
  const objectsRunning = objectsJob?.status === "running" || objectsJob?.status === "cancelled";
  // The tracked object's box on the shown frame: solid when the camera saw it
  // there, dashed when the position is the FK fingertip (held) or carried forward.
  const objectBox = useMemo(() => {
    if (!track || !shownOverlay || track.episode !== shownOverlay.episode || track.camera !== camera) return null;
    const f = Math.min(track.xyz.length - 1, Math.max(0, shownOverlay.frame));
    const source = track.source[f];
    const bbox = track.bbox[f];
    const colour = `rgb(${track.rgb.join(",")})`;
    if (source === "vision" && bbox) return { source, x: bbox[0], y: bbox[1], w: bbox[2], h: bbox[3], colour };
    // Not seen: draw the box where the last sighting was, or at the fingertip.
    const c = track.centroid[f];
    const side = Math.sqrt(objectsForLock?.prior.area_px ?? 900);
    return c ? { source, x: c[0] - side / 2, y: c[1] - side / 2, w: side, h: side, colour } : null;
  }, [track, shownOverlay, camera, objectsForLock]);

  if (!dataset) return null;

  if (!lock) {
    return (
      <EmptyState
        title="No lock to save fits with"
        actions={
          <Button variant="primary" onClick={onGoCalibrate}>
            Go to Calibrate
          </Button>
        }
      >
        Camera, scene and object fits are saved per lock. This dataset has none yet: calibrate first.
      </EmptyState>
    );
  }

  const fixedCams = cameras.filter((c) => c.fixed);
  const movingCams = cameras.filter((c) => !c.fixed);
  // One primary button per view: the next step that isn't done yet.
  const next: "camera" | "scene" | "objects" | null = !anyFit ? "camera" : !sceneForLock ? "scene" : !objectsForLock ? "objects" : null;
  const fitBlocker = busy ? "Paused while a calibration runs." : !info?.fixed ? "Pick a fixed camera: wrist cameras move with the arm." : null;
  const emptyText = (playingText: string) =>
    playing ? (overlayBusy ? "Loading frame…" : playingText) : overlayBusy ? "Loading frame…" : overlayError ?? "No frame";

  const overlayPane = (
    <div className="cam-view" style={{ aspectRatio: `${w} / ${h}` }}>
      {shownOverlay && <img src={shownOverlay.image} alt={`${camera} camera, episode ${shownOverlay.episode} frame ${shownOverlay.frame}`} />}
      {shownOverlay && (
        <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" aria-hidden="true">
          {realPolys.map((p, i) => (
            <polygon key={`r${i}`} points={p} className="cam-real" vectorEffect="non-scaling-stroke" />
          ))}
          {polygons.map((p, i) => (
            <polygon key={i} points={p} className="cam-sim" vectorEffect="non-scaling-stroke" />
          ))}
          {objectBox && (
            <g>
              <rect
                x={objectBox.x}
                y={objectBox.y}
                width={objectBox.w}
                height={objectBox.h}
                className={`cam-obj ${objectBox.source}`}
                style={{ stroke: objectBox.colour }}
                vectorEffect="non-scaling-stroke"
              />
              <circle cx={objectBox.x + objectBox.w / 2} cy={objectBox.y + objectBox.h / 2} r={3} className="cam-obj-dot" style={{ fill: objectBox.colour }} />
            </g>
          )}
        </svg>
      )}
      {!shownOverlay && <div className="cam-empty">{emptyText(fit ? "Pause to see the overlay for this fit" : "Pause to see the frame")}</div>}
      {shownOverlay && overlayBusy && (
        <span className="cam-dot">
          <Spinner />
        </span>
      )}
      {shownOverlay && playing && <div className="cam-tag bottom">frame {shownOverlay.frame} · pause to follow the playhead</div>}
      {mode === "both" && shownOverlay && <div className="cam-tag">real</div>}
    </div>
  );

  const simPane = (
    <div className="cam-view" style={{ aspectRatio: `${w} / ${h}` }}>
      {shownView && <img src={shownView.sim_image} alt={`sim view of ${camera}, episode ${shownView.episode} frame ${shownView.frame}`} />}
      {!shownView && (
        <div className="cam-empty">
          {!fit
            ? "Needs: camera fit"
            : !backdrop
              ? "Needs: scene fit with this camera's backdrop"
              : playing
                ? viewBusy
                  ? "Compositing…"
                  : "Pause to see the sim view"
                : viewBusy
                  ? "Compositing…"
                  : viewError ?? "No frame"}
        </div>
      )}
      {shownView && viewBusy && (
        <span className="cam-dot">
          <Spinner />
        </span>
      )}
      {shownView && <div className="cam-tag">sim view</div>}
    </div>
  );

  return (
    <>
      <div className="section-hint lock-line" title={lock}>
        Fits saved with <b>{lockLabel}</b>
        {borrowed && " (the stock model has none of its own, so this is the dataset's newest lock)"}
      </div>

      <Step n={1} title="Camera">
        {cameras.length ? (
          <SegmentedControl
            label="Camera"
            value={camera}
            onChange={setCamera}
            options={cameras.map((c) => ({
              value: c.name,
              label: (
                <>
                  {c.name}
                  {fitsReady && fits[c.name] ? " ✓" : ""}
                </>
              ),
              title: c.fixed ? `${c.key} · ${c.width}×${c.height}${fitsReady && fits[c.name] ? " · fitted" : ""}` : `${c.key} moves with the arm; only fixed cameras can be fitted`,
              disabled: !c.fixed || running,
            }))}
          />
        ) : (
          <div className="section-hint">No video streams listed for this dataset.</div>
        )}
        {movingCams.length > 0 && (
          <div className="section-hint">
            {movingCams.map((c) => c.name).join(", ")} {movingCams.length === 1 ? "moves" : "move"} with the arm and can't be fitted.
          </div>
        )}

        {!fitsReady ? (
          <div className="section-hint">
            <Spinner /> Checking fits saved with {lockLabel}…
          </div>
        ) : !anyFit ? (
          <EmptyState
            title="No camera fit yet"
            actions={
              <>
                {!running ? (
                  <Button variant="primary" onClick={startFit} disabled={!!fitBlocker || !fixedCams.length}>
                    Fit camera from video
                  </Button>
                ) : (
                  <Button variant="primary" busy>
                    Fitting camera…
                  </Button>
                )}
                {altLock && (
                  <Button onClick={() => onSwitchLock(altLock.name)} disabled={!canSwitch} title={canSwitch ? `Select ${altLock.name} in Sim physics` : "Stop the calibration first"}>
                    Switch to {altLock.label}
                  </Button>
                )}
              </>
            }
          >
            <span>
              Camera, scene and object fits are saved per lock. <b>{lockLabel}</b> has none yet.
            </span>
            {fitBlocker && <div className="section-hint">{fitBlocker}</div>}
            {altLock && !canSwitch && <div className="section-hint">Switching locks waits for the calibration to finish.</div>}
          </EmptyState>
        ) : (
          <>
            {!fit && camera && (
              <InlineNotice tone="info">
                The {camera} camera has no fit with {lockLabel}; another camera does ({Object.keys(fits).join(", ")}).
              </InlineNotice>
            )}
            <SegmentedControl<ViewMode>
              label="View"
              value={mode}
              onChange={setMode}
              options={[
                { value: "overlay", label: "Real + outline", title: "The real frame with the sim arm's outline" },
                { value: "sim", label: "Sim view", title: "The sim's own view from this camera (needs the scene fit)" },
                { value: "both", label: "Both", title: "Real frame and sim view next to each other (needs the scene fit)" },
              ]}
            />
            {mode !== "overlay" && !sceneForLock && <div className="section-hint">The sim view needs the scene fit (step 2).</div>}
            {mode === "overlay" ? overlayPane : mode === "sim" ? simPane : <div className="cam-pair">{overlayPane}{simPane}</div>}
            <div className="cam-stats">
              {fit ? (
                <span title="IoU of the sim outline and the segmented real arm, averaged over the fit's frames and over held-out frames">
                  fit <span className="num">{pct(fit.fit_iou)}</span> · held-out <span className="num">{pct(fit.heldout_iou)}</span>
                  {fit.cancelled && (
                    <>
                      {" "}
                      <Badge tone="warn" glyph="◐" title="This camera fit was stopped early: the best pose so far was saved.">
                        partial
                      </Badge>
                    </>
                  )}
                </span>
              ) : (
                <span className="muted">not fitted</span>
              )}
              {mode === "overlay" && shownOverlay && shownOverlay.iou != null && (
                <span title="IoU of the sim outline and the segmented arm on the frame shown">
                  this frame <span className="num">{pct(shownOverlay.iou)}</span>
                </span>
              )}
              {mode !== "overlay" && shownView && (
                <span title="IoU of the sim view's arm and the segmented real arm on the frame shown">
                  this frame <span className="num">{pct(shownView.iou)}</span>
                </span>
              )}
            </div>
            <div className="legend-row">
              <LegendItem k={<LineKey tone="real" />}>segmented real arm</LegendItem>
              <LegendItem k={<LineKey tone="sim" dashed />}>sim arm outline</LegendItem>
            </div>
            {!running ? (
              <Button onClick={startFit} disabled={!!fitBlocker} title={`saved to locks/${lock}.camera.json`}>
                {fit ? `Fit ${camera} camera again` : `Fit ${camera} camera from video`}
              </Button>
            ) : null}
            {fitBlocker && <div className="section-hint">{fitBlocker}</div>}
          </>
        )}
        {job && job.camera === camera && (
          <div className="run">
            <div className="run-head">
              <span>
                {job.status === "running" && `${stageLabel(job.last)} · ${job.last?.elapsed_s?.toFixed(0) ?? 0}s`}
                {job.status === "cancelled" && "Stopping after this generation"}
                {job.status === "done" && `Saved with ${lockLabel}`}
                {job.status === "error" && <span className="tone-critical">{job.error}</span>}
              </span>
              <span className="num">best IoU {pct(job.last?.best_iou)}</span>
            </div>
            {running && <ProgressBar value={null} label="Camera fit in progress" />}
            <Sparkline values={job.best} total={job.best.length} label="Best IoU so far" />
            {running && (
              <Button block busy={job.status === "cancelled"} onClick={stopFit}>
                {job.status === "cancelled" ? "Stopping…" : "Stop and save best so far"}
              </Button>
            )}
          </div>
        )}
      </Step>

      <Step n={2} title="Scene" needs={anyFit ? null : "camera fit"}>
        <div className="section-hint">Table height and workspace from the grasps in the logs; the camera fit gives the backdrop.</div>
        {sceneForLock && (
          <dl className="result">
            <div>
              <dt>Table height</dt>
              <dd
                className="num"
                title={`fingertips at ${sceneForLock.table.n_on_table} of ${sceneForLock.table.n_grasps} grasps sat at ${mm(sceneForLock.table.z_grasp)}, minus an assumed object half-height of ${mm(sceneForLock.table.object_half_height)}`}
              >
                {mm(sceneForLock.table.z)}
              </dd>
            </div>
            <div>
              <dt>Tilt</dt>
              <dd
                className="num"
                title={sceneForLock.table.horizontal ? "assumed horizontal; the fitted plane through the grasp points tilts this much" : "the grasp points disagree with a horizontal table; the fitted plane is used"}
              >
                {sceneForLock.table.tilt_deg.toFixed(1)}°{sceneForLock.table.horizontal ? "" : " (used)"}
              </dd>
            </div>
            <div>
              <dt>Workspace</dt>
              <dd className="num">
                {cm(sceneForLock.workspace.size[0])}×{cm(sceneForLock.workspace.size[1])}×{cm(sceneForLock.workspace.size[2])} cm
              </dd>
            </div>
            <div>
              <dt>Picks / places</dt>
              <dd className="num">
                {sceneForLock.grasp_points.length} / {sceneForLock.release_points.length}
              </dd>
            </div>
            {backdrop && (
              <div className="span2">
                <dt>Table plane vs {camera} image</dt>
                <dd className="num" title="share of the projected table plane that lands on the backdrop's dominant colour region · IoU of the two">
                  {pct(backdrop.plane_in_region)} on the table colour · IoU {pct(backdrop.iou)}
                </dd>
              </div>
            )}
          </dl>
        )}
        {anyFit && (
          <Button
            variant={next === "scene" ? "primary" : "secondary"}
            busy={sceneBusy}
            onClick={fitScene}
            disabled={busy || running}
            title={`saved to locks/${lock}.scene.json and locks/scene/`}
          >
            {sceneBusy ? "Fitting scene…" : sceneForLock ? "Fit scene again" : "Fit scene from logs"}
          </Button>
        )}
        {anyFit && (busy || running) && <div className="section-hint">{busy ? "Paused while a calibration runs." : "Waits for the camera fit to finish."}</div>}
      </Step>

      <Step n={3} title="Objects" needs={!anyFit ? "camera fit" : !sceneForLock ? "scene fit" : null}>
        <div className="section-hint">Find the object in the video, track it through every episode, and weigh it from the residual.</div>
        {objectsForLock && (
          <>
            <div className="section-hint">
              <span className="key arm-key" style={{ ["--key" as string]: `rgb(${objectsForLock.prior.rgb.join(",")})` }} aria-hidden="true" /> {objectsForLock.description} · seen
              in {objectsForLock.camera}, {objectsForLock.n_tracked} of {objectsForLock.n_episodes} episodes
            </div>
            <dl className="result">
              <div>
                <dt>Footprint</dt>
                <dd className="num" title="oriented extent of the object's pixels back-projected onto the table plane, median over episodes (height assumed from the scene fit)">
                  {objectsForLock.footprint_m?.long && objectsForLock.footprint_m?.short
                    ? `${(objectsForLock.footprint_m.long.median * 100).toFixed(1)} × ${(objectsForLock.footprint_m.short.median * 100).toFixed(1)} cm`
                    : "--"}
                </dd>
              </div>
              <div>
                <dt>Vision vs FK</dt>
                <dd
                  className="num"
                  title={`distance between where the camera saw the object before the grasp and where FK puts the fingertip at the grasp, median over episodes${
                    objectsForLock.vision_vs_fk_offset_m ? ` (of which a fixed offset of ${objectsForLock.vision_vs_fk_offset_m.map((v) => (v * 100).toFixed(1)).join(", ")} cm)` : ""
                  }`}
                >
                  {objectsForLock.vision_vs_fk_m ? `${(objectsForLock.vision_vs_fk_m.median * 100).toFixed(1)} cm` : "--"}
                </dd>
              </div>
              <div>
                <dt>Start pose spread</dt>
                <dd className="num" title="std of the vision start poses over episodes (x, y)">
                  {objectsForLock.start ? `${(objectsForLock.start.std[0] * 100).toFixed(1)} × ${(objectsForLock.start.std[1] * 100).toFixed(1)} cm` : "--"}
                </dd>
              </div>
              <div>
                <dt>Mass from the residual</dt>
                <dd className="num" title={objectsForLock.payload?.verdict ?? "payload estimate not run"}>
                  {objectsForLock.payload?.mass_kg != null && objectsForLock.payload.verdict.startsWith("consistent")
                    ? `${grams(objectsForLock.payload.mass_kg)} ± ${grams((objectsForLock.payload.mass_iqr_kg ?? 0) / 2)}`
                    : objectsForLock.payload
                      ? "below the noise"
                      : "--"}
                </dd>
              </div>
            </dl>
            {objectsForLock.payload && !objectsForLock.payload.verdict.startsWith("consistent") && (
              <div className="section-hint">{objectsForLock.payload.verdict}</div>
            )}
            <div className="section-hint">Box on the frame: solid where the camera sees the object, dashed while it is held (fingertip from FK) or hidden.</div>
          </>
        )}
        {sceneForLock &&
          (!objectsRunning ? (
            <Button
              variant={next === "objects" ? "primary" : "secondary"}
              onClick={startObjectsFit}
              disabled={busy || running || sceneBusy}
              title={`saved to locks/${lock}.objects.json (a few minutes: every frame of every episode)`}
            >
              {objectsForLock ? "Track object again" : "Find and track the object"}
            </Button>
          ) : (
            <Button block busy={objectsJob?.status === "cancelled"} onClick={stopObjectsFit}>
              {objectsJob?.status === "cancelled" ? "Stopping…" : "Stop and keep what is tracked"}
            </Button>
          ))}
        {sceneForLock && !objectsRunning && (busy || running || sceneBusy) && (
          <div className="section-hint">{busy ? "Paused while a calibration runs." : running ? "Waits for the camera fit to finish." : "Waits for the scene fit."}</div>
        )}
        {objectsJob && (
          <div className="section-hint">
            {objectsJob.status === "running" && (
              <>
                <Spinner />{" "}
                {objectsJob.stage === "track"
                  ? `Tracking episode ${objectsJob.i} / ${objectsJob.n}`
                  : objectsJob.stage === "payload"
                    ? `Weighing: ${objectsJob.i} / ${objectsJob.n} episodes replayed`
                    : objectsJob.stage === "cross"
                      ? "Cross-camera check"
                      : "Learning the object's colour"}
              </>
            )}
            {objectsJob.status === "cancelled" && "Stopping after this episode"}
            {objectsJob.status === "done" && `Saved with ${lockLabel}`}
            {objectsJob.status === "error" && <span className="tone-critical">{objectsJob.error}</span>}
          </div>
        )}
        {objectsRunning && objectsJob && objectsJob.n > 0 && <ProgressBar value={objectsJob.i / objectsJob.n} label="Object tracking progress" />}
      </Step>
    </>
  );
}
