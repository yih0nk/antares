import { useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  type CameraFit,
  type CameraInfo,
  type CameraJobEvent,
  type CameraOverlay,
  type CameraProgressEvent,
  type SceneFit,
  type SceneView,
} from "./api";
import { Sparkline } from "./Charts";

// The dataset's fixed cameras, fitted to the robot's own logs: pick a
// camera, fit it (a job streamed over SSE), and see the sim arm's outline
// drawn over the real frame at the playhead. The overlay is fetched when
// the playhead pauses or a scrub settles, never per frame while playing.
// The Scene block below it recovers the table and workspace from the logs
// and swaps the overlay for the sim's own view of the same instant.

type Props = {
  dataset: string;
  episode: number | null;
  frame: number;
  playing: boolean;
  // Lock the camera fits belong to (the ghost's lock, or the dataset's default).
  lock: string | null;
  // Another job (calibration) is running; fits are disabled meanwhile.
  busy: boolean;
  // The scene fit saved for `lock`, owned by the app (the 3D viewer draws it too).
  scene: SceneFit | null;
  onSceneChange: (scene: SceneFit | null) => void;
  onError: (message: string) => void;
};

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
  if (!ev) return "starting";
  if (ev.stage === "sample") return `sampled ${ev.stage_evaluation} poses`;
  if (ev.stage === "start") return `start ${ev.start + 1} / ${ev.n_starts}`;
  if (ev.stage === "refine") return "refining at full resolution";
  return "scoring held-out frames";
};

const mm = (v: number | null | undefined) => (v == null || !isFinite(v) ? "--" : `${(v * 1000).toFixed(1)} mm`);
const cm = (v: number) => `${(v * 100).toFixed(0)}`;

export function CameraView({ dataset, episode, frame, playing, lock, busy, scene, onSceneChange, onError }: Props) {
  const [cameras, setCameras] = useState<CameraInfo[]>([]);
  const [camera, setCamera] = useState<string>("");
  const [fits, setFits] = useState<Record<string, CameraFit>>({});
  const [job, setJob] = useState<Job | null>(null);
  const [overlay, setOverlay] = useState<CameraOverlay | null>(null);
  const [overlayBusy, setOverlayBusy] = useState(false);
  const [overlayError, setOverlayError] = useState<string | null>(null);
  const [mode, setMode] = useState<ViewMode>("overlay");
  const [view, setView] = useState<SceneView | null>(null);
  const [viewBusy, setViewBusy] = useState(false);
  const [viewError, setViewError] = useState<string | null>(null);
  const [sceneBusy, setSceneBusy] = useState(false);
  const sourceRef = useRef<EventSource | null>(null);
  useEffect(() => () => sourceRef.current?.close(), []);

  // ---- cameras in the dataset's info -----------------------------------------
  useEffect(() => {
    if (!dataset) return;
    let live = true;
    setCameras([]);
    setOverlay(null);
    api.cameras(dataset).then(
      (cs) => {
        if (!live) return;
        setCameras(cs);
        setCamera((cur) => (cs.some((c) => c.name === cur) ? cur : cs.find((c) => c.fixed)?.name ?? ""));
      },
      () => live && setCameras([]),
    );
    return () => {
      live = false;
    };
  }, [dataset]);

  // ---- fits saved for the lock ------------------------------------------------
  const refreshFits = (name: string | null) => {
    if (!name) {
      setFits({});
      return;
    }
    api.cameraFits(name).then((r) => setFits(r.cameras), () => setFits({}));
  };
  useEffect(() => refreshFits(lock), [lock]);

  // ---- overlay at the playhead, only while paused ---------------------------
  const running = job?.status === "running" || job?.status === "cancelled";
  const fit = camera ? fits[camera] : undefined;
  const info = cameras.find((c) => c.name === camera);
  useEffect(() => {
    if (!dataset || episode == null || !camera || !info?.fixed || playing) return;
    let live = true;
    const id = setTimeout(() => {
      setOverlayBusy(true);
      setOverlayError(null);
      api
        .cameraOverlay(dataset, camera, episode, frame, fit ? lock : null)
        .then(
          (o) => live && setOverlay(o),
          (e) => live && setOverlayError((e as Error).message),
        )
        .finally(() => live && setOverlayBusy(false));
    }, 180);
    return () => {
      live = false;
      clearTimeout(id);
    };
  }, [dataset, episode, camera, frame, playing, lock, fit, info?.fixed]);

  // ---- sim view at the playhead: needs the lock's scene fit with this camera's backdrop ----
  const backdrop = scene && scene.lock === lock && camera ? scene.backdrops[camera] : undefined;
  const wantView = mode !== "overlay" && !!backdrop && !!fit;
  useEffect(() => {
    if (!wantView || !dataset || episode == null || !camera || !lock || playing) return;
    let live = true;
    const id = setTimeout(() => {
      setViewBusy(true);
      setViewError(null);
      api
        .sceneView(dataset, camera, episode, frame, lock)
        .then(
          (v) => live && setView(v),
          (e) => live && setViewError((e as Error).message),
        )
        .finally(() => live && setViewBusy(false));
    }, 180);
    return () => {
      live = false;
      clearTimeout(id);
    };
  }, [wantView, dataset, episode, camera, frame, playing, lock, scene?.created_at]);

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

  // ---- fit job ------------------------------------------------------------------
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
    src.onmessage = (msg) => {
      const ev = JSON.parse(msg.data) as CameraJobEvent;
      if (ev.type === "progress") {
        setJob((j) => (j && j.jobId === jobId ? { ...j, last: ev, best: [...j.best, ev.best_iou] } : j));
      } else if (ev.type === "done") {
        src.close();
        setJob((j) => (j && j.jobId === jobId ? { ...j, status: "done" } : j));
        refreshFits(lock);
      } else if (ev.type === "error") {
        src.close();
        setJob((j) => (j && j.jobId === jobId ? { ...j, status: "error", error: ev.message } : j));
      }
    };
    src.onerror = () => {
      if (src.readyState === EventSource.CLOSED) {
        setJob((j) => (j && j.jobId === jobId && j.status === "running" ? { ...j, status: "error", error: "lost the progress stream" } : j));
      }
    };
  };

  const stopFit = async () => {
    if (!job) return;
    await api.cancelJob(job.jobId).catch(() => {});
    setJob((j) => (j ? { ...j, status: j.status === "running" ? "cancelled" : j.status } : j));
  };

  const polygons = useMemo(
    () => (overlay?.sim_polygons ?? []).map((p) => p.map(([x, y]) => `${x},${y}`).join(" ")),
    [overlay],
  );
  const realPolys = useMemo(
    () => (overlay?.real_mask_polygon ?? []).map((p) => p.map(([x, y]) => `${x},${y}`).join(" ")),
    [overlay],
  );
  const showOverlay = overlay && overlay.camera === camera && overlay.dataset === dataset;
  const showView = wantView && view && view.camera === camera && view.dataset === dataset && view.lock === lock;
  const w = overlay?.width ?? info?.width ?? 640;
  const h = overlay?.height ?? info?.height ?? 480;
  const anyFit = Object.keys(fits).length > 0;
  const sceneForLock = scene && scene.lock === lock ? scene : null;

  if (!dataset) return null;

  const overlayPane = (
    <div className="cam-view" style={{ aspectRatio: `${w} / ${h}` }}>
      {showOverlay && <img src={overlay.image} alt={`${camera} camera, episode ${overlay.episode} frame ${overlay.frame}`} />}
      {showOverlay && (
        <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none">
          {realPolys.map((p, i) => (
            <polygon key={`r${i}`} points={p} className="cam-real" vectorEffect="non-scaling-stroke" />
          ))}
          {polygons.map((p, i) => (
            <polygon key={i} points={p} className="cam-sim" vectorEffect="non-scaling-stroke" />
          ))}
        </svg>
      )}
      {!showOverlay && (
        <div className="cam-empty muted small">
          {playing ? "pause to see the frame" : overlayBusy ? "loading frame…" : overlayError ?? "no frame"}
        </div>
      )}
      {showOverlay && overlayBusy && <span className="dot cam-dot" />}
      {showOverlay && playing && <div className="cam-paused muted small">paused frame {overlay.frame}</div>}
      {mode === "both" && showOverlay && <div className="cam-tag muted small">real</div>}
    </div>
  );

  const simPane = (
    <div className="cam-view" style={{ aspectRatio: `${w} / ${h}` }}>
      {showView && <img src={view.sim_image} alt={`sim view of ${camera}, episode ${view.episode} frame ${view.frame}`} />}
      {!showView && (
        <div className="cam-empty muted small">
          {!fit
            ? "fit the camera first"
            : !backdrop
              ? "fit the scene first: the sim view needs this camera's backdrop"
              : playing
                ? "pause to see the sim view"
                : viewBusy
                  ? "compositing…"
                  : viewError ?? "no frame"}
        </div>
      )}
      {showView && viewBusy && <span className="dot cam-dot" />}
      {showView && <div className="cam-tag muted small">sim view</div>}
    </div>
  );

  return (
    <section className="camera">
      <div className="label">Camera</div>
      {cameras.length ? (
        <div className="segmented">
          {cameras.map((c) => (
            <button
              key={c.name}
              className={c.name === camera ? "on" : ""}
              onClick={() => setCamera(c.name)}
              disabled={!c.fixed || running}
              title={c.fixed ? `${c.key} · ${c.width}×${c.height}` : `${c.key} moves with the arm; only fixed cameras can be fitted`}
            >
              {c.name}
              {fits[c.name] ? " ✓" : ""}
            </button>
          ))}
        </div>
      ) : (
        <div className="muted small">no video streams listed for this dataset</div>
      )}

      {camera && (
        <>
          {mode === "overlay" ? overlayPane : mode === "sim" ? simPane : <div className="cam-pair">{overlayPane}{simPane}</div>}
          <div className="row">
            <span className="muted small">
              {fit ? (
                <>
                  fit IoU <span className="mono">{pct(fit.fit_iou)}</span> · held-out{" "}
                  <span className="mono">{pct(fit.heldout_iou)}</span>
                </>
              ) : (
                "not fitted yet: sim outline needs a camera pose"
              )}
            </span>
            {mode === "overlay" && showOverlay && overlay.iou != null && (
              <span className="mono small" title="IoU of the sim outline and the segmented arm on this frame">
                {pct(overlay.iou)}
              </span>
            )}
            {mode !== "overlay" && showView && (
              <span className="mono small" title="IoU of the sim view's arm and the segmented real arm on this frame">
                {pct(view.iou)}
              </span>
            )}
          </div>
          {!running ? (
            <button className="cta secondary" onClick={startFit} disabled={busy || !lock || !info?.fixed} title={lock ? `saved to locks/${lock}.camera.json` : "calibrate first so there is a lock to attach the camera to"}>
              {fit ? "Fit again from logs" : "Fit from logs"}
            </button>
          ) : (
            <button className="cta secondary" onClick={stopFit} disabled={job?.status === "cancelled"}>
              {job?.status === "cancelled" ? "Stopping…" : "Stop and keep best"}
            </button>
          )}
          {job && job.camera === camera && (
            <div className="calib-body">
              <div className="row">
                <span className="muted small">
                  {job.status === "running" && `${stageLabel(job.last)} · ${job.last?.elapsed_s?.toFixed(0) ?? 0}s`}
                  {job.status === "cancelled" && "stopping after this generation"}
                  {job.status === "done" && `saved for lock ${lock}`}
                  {job.status === "error" && <span className="bad">{job.error}</span>}
                </span>
                <span className="mono small">best IoU {pct(job.last?.best_iou)}</span>
              </div>
              <Sparkline values={job.best} total={job.best.length} />
            </div>
          )}
          <div className="muted small">
            <i className="swatch sim" /> sim arm outline (convex hulls) · <i className="swatch warm" /> segmented real arm
          </div>

          <div className="label">Scene</div>
          <div className="segmented" title="Real frame with the sim outline, the sim's own view from this camera, or both">
            {(["overlay", "sim", "both"] as ViewMode[]).map((m) => (
              <button key={m} className={m === mode ? "on" : ""} onClick={() => setMode(m)} disabled={m !== "overlay" && !sceneForLock}>
                {m === "overlay" ? "real + outline" : m === "sim" ? "sim view" : "side by side"}
              </button>
            ))}
          </div>
          {sceneForLock ? (
            <div className="scene-stats">
              <div>
                <span className="muted small">table height</span>
                <div className="mono" title={`fingertips at ${sceneForLock.table.n_on_table} of ${sceneForLock.table.n_grasps} grasps sat at ${mm(sceneForLock.table.z_grasp)}, minus an assumed object half-height of ${mm(sceneForLock.table.object_half_height)}`}>
                  {mm(sceneForLock.table.z)}
                </div>
              </div>
              <div>
                <span className="muted small">tilt</span>
                <div className="mono" title={sceneForLock.table.horizontal ? "assumed horizontal; the fitted plane through the grasp points tilts this much" : "the grasp points disagree with a horizontal table; the fitted plane is used"}>
                  {sceneForLock.table.tilt_deg.toFixed(1)}°{sceneForLock.table.horizontal ? "" : " (used)"}
                </div>
              </div>
              <div>
                <span className="muted small">workspace</span>
                <div className="mono">
                  {cm(sceneForLock.workspace.size[0])}×{cm(sceneForLock.workspace.size[1])}×{cm(sceneForLock.workspace.size[2])} cm
                </div>
              </div>
              <div>
                <span className="muted small">picks / places</span>
                <div className="mono">
                  {sceneForLock.grasp_points.length} / {sceneForLock.release_points.length}
                </div>
              </div>
              {backdrop && (
                <div style={{ gridColumn: "1 / -1" }}>
                  <span className="muted small">table plane vs {camera} image</span>
                  <div className="mono" title="share of the projected table plane that lands on the backdrop's dominant colour region · IoU of the two">
                    {pct(backdrop.plane_in_region)} on the table colour · IoU {pct(backdrop.iou)}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="muted small">
              {anyFit ? "table height and workspace from the grasps in the logs" : "fit a camera first; the scene needs its backdrop"}
            </div>
          )}
          <button
            className="cta secondary"
            onClick={fitScene}
            disabled={busy || sceneBusy || running || !lock || !anyFit}
            title={lock ? `saved to locks/${lock}.scene.json and locks/scene/` : "calibrate first so there is a lock to attach the scene to"}
          >
            {sceneBusy ? "Fitting scene…" : sceneForLock ? "Fit scene again" : "Fit scene from logs"}
          </button>
        </>
      )}
    </section>
  );
}
