import { useEffect, useMemo, useRef, useState } from "react";
import { useThree } from "@react-three/fiber";
import * as THREE from "three";
import { api, type CameraFit, type CameraInfo, type CameraJobEvent, type CameraProgressEvent } from "./api";
import type { Playhead } from "./Arm";

// The dataset's own video drawn behind the 3D view. With a fitted camera (the
// Camera panel), the 3D camera takes the fitted pose and field of view, so the
// real arm and the ghost land on the arm in the footage. A fixed camera with no
// fit anywhere in the dataset is fitted automatically the first time the video
// is shown (about a minute), and the view snaps to it when the fit is saved;
// until then, and for a wrist camera, the video is a plain backdrop and the
// view still orbits. The video follows the playhead: it plays at the viewer's speed and
// is re-seeked whenever it drifts, and seeks frame by frame while paused.

export type Underlay = {
  camera: CameraInfo;
  // The fitted pose for this camera: the lock's own, else the best one saved
  // under any lock of the dataset (the pose belongs to the recording, not to
  // the dynamics), with the lock it was saved under.
  fit: (CameraFit & { lock?: string }) | null;
  src: string;
  // Frames the video lags the joint log (positive: the camera is behind).
  delay: number;
};

// The automatic camera fit that lines the view up with a camera nobody fitted.
export type LineUp = {
  jobId: string | null;
  dataset: string;
  camera: string;
  // "done": saved, until the refetched fits bring it in.
  status: "running" | "done" | "error" | "cancelled";
  last?: CameraProgressEvent;
  // Set during the second pass: the fitted camera whose field of view it holds.
  sibling?: string;
  error?: string;
};

// Cameras of the dataset and the fits saved for the lock, refetched when the
// underlay is turned on or `fitsRev` changes (a fit saved in the Camera panel).
// `busy` (a calibration running) holds the automatic camera fit back.
export function useUnderlay(
  dataset: string,
  episode: number | null,
  lock: string | null,
  on: boolean,
  fitsRev = 0,
  busy = false,
) {
  const [cameras, setCameras] = useState<CameraInfo[]>([]);
  const [camera, setCamera] = useState("");
  const [fits, setFits] = useState<Record<string, CameraFit>>({});
  const [datasetFits, setDatasetFits] = useState<Record<string, CameraFit & { lock: string }>>({});
  // The dataset whose fits `datasetFits` holds, once they have arrived.
  const [datasetFitsFor, setDatasetFitsFor] = useState<string | null>(null);
  const [delay, setDelay] = useState(0);
  const [lineUp, setLineUp] = useState<LineUp | null>(null);
  const [lineUpRev, setLineUpRev] = useState(0);
  const tried = useRef(new Set<string>());
  const sourceRef = useRef<EventSource | null>(null);
  useEffect(() => () => sourceRef.current?.close(), []);
  const rev = fitsRev + lineUpRev;

  useEffect(() => {
    if (!dataset) return;
    let live = true;
    setCameras([]);
    api.cameras(dataset).then(
      (cs) => {
        if (!live) return;
        setCameras(cs);
        setCamera((cur) => (cs.some((c) => c.name === cur) ? cur : (cs.find((c) => c.fixed) ?? cs[0])?.name ?? ""));
      },
      () => live && setCameras([]),
    );
    return () => {
      live = false;
    };
  }, [dataset]);

  useEffect(() => {
    if (!on || !lock) {
      setFits({});
      return;
    }
    let live = true;
    api.cameraFits(lock).then(
      (r) => live && setFits(r.cameras),
      () => live && setFits({}),
    );
    return () => {
      live = false;
    };
  }, [on, lock, camera, rev]);

  useEffect(() => {
    if (!on || !dataset) {
      setDatasetFits({});
      setDatasetFitsFor(null);
      return;
    }
    let live = true;
    api.cameraFitsForDataset(dataset).then(
      (r) => {
        if (!live) return;
        setDatasetFits(r.cameras);
        setDatasetFitsFor(dataset);
      },
      () => live && setDatasetFits({}),
    );
    return () => {
      live = false;
    };
  }, [on, dataset, camera, rev]);

  const info = cameras.find((c) => c.name === camera) ?? null;
  const fixed = !!info?.fixed;
  const own = fixed ? fits[camera] : undefined;
  // Memoised so the pinned camera is only re-posed when the fit changes.
  const fit = useMemo(
    () =>
      own && (!own.dataset || own.dataset === dataset)
        ? { ...own, lock: lock ?? undefined }
        : fixed
          ? datasetFits[camera] ?? null
          : null,
    [own, dataset, lock, fixed, datasetFits, camera],
  );
  // ---- automatic fit for a fixed camera no lock of the dataset has fitted ----
  // Once per dataset and camera per page load, so a stop or a failure is not retried.
  useEffect(() => {
    if (!on || !fixed || fit || !lock || busy || datasetFitsFor !== dataset) return;
    const key = `${dataset}|${camera}`;
    if (tried.current.has(key)) return;
    tried.current.add(key);
    // Not cancelled by a re-render: the job runs to the end either way and
    // saves its fit, so its progress is followed whatever changes meanwhile.
    setLineUp({ jobId: null, dataset, camera, status: "running" });
    api.cameraFit(dataset, camera, lock).then(
      ({ job_id: jobId }) => {
        setLineUp((l) => (l && l.dataset === dataset && l.camera === camera && !l.jobId ? { ...l, jobId } : l));
        sourceRef.current?.close();
        const src = api.jobEvents(jobId);
        sourceRef.current = src;
        const update = (f: (l: LineUp) => LineUp | null) => setLineUp((l) => (l && l.jobId === jobId ? f(l) : l));
        src.onmessage = (msg) => {
          const ev = JSON.parse(msg.data) as CameraJobEvent;
          if (ev.type === "progress") {
            update((l) => ({ ...l, last: ev }));
          } else if (ev.type === "pass") {
            update((l) => ({ ...l, sibling: ev.sibling, last: undefined }));
          } else if (ev.type === "done") {
            src.close();
            update((l) => ({ ...l, status: "done" }));
            setLineUpRev((r) => r + 1);
          } else if (ev.type === "error") {
            src.close();
            update((l) => ({ ...l, status: "error", error: ev.message }));
          }
        };
        src.onerror = () => {
          if (src.readyState === EventSource.CLOSED) {
            update((l) => (l.status === "running" ? { ...l, status: "error", error: "lost the progress stream" } : l));
          }
        };
      },
      (e) => setLineUp({ jobId: null, dataset, camera, status: "error", error: (e as Error).message }),
    );
  }, [on, fixed, fit, lock, busy, datasetFitsFor, dataset, camera]);

  const stopLineUp = () => {
    if (!lineUp?.jobId) return;
    api.cancelJob(lineUp.jobId).catch(() => {});
    setLineUp((l) => (l ? { ...l, status: "cancelled" } : l));
  };

  const underlay: Underlay | null =
    on && info && episode != null
      ? {
          camera: info,
          fit,
          src: api.cameraClipUrl(dataset, info.name, episode),
          delay,
        }
      : null;
  return {
    cameras,
    camera,
    setCamera,
    delay,
    setDelay,
    underlay,
    // Bumped when an automatic fit is saved (the Camera panel refetches its list).
    fitsRev: lineUpRev,
    // The automatic fit of the camera shown, when one is running or ended without a fit.
    lineUp: lineUp && lineUp.dataset === dataset && lineUp.camera === camera && !fit ? lineUp : null,
    stopLineUp,
  };
}

// Re-seek while playing only past this drift; small drift is left alone
// because every seek stalls the video for a moment.
const DRIFT_S = 0.12;

export function VideoBackdrop({
  underlay,
  playhead,
  playing,
  speed,
  onError,
}: {
  underlay: Underlay;
  playhead: Playhead;
  playing: boolean;
  speed: number;
  onError: (message: string | null) => void;
}) {
  const ref = useRef<HTMLVideoElement>(null);
  const { src, delay } = underlay;
  const fps = underlay.camera.fps || 30;

  useEffect(() => onError(null), [src, onError]);

  useEffect(() => {
    const v = ref.current;
    if (!v) return;
    let raf = 0;
    // A new delay, clip or speed seeks once whatever the drift, so a
    // one-frame nudge shows while playing too.
    let resync = true;
    const tick = () => {
      raf = requestAnimationFrame(tick);
      if (v.readyState < 1 || !isFinite(v.duration)) return;
      const end = Math.max(0, v.duration - 0.5 / fps);
      // Frame f of the log is frame f + delay of the video (centre of the frame).
      const raw = (Math.floor(playhead.frame) + delay + 0.5) / fps;
      const t = Math.min(Math.max(0, raw), end);
      // Past either end of the clip (a delay, or a log longer than the video)
      // the video holds its first or last frame: play() on an ended video
      // would restart it from frame 0.
      if (playing && raw > 0 && raw < end && !v.ended) {
        if (v.playbackRate !== speed) v.playbackRate = speed;
        if (v.paused) v.play().catch(() => {});
        if (!v.seeking && (resync || Math.abs(v.currentTime - t) > DRIFT_S * Math.max(1, speed))) {
          v.currentTime = t;
          resync = false;
        }
      } else {
        if (!v.paused) v.pause();
        if (!v.seeking && Math.abs(v.currentTime - t) > 0.5 / fps) v.currentTime = t;
      }
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playhead, playing, speed, fps, delay, src]);

  return (
    <video
      ref={ref}
      className="underlay-video"
      src={src}
      muted
      playsInline
      preload="auto"
      onError={() => onError("Could not load this camera's video for the episode.")}
    />
  );
}

// MuJoCo is z-up and the arm group is turned -90° about x into three's y-up
// frame; the fitted camera (MuJoCo's convention, which is three's: looks down
// -z, y up) gets the same turn.
const Z_UP_TO_Y_UP = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2);

export function MatchCamera({ fit }: { fit: CameraFit }) {
  const camera = useThree((s) => s.camera) as THREE.PerspectiveCamera;
  useEffect(() => {
    const fov = camera.fov;
    const [x, y, z] = fit.pos;
    const [w, qx, qy, qz] = fit.quat;
    camera.position.set(x, y, z).applyQuaternion(Z_UP_TO_Y_UP);
    camera.quaternion.copy(Z_UP_TO_Y_UP).multiply(new THREE.Quaternion(qx, qy, qz, w));
    camera.fov = fit.fovy;
    camera.updateProjectionMatrix();
    // FitCamera and the orbit controls restore the pose, not the field of view.
    return () => {
      camera.fov = fov;
      camera.updateProjectionMatrix();
    };
  }, [camera, fit]);
  return null;
}
