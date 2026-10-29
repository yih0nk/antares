import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useThree } from "@react-three/fiber";
import * as THREE from "three";
import { api, followJob, type CameraFit, type CameraInfo, type CameraJobEvent, type CameraProgressEvent, pickCamera } from "./api";
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

// The camera shown (of `cameras`, the dataset's, fetched once by the app) and
// the fits saved for the lock, refetched when the underlay is turned on or
// `fitsRev` changes (a fit saved in the Camera panel). `busy` (a calibration
// running) holds the automatic camera fit back.
export function useUnderlay(
  dataset: string,
  cameras: CameraInfo[],
  episode: number | null,
  lock: string | null,
  on: boolean,
  fitsRev = 0,
  busy = false,
) {
  const [camera, setCamera] = useState("");
  const [fits, setFits] = useState<Record<string, CameraFit>>({});
  // The lock whose fits `fits` holds, once they have arrived.
  const [fitsFor, setFitsFor] = useState<string | null>(null);
  const [datasetFits, setDatasetFits] = useState<Record<string, CameraFit & { lock: string }>>({});
  // The dataset whose fits `datasetFits` holds, once they have arrived.
  const [datasetFitsFor, setDatasetFitsFor] = useState<string | null>(null);
  // The video's lag behind the joint log is a property of the recording, so
  // it is kept per dataset and camera: switching camera (or dataset) starts
  // from 0, and switching back finds the delay set before.
  const [delays, setDelays] = useState<Record<string, number>>({});
  const [lineUp, setLineUp] = useState<LineUp | null>(null);
  const [lineUpRev, setLineUpRev] = useState(0);
  const tried = useRef(new Set<string>());
  const sourceRef = useRef<EventSource | null>(null);
  useEffect(() => () => sourceRef.current?.close(), []);
  const rev = fitsRev + lineUpRev;

  useEffect(() => setCamera((cur) => pickCamera(cameras, cur)), [cameras]);

  const delayKey = `${dataset}|${camera}`;
  const delay = delays[delayKey] ?? 0;
  const setDelay = useCallback(
    (next: number | ((d: number) => number)) =>
      setDelays((ds) => ({ ...ds, [delayKey]: typeof next === "function" ? next(ds[delayKey] ?? 0) : next })),
    [delayKey],
  );

  useEffect(() => {
    if (!on || !lock) {
      setFits({});
      setFitsFor(null);
      return;
    }
    let live = true;
    api.cameraFits(lock).then(
      (r) => {
        if (!live) return;
        setFits(r.cameras);
        setFitsFor(lock);
      },
      () => {
        // A failed fetch still belongs to this lock: no fit, rather than the last lock's.
        if (!live) return;
        setFits({});
        setFitsFor(lock);
      },
    );
    return () => {
      live = false;
    };
  }, [on, lock, rev]);

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
      () => {
        if (!live) return;
        setDatasetFits({});
        setDatasetFitsFor(dataset);
      },
    );
    return () => {
      live = false;
    };
  }, [on, dataset, rev]);

  const info = cameras.find((c) => c.name === camera) ?? null;
  const fixed = !!info?.fixed;
  // The lock's own fit of the camera, whatever dataset id it was saved with:
  // the same choice as the Camera panel, so the two never disagree on
  // whether the camera is fitted.
  // Fits fetched for another lock or dataset (just after a switch) are not used.
  const own = fixed && fitsFor === lock ? fits[camera] : undefined;
  // Memoised so the pinned camera is only re-posed when the fit changes.
  const fit = useMemo(
    () => (own ? { ...own, lock: lock ?? undefined } : fixed && datasetFitsFor === dataset ? datasetFits[camera] ?? null : null),
    [own, lock, fixed, datasetFits, datasetFitsFor, dataset, camera],
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
        followJob<CameraJobEvent>(src, (ev) => {
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
        });
        src.onerror = () => {
          // EventSource reconnects on its own; once it gives up, the server
          // says how the fit ended.
          if (src.readyState !== EventSource.CLOSED) return;
          const lost = (l: LineUp) =>
            l.status === "running" || l.status === "cancelled" ? { ...l, status: "error" as const, error: "lost the progress stream" } : l;
          api.jobStatus(jobId).then(
            (st) => {
              if (st.status === "done") {
                update((l) => ({ ...l, status: "done" }));
                setLineUpRev((r) => r + 1);
              } else if (st.status === "error") {
                update((l) => ({ ...l, status: "error", error: st.error ?? "the fit failed" }));
              } else {
                update(lost);
              }
            },
            () => update(lost),
          );
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

  // Memoised: the viewer re-poses its camera and re-seeks the video when it changes.
  const underlay = useMemo<Underlay | null>(
    () => (on && info && episode != null ? { camera: info, fit, src: api.cameraClipUrl(dataset, info.name, episode), delay } : null),
    [on, info, episode, fit, dataset, delay],
  );
  return {
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
