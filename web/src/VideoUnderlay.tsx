import { useEffect, useRef, useState } from "react";
import { useThree } from "@react-three/fiber";
import * as THREE from "three";
import { api, type CameraFit, type CameraInfo } from "./api";
import type { Playhead } from "./Arm";

// The dataset's own video drawn behind the 3D view. With a camera fitted for
// the lock (the Cameras panel), the 3D camera takes the fitted pose and field
// of view, so the real arm and the ghost land on the arm in the footage; with
// no fit (or a wrist camera) the video is a plain backdrop and the view still
// orbits. The video follows the playhead: it plays at the viewer's speed and
// is re-seeked whenever it drifts, and seeks frame by frame while paused.

export type Underlay = {
  camera: CameraInfo;
  // The fitted pose for this camera, when the lock has one for this dataset.
  fit: CameraFit | null;
  src: string;
  // Frames the video lags the joint log (positive: the camera is behind).
  delay: number;
};

// Cameras of the dataset and the fits saved for the lock, refetched when the
// underlay is turned on so a fit made in the Cameras panel is picked up.
export function useUnderlay(dataset: string, episode: number | null, lock: string | null, on: boolean) {
  const [cameras, setCameras] = useState<CameraInfo[]>([]);
  const [camera, setCamera] = useState("");
  const [fits, setFits] = useState<Record<string, CameraFit>>({});
  const [delay, setDelay] = useState(0);

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
  }, [on, lock, camera]);

  const info = cameras.find((c) => c.name === camera) ?? null;
  const fit = info?.fixed ? fits[camera] : undefined;
  const underlay: Underlay | null =
    on && info && episode != null
      ? {
          camera: info,
          fit: fit && (!fit.dataset || fit.dataset === dataset) ? fit : null,
          src: api.cameraClipUrl(dataset, info.name, episode),
          delay,
        }
      : null;
  return { cameras, camera, setCamera, delay, setDelay, underlay };
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
    const [x, y, z] = fit.pos;
    const [w, qx, qy, qz] = fit.quat;
    camera.position.set(x, y, z).applyQuaternion(Z_UP_TO_Y_UP);
    camera.quaternion.copy(Z_UP_TO_Y_UP).multiply(new THREE.Quaternion(qx, qy, qz, w));
    camera.fov = fit.fovy;
    camera.updateProjectionMatrix();
  }, [camera, fit]);
  return null;
}
