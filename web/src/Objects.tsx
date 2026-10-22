import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import type { ObjectTrack } from "./api";

// The manipulated object in the 3D viewer: a small box with the vision
// footprint, coloured like the object, following the tracked position at
// the playhead. Read every render-loop tick from the shared mutable
// playhead, like the arms, so playback never goes through React state.
// Always drawn solid so it reads as a thing, not as more chrome:
//  - seen by the camera on the table ("vision"): opaque;
//  - between the jaws ("held", position from FK): opaque with a dark
//    outline, so it stays visible inside the translucent ghost;
//  - out of sight ("hidden", last position carried forward): faded.
// Held, its centre sits at the FK fingertip, as on the table at the grasp
// frame (the track is continuous there), so the lower half of the box
// shows below the jaw tips: the fingers close around the object's middle.
//
// The raw track is per-frame vision, so it is cleaned once per track before
// drawing: hidden runs shorter than MIN_HIDDEN_FRAMES (the arm passing in
// front for a moment) keep the state around them, and vision positions are
// averaged over a short centred window. Both depend only on the frame, so
// scrubbing shows the same picture as playback.

type Props = { track: ObjectTrack | null; playhead: { frame: number } };

const DEFAULT_FOOTPRINT: [number, number] = [0.03, 0.03];
const HIDDEN_OPACITY = 0.3;
const MIN_HIDDEN_FRAMES = 8;
const SMOOTH_HALF_WINDOW = 3;
// The outline sits just outside the faces so it never z-fights with them.
const OUTLINE_SCALE = 1.04;

type Source = "vision" | "held" | "hidden";

function cleanTrack(track: ObjectTrack): { xyz: number[][]; source: Source[] } {
  const n = track.xyz.length;
  const source = track.source.slice(0, n) as Source[];
  // Short hidden runs inherit the state before them (or after, at the start).
  for (let i = 0; i < n; ) {
    if (source[i] !== "hidden") {
      i++;
      continue;
    }
    let j = i;
    while (j < n && source[j] === "hidden") j++;
    const fill = i > 0 ? source[i - 1] : j < n ? source[j] : "hidden";
    if (j - i < MIN_HIDDEN_FRAMES && fill !== "hidden") for (let k = i; k < j; k++) source[k] = fill;
    i = j;
  }
  // Average vision frames over a centred window of vision frames only, so
  // the box never blends across a grasp or a release.
  const xyz = track.xyz.map((p, i) => {
    if (track.source[i] !== "vision") return p;
    const sum = [0, 0, 0];
    let m = 0;
    for (let k = Math.max(0, i - SMOOTH_HALF_WINDOW); k <= Math.min(n - 1, i + SMOOTH_HALF_WINDOW); k++) {
      if (track.source[k] !== "vision") continue;
      sum[0] += track.xyz[k][0];
      sum[1] += track.xyz[k][1];
      sum[2] += track.xyz[k][2];
      m++;
    }
    return m ? sum.map((v) => v / m) : p;
  });
  return { xyz, source };
}

export function ObjectMarker({ track, playhead }: Props) {
  const solid = useRef<THREE.Mesh>(null);
  const outline = useRef<THREE.LineSegments>(null);
  const clean = useMemo(() => (track && track.xyz.length ? cleanTrack(track) : null), [track]);
  const size = useMemo<[number, number, number]>(() => {
    const fp = track?.footprint_m ?? DEFAULT_FOOTPRINT;
    const h = track?.half_height ?? 0.015;
    return [fp[0], fp[1], 2 * h];
  }, [track]);
  const colour = useMemo(() => {
    const [r, g, b] = track?.rgb ?? [220, 80, 70];
    return new THREE.Color(r / 255, g / 255, b / 255);
  }, [track]);
  // A slight self-glow keeps the object's colour readable on the dark table
  // and inside the translucent ghost; the outline is the same hue, darker.
  const material = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: colour,
        emissive: colour.clone().multiplyScalar(0.35),
        roughness: 0.55,
        metalness: 0,
        transparent: true,
      }),
    [colour],
  );
  const edges = useMemo(() => {
    const geom = new THREE.BoxGeometry(...size.map((v) => v * OUTLINE_SCALE) as [number, number, number]);
    const e = new THREE.EdgesGeometry(geom);
    geom.dispose();
    return e;
  }, [size]);
  const edgeColour = useMemo(() => colour.clone().multiplyScalar(0.45), [colour]);
  const opacity = useRef(1);

  useFrame((_, dt) => {
    const s = solid.current;
    const o = outline.current;
    if (!clean || !s || !o) return;
    const f = Math.min(clean.xyz.length - 1, Math.max(0, Math.floor(playhead.frame)));
    const p = clean.xyz[f];
    const src = clean.source[f];
    s.position.set(p[0], p[1], p[2]);
    o.position.set(p[0], p[1], p[2]);
    // Ease the fade so a state change reads as a change, not a flash.
    const target = src === "hidden" ? HIDDEN_OPACITY : 1;
    opacity.current += (target - opacity.current) * Math.min(1, dt * 10);
    material.opacity = opacity.current;
    material.depthWrite = opacity.current > 0.95;
    o.visible = src === "held";
  });

  if (!track) return null;
  return (
    <group>
      <mesh ref={solid} material={material} castShadow renderOrder={2}>
        <boxGeometry args={size} />
      </mesh>
      <lineSegments ref={outline} geometry={edges} renderOrder={3}>
        <lineBasicMaterial color={edgeColour} />
      </lineSegments>
    </group>
  );
}
