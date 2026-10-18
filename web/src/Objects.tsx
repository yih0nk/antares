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

type Props = { track: ObjectTrack | null; playhead: { frame: number } };

const DEFAULT_FOOTPRINT: [number, number] = [0.03, 0.03];
const HIDDEN_OPACITY = 0.3;

export function ObjectMarker({ track, playhead }: Props) {
  const solid = useRef<THREE.Mesh>(null);
  const outline = useRef<THREE.LineSegments>(null);
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
    const geom = new THREE.BoxGeometry(...size);
    const e = new THREE.EdgesGeometry(geom);
    geom.dispose();
    return e;
  }, [size]);
  const edgeColour = useMemo(() => colour.clone().multiplyScalar(0.45), [colour]);

  useFrame(() => {
    const s = solid.current;
    const o = outline.current;
    if (!track || !track.xyz.length || !s || !o) return;
    const f = Math.min(track.xyz.length - 1, Math.max(0, Math.floor(playhead.frame)));
    const p = track.xyz[f];
    const src = track.source[f];
    s.position.set(p[0], p[1], p[2]);
    o.position.set(p[0], p[1], p[2]);
    material.opacity = src === "hidden" ? HIDDEN_OPACITY : 1;
    material.depthWrite = src !== "hidden";
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
