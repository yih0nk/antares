import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import type { ObjectTrack } from "./api";

// The manipulated object in the 3D viewer: a small box with the vision
// footprint, coloured like the object, following the tracked position at
// the playhead. Read every render-loop tick from the shared mutable
// playhead, like the arms, so playback never goes through React state.
// Solid while the camera sees it on the table, a wireframe while it is
// between the jaws (position from FK) or hidden (last position carried
// forward).

type Props = { track: ObjectTrack | null; playhead: { frame: number } };

const DEFAULT_FOOTPRINT: [number, number] = [0.03, 0.03];

export function ObjectMarker({ track, playhead }: Props) {
  const solid = useRef<THREE.Mesh>(null);
  const wire = useRef<THREE.LineSegments>(null);
  const size = useMemo<[number, number, number]>(() => {
    const fp = track?.footprint_m ?? DEFAULT_FOOTPRINT;
    const h = track?.half_height ?? 0.015;
    return [fp[0], fp[1], 2 * h];
  }, [track]);
  const colour = useMemo(() => {
    const [r, g, b] = track?.rgb ?? [220, 80, 70];
    return new THREE.Color(r / 255, g / 255, b / 255);
  }, [track]);
  const edges = useMemo(() => {
    const geom = new THREE.BoxGeometry(...size);
    const e = new THREE.EdgesGeometry(geom);
    geom.dispose();
    return e;
  }, [size]);

  useFrame(() => {
    const s = solid.current;
    const w = wire.current;
    if (!track || !track.xyz.length || !s || !w) return;
    const f = Math.min(track.xyz.length - 1, Math.max(0, Math.floor(playhead.frame)));
    const p = track.xyz[f];
    const seen = track.source[f] === "vision";
    s.position.set(p[0], p[1], p[2]);
    w.position.set(p[0], p[1], p[2]);
    s.visible = seen;
    w.visible = !seen;
  });

  if (!track) return null;
  return (
    <group>
      <mesh ref={solid} castShadow>
        <boxGeometry args={size} />
        <meshStandardMaterial color={colour} roughness={0.6} metalness={0} />
      </mesh>
      <lineSegments ref={wire} geometry={edges}>
        <lineBasicMaterial color={colour} transparent opacity={0.8} />
      </lineSegments>
    </group>
  );
}
