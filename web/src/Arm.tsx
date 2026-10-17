import { useEffect, useMemo, useRef } from "react";
import { useFrame, useLoader } from "@react-three/fiber";
import * as THREE from "three";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import { api, type Geometry } from "./api";

// One SO-101 drawn from the API's geometry: a group per MuJoCo body, posed
// every frame from (T, nbody, 7) world poses. When `poses` changes the arm
// eases from the old trajectory to the new one, which is the ghost snapping
// onto the real arm as calibration finds better parameters.

export type Playhead = { frame: number };

type Props = {
  geometry: Geometry;
  poses: number[][][] | null;
  playhead: Playhead;
  material: THREE.Material;
  snapSeconds?: number;
};

// MuJoCo quaternions are [w, x, y, z].
const toQuat = (q: number[]) => new THREE.Quaternion(q[1], q[2], q[3], q[0]);

const easeOut = (t: number) => 1 - Math.pow(1 - t, 3);

export function Arm({ geometry, poses, playhead, material, snapSeconds = 0.6 }: Props) {
  const files = useMemo(() => [...new Set(geometry.geoms.map((g) => g.mesh))], [geometry]);
  const loaded = useLoader(STLLoader, files.map(api.meshUrl));
  const meshes = useMemo(() => new Map(files.map((f, i) => [f, loaded[i]])), [files, loaded]);

  const bodyRefs = useRef<(THREE.Group | null)[]>([]);
  const prev = useRef<number[][][] | null>(null);
  const shown = useRef<number[][][] | null>(null);
  const tween = useRef({ start: 0, active: false });

  useEffect(() => {
    if (!poses) return;
    // Snap from whatever was on screen; the very first trajectory just appears.
    prev.current = shown.current;
    shown.current = poses;
    tween.current = { start: performance.now(), active: prev.current !== null };
  }, [poses]);

  const tmp = useMemo(
    () => ({ p0: new THREE.Vector3(), p1: new THREE.Vector3(), q0: new THREE.Quaternion(), q1: new THREE.Quaternion() }),
    [],
  );

  useFrame(() => {
    const cur = shown.current;
    if (!cur || !cur.length) return;
    const f = Math.min(cur.length - 1, Math.max(0, Math.floor(playhead.frame)));
    let alpha = 1;
    const tw = tween.current;
    if (tw.active && prev.current) {
      const t = (performance.now() - tw.start) / 1000 / snapSeconds;
      if (t >= 1) tw.active = false;
      else alpha = easeOut(t);
    }
    const old = tw.active && prev.current ? prev.current : null;
    const fo = old ? Math.min(old.length - 1, f) : 0;
    for (let b = 0; b < cur[f].length; b++) {
      const g = bodyRefs.current[b];
      if (!g) continue;
      const p = cur[f][b];
      tmp.p1.set(p[0], p[1], p[2]);
      tmp.q1.set(p[4], p[5], p[6], p[3]);
      if (old && old[fo]?.[b]) {
        const o = old[fo][b];
        tmp.p0.set(o[0], o[1], o[2]);
        tmp.q0.set(o[4], o[5], o[6], o[3]);
        tmp.p1.lerpVectors(tmp.p0, tmp.p1, alpha);
        tmp.q1.slerpQuaternions(tmp.q0, tmp.q1, alpha);
      }
      g.position.copy(tmp.p1);
      g.quaternion.copy(tmp.q1);
    }
  });

  const geomsByBody = useMemo(() => {
    const by = new Map<number, Geometry["geoms"]>();
    for (const g of geometry.geoms) by.set(g.body, [...(by.get(g.body) ?? []), g]);
    return by;
  }, [geometry]);

  return (
    <group visible={!!poses}>
      {geometry.bodies.map((body) => (
        <group key={body.id} ref={(el) => (bodyRefs.current[body.id] = el)}>
          {(geomsByBody.get(body.id) ?? []).map((g, i) => (
            <mesh
              key={i}
              geometry={meshes.get(g.mesh)}
              material={material}
              position={g.pos as [number, number, number]}
              quaternion={toQuat(g.quat)}
              castShadow
              receiveShadow
            />
          ))}
        </group>
      ))}
    </group>
  );
}

export const realMaterial = new THREE.MeshStandardMaterial({
  color: "#d9dce4",
  roughness: 0.55,
  metalness: 0.1,
});

export const ghostMaterial = new THREE.MeshStandardMaterial({
  color: "#5b8cff",
  emissive: "#2b4dbf",
  emissiveIntensity: 0.35,
  roughness: 0.4,
  transparent: true,
  opacity: 0.38,
  depthWrite: false,
});
