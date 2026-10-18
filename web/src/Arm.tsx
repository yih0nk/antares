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
  // Identifies the episode; new poses for the same clip ease in (the snap),
  // a different clip just appears.
  clip: string;
  playhead: Playhead;
  material: THREE.Material;
  snapSeconds?: number;
  // Half-open [start, end) frame spans where the arm runs warm: while the
  // playhead is inside one, the material's emissive eases toward `warm`.
  // Only meaningful for a MeshStandardMaterial.
  hotSpans?: number[][] | null;
  // Hidden by the viewer's real/sim toggle; keeps the trajectory loaded.
  hidden?: boolean;
};

const WARM = new THREE.Color("#ff8a5b");
const WARM_INTENSITY = 0.9;

// MuJoCo quaternions are [w, x, y, z].
const toQuat = (q: number[]) => new THREE.Quaternion(q[1], q[2], q[3], q[0]);

const easeOut = (t: number) => 1 - Math.pow(1 - t, 3);

export function Arm({ geometry, poses, clip, playhead, material, snapSeconds = 0.6, hotSpans = null, hidden = false }: Props) {
  const files = useMemo(() => [...new Set(geometry.geoms.map((g) => g.mesh))], [geometry]);
  const loaded = useLoader(
    STLLoader,
    files.map((f) => api.meshUrl(f, geometry.robot)),
  );
  const meshes = useMemo(() => new Map(files.map((f, i) => [f, loaded[i]])), [files, loaded]);

  const bodyRefs = useRef<(THREE.Group | null)[]>([]);
  const prev = useRef<number[][][] | null>(null);
  const shown = useRef<number[][][] | null>(null);
  const tween = useRef({ start: 0, active: false });
  const shownClip = useRef<string | null>(null);

  useEffect(() => {
    if (!poses) return;
    // Snap from whatever was on screen, but only within the same episode; a
    // new episode (or the very first trajectory) just appears.
    const sameClip = shownClip.current === clip;
    shownClip.current = clip;
    prev.current = shown.current;
    shown.current = poses;
    tween.current = { start: performance.now(), active: sameClip && prev.current !== null };
  }, [poses, clip]);

  const tmp = useMemo(
    () => ({ p0: new THREE.Vector3(), p1: new THREE.Vector3(), q0: new THREE.Quaternion(), q1: new THREE.Quaternion() }),
    [],
  );

  // The material's resting emissive, captured once so heat can ease back to it.
  const std = material instanceof THREE.MeshStandardMaterial ? material : null;
  const rest = useMemo(
    () => (std ? { emissive: std.emissive.clone(), intensity: std.emissiveIntensity } : null),
    [std],
  );
  const heat = useRef(0);
  const spans = useRef<number[][] | null>(null);
  spans.current = hotSpans;

  useFrame((_, dt) => {
    const cur = shown.current;
    if (!cur || !cur.length) return;
    const f = Math.min(cur.length - 1, Math.max(0, Math.floor(playhead.frame)));
    if (std && rest) {
      const s = spans.current;
      let hot = false;
      if (s) for (let i = 0; i < s.length && !hot; i++) hot = f >= s[i][0] && f < s[i][1];
      const target = hot ? 1 : 0;
      const k = Math.min(1, dt * 8);
      heat.current += (target - heat.current) * k;
      if (Math.abs(heat.current - target) < 0.005) heat.current = target;
      std.emissive.lerpColors(rest.emissive, WARM, heat.current);
      std.emissiveIntensity = rest.intensity + (WARM_INTENSITY - rest.intensity) * heat.current;
    }
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
    <group visible={!!poses && !hidden}>
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

// The sim driven by lag-compensated commands: where the arm would go if it
// were sent what the calibrated model says it needs.
export const compMaterial = new THREE.MeshStandardMaterial({
  color: "#3fd6b8",
  emissive: "#1c8f78",
  emissiveIntensity: 0.3,
  roughness: 0.4,
  transparent: true,
  opacity: 0.3,
  depthWrite: false,
});

// The policy driving the stock sim (grey) and the calibrated sim (green):
// where each simulator says the same policy would take the arm.
export const policyDefaultMaterial = new THREE.MeshStandardMaterial({
  color: "#aeb4c2",
  emissive: "#555b68",
  emissiveIntensity: 0.25,
  roughness: 0.5,
  transparent: true,
  opacity: 0.32,
  depthWrite: false,
});

export const policyLockMaterial = new THREE.MeshStandardMaterial({
  color: "#6fe07a",
  emissive: "#2f9a3c",
  emissiveIntensity: 0.3,
  roughness: 0.4,
  transparent: true,
  opacity: 0.34,
  depthWrite: false,
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
