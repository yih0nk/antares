import { useEffect, useMemo, useRef } from "react";
import { useFrame, useLoader } from "@react-three/fiber";
import * as THREE from "three";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader.js";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { api, type Geometry } from "./api";
import { palette, prefersReducedMotion, type SimTone } from "./theme";

// An OBJ file's meshes as one geometry (Menagerie models ship OBJ as well as STL).
function objGeometry(group: THREE.Group): THREE.BufferGeometry {
  const parts: THREE.BufferGeometry[] = [];
  group.traverse((o) => {
    if (o instanceof THREE.Mesh) parts.push((o.geometry as THREE.BufferGeometry).clone().toNonIndexed());
  });
  if (parts.length === 1) return parts[0];
  // mergeGeometries needs every part to carry the same attributes: keep
  // position and normal, and give a part without normals computed ones.
  for (const p of parts) {
    for (const k of Object.keys(p.attributes)) if (k !== "position" && k !== "normal") p.deleteAttribute(k);
    if (!p.getAttribute("normal")) p.computeVertexNormals();
  }
  let merged = mergeGeometries(parts);
  if (!merged) {
    // Still mismatched (e.g. normal item sizes differ): merge positions alone.
    for (const p of parts) p.deleteAttribute("normal");
    merged = mergeGeometries(parts);
    merged?.computeVertexNormals();
  }
  return merged ?? new THREE.BufferGeometry();
}

const isObj = (f: string) => f.toLowerCase().endsWith(".obj");

// One arm drawn from the API's geometry: a group per MuJoCo body, posed
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
  // Half-open [start, end, kind?) frame spans where the arm glows: while the
  // playhead is inside one, the material eases toward the span's status
  // colour (kind 0 or absent: contact, critical; kind 1: anomaly, warn).
  // Only meaningful for a MeshStandardMaterial.
  hotSpans?: number[][] | null;
  // Hidden by the viewer's real/sim toggle; keeps the trajectory loaded.
  hidden?: boolean;
};

// Glow where the sim disagrees with the arm: contact (critical) or anomaly (warn).
const GLOW = [new THREE.Color(palette.critical), new THREE.Color(palette.warn)];
const WARM_INTENSITY = 0.25;
const WARM_TINT = 0.25;

// MuJoCo quaternions are [w, x, y, z].
const toQuat = (q: number[]) => new THREE.Quaternion(q[1], q[2], q[3], q[0]);

const easeOut = (t: number) => 1 - Math.pow(1 - t, 3);

export function Arm({ geometry, poses, clip, playhead, material, snapSeconds = 0.6, hotSpans = null, hidden = false }: Props) {
  // prefers-reduced-motion: no tween onto a new fit, no eased glow.
  const still = useMemo(prefersReducedMotion, []);
  const files = useMemo(() => [...new Set(geometry.geoms.map((g) => g.mesh))], [geometry]);
  const stlFiles = useMemo(() => files.filter((f) => !isObj(f)), [files]);
  const objFiles = useMemo(() => files.filter(isObj), [files]);
  const stl = useLoader(
    STLLoader,
    stlFiles.map((f) => api.meshUrl(f, geometry.robot)),
  );
  const obj = useLoader(
    OBJLoader,
    objFiles.map((f) => api.meshUrl(f, geometry.robot)),
  );
  const meshes = useMemo(
    () =>
      new Map<string, THREE.BufferGeometry>([
        ...stlFiles.map((f, i) => [f, stl[i]] as const),
        ...objFiles.map((f, i) => [f, objGeometry(obj[i])] as const),
      ]),
    [stlFiles, objFiles, stl, obj],
  );

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
    tween.current = { start: performance.now(), active: !still && snapSeconds > 0 && sameClip && prev.current !== null };
  }, [poses, clip, still, snapSeconds]);

  const tmp = useMemo(
    () => ({ p0: new THREE.Vector3(), p1: new THREE.Vector3(), q0: new THREE.Quaternion(), q1: new THREE.Quaternion() }),
    [],
  );

  // The material's resting emissive, captured once so heat can ease back to it.
  const std = material instanceof THREE.MeshStandardMaterial ? material : null;
  // Read from userData when the material recorded it at creation: a material
  // the viewer swaps out mid-glow (stock -> calibrated) would otherwise hand a
  // warm emissive to the next Arm that captures it.
  const rest = useMemo(
    () =>
      std
        ? ((std.userData.rest as { color: THREE.Color; emissive: THREE.Color; intensity: number } | undefined) ?? {
            color: std.color.clone(),
            emissive: std.emissive.clone(),
            intensity: std.emissiveIntensity,
          })
        : null,
    [std],
  );
  const heat = useRef(0);
  const glow = useRef(GLOW[0]);
  const spans = useRef<number[][] | null>(null);
  spans.current = hotSpans;

  useFrame((_, dt) => {
    const cur = shown.current;
    if (!cur || !cur.length) return;
    const f = Math.min(cur.length - 1, Math.max(0, Math.floor(playhead.frame)));
    if (std && rest) {
      const s = spans.current;
      let hot = false;
      if (s)
        for (let i = 0; i < s.length && !hot; i++) {
          hot = f >= s[i][0] && f < s[i][1];
          if (hot) glow.current = GLOW[s[i][2] === 1 ? 1 : 0];
        }
      const target = hot ? 1 : 0;
      const k = still ? 1 : Math.min(1, dt * 8);
      heat.current += (target - heat.current) * k;
      if (Math.abs(heat.current - target) < 0.005) heat.current = target;
      // The base colour warms too: a red emissive over a blue arm alone would
      // read as magenta, the lag-compensated arm's colour.
      std.color.lerpColors(rest.color, glow.current, heat.current * WARM_TINT);
      std.emissive.lerpColors(rest.emissive, glow.current, heat.current);
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
              scale={(g.scale ?? [1, 1, 1]) as [number, number, number]}
              castShadow
              receiveShadow
            />
          ))}
        </group>
      ))}
    </group>
  );
}

// Materials. Colours come from theme.ts; the encoding rules (web/DESIGN.md):
// the real arm is solid and opaque, every sim arm is translucent, and a sim's
// colour says which physics it runs on (stock grey or calibrated blue).

export const realMaterial = new THREE.MeshStandardMaterial({
  color: palette.real,
  roughness: 0.55,
  metalness: 0.1,
});

function translucent(hex: string, opacity: number): THREE.MeshStandardMaterial {
  const color = new THREE.Color(hex);
  const m = new THREE.MeshStandardMaterial({
    color,
    emissive: color.clone().multiplyScalar(0.45),
    emissiveIntensity: 0.35,
    roughness: 0.4,
    transparent: true,
    opacity,
    depthWrite: false,
  });
  m.userData.rest = { color: m.color.clone(), emissive: m.emissive.clone(), intensity: m.emissiveIntensity };
  return m;
}

// The replay ghost. A second pane gets its own copies (the hot-span glow
// animates the material, which would otherwise light both panes).
export const makeGhostMaterial = (tone: SimTone) => translucent(palette[tone], tone === "sim" ? 0.4 : 0.36);
export const ghostMaterials: Record<SimTone, THREE.MeshStandardMaterial> = {
  stock: makeGhostMaterial("stock"),
  sim: makeGhostMaterial("sim"),
};

// The sim driven by lag-compensated commands: where the arm would go if it
// were sent what the calibrated model says it needs.
export const compMaterial = translucent(palette.comp, 0.32);

// The policy driving the stock sim and the calibrated sim: where each
// simulator says the same policy would take the arm. Same colours as the
// replay ghost's physics, so the viewer hides that ghost while these show.
export const policyDefaultMaterial = translucent(palette.stock, 0.34);
export const policyLockMaterial = translucent(palette.sim, 0.36);
