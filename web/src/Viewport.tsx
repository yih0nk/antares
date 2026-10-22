import { Suspense, useEffect, useRef, type ReactNode } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Grid, OrbitControls } from "@react-three/drei";
import type * as THREE from "three";
import type { Geometry } from "./api";
import { Arm, compMaterial, realMaterial, type Playhead } from "./Arm";
import { palette } from "./theme";
import { MatchCamera, type Underlay } from "./VideoUnderlay";

// One 3D pane: the real arm, the sim ghost and (optionally) the
// lag-compensated sim for one episode, with its own playhead and camera.

export type Show = "both" | "real" | "sim";

// Advances the shared playhead inside the render loop so playback never
// waits on React; the UI copy of the frame is updated when it changes.
function Driver({
  playhead,
  playing,
  fps,
  frames,
  speed,
  onFrame,
}: {
  playhead: Playhead;
  playing: boolean;
  fps: number;
  frames: number;
  speed: number;
  onFrame: (f: number) => void;
}) {
  const last = useRef(-1);
  useFrame((_, dt) => {
    if (frames < 2) return;
    // Cap dt only against long stalls (a backgrounded tab), so slow renders
    // still play back in real time.
    if (playing) playhead.frame = (playhead.frame + Math.min(dt, 0.5) * fps * speed) % frames;
    const f = Math.floor(playhead.frame);
    if (f !== last.current) {
      last.current = f;
      onFrame(f);
    }
  });
  return null;
}

const CAMERA_POSITION: [number, number, number] = [0.62, 0.48, 0.72];
const CAMERA_TARGET: [number, number, number] = [0, 0.16, 0];
// Widest view the default camera frames the whole arm in; narrower panes
// (comparing two episodes) pull the camera back so the arm still fits.
const FRAMED_ASPECT = 1.6;

function FitCamera() {
  const camera = useThree((s) => s.camera);
  const aspect = useThree((s) => s.size.width / Math.max(1, s.size.height));
  // The pull-back factor last applied; a later aspect change rescales the
  // camera's current offset from the target, so an orbit the user made survives
  // a resize or the compare toggle.
  const lastK = useRef<number | null>(null);
  useEffect(() => {
    const k = Math.max(1, FRAMED_ASPECT / aspect);
    const [tx, ty, tz] = CAMERA_TARGET;
    if (lastK.current == null) {
      const [x, y, z] = CAMERA_POSITION;
      camera.position.set(tx + (x - tx) * k, ty + (y - ty) * k, tz + (z - tz) * k);
    } else if (lastK.current !== k) {
      const r = k / lastK.current;
      const { x, y, z } = camera.position;
      camera.position.set(tx + (x - tx) * r, ty + (y - ty) * r, tz + (z - tz) * r);
    }
    lastK.current = k;
  }, [camera, aspect]);
  return null;
}

// A pane that mirrors another pane's playhead: the second episode is a pure
// function of the first one's clock (scaled by frame rate, wrapped to its
// own length), so the two never drift apart.
function Follow({
  lead,
  leadFps,
  playhead,
  fps,
  frames,
}: {
  lead: Playhead;
  leadFps: number;
  playhead: Playhead;
  fps: number;
  frames: number;
}) {
  useFrame(() => {
    if (frames < 2) return;
    playhead.frame = ((lead.frame / Math.max(1, leadFps)) * fps) % frames;
  });
  return null;
}

export function ArmScene({
  geometry,
  realPoses,
  simPoses,
  compPoses,
  hotSpans,
  clip,
  show,
  hideGhost = false,
  playhead,
  playing,
  fps,
  frames,
  speed,
  onFrame,
  ghostMat,
  children,
  follow,
  underlay,
  label,
}: {
  geometry: Geometry | null;
  realPoses: number[][][] | null;
  simPoses: number[][][] | null;
  compPoses: number[][][] | null;
  hotSpans: number[][] | null;
  clip: string;
  show: Show;
  // Hide the replay ghost (while the policy's ghosts take its place).
  hideGhost?: boolean;
  playhead: Playhead;
  playing: boolean;
  fps: number;
  frames: number;
  speed: number;
  onFrame: (f: number) => void;
  // The ghost's material: its colour follows the physics (stock or calibrated).
  ghostMat: THREE.Material;
  // Extra things drawn in the arm's frame (policy ghosts, the fitted scene, the tracked object).
  children?: ReactNode;
  // Follow another pane's playhead (scaled by fps) instead of running its own clock.
  follow?: { lead: Playhead; leadFps: number };
  // The dataset video drawn behind a transparent canvas; a fitted camera pins the view to it.
  underlay?: Underlay | null;
  // Accessible description of the pane.
  label: string;
}) {
  const pinned = underlay?.fit ?? null;
  return (
    <Canvas shadows camera={{ position: CAMERA_POSITION, fov: 38, near: 0.01, far: 20 }} aria-label={label} role="img">
      {pinned ? <MatchCamera fit={pinned} /> : <FitCamera />}
      {!underlay && <color attach="background" args={[palette.bg]} />}
      <ambientLight intensity={0.55} />
      <directionalLight position={[0.8, 1.6, 0.6]} intensity={1.3} castShadow shadow-mapSize={[1024, 1024]} />
      <directionalLight position={[-1, 0.6, -0.8]} intensity={0.35} />
      {!underlay && (
        <Grid infiniteGrid cellSize={0.05} sectionSize={0.25} cellColor={palette.raised} sectionColor={palette.border} fadeDistance={3} />
      )}
      {geometry && (
        <Suspense fallback={null}>
          {/* MuJoCo is z-up; three.js is y-up. Keyed by model so a robot switch swaps the meshes cleanly. */}
          <group key={geometry.robot} rotation={[-Math.PI / 2, 0, 0]}>
            <Arm geometry={geometry} poses={realPoses} clip={clip} playhead={playhead} material={realMaterial} hidden={show === "sim"} />
            <Arm
              geometry={geometry}
              poses={simPoses}
              clip={clip}
              playhead={playhead}
              material={ghostMat}
              hotSpans={hotSpans}
              hidden={show === "real" || hideGhost}
            />
            <Arm geometry={geometry} poses={compPoses} clip={clip} playhead={playhead} material={compMaterial} hidden={show === "real"} />
            {children}
          </group>
        </Suspense>
      )}
      {follow ? (
        <Follow lead={follow.lead} leadFps={follow.leadFps} playhead={playhead} fps={fps} frames={frames} />
      ) : (
        <Driver playhead={playhead} playing={playing} fps={fps} frames={frames} speed={speed} onFrame={onFrame} />
      )}
      {!pinned && <OrbitControls makeDefault target={CAMERA_TARGET} />}
    </Canvas>
  );
}
