import { useMemo } from "react";
import * as THREE from "three";
import type { SceneFit } from "./api";
import { palette } from "./theme";

// The static scene a lock's scene fit recovered from the logs, drawn in the
// 3D viewer (MuJoCo z-up frame; the caller applies the y-up rotation): the
// table as a thin dark plane at the fitted height, and the fingertip
// workspace as a faint dashed grey box (chrome, not a sim, so not blue).

// Same margin antares.scene uses around the workspace for the table geom.
const TABLE_MARGIN = 0.1;
const TABLE_THICKNESS = 0.006;

type Props = { scene: SceneFit | null };

export function SceneGeometry({ scene }: Props) {
  const table = useMemo(() => {
    if (!scene) return null;
    const { min, max } = scene.workspace;
    const sx = max[0] - min[0] + 2 * TABLE_MARGIN;
    const sy = max[1] - min[1] + 2 * TABLE_MARGIN;
    const cx = (min[0] + max[0]) / 2;
    const cy = (min[1] + max[1]) / 2;
    return { size: [sx, sy, TABLE_THICKNESS] as [number, number, number], pos: [cx, cy, scene.table.z - TABLE_THICKNESS / 2] as [number, number, number] };
  }, [scene]);
  const box = useMemo(() => {
    if (!scene) return null;
    const { min, max } = scene.workspace;
    const geom = new THREE.BoxGeometry(max[0] - min[0], max[1] - min[1], max[2] - min[2]);
    const edges = new THREE.EdgesGeometry(geom);
    geom.dispose();
    return { edges, pos: [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2] as [number, number, number] };
  }, [scene]);
  if (!scene || !table || !box) return null;
  return (
    <group>
      <mesh position={table.pos} receiveShadow>
        <boxGeometry args={table.size} />
        <meshStandardMaterial color={palette.raised} roughness={0.9} metalness={0} />
      </mesh>
      {/* a faint dashed guide, so the workspace never reads as an object */}
      <lineSegments geometry={box.edges} position={box.pos} ref={(l) => l?.computeLineDistances()}>
        <lineDashedMaterial color={palette.textMuted} transparent opacity={0.2} dashSize={0.01} gapSize={0.008} />
      </lineSegments>
    </group>
  );
}
