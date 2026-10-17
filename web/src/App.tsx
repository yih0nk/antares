import { useEffect, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { Grid, OrbitControls } from "@react-three/drei";

// Layout: 3D viewer (real arm + ghost sim arm) on the left, gap panel on
// the right, episode timeline along the bottom. The arm meshes, replay
// poses, and live calibration stream land once antares.sim is in.
export default function App() {
  const [datasets, setDatasets] = useState<string[]>([]);

  useEffect(() => {
    fetch("/api/datasets")
      .then((r) => r.json())
      .then(setDatasets)
      .catch(() => setDatasets([]));
  }, []);

  return (
    <div className="shell">
      <header className="topbar">
        <span className="brand">antares</span>
        <span className="tagline">sim, calibrated to this robot</span>
        <select className="picker" disabled={!datasets.length}>
          {datasets.map((d) => (
            <option key={d}>{d}</option>
          ))}
        </select>
      </header>

      <main className="viewer">
        <Canvas camera={{ position: [0.45, 0.35, 0.45], fov: 40 }}>
          <color attach="background" args={["#0b0d12"]} />
          <ambientLight intensity={0.6} />
          <directionalLight position={[1, 2, 1]} intensity={1.2} />
          <Grid
            infiniteGrid
            cellSize={0.05}
            sectionSize={0.25}
            cellColor="#1c2230"
            sectionColor="#2a3346"
            fadeDistance={3}
          />
          <OrbitControls makeDefault target={[0, 0.1, 0]} />
        </Canvas>
      </main>

      <aside className="panel">
        <div className="label">Sim-to-real gap</div>
        <div className="score">--</div>
        <button className="cta" disabled>
          Calibrate
        </button>
      </aside>

      <footer className="timeline">
        <div className="label">Episode timeline</div>
      </footer>
    </div>
  );
}
