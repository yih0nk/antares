// Small SVG charts for the panel and timeline. They size to their container
// width through viewBox + preserveAspectRatio="none" and keep strokes crisp
// with vector-effect.

type TraceProps = {
  real: number[];
  sim: number[];
  frame: number;
  onSeek?: (frame: number) => void;
};

function path(values: number[], n: number, lo: number, hi: number, w: number, h: number) {
  const span = hi - lo || 1;
  let d = "";
  for (let i = 0; i < values.length; i++) {
    const x = (i / Math.max(1, n - 1)) * w;
    const y = h - ((values[i] - lo) / span) * h;
    d += `${i ? "L" : "M"}${x.toFixed(2)},${y.toFixed(2)}`;
  }
  return d;
}

export function JointTrace({ real, sim, frame, onSeek }: TraceProps) {
  const W = 1000;
  const H = 100;
  const all = real.concat(sim);
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const pad = (hi - lo) * 0.08 || 1;
  const n = Math.max(real.length, sim.length);
  const x = (frame / Math.max(1, n - 1)) * W;

  const seek = (e: React.PointerEvent<SVGSVGElement>) => {
    if (!onSeek || (e.type === "pointermove" && e.buttons !== 1)) return;
    const r = e.currentTarget.getBoundingClientRect();
    const t = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
    onSeek(t * (n - 1));
  };

  return (
    <svg
      className="trace"
      viewBox={`0 0 ${W} ${H}`}
      preserveAspectRatio="none"
      onPointerDown={seek}
      onPointerMove={seek}
    >
      <path d={path(real, n, lo - pad, hi + pad, W, H)} className="trace-real" vectorEffect="non-scaling-stroke" />
      <path d={path(sim, n, lo - pad, hi + pad, W, H)} className="trace-sim" vectorEffect="non-scaling-stroke" />
      <line x1={x} x2={x} y1={0} y2={H} className="trace-head" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

type SparkProps = { values: number[]; total: number; baseline?: number | null };

// Best-so-far error over evaluations, on a fixed x range so the line grows.
export function Sparkline({ values, total, baseline }: SparkProps) {
  const W = 280;
  const H = 64;
  if (!values.length) return <svg className="spark" viewBox={`0 0 ${W} ${H}`} />;
  const hi = Math.max(baseline ?? 0, ...values);
  const lo = 0;
  const n = Math.max(total, values.length);
  const y = (v: number) => H - 2 - ((v - lo) / (hi - lo || 1)) * (H - 4);
  const d = values.map((v, i) => `${i ? "L" : "M"}${((i / Math.max(1, n - 1)) * W).toFixed(1)},${y(v).toFixed(1)}`).join("");
  const lastX = ((values.length - 1) / Math.max(1, n - 1)) * W;
  return (
    <svg className="spark" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
      {baseline != null && (
        <line x1={0} x2={W} y1={y(baseline)} y2={y(baseline)} className="spark-base" vectorEffect="non-scaling-stroke" />
      )}
      <path d={`${d}L${lastX.toFixed(1)},${H}L0,${H}Z`} className="spark-fill" />
      <path d={d} className="spark-line" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}
